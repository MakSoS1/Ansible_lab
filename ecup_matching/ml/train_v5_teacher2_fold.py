from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
from .data_subset import select_items_by_ids
from .features import normalize_items
from .run_v5_pretrained_biencoder import development_rows_and_folds
from .train_v1 import attach_pair_category
from .train_v2_structured import prefilter_weak_candidates_parquet
from .train_v5_teacher_fold import teacher_accumulation_should_step,teacher_trainable_layer_indices,_find_bert_layers
from .v5_contrastive_data import select_fold_contrastive_pairs
from .v5_evaluation import macro_ap_report
from .v5_item_text import serialize_item_v5
from .v5_teacher2_objective import source_loss_weights,torch_category_ranking_loss
from .v5_validation import manifest_sha256
from .v5_weak_specialists import forbidden_weak_item_ids
from .weak_labels import prepare_weak_pairs,remove_human_conflicts,sample_weak_training

def train_fold(*,human_items_path:Path,full_items_path:Path,matches_path:Path,weak_matches_path:Path,manifest_path:Path,base_oof_path:Path,model_dir:Path,output_dir:Path,expected_split_sha:str,held_fold:int,device:str='mps',weak_presample_rows:int=180000,weak_final_rows:int=100000,physical_batch_size:int=8,effective_batch_size:int=32,max_length:int=128,last_n_layers:int=4,max_steps:int=800,learning_rate:float=2e-5,ranking_weight:float=.25,seed:int=2026)->dict:
 import torch
 from torch.utils.data import DataLoader,Dataset,Sampler
 from transformers import AutoModelForSequenceClassification,AutoTokenizer
 started=time.perf_counter();output_dir.mkdir(parents=True,exist_ok=True);manifest=json.loads(manifest_path.read_text())
 if manifest_sha256(manifest)!=expected_split_sha:raise ValueError('split SHA mismatch')
 matches=pd.read_parquet(matches_path,columns=['id1','id2','target']);rows,folds=development_rows_and_folds(manifest,total_rows=len(matches));dev=matches.iloc[rows].reset_index(drop=True);human_items=pd.read_parquet(human_items_path,columns=['id','name','attributes','category']);dev=attach_pair_category(dev,human_items)
 base=pd.read_parquet(base_oof_path,columns=['row_index','score']).sort_values('row_index');
 if base.row_index.astype(np.int64).tolist()!=rows.tolist():raise ValueError('base OOF mismatch')
 base_scores=base.score.to_numpy(float);human_train=select_fold_contrastive_pairs(dev,folds,base_scores,held_fold=held_fold,max_negative_to_positive=2.0,hard_negative_fraction=.5,seed=seed);human_train=human_train[['id1','id2','target','category']].copy();human_train['source']='human';human_train['weak_weight']=1.0
 held_mask=folds==held_fold;held=dev.loc[held_mask].reset_index(drop=True);held_rows=rows[held_mask]
 forbidden=forbidden_weak_item_ids(matches,manifest,held_fold=held_fold);weak,weak_input=prefilter_weak_candidates_parquet(weak_matches_path,validation_item_ids=forbidden,max_presample_rows=weak_presample_rows,seed=seed+held_fold);weak,prep=prepare_weak_pairs(weak[['id1','id2','target']]);weak,conflicts=remove_human_conflicts(weak,human_train[['id1','id2','target']]);weak_ids=set(weak.id1)|set(weak.id2);weak_items=select_items_by_ids(full_items_path,weak_ids,include_attributes=True);weak=attach_pair_category(weak,weak_items);weak=sample_weak_training(weak,max_rows=weak_final_rows,seed=seed+held_fold);final_weak_ids=set(weak.id1)|set(weak.id2)
 if final_weak_ids & forbidden:raise RuntimeError('held/gold item leaked into weak teacher')
 weak=weak[['id1','id2','target','category','weak_weight']].copy();weak['source']='weak';pairs=pd.concat([human_train,weak],ignore_index=True)
 needed=set(pairs.id1)|set(pairs.id2)|set(held.id1)|set(held.id2);human_subset=human_items[human_items.id.isin(needed)].copy();missing=needed-set(human_subset.id);extra=select_items_by_ids(full_items_path,missing,include_attributes=True) if missing else human_subset.iloc[:0].copy();items=pd.concat([human_subset,extra],ignore_index=True).drop_duplicates('id',keep='first');cache=normalize_items(items);texts={i:f'[CAT] {n.category}\n'+serialize_item_v5(n,max_chars=850) for i,n in cache.items()}
 if not needed<=set(texts):raise RuntimeError('missing teacher texts')
 cat_names=sorted(pairs.category.astype(str).unique());cat_to_id={c:i for i,c in enumerate(cat_names)};pairs['_cat_id']=pairs.category.astype(str).map(cat_to_id).astype(int);held_cat=held.category.astype(str).map(cat_to_id).fillna(-1).astype(int)
 tokenizer=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True);model=AutoModelForSequenceClassification.from_pretrained(str(model_dir),local_files_only=True,num_labels=1,ignore_mismatched_sizes=True).to(device)
 for p in model.parameters():p.requires_grad=False
 layers=_find_bert_layers(model)
 for li in teacher_trainable_layer_indices(len(layers),last_n=last_n_layers):
  for p in layers[li].parameters():p.requires_grad=True
 for name,p in model.named_parameters():
  if name.startswith('classifier') or '.classifier.' in name or 'pooler' in name or name.endswith('LayerNorm.weight') or name.endswith('LayerNorm.bias'):p.requires_grad=True
 trainable=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(trainable,lr=learning_rate,weight_decay=.01);acc_steps=int(np.ceil(effective_batch_size/physical_batch_size))
 class DS(Dataset):
  def __init__(self,f):self.f=f.reset_index(drop=True)
  def __len__(self):return len(self.f)
  def __getitem__(self,i):
   r=self.f.iloc[i];return r.id1,r.id2,float(r.target),str(r.source),float(r.weak_weight),int(r._cat_id) if '_cat_id' in self.f else -1
 class CatBatch(Sampler):
  def __init__(self,f,batch,seed):self.batches=[];rng=np.random.default_rng(seed)
  def __iter__(self):
   batches=[]
   for _,g in self.f.groupby('_cat_id',sort=True):
    idx=g.index.to_numpy();rng.shuffle(idx);batches.extend([idx[s:s+physical_batch_size].tolist() for s in range(0,len(idx),physical_batch_size)])
   rng.shuffle(batches);return iter(batches)
  def __len__(self):return int(np.ceil(len(self.f)/physical_batch_size))
 def collate(batch):
  a,b,y,src,conf,cat=zip(*batch);tok=tokenizer([texts[x] for x in a],[texts[x] for x in b],padding=True,truncation=True,max_length=max_length,return_tensors='pt');w=source_loss_weights(np.asarray(src),np.asarray(conf),human_weight=1.0,weak_scale=.1);return tok,torch.tensor(y,dtype=torch.float32),torch.tensor(w,dtype=torch.float32),torch.tensor(cat,dtype=torch.long)
 loader=DataLoader(DS(pairs),batch_sampler=CatBatch(pairs,physical_batch_size,seed+held_fold),num_workers=0,collate_fn=collate);model.train();opt.zero_grad(set_to_none=True);steps=0;losses=[]
 while steps<max_steps:
  accumulated=0
  for bi,(tok,y,w,cat) in enumerate(loader):
   tok={k:v.to(device) for k,v in tok.items()};y=y.to(device);w=w.to(device);cat=cat.to(device);logit=model(**tok).logits.squeeze(-1);bce=torch.nn.functional.binary_cross_entropy_with_logits(logit,y,reduction='none');bce=(bce*w).sum()/w.sum().clamp_min(1e-6);rank=torch_category_ranking_loss(logit,y,cat);loss=bce+ranking_weight*rank;(loss/acc_steps).backward();losses.append(float(loss.detach().cpu()));accumulated+=1
   if teacher_accumulation_should_step(accumulated,accumulation_steps=acc_steps,is_last_microbatch=bi==len(loader)-1):torch.nn.utils.clip_grad_norm_(trainable,1.0);opt.step();opt.zero_grad(set_to_none=True);steps+=1;accumulated=0
   if steps>=max_steps:break
 model.eval();held2=held.copy();held2['_cat_id']=held_cat;pred=[]
 with torch.no_grad():
  for s in range(0,len(held2),48):
   q=held2.iloc[s:s+48];tok=tokenizer([texts[x] for x in q.id1],[texts[x] for x in q.id2],padding=True,truncation=True,max_length=max_length,return_tensors='pt');tok={k:v.to(device) for k,v in tok.items()};pred.append(torch.sigmoid(model(**tok).logits.squeeze(-1)).cpu().numpy())
 score=np.concatenate(pred);report=macro_ap_report(held,score);pd.DataFrame({'row_index':held_rows,'fold':held_fold,'teacher2_score':score}).sort_values('row_index').to_parquet(output_dir/f'v5g-teacher2-fold-{held_fold}-oof.parquet',index=False);payload={'version':'v5g-field-aware-weak-ranking-teacher','held_fold':held_fold,'gold_metric_opened':False,'gold_rows_used':0,'human_rows':len(human_train),'weak_rows':len(weak),'weak_input_rows':int(weak_input),'weak_prepare':prep,'weak_conflicts':conflicts,'steps':steps,'macro_average_precision':report['macro_average_precision'],'per_category_ap':report['per_category_ap'],'elapsed_seconds':time.perf_counter()-started};(output_dir/f'v5g-teacher2-fold-{held_fold}-metrics.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True));return payload

def main():
 p=argparse.ArgumentParser();
 for n in ('human-items','full-items','matches','weak-matches','manifest','base-oof','model-dir','output-dir'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--expected-split-sha',required=True);p.add_argument('--held-fold',type=int,required=True);a=p.parse_args();print(json.dumps(train_fold(human_items_path=a.human_items,full_items_path=a.full_items,matches_path=a.matches,weak_matches_path=a.weak_matches,manifest_path=a.manifest,base_oof_path=a.base_oof,model_dir=a.model_dir,output_dir=a.output_dir,expected_split_sha=a.expected_split_sha,held_fold=a.held_fold),ensure_ascii=False,sort_keys=True))
if __name__=='__main__':main()
