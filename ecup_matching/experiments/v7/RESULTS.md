# v7 Results

## Baselines retained

- strict quality reference: v5 category-shrunk HGB + equal-rank fusion = `0.6018115534135564`
- fast runtime reference: v6 gate95 = `0.6006003614522999`
- sealed gold opened: **no**

## 2026-08-12 — contract phase

Status: **IN PROGRESS**.

Evidence before implementation:
- retained final v5 teacher signal is teacher2 on `ai-forever/ruBert-base`;
- teacher2 used `max_length=128`, `max_steps=800`, and at most 100k weak rows per outer fold;
- v5 item text placed `[NUMERIC]` before `[ATTR]`, creating a context-allocation failure for canonical typed attributes;
- v7 RED CI run `31546090474` failed in the new targeted unit-test step before production modules existed.

Implemented after RED:
- isolated `v7_item_text.py` identity-first serializer; v5/v6 serializer unchanged;
- isolated `v7_teacher_contract.py` with 256-token minimum, full requested curriculum exposure contract, and explicit forbidden-item weak-pair filtering;
- macro-balanced pair batches: equal training exposure per official category and mixed positive/negative examples inside each category batch;
- five-fold driver reconstructs and verifies immutable split SHA `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`, zero overlap, `285210` development rows and `80444` unopened gold rows;
- shared weak pretraining excludes the complete human item universe before any fold-specific fine-tuning.

## 2026-08-12 — RTX 2060 CUDA gate

All measurements below used the exact pinned `ai-forever/ruBert-base` revision `43be4261797042e172adf7476c558734f3cbb2a0`, `max_length=256`, CUDA fp16 and real E-CUP human pairs on `NVIDIA GeForce RTX 2060 SUPER`.

| Run | Inference batch | Train physical/effective batch | Inference pairs/s | Train examples/s | Peak training allocated VRAM |
|---|---:|---:|---:|---:|---:|
| `31547153312` | 16 | 2 / 32 | `371.9400` | `26.1084` | `~1.65 GB` |
| `31547513168` | 64 | 8 / 32 | **`400.4413`** | `70.8311` | `~1.70 GB` |
| `31547717440` | 256 | 32 / 32 | `394.6975` | **`84.0242`** | `~1.96 GB` |

Decisions from measured evidence:
- **KEEP training physical batch 32 / effective batch 32.** It is `3.22x` faster than the original physical batch 2 benchmark and still uses far below the 8 GiB card limit.
- **KEEP inference batch 64** among the measured choices; batch 256 is slightly slower, so larger batches are not assumed better.
- The RTX 2060 batch-64 neural-only projection is about `287 s` for 115k pairs and `687 s` for 275k pairs. These are local-GPU projections only; they are not H100 measurements and do not by themselves prove the organizer runtime gate.
- Because batch 32 still had large VRAM headroom, a final bounded benchmark without gradient checkpointing was opened on source `b4bea6922c79ef5bf93ed2aead5de6e2144f59f2`. Production OOF will use that mode only if the measured speed/VRAM result is better and safe.

Dispatcher TDD/security evidence:
- initial v7 dispatcher RED `31546512720`, GREEN `31546790051`;
- batch 8/64 RED `31547407943`, GREEN `31547458575`;
- maximum batch probe RED `31547614100`, GREEN `31547656554`;
- retained train-batch-32 contract RED `31547793230`, GREEN `31547909317`;
- the v7 dispatcher accepts only exact SHAs reachable from `ecup-v7-neural` and fixed `v7-benchmark`/`v7-train` profiles inside the existing offline/read-only container contract.

## 2026-08-12 — weak attribute bug found before final OOF

During a preflight review of the expensive driver, the first implementation was found to call `select_items_by_ids(..., include_attributes=False)` and then serialize those rows. That helper deliberately substitutes `"{}"` for attributes, so the 600k weak curriculum would have seen names/categories but **none of the canonical typed attributes that v7 was designed to exploit**.

This was not a crash; it was a silent quality bug. It was fixed before accepting any v7 OOF result:
- intermediate weak sampling still uses a lightweight category-only scan where attributes are not needed;
- after the final 600k weak pairs are selected, `build_v7_text_cache_from_parquet` performs a streaming scan of the full item parquet for only the selected IDs, keeps the real attributes, canonicalizes them and stores only the serialized strings/category map;
- the driver checks that category identity is unchanged between the lightweight and full-attribute scans and that no human item survived the leakage filter.

TDD evidence:
- RED `31548047542` required real canonical weak attributes in the streamed cache;
- GREEN `31548225849` on source `6e12faa62b20626371b7ea265b2ec13cfbac553b`, including full v7 driver imports and `memory_policy.py`.

Runs started before this fix are not quality evidence. In particular, the older name-only weak OOF run `31547962566` was cancelled as superseded. The first full-attribute run `31548340838` was also cancelled only to perform the bounded no-gradient-checkpointing speed probe before restarting the same canonical five-fold experiment.

## 2026-08-12 — first immutable fold-0 quality gate

The first completed leakage-safe v7 research gate is private GPU run `31550137850`, job `93970869826`, exact source `bf8d6c79105d8454b60857a951f9fe08288c0c1f`.

Configuration: `ai-forever/ruBert-base`, `max_length=256`, full real attributes, 600k leakage-safe weak curriculum, weak `0.10` epoch, human `0.50` epoch, last 8 encoder layers trainable, macro-balanced batches, batch `32`, ranking weight `0.25`.

Result:
- immutable fold-0 standalone Macro AP: **`0.6791967999009738`**;
- exact retained teacher2 fold-0 reference: `0.4330985437448661`;
- delta vs teacher2: **`+0.2460982561561077`**;
- held rows: `57042`;
- sealed gold opened: **no**;
- gold rows scored: **0**;
- cross-split item overlap: **0**.

This number is intentionally labelled **diagnostic only**, not strict five-fold OOF. Fold 0 is now a research-selection gate; any configuration retained from it must still complete all five outer folds, and folds 1–4 should be reported separately as confirmatory evidence because fold 0 has been used for model selection.

The weakest v7 fold-0 categories are `Одежда 0.4143`, `Ювелирные изделия 0.4656`, `Обувь 0.4781`, `Мебель 0.5357`, `Галантерея и аксессуары 0.5699`. Most other categories are already roughly `0.64–0.88`, so the remaining gap is concentrated rather than uniform.

## 2026-08-12 — identity-v2 context repair

A composed-path review found that v7 already prepends category as `[CAT]`; therefore a proposed duplicate category token was rejected before GPU use. The useful remaining context defect was category-specific identity evidence: material/composition, gender, season, jewelry hallmark/karat and stone/insert were still residual attributes behind the numeric section.

`identity-v2` promotes those keys into the front identity packet. TDD evidence: RED `31573302453`, targeted GREEN `31573405437`, full repository suite + memory policy GREEN `31574407135`. The production change is now on canonical `ecup-v7-neural`; canonical v7 CI `31575399464` is GREEN.

Exact ruBERT tokenizer visibility on real immutable fold-0 pairs (`max_length=256`, deterministic <=2500 pairs/category, no label-based sampling) changed as follows:

| Category | old critical-attribute token visibility | identity-v2 |
|---|---:|---:|
| Одежда | `72.1%` | **`97.3%`** |
| Обувь | `73.9%` | **`90.5%`** |
| Ювелирные изделия | `53.9%` | **`89.9%`** |
| Галантерея и аксессуары | `61.0%` | **`85.0%`** |
| Мебель | `18.2%` | **`37.5%`** |

This is a real context-allocation defect, not a synthetic-only example. `identity-v2` is therefore the next neural candidate if the already-running one-epoch baseline does not give sufficient margin.

## 2026-08-12 — next controlled gates

- one-epoch baseline probe is running as private run `31573080203`; it differs from the `0.6791968` run only by human epochs `0.50 -> 1.00` and remains pinned to the pre-identity-v2 source for causal comparison;
- fixed v7/v5 equal-rank fusion evaluator exists with strict row/fold/gold alignment checks; the first GitHub-hosted CPU attempt failed before evaluation because `gpu-dispatch` hosted jobs do not receive `HF_TOKEN`, so that run is infrastructure evidence only;
- target-free hard-negative human sampling was opened because the retained human sampler chooses negatives uniformly. RED `31576201464`; implementation is isolated from the retained sampler and uses only serialized outer-train text similarity. It is not yet a quality result.

## Quality status

Best measured v7 quality evidence is currently **fold-0 diagnostic `0.6791967999009738`**. No v7 strict five-fold OOF metric has been claimed yet. The target `0.70` is close enough to pursue directly, but it will only be called reached for the requested local validation after a frozen candidate completes the strict outer-OOF gate.