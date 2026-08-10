# E-CUP Matching — Iteration v3 Results

Date: 2026-08-10/11
Status: final packaging verification in progress

## Baseline and fixed validation

- retained v2b Macro AP: `0.5010008994958702`
- fixed validation rows: `73,131`
- validation train/item overlap: `0`
- exact same item-disjoint split as v1/v2
- v2 organizer benchmark: `334 s / 275k pairs / 537,300 items`

## Free GPU backend actually used

The retained neural candidate was trained on a standard GitHub-hosted `macos-15`
Apple Silicon runner. A real PyTorch MPS matrix multiplication probe succeeded
before training, so this is an actually verified GPU backend rather than a
paper-only option.

Rejected infrastructure probes:

- Hugging Face ZeroGPU: HTTP 402 before compute allocation;
- Lightning Studio: HTTP 403 on Studio creation;
- Lightning direct Docker Job API: authentication and Teamspace discovery worked,
  but even a tiny CPU `Job.run` returned HTTP 403 unauthorized.

No secret credentials or raw competition data were committed to public Git.

## Prepared neural data

Fast authoritative-human preparation run: `31434855373` / job `93606589743`.

Private HF prefix: `experiments/v3/prepared/13edb087498b`.

- rows before compaction: `292,523`
- compact train rows: `180,000`
- validation rows: `73,131`
- validation item overlap: `0`
- authoritative human positives retained: `77,515`
- weak rows in this first retained neural curriculum: `0`
- train source: 100% human
- serialized train parquet: `87,196,520` bytes
- serialized validation parquet: `34,958,247` bytes

The earlier human+weak preparation path was rejected after a hosted runner was
shut down during the selective scan over the 4.1 GiB full item table. v3 therefore
uses the faster human-only first neural curriculum; v2b remains the structured
weak-label anchor in the final blend.

## Neural stage 1 and model-mined stage 2

Retained production training run: `31437623156` / job `93615189602`.

Private HF prefix: `experiments/v3/neural/2d31cb18a06e`.

Model: `cointegrated/rubert-tiny2` pairwise sequence classifier.

Training:

- accelerator: Apple M1 MPS
- stage 1: `1,600` optimization steps
- stage 2: `300` optimization steps
- total neural experiment time: `1327.387913542 s`
- validation item overlap: `0`

Real model-mined hard-negative stage 2:

- authoritative human rows scored by stage 1: `180,000`
- mined hard negatives: `12,000`
- paired stage-2 positives: `12,000`
- priority-category mined negatives: `8,400`
- stage-2 focused rows: `24,000`

The hard-negative stage was genuinely trained and evaluated, but the fixed
validation rejected its checkpoint: stage 1 remained substantially stronger.
This is retained as evidence that "hard-negative mining" was tested rather than
silently claimed.

## Blend selection

Four candidate families were compared on the unchanged 73,131-row validation:

- stage-1 global blend;
- stage-1 priority-category blend;
- stage-2 global blend;
- stage-2 priority-category blend.

Selected candidate: **`stage1-global`**.

- selected neural checkpoint: stage 1
- neural blend weight: **`0.45`**
- structured v2b weight: `0.55`
- selected Macro AP: **`0.5254642645846543`**
- v2b Macro AP: `0.5010008994958702`
- absolute improvement: **`+0.024463365088784106`**
- relative improvement over v2b: about **`+4.88%`**

This strictly passes the v3 quality gate.

## Selected per-category AP

| Category | v3 AP |
|---|---:|
| Автотовары | 0.5053259765 |
| Аптека | 0.5566355495 |
| Бытовая техника | 0.6663517955 |
| Бытовая химия | 0.6825961876 |
| Галантерея и аксессуары | 0.3953930428 |
| Детские товары | 0.7699751585 |
| Дом и сад | 0.5419038118 |
| Канцелярские товары | 0.5493489696 |
| Красота и гигиена | 0.6053912861 |
| Мебель | 0.3703375892 |
| Музыкальные инструменты | 0.6080430801 |
| Обувь | 0.3039013681 |
| Одежда | 0.2902099802 |
| Продукты питания | 0.5665505227 |
| Спорт и отдых | 0.4834064834 |
| Строительство и ремонт | 0.5184692334 |
| Товары для животных | 0.6051009805 |
| Хобби и творчество | 0.8142101178 |
| Электроника | 0.3299239214 |
| Ювелирные изделия | 0.3462102372 |

## Submission package and organizer compatibility

Exact organizer image: `odsai/ecup26-matching-baseline:1.0`.

A corrected environment probe verified that the image already contains a CUDA
PyTorch/Transformers stack (`torch 2.10.0+cu128`, Transformers, tokenizers and
safetensors), so v3 does not vendor Python wheels into the submission ZIP.

Independent final neural smoke run `31440472151` / job `93623920970` is **green**:

- ZIP built inside the exact organizer image;
- network disabled during prediction;
- `1,000 / 1,000` pairs were actually sent through the neural reranker;
- runtime log: `neural device=cpu rows=1,000` and `neural_pairs=1,000`;
- output schema/range/finite-score checks passed;
- `final_neural_smoke_verified`;
- runner cleanup passed.

The smoke is important because an earlier category-gated sprint exposed a real
case-normalization bug that produced `neural_pairs=0`. That sprint package was
rejected as a diagnostic. A RED/GREEN regression test now covers normalized
category routing; the corrected source has `108 passed` tests plus a passing
memory policy. The retained global blend was not affected by that category-only
routing mismatch because its `__global__` alpha routes all pairs to the neural
model.

## Full 275k offline benchmark

Final package run: `31439648374` / job `93621437335`.

Completed gates so far:

- exact source and memory policy: PASS;
- private model/data fetch and verification: PASS;
- exact fixed-validation score gate: PASS;
- final ZIP build inside organizer image: PASS;
- exact 275,000-pair benchmark slice: PASS;
- full `--network none` prediction: running at the time of this checkpoint.

The hosted Ubuntu benchmark has no NVIDIA driver, so the global RuBERT branch is
forced onto CPU for all 275k pairs. This is intentionally a correctness/stress
check, not a faithful estimate of the organizer H100 inference speed.

Final ZIP bytes, SHA-256, full 275k wall runtime and canonical private HF upload
are recorded here only after run `31439648374` completes successfully.

## Current decision

v3 is the **quality winner** and the intended retained model. It is not marked
`completed` until the long 275k offline run finishes, the canonical
`submissions/v3/` artifacts are verified in private HF, the public memory/state
files are updated, and a new private Memora checkpoint passes integrity/secret
checks.
