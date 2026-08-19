# E-CUP Matching — Iteration v3 Results

Date: 2026-08-10/11
Status: **completed / retained**

## Baseline and fixed validation

- retained v2b Macro AP: `0.5010008994958702`
- fixed validation rows: `73,131`
- validation train/item overlap: `0`
- exact same item-disjoint split as v1/v2
- v2 organizer benchmark: `334 s / 275k pairs / 537,300 items`

## Free GPU backend actually used

The retained neural candidate was trained on a standard GitHub-hosted `macos-15`
Apple Silicon runner. A real PyTorch MPS matrix multiplication probe succeeded
before training, so this was an actually verified free GPU backend.

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

- accelerator: GitHub `macos-15` Apple M1 MPS
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
The retained v3 therefore uses the stage-1 checkpoint while preserving the
hard-negative result as a tested/rejected ablation rather than claiming it as an
unmeasured improvement.

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

## Organizer compatibility and neural execution

Exact organizer image: `odsai/ecup26-matching-baseline:1.0`.

A corrected environment probe verified that the image already contains a CUDA
PyTorch/Transformers stack (`torch 2.10.0+cu128`, Transformers, tokenizers and
safetensors), so v3 does not vendor Python wheels into the submission ZIP.

Independent final neural smoke run `31440472151` / job `93623920970` is green:

- ZIP built inside the exact organizer image;
- network disabled during prediction;
- `1,000 / 1,000` pairs were actually sent through the neural reranker;
- runtime log contained `neural device=cpu rows=1,000` and `neural_pairs=1,000`;
- output schema/range/finite-score checks passed;
- `final_neural_smoke_verified`;
- cleanup passed.

An earlier category-gated sprint exposed a real case-normalization bug that
produced `neural_pairs=0`. That sprint package was rejected as a diagnostic. A
RED/GREEN regression test was added and the runtime now normalizes both manifest
and item categories. Latest canonical-source verification reports **108 tests
passed** plus `memory_policy.py` PASS.

## Canonical final submission

Canonical packaging run: **`31440971110` / job `93625406492`**.

Source SHA: `de4141af04e33170777d2de56ae059ebe52bb806`.

Package:

- private artifact: `submissions/v3/ecup-v3-submission.zip`
- private evidence: `submissions/v3/v3-package-metrics.json`
- ZIP size: **`109,185,253 bytes`**
- ZIP files: `21`
- SHA-256: **`b833ceb203f8cc7d87517257df8ee5e0a2590075db0ecd2932b8281950015660`**
- private HF presence verified after upload
- runner temporary data cleaned

Canonical offline correctness run inside the exact organizer image:

- network: disabled with `--network none`
- pairs: `10,000`
- actual neural pairs: **`10,000 / 10,000`**
- items in slice: `19,822`
- wall runtime on GitHub Ubuntu CPU: `103 s`
- measured feature time: `6.56 s`
- measured neural time: `89.55 s`
- measured total inside runtime: `101.63 s`
- output rows: `10,000`
- unique scores: `9,996`
- schema/order/range/finite checks: PASS
- canonical private-HF verification: PASS

The GitHub Ubuntu runner has no NVIDIA driver, so this CPU timing is a
correctness/stress measurement, not an estimate of organizer H100 throughput.
The official submission image detects CUDA and the final runtime selects CUDA
when available. The separate full 275k CPU stress run `31439648374` was left as
non-blocking evidence because forcing all 275k global-reranker pairs through CPU
is not representative of the organizer H100 target.

## Retained decision

**v3 is retained as the current best submission candidate.**

It passed the comparable item-disjoint quality gate, real GPU training,
model-mined-hard-negative ablation, exact-image packaging, real offline neural
execution, canonical ZIP integrity, private artifact verification and repository
test/memory-policy gates. Public Git contains source/tests/docs only; the model,
raw data and submission ZIP remain private.
