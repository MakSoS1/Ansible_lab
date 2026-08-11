# E-CUP Matching v4 Strong Reranker — Design

Date: 2026-08-11
Status: approved for implementation

## Goal

Produce a v4 submission candidate that strictly improves the retained v3 Macro Average Precision of `0.5254642645846543` on the unchanged 73,131-row item-disjoint human validation split, while preserving v3 as an immutable fallback and remaining compatible with the organizer's offline H100 runtime.

## Non-negotiable invariants

- Keep `submissions/v3/canonical/b833ceb203f8cc7d87517257df8ee5e0a2590075db0ecd2932b8281950015660/ecup-v3-submission.zip` immutable.
- Never modify or merge `main`.
- Raw competition parquet, model weights, private metrics and submission ZIPs remain in private Hugging Face storage only.
- Primary model-selection metric is the official unweighted category Macro Average Precision.
- Validation remains the exact existing 73,131-row connected-component item-disjoint split with zero train/validation item overlap.
- A v4 component is retained only when it improves the fixed validation or is required to enable a later independently measured improvement.
- No runtime optimization may change predictions without a separate measured ablation.
- Final inference must work offline in `odsai/ecup26-matching-baseline:1.0` and use CUDA automatically when H100 is available.

## Baseline

Retained v3:

- structured anchor: `v2b-weak-curriculum`;
- neural model: `cointegrated/rubert-tiny2` stage-1 checkpoint;
- global blend: `0.55 * v2b + 0.45 * neural`;
- fixed-validation Macro AP: `0.5254642645846543`;
- canonical ZIP SHA-256: `b833ceb203f8cc7d87517257df8ee5e0a2590075db0ecd2932b8281950015660`.

The largest unused supervision reserve is that the retained v3 neural branch trained on only 180,000 authoritative-human rows and zero LLM weak rows, while the leakage-safe outer human train contains 292,523 rows and the competition provides more than 11 million soft LLM-labelled pairs.

## Selected base encoder

Use `ai-forever/ruBert-base` as the primary v4 cross-encoder base.

Reasons:

- Apache-2.0 model license;
- Russian BERT encoder;
- 12 layers, hidden size 768, 12 attention heads;
- approximately 178M parameters;
- materially larger representational capacity than `rubert-tiny2` while remaining practical on an RTX 2060 SUPER 8 GB through FP16, small micro-batches and gradient accumulation;
- normal Transformers `BertForSequenceClassification` compatibility via `AutoModelForSequenceClassification` with a newly initialized classification head.

Pin the exact Hugging Face revision used by the trusted GPU image or preparation workflow and record it in v4 results. Never depend on the moving `main` revision for a retained artifact.

## Experiment ladder

### v4a — full-human stronger cross-encoder

Purpose: isolate the value of stronger model capacity and removal of the v3 180k compaction.

Training data:

- all 292,523 leakage-safe human training pairs;
- no LLM weak rows;
- fixed validation unchanged;
- category-balanced sampling/weighting may be used, but every category remains represented and no validation item is allowed into training.

Model/input:

- `ai-forever/ruBert-base`;
- existing deterministic compact product-pair serialization from the v3 path;
- max length 256 unless the measured RTX memory gate requires 192;
- FP16 CUDA training;
- gradient checkpointing is allowed if required by 8 GB VRAM;
- evaluation always uses the complete fixed validation.

Selection:

- evaluate raw neural AP;
- sweep global blend alpha with v2b;
- evaluate category-aware blend only as an explicit separate candidate;
- retain the best candidate by fixed Macro AP.

### v4b — weak-label neural curriculum

Purpose: transfer information from the >11M LLM-labelled pool into the stronger neural branch without allowing noisy pseudo-labels to dominate human truth.

Start from the best v4a checkpoint.

Weak-label policy reuses the established v2 confidence function:

- target `<=0.03` or `>=0.97`: weak weight `1.0`;
- target `0.03..0.15` or `0.85..0.97`: weak weight `0.6`;
- target `0.15..0.30` or `0.70..0.85`: weak weight `0.3`;
- target in `(0.30, 0.70)`: excluded from direct supervision.

Additional controls:

- canonicalize pair direction;
- deduplicate exact weak pairs;
- remove exact human-pair collisions;
- remove weak false negatives inside authoritative human positive components;
- reject every weak pair touching a validation item;
- category/class-balanced deterministic sampling;
- human rows have dominant sample weight;
- weak sampling is capped initially at 600,000 rows so the first real run is measurable and reproducible on the home GPU;
- if v4b improves v4a and GPU wall time is comfortable, an 800,000-row weak ablation may be evaluated as a separate immutable run, not by overwriting the 600k result.

Training curriculum:

1. warm start from retained v4a;
2. mix all human rows with selected weak rows;
3. preserve at least 50% human examples in every effective optimization window through sampler weighting or explicit batch composition;
4. use weighted soft BCE for weak probabilities and hard BCE targets for human labels;
5. use a lower learning rate than the v4a base fine-tune.

### v4c — model-mined hard negatives with replay

Purpose: improve difficult ranking cases without reproducing the v3 stage-2 regression caused by over-focusing on a small hard set.

Start from the best v4a/v4b checkpoint, whichever has higher fixed Macro AP.

Mining pool:

- authoritative human train negatives;
- high-confidence weak negatives not touching validation items;
- optional low-scoring human positives as hard positives.

Mining score:

- current neural score;
- structured/neural disagreement;
- explicit model/SKU/numeric contradiction indicators when available.

Selection:

- global hard-negative quota plus per-category minimum quotas;
- extra quota for the six historically weak categories: Electronics, Apparel, Footwear, Jewelry, Accessories and Furniture;
- do not mine only false positives: retain representative positives and ordinary/easy negatives.

Replay curriculum:

- 25% mined hard negatives;
- 25% matched positives/hard positives;
- 50% ordinary replay sampled from the parent v4a/v4b curriculum;
- low learning rate;
- short fine-tune;
- compare every checkpoint against its parent on complete validation.

Reject v4c if its best validation Macro AP does not exceed the parent checkpoint.

## Blend and category specialization

For every retained neural checkpoint, evaluate:

1. neural only;
2. global alpha blend with v2b over a deterministic alpha grid;
3. category-specific alpha vector with a shrinkage rule toward the global alpha so small categories cannot overfit freely.

Category specialization is allowed only as a lightweight residual/blend decision in v4. Do not train 20 independent BERTs in this iteration.

The final score written to `predict` remains continuous; no classification threshold is used for submission.

## GPU strategy

Training backend priority:

1. home NVIDIA RTX 2060 SUPER through private `MakSoS1/gpu-dispatch`;
2. GitHub-hosted Apple M1 MPS only as a fallback/benchmark.

Before the long v4a train, run the same short model/tokenization workload on RTX and compare examples/second against the historical M1 path. Backend choice is based on training throughput and reliability, not on organizer inference timing.

The organizer target has H100 80 GB, 20 CPU cores and 200 GB RAM. Therefore local CPU/M1 inference time is not a retention criterion for intermediate v4 training. Runtime becomes a hard gate only after a quality winner exists.

## Artifact layout

Every actual training run writes to an immutable run namespace:

- `experiments/v4/runs/<source-sha>/<run-id>/v4a/...`
- `experiments/v4/runs/<source-sha>/<run-id>/v4b/...`
- `experiments/v4/runs/<source-sha>/<run-id>/v4c/...`

A run must never upload directly over another run's model or metrics.

Only after final verification is the winning package copied to:

- `submissions/v4/canonical/<submission-sha256>/ecup-v4-submission.zip`
- `submissions/v4/canonical/<submission-sha256>/v4-package-metrics.json`

A mutable convenience alias may point to the canonical bytes:

- `submissions/v4/ecup-v4-submission.zip`

The canonical SHA-256 path is the source of truth.

## Quality gates

v4 is retained only if all are true:

- validation rows: exactly 73,131;
- validation item overlap: exactly 0;
- final Macro AP strictly greater than `0.5254642645846543`;
- all 20 category AP values recorded;
- comparison against v3 uses identical validation rows/order/labels;
- model/license/revision recorded;
- final package works with network disabled in exact organizer image;
- CUDA path is exercised when a compatible NVIDIA device is available;
- output schema/order/range/finite checks pass;
- submission ZIP <5 GB;
- Docker image remains <15 GB;
- private artifact presence and SHA-256 are verified;
- repository tests, memory policy, Memora ingest and v4 checkpoint all pass.

Target score bands:

- minimum useful v4: `>0.5254642646`;
- target: `>=0.54`;
- stretch: `>=0.55`.

## Runtime gate

Do not reject an intermediate v4 model based on GitHub CPU or M1 inference timing.

After selecting the quality winner:

1. build the exact organizer package;
2. run a network-disabled correctness smoke;
3. measure CUDA throughput on the home RTX as a conservative functional check;
4. estimate/check H100 feasibility using batch sizes supported by the exact organizer stack;
5. if the all-pair cross-encoder path is unnecessarily expensive, add uncertainty/category gating only if the gate preserves the selected validation score within a separately measured ablation.

The official limits remain the final authority: Check 1 minute, Public 6 minutes, Private 13 minutes.

## Failure policy

- Any validation leakage: abort the affected run.
- Any artifact overwrite ambiguity: reject the affected artifact and publish a new immutable run.
- v4a <= v3: v4a is a negative result; do not proceed to weak/hard stages as if it were an improvement unless the stronger encoder is still useful as an initialization ablation explicitly documented.
- v4b <= v4a: retain v4a and reject v4b.
- v4c <= parent: retain parent and reject v4c.
- CUDA OOM: lower micro-batch first, then enable gradient checkpointing, then reduce max length from 256 to 192; do not silently change validation or remove human rows.
- Any requirement to expose secrets or publicize private competition data: reject that execution path.
