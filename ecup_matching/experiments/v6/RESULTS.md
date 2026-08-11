# E-CUP Matching — Iteration v6 Runtime-Constrained Results

Date: 2026-08-11

## Objective

v5 exceeded the local quality target but timed out on the competition platform. v6 therefore uses a hard Pareto rule:

1. retain only architectures with honest strict component-disjoint OOF Macro AP `>= 0.6000`;
2. among passing architectures, minimize real end-to-end inference runtime;
3. keep the sealed gold split unopened.

The pure-quality v5 best remains `0.6018115534135564`. v6 is not claimed to improve that number; it is the runtime-constrained production line.

## Selected v6 quality point

**Strict outer OOF Macro AP:** `0.6006003614522999`

Selected architecture:

- weak category specialist;
- sparse TF-IDF specialist;
- explicit per-key specialist;
- supervised contrastive item score;
- typed/canonicalized explicit specialist;
- pair teacher evaluated only for the target-free top-disagreement `95%` of pairs within each official category;
- for the remaining pairs, teacher is replaced by the unweighted mean of the five target-free percentile ranks;
- selected teacher scores are percentile-ranked only over selected teacher rows;
- final meta layer remains a full outer-cross-fitted category-shrunk simplex plus fixed HGB, fused as `0.5 * percentile_rank(category_shrunk) + 0.5 * percentile_rank(HGB)`.

Requested teacher coverage: `0.95`.
Actual development teacher fraction: `0.9500262964131693`.

Selection evidence:

- workflow run `31531141700`;
- job `93911179929`;
- source commit `fb15ec43a90c892c416acb2d10fe04cc126a4398`;
- private HF prefix `experiments/v6/teacher-gate/95/fb15ec43a90c`.

## Validation contract

- development rows: `285,210`;
- immutable five outer folds;
- component/item-disjoint validation;
- split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- sealed gold opened: **false**;
- sealed gold rows scored: **0**;
- strict metric: unweighted mean of `average_precision_score` over exactly the 20 official categories;
- gate selection uses only predictions and official category, never target labels;
- meta models are fit only on outer-train rows for each held fold;
- no post-result weight search is used for the retained formula.

## Runtime/quality exploration

### Remove expensive signals and refit meta

| Architecture | Strict OOF Macro AP | Decision |
|---|---:|---|
| structured only | `0.5808404005946962` | reject |
| no teacher | `0.5931387077244183` | reject |
| no contrastive | `0.5928725263319000` | reject |

Simple deletion cannot satisfy the `0.6000` quality gate.

### Target-free partial teacher

| Teacher coverage | Strict OOF Macro AP | Decision |
|---:|---:|---|
| 25% | `0.5929214688140778` | reject |
| 55% | `0.5966896566149946` | reject |
| 85% | `0.5999300791828578` | reject narrowly |
| **95%** | **`0.6006003614522999`** | **retain** |

### Teacher distillation

A leakage-safe HGB student was trained on teacher percentile ranks using only outer-train teacher values for each held fold.

Strict OOF: `0.5931935841654697` — reject.

### Student + real-teacher hybrid

| Real teacher coverage | Strict OOF Macro AP | Decision |
|---:|---:|---|
| 25% | `0.5955200995988962` | reject |
| 40% | `0.5964542084428204` | reject |
| 55% | `0.5974645385215002` | reject |
| 70% | `0.5985536258037350` | reject |
| 85% | `0.5998746122650258` | reject narrowly |

The hybrid did not preserve enough teacher ordering to justify replacing the simpler gate95 architecture.

## Production runtime implementation

The selected model keeps FP32 neural semantics and obtains speed from implementation changes rather than mixed precision, because the retained OOF margin over `0.6000` is small.

Implemented changes:

- target-free teacher gate avoids approximately 5% of pair-teacher calls;
- contrastive items are stable length-bucketed before tokenization to reduce padding;
- selected teacher pairs are stable length-bucketed before cross-encoder inference;
- CUDA batch sizes are scaled by detected VRAM;
- RTX-class 8 GiB default: contrastive `256`, teacher `96`;
- 20+ GiB default: contrastive `512`, teacher `192`;
- 60+ GiB default: contrastive `1024`, teacher `384`;
- CUDA OOM fallback halves batches safely;
- non-blocking CUDA transfers;
- PyTorch SDPA is requested where supported, with eager fallback;
- structured feature chunk size increased to `50,000`;
- phase-level runtime telemetry is emitted for load, structured, contrastive, gate, teacher, meta and write stages;
- no network is required at inference.

## Production equivalence safeguards

Tests verify that assembling a teacher signal from only the selected teacher rows is numerically identical to the validation formula that receives a full teacher vector and then masks it. Changing unselected teacher values cannot change gated predictions.

## Final packaging status

Final archive target:

`ecup-v6-fast-gate95-0.6006003615-submission.zip`

Production packaging, exact organizer-image smoke, full tests, SHA-256, RTX 2060 benchmark and final artifact provenance are recorded only after those steps actually pass. No runtime figure is inferred from the OOF experiments.

## Leaderboard status

Public leaderboard AP: unknown.
Private leaderboard AP: unknown.

Local OOF must remain separate from leaderboard evidence.
