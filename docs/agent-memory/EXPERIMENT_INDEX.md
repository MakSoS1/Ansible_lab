# E-CUP Matching — Experiment Index

Canonical short registry. Detailed rationale, exact fold evidence and rejected branches live in `DECISIONS.md`, `ecup_matching/experiments/v*/PLAN.md`, `RESULTS.md`, `SAFE_METRICS.json` and private immutable OOF artifacts.

## Version summary

| Version | Status | Validation | Best evidence | Interpretation |
|---|---|---|---:|---|
| v1 | historical | old item-disjoint holdout | hidden `0.2345852292` | historical |
| v2 | historical verified platform fallback | old holdout | hidden `0.2583231811` | previous hidden anchor |
| v3 | historical | old holdout | hidden canonical `0.2481015189` | historical |
| v4 | historical | old holdout/cross-fit | hidden canonical `0.2531285195` | historical |
| v5 | completed quality-first production | 285,210 dev / 80,444 sealed gold / 5 folds / zero item overlap | strict OOF `0.6018115534` | best strict local quality; organizer-smoked, but too slow for v6 runtime objective |
| v6 | in_progress | same immutable v5 component-disjoint protocol | strict OOF `0.6006003615` | current runtime-constrained gate95 candidate; exact RTX full-run gate pending |

`0.6018115534` and `0.6006003615` are strict local OOF measurements, not Public/Private leaderboard claims. The sealed gold split remains unopened and platform scores remain a separate evidence axis.

## v5 retained ladder

| Step | Status | Strict OOF Macro AP | Key interpretation |
|---|---|---:|---|
| human structured audit | BASE | `0.5315527709` | immutable baseline |
| category specialists | KEEP | `0.5476780661` | category structure matters |
| weak specialists | KEEP | `0.5514237339` | leakage-safe weak labels help |
| sparse TF-IDF specialists | KEEP signal | `0.5651306839` | rare SKU/model tokens are strong |
| explicit per-key attributes | KEEP signal | `0.5683065131` | explicit key identity helps |
| supervised contrastive | KEEP signal | `0.5662217063` | supervised item space adds diversity |
| 4-signal equal-rank | KEEP intermediate | `0.5870570848` | heterogeneous signals combine strongly |
| + pair teacher | KEEP signal | `0.5952697490` | pair teacher is valuable but expensive |
| + typed explicit = six-signal | VERIFIED FALLBACK | `0.5975445721` | stable six-signal package |
| outer-cross-fitted global simplex | KEEP intermediate | `0.5992720660` | honest learned weighting helps |
| frozen global-meta + category logistic equal-rank | KEEP intermediate | `0.5995921710` | diversity fusion helps |
| fixed category-shrunk simplex | VERIFIED FALLBACK | `0.6009542418` | first honest crossing of `0.60` |
| fixed nonlinear HGB meta stack | complementary | `0.6006290885` | nonlinear category interactions help |
| category-shrunk + HGB equal-rank | V5 QUALITY BEST | `0.6018115534` | best strict OOF; all five folds improve vs category-shrunk |

### v5 verified package

- ZIP: `ecup-v5-category-hgb-fusion-0.6018115534-submission.zip`;
- SHA-256: `442769bd2c92d43730d7034fb91d8a83e596a8445ae3c3f887783890e90284d5`;
- private HF: `submissions/v5/0.6018115534`;
- workflow run `31526323018`;
- Actions artifact `9116032675`;
- exact organizer-image offline smoke: passed;
- platform Public/Private result: unknown.

v5 remains the quality reference and a reproducible fallback. v6 exists because the production objective now includes a hard inference-runtime constraint.

## v6 runtime-constrained ladder

Hard Pareto rule: first require strict OOF `>= 0.6000`; among passing candidates, minimize measured end-to-end runtime.

| Candidate | Strict OOF Macro AP | Decision |
|---|---:|---|
| structured only | `0.5808404006` | reject |
| no teacher | `0.5931387077` | reject |
| no contrastive | `0.5928725263` | reject |
| teacher gate 25% | `0.5929214688` | reject |
| teacher gate 55% | `0.5966896566` | reject |
| teacher gate 85% | `0.5999300792` | reject narrowly |
| teacher distillation | `0.5931935842` | reject |
| student + teacher hybrid 85% | `0.5998746123` | reject narrowly |
| **teacher gate 95%** | **`0.6006003614522999`** | **retain as current v6 candidate** |

### Current v6 architecture

- weak, sparse, explicit, supervised contrastive and typed-explicit signals are retained;
- target-free disagreement is computed from the five non-teacher percentile-rank signals within each category;
- real pair teacher is evaluated for the highest-disagreement `95%` of pairs per category;
- remaining teacher values use the mean rank of the five cheap signals;
- selected teacher scores are percentile-ranked only over selected teacher rows;
- final target-fitted meta is fully outer-cross-fitted category-shrunk simplex + fixed HGB, frozen 50/50 rank fusion;
- actual development teacher fraction: `0.9500262964131693`.

Selection evidence:

- run `31531141700`, job `93911179929`;
- source `fb15ec43a90c892c416acb2d10fe04cc126a4398`;
- private HF `experiments/v6/teacher-gate/95/fb15ec43a90c`.

### Current v6 packaging/runtime evidence

Latest candidate packaging run `31535674086` reached all model/runtime gates before the repository documentation policy:

- production meta refit: passed;
- exact organizer-image offline/read-only smoke: passed;
- 64-row CPU smoke total: `24.14s`;
- smoke output schema `id1,id2,predict`: passed;
- candidate ZIP before documentation gate: `1,143,630,143` bytes;
- candidate SHA-256 before documentation gate: `20c5f128e43c5303893301f012726381df06a4e20d027ea054acf36e0f6aae40`;
- test result after smoke: `264 passed, 1 failed, 91 warnings`;
- the sole failure was the stale repository documentation policy state, not inference/model behavior.

This candidate is **not yet the final artifact**. The final ZIP must be rebuilt after policy becomes GREEN and the exact rebuilt bytes must then pass the RTX 2060 benchmark.

## Immutable validation facts

- metric: unweighted mean of sklearn `average_precision_score` over exactly 20 official categories;
- split SHA: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- development rows: `285,210`;
- sealed gold rows: `80,444`;
- item/component overlap across splits: `0`;
- sealed gold opened: **false**;
- sealed gold rows scored: **0**.

## Runtime implementation retained in v6

- FP32 neural semantics while the quality margin over `0.6000` is small;
- stable length bucketing for contrastive item texts and teacher pairs;
- VRAM-aware CUDA batches (`256` contrastive / `96` teacher on 8 GiB RTX class);
- CUDA OOM batch-halving fallback;
- non-blocking device transfer;
- SDPA where supported with eager fallback;
- offline/local-files-only inference;
- phase telemetry for load, structured, contrastive, gate, teacher, meta and write.

## Failure lessons that remain binding

- infrastructure/OOM/API failures are not model-quality evidence;
- production refit scores are not validation scores;
- learned meta layers require outer cross-fitting;
- direct attribute likelihood shift was harmful even though explicit attribute estimator features were useful;
- pretrained-only embeddings were insufficient; supervised contrastive was the useful neural signal;
- do not tune post-result fusion weights on the same held labels without another nested layer;
- do not use sealed gold to recover runtime-induced quality loss.

## Next gate

Rebuild the exact gate95 archive from a GREEN repository, publish immutable SHA/provenance, benchmark those exact bytes on `ecup-rtx2060` inside the organizer image including a full reference `matches.parquet` run, and retain it only if runtime fits while strict OOF remains `>= 0.6000`.
