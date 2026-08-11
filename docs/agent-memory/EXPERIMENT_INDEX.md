# E-CUP Matching — Experiment Index

Canonical short registry. Detailed rationale, exact fold evidence and rejected branches live in `DECISIONS.md`, `ecup_matching/experiments/v*/PLAN.md`, `RESULTS.md`, `SAFE_METRICS.json` and private immutable OOF artifacts.

## Version summary

| Version | Status | Validation | Best evidence | Interpretation |
|---|---|---|---:|---|
| v1 | historical | old item-disjoint holdout | hidden `0.2345852292` | historical |
| v2 | historical verified platform fallback | old holdout | hidden `0.2583231811` | previous hidden anchor |
| v3 | historical | old holdout | hidden canonical `0.2481015189` | historical |
| v4 | historical | old holdout/cross-fit | hidden canonical `0.2531285195` | historical |
| v5 | completed quality-first production | 285,210 dev / 80,444 sealed gold / 5 folds / zero item overlap | strict OOF `0.6018115534` | best retained strict local quality |
| v6 | in_progress runtime reference | same immutable component-disjoint protocol | strict OOF `0.6006003615` | gate95 + prediction-preserving runtime optimization; exact full runtime gate pending |
| v7 | in_progress quality sprint | same immutable component-disjoint protocol | no strict OOF yet | identity-first 256-token ruBERT-base teacher; `0.70` is a stretch target only |

`0.6018115534` and `0.6006003615` are strict local OOF measurements, not Public/Private leaderboard claims. The sealed gold split remains unopened and platform scores remain a separate evidence axis. v7 has no quality result yet.

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

v5 remains the quality reference. Its ladder also shows why another lexical/meta micro-tuning pass is unlikely to supply the `~0.10` absolute gain required for the v7 stretch target.

## v6 runtime-constrained ladder

Hard v6 Pareto rule: first require strict OOF `>= 0.6000`; among passing candidates, minimize measured end-to-end runtime.

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
| **teacher gate 95%** | **`0.6006003614522999`** | **retain as v6 runtime reference** |

### v6 architecture/runtime evidence

- weak, sparse, explicit, supervised contrastive and typed-explicit signals are retained;
- target-free disagreement selects the highest-disagreement 95% per category for the real pair teacher;
- remaining teacher values use the mean percentile rank of the five cheap signals;
- final target-fitted meta is fully outer-cross-fitted category-shrunk simplex + fixed HGB, frozen 50/50 rank fusion;
- actual development teacher fraction: `0.9500262964131693`;
- selection run `31531141700`, job `93911179929`, source `fb15ec43a90c892c416acb2d10fe04cc126a4398`;
- private selection prefix `experiments/v6/teacher-gate/95/fb15ec43a90c`.

Prediction-preserving runtime engineering, 2026-08-12:

| Structured path | us/pair | Note |
|---|---:|---|
| as submitted | `2210.1` | single-threaded, duplicated work |
| shared difflib between passes | `1770.5` | bitwise identical |
| + fork worker pool | `487.0` | bitwise identical |

Projected structured phase on the organizer host: public `~254s -> ~22s` of `360s`, private `~608s -> ~44s` of `780s`. `select_items_by_ids` measured `13.5x` faster on a 2,000,000-row item file. These are implementation/profile projections, not the final end-to-end exact-byte benchmark.

Latest pre-policy packaging evidence from run `31535674086`:

- production meta refit: passed;
- exact organizer-image offline/read-only 64-row smoke: passed;
- candidate ZIP before documentation gate: `1,143,630,143` bytes;
- candidate SHA-256 before documentation gate: `20c5f128e43c5303893301f012726381df06a4e20d027ea054acf36e0f6aae40`;
- full suite at that point: `264 passed, 1 failed, 91 warnings`;
- sole failure was stale documentation-memory state.

That pre-policy ZIP is not retained as final. The exact full RTX/runtime gate remains pending.

## v7 neural quality sprint

### Why v7 exists

The final v5 teacher signal was already `ai-forever/ruBert-base`; therefore a simple encoder-name swap is not a meaningful new hypothesis. The retained `teacher2` path nevertheless had four concrete constraints:

1. `max_length=128` for the item pair;
2. `max_steps=800`;
3. at most `100,000` weak rows per outer fold;
4. item serialization ordered generic `[NUMERIC]` before `[ATTR]`, allowing noisy numbers to consume context before canonical typed attributes.

At the same time, the v5 ladder shows the pair teacher is the only retained signal that jointly reads both items and it contributes real ensemble diversity. v7 therefore attacks the pairwise signal directly rather than adding another correlated lexical feature or another meta layer.

### Candidate A contract

- branch: `ecup-v7-neural`, created from accelerated runtime branch commit `0580eeed2fb04f363951a5a325442430e4639e0c`;
- encoder: `ai-forever/ruBert-base`, revision `43be4261797042e172adf7476c558734f3cbb2a0`;
- pair context: `max_length=256`;
- identity-first text: `[NAME]`, `[BRAND]`, canonical `[MODEL]`, canonical typed identity attributes, then residual numeric/attributes;
- weak data: explicit forbidden-item filtering and confidence weighting;
- training exposure: must cover the declared curriculum rather than silently truncate at the legacy 800-step limit;
- validation: same five immutable outer folds, no sealed-gold scoring;
- stretch target: strict OOF `0.70`;
- minimum KEEP threshold: beat v5 quality reference `0.6018115534135564` before runtime/packaging consideration.

### v7 implementation evidence so far

RED-by-design:

- Actions run `31546090474`, job `93958793656` failed in the newly introduced v7 unit-test step before the v7 production modules existed.

GREEN code contract after minimal implementation:

- `ecup_matching/ml/v7_item_text.py` created as an isolated serializer so v5/v6 semantics remain unchanged;
- `ecup_matching/ml/v7_teacher_contract.py` created with 256-token minimum, optimizer-step exposure validation and forbidden weak-endpoint filtering;
- `ecup_matching/tests/test_v7_neural_contract.py` covers deterministic hard bounds, visible canonical model code, `128 GB == 0.128 TB`, critical typed values, weak leakage exclusion and legacy `128/800` rejection;
- Actions run `31546214410` passed the targeted v7 code tests and failed only the documentation-memory policy because canonical docs still named v6. The docs are now updated as part of the same v7 iteration.

**No v7 OOF metric is recorded yet.** Do not convert the target into a claim.

### v7 GPU/runtime next gate

Private `MakSoS1/gpu-dispatch` is still intentionally locked to an older allowed source branch and fixed v4-era profiles. Extend that trusted dispatcher narrowly for a fixed v7 benchmark/train profile on the dedicated v7 branch; do not grant arbitrary commands. Then measure 256-token ruBERT-base throughput/VRAM on `ecup-rtx2060`, and run the five-fold Candidate A experiment with progress/timing telemetry.

Candidate B — aligned shared-key pair serialization — is only considered if Candidate A is insufficient, and must earn retention independently.

## Immutable validation facts

- metric: unweighted mean of sklearn `average_precision_score` over exactly 20 official categories;
- split SHA: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- development rows: `285,210`;
- sealed gold rows: `80,444`;
- item/component overlap across splits: `0`;
- sealed gold opened: **false**;
- sealed gold rows scored: **0**.

## Failure lessons that remain binding

- infrastructure/OOM/API failures are not model-quality evidence;
- production refit scores are not validation scores;
- learned meta layers require outer cross-fitting;
- direct attribute likelihood shift was harmful even though explicit attribute estimator features were useful;
- pretrained-only embeddings were insufficient; supervised contrastive was the useful neural signal;
- the final pair teacher already used ruBERT-base; model-name substitution is not a sufficient hypothesis;
- do not tune post-result fusion weights on the same held labels without another nested layer;
- do not use sealed gold to recover runtime-induced quality loss;
- a fixed-overhead smoke is not runtime evidence;
- the submission file list must be derived from the import graph, never hand-maintained;
- never claim the v7 `0.70` target until a complete comparable five-fold aggregate proves it.

## Next gate

Make `scripts/memory_policy.py` GREEN for v7, extend the trusted private GPU dispatcher with a narrow v7 profile, benchmark the real 256-token path, then produce all five held-fold Candidate A vectors and strict aggregate metrics. Every KEEP/REJECT/FAIL must update v7 results, safe metrics, current state and Memora before the next hypothesis.