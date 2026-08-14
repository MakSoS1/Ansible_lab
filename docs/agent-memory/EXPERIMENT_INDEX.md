# E-CUP Matching — Experiment Index

Canonical short registry. Detailed rationale and rejected branches live in `DECISIONS.md`, `ecup_matching/experiments/v*/RESULTS.md`, plans/specs and private OOF artifacts.

## Version summary

| Version | Status | Validation | Best evidence | Interpretation |
|---|---|---|---:|---|
| v1 | historical | old item-disjoint holdout | hidden `0.2345852292` | historical |
| v2 | historical verified platform fallback | old holdout | hidden `0.2583231811` | previous production anchor |
| v3 | historical | old holdout | hidden canonical `0.2481015189` | historical |
| v4 | historical | old holdout/cross-fit | hidden canonical `0.2531285195` | historical |
| **v5** | **current verified production** | **285,210 dev / 80,444 sealed gold / 5 folds / zero item overlap**, SHA `aae58f...eb55b` | **strict OOF `0.6018115534`** | **submit verified category-shrunk + HGB package** |

The old v2 hidden result must no longer be reported as “current production best”: v5 now has a fully organizer-smoked submission artifact. Platform Public/Private score for v5 is still unknown until submission.

## v5 retained ladder

| Step | Status | Strict OOF Macro AP | Key interpretation |
|---|---|---:|---|
| human structured audit | BASE | `0.5315527709` | immutable v5 baseline |
| category specialists | KEEP | `0.5476780661` | category structure matters |
| weak specialists | KEEP | `0.5514237339` | leakage-safe weak labels help |
| sparse TF-IDF specialists | KEEP signal | `0.5651306839` | rare SKU/model tokens are strong |
| supervised contrastive | KEEP signal | `0.5662217063` | task supervision beats pretrained-only embeddings |
| explicit per-key attributes | KEEP signal | `0.5683065131` | explicit key identity helps |
| 4-signal equal-rank | KEEP intermediate | `0.5870570848` | heterogeneous signals combine strongly |
| + pair teacher | KEEP signal | `0.5952697490` | teacher improves every fold |
| + typed explicit = six-signal | VERIFIED FALLBACK | `0.5975445721` | first stable six-signal package |
| outer-cross-fitted global simplex | KEEP intermediate | `0.5992720660` | honest learned weighting helps |
| fully nested category logistic | reject standalone | `0.5988060044` | diversity only |
| frozen global-meta + logistic equal-rank | KEEP intermediate | `0.5995921710` | label-free diversity fusion helps |
| fixed category-shrunk simplex | VERIFIED FALLBACK | `0.6009542418` | first honest crossing of 0.60 |
| fixed nonlinear HGB meta stack | complementary | `0.6006290885` | nonlinear category interactions help |
| **frozen category-shrunk + HGB equal-rank** | **CURRENT BEST / VERIFIED SUBMISSION** | **`0.6018115534`** | **all 5 folds improve vs category-shrunk** |

## Current best fold AP

Final frozen 50/50 rank fusion:

- fold 0: `0.600317954001536`;
- fold 1: `0.6073630562662657`;
- fold 2: `0.6122052716465903`;
- fold 3: `0.5973819202189384`;
- fold 4: `0.6105222735923926`.

Research evidence:

- run `31525549063`;
- artifact `9114783508`, digest `cca521d4c402fbd9d1aa9bce17902a1499d97b8fac97d681190c5365098ef8e0`;
- private HF `experiments/v5/category-hgb-fusion/79de99434912`.

## Current verified production submission

- ZIP: `ecup-v5-category-hgb-fusion-0.6018115534-submission.zip`;
- SHA-256: `442769bd2c92d43730d7034fb91d8a83e596a8445ae3c3f887783890e90284d5`;
- private HF: `submissions/v5/0.6018115534`;
- final workflow run `31526323018`, job `93895429369`;
- Actions artifact `9116032675`, digest `fc6a72f63146df414c5ff4de4aef62a4568e516a12d465830492941348824a46`;
- exact organizer-image offline smoke: passed;
- HGB joblib organizer-image load: passed;
- full tests after smoke: `230 passed, 1 warning`.

Verified fallbacks:

- category-shrunk `0.6009542418`: HF `submissions/v5/0.6009542418`, Actions `9114889240`;
- six-signal `0.5975445721`: HF `submissions/v5/0.5975445721`, Actions `9112337546`.

## Immutable validation facts

- metric: unweighted mean of sklearn `average_precision_score` over exactly 20 official categories;
- split SHA: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- development rows: `285,210`;
- sealed gold rows: `80,444`;
- item/component overlap across splits: `0`;
- sealed gold opened: **false**;
- sealed gold rows scored: **0**.

`0.6018115534` is strict local OOF, not a Public/Private leaderboard claim. Record platform results as a separate evidence axis after the exact verified ZIP is submitted.

## Important rejected/failure lessons

- direct attribute likelihood shift regressed to `0.5232189037`; use explicit estimator features instead;
- pretrained multilingual bi-encoder alone was near baseline (`0.5318080650`); supervised contrastive was the useful neural signal;
- first ruBERT teacher attempt was an integration failure before metrics, not a model-quality rejection;
- OOM/runner/API failures are infrastructure evidence, never metric evidence;
- learned meta layers require outer cross-fitting; full-development production refit scores are not validation;
- do not tune post-result fusion weights on the same OOF labels unless an additional nested layer is used.
