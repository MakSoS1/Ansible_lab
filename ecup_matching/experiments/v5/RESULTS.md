# E-CUP Matching — Iteration v5 Results

Date: 2026-08-11  
Status: **target exceeded; production submission verified; sealed gold unopened**

## Final result

The v5 engineering target of strict local Macro AP `>= 0.60` was reached and exceeded without opening sealed gold.

**Current honest strict OOF best:**

`0.6018115534135564`

Final retained model-selection formula:

`0.5 * percentile_rank(category_shrunk_oof) + 0.5 * percentile_rank(hgb_stack_oof)`

The 50/50 formula was frozen before its real OOF metric was inspected. Both component vectors are fully outer-cross-fitted. No post-result weight search was performed.

Fold Macro AP:

| Fold | Final AP |
|---:|---:|
| 0 | `0.600317954001536` |
| 1 | `0.6073630562662657` |
| 2 | `0.6122052716465903` |
| 3 | `0.5973819202189384` |
| 4 | `0.6105222735923926` |

All five folds improve over the category-shrunk component alone.

Research evidence:

- workflow run `31525549063`;
- artifact ID `9114783508`;
- artifact digest `cca521d4c402fbd9d1aa9bce17902a1499d97b8fac97d681190c5365098ef8e0`;
- private HF `experiments/v5/category-hgb-fusion/79de99434912`.

## Validation contract

The comparison protocol remained fixed throughout the retained v5 ladder:

- human rows: `365,654`;
- connected item components: `345,654`;
- development rows: `285,210`;
- sealed-gold rows: `80,444`;
- five immutable development folds;
- cross-split item overlap: `0`;
- split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- sealed gold opened: **false**;
- sealed-gold rows scored: **0**.

The strict metric is the organizer metric: `sklearn.metrics.average_precision_score` for each of the 20 official categories followed by their unweighted mean. Strict mode requires exactly those 20 categories and both classes.

The component-disjoint split is stricter than random pair splitting because an item/component cannot occur in both meta-train and held fold. This makes local deltas suitable for model selection on unseen products. It does **not** imply that local `0.6018115534` will numerically equal Public or Private leaderboard AP; test distribution can differ.

## Retained ladder

| Stage | Strict OOF Macro AP | Decision |
|---|---:|---|
| human structured audit | `0.5315527708634168` | baseline |
| category specialists | `0.5476780661335778` | keep |
| weak specialists | `0.5514237338676234` | keep |
| sparse TF-IDF specialists | `0.5651306838802859` | keep signal |
| supervised contrastive | `0.5662217062664492` | keep signal |
| explicit per-key attributes | `0.5683065131240066` | keep signal |
| four-signal equal-rank | `0.5870570848443828` | keep intermediate |
| + pair teacher | `0.5952697490140912` | keep signal |
| six-signal + typed explicit | `0.5975445721449741` | verified fallback |
| outer-cross-fitted global simplex | `0.5992720660193247` | keep intermediate |
| fully nested category logistic | `0.5988060044248327` | reject standalone |
| frozen global-meta + logistic rank fusion | `0.5995921709945611` | keep intermediate |
| fixed category-shrunk simplex | `0.60095424180184` | verified fallback; first `>0.60` |
| fixed HGB meta stack | `0.6006290884983169` | complementary source |
| **frozen category-shrunk + HGB rank fusion** | **`0.6018115534135564`** | **current best / verified submission** |

## Six retained base signals

The final package preserves the same six underlying production signals already validated in the `0.5975445721` package:

1. weak category specialist;
2. sparse TF-IDF specialist;
3. explicit per-key attribute specialist;
4. supervised contrastive item score;
5. pair-teacher score;
6. typed/canonicalized explicit specialist.

Each is converted to a target-free percentile rank before meta scoring.

## Meta component A — fixed category-shrunk simplex

Predeclared before its result:

- global nonnegative simplex fit on the other four outer folds by direct Macro AP coordinate ascent;
- local category simplex fit only on the same outer-train rows;
- fixed shrinkage prior `8000`:
  `(support * local + 8000 * global) / (support + 8000)`;
- no post-result prior grid or category-specific tuning.

Strict outer OOF:

**`0.60095424180184`**

It was reproduced exactly after refactoring. Evidence run `31524781399`, artifact `9114649149`, private HF `experiments/v5/category-shrunk/efa629cc0435`.

A standalone organizer-verified fallback submission was also retained:

- ZIP `ecup-v5-category-shrunk-0.6009542418-submission.zip`;
- SHA-256 `3a5341c42346727793ab8877ee6bc8f07e3ac4f18f97c32a9d39d76b5e0609c1`;
- private HF `submissions/v5/0.6009542418`;
- Actions artifact `9114889240`;
- organizer smoke passed;
- full tests `225 passed, 1 warning`.

## Meta component B — fixed nonlinear HGB

Predeclared fixed `HistGradientBoostingClassifier` over the same six percentile ranks plus official category as a categorical feature:

- learning rate `0.05`;
- max iter `160`;
- max leaf nodes `15`;
- max depth `3`;
- min samples leaf `200`;
- L2 regularization `5.0`;
- early stopping disabled;
- random state `20260811`;
- category-balanced sample weights;
- no HGB parameter grid after seeing OOF.

Strict outer OOF:

**`0.6006290884983169`**

Standalone HGB was below category-shrunk, but its different nonlinear inductive bias was complementary. Private HF `experiments/v5/hgb-meta-stack/84a934484619`.

## Leakage safeguards for target-fitted meta models

Tests explicitly enforce that changing labels inside a held outer fold cannot change predictions/weights for that same fold.

Additional safeguards:

- category-logistic hyperparameter selection was fully nested inside outer-train;
- category shrinkage prior was frozen before real evaluation;
- HGB hyperparameters were frozen before real evaluation;
- final 50/50 fusion was frozen before its result;
- full-development refits are production-only and their in-sample scores are never reported as validation;
- no sealed-gold labels/items participate in development model choice.

## Final verified production submission

**ZIP:** `ecup-v5-category-hgb-fusion-0.6018115534-submission.zip`  
**ZIP SHA-256:** `442769bd2c92d43730d7034fb91d8a83e596a8445ae3c3f887783890e90284d5`  
**ZIP size:** `1,144,877,898` bytes

Private Hugging Face:

- repo `Maksim123321/e-cup-2026-matching-private`;
- prefix `submissions/v5/0.6018115534`.

Final production verification:

- workflow run `31526323018`;
- job `93895429369`;
- source commit `695ea6ec38e47048007b58389d15c611899bdfe5`;
- Actions artifact ID `9116032675`;
- Actions artifact digest `fc6a72f63146df414c5ff4de4aef62a4568e516a12d465830492941348824a46`;
- organizer image `odsai/ecup26-matching-baseline:1.0`;
- CI sklearn = organizer sklearn = `1.9.0`;
- HGB joblib loaded successfully inside organizer image before packaging;
- final ZIP built by patching only the selected meta runtime/artifacts over the byte-verified six-signal package;
- end-to-end `run.py` smoke passed inside organizer image with `--network none` and read-only filesystem;
- smoke output exactly `id1,id2,predict`, all rows present, finite and nonconstant;
- full repository tests after smoke: `230 passed, 1 warning`;
- exact smoked ZIP uploaded to HF and retained as Actions artifact.

The 64-pair CPU smoke took about `72.75 s`; this is a small CPU organizer-image correctness smoke, not the contest H100 full-dataset runtime benchmark. The heavy neural paths use CUDA when available, and the final HGB/category meta layer is negligible relative to the six base signals.

## Six-signal verified fallback

The previous fallback remains preserved:

- strict OOF `0.5975445721449741`;
- ZIP `ecup-v5-six-signal-0.5975445721-submission.zip`;
- SHA-256 `ee6fec40fe7e79095c33b5a2ed8a1c6cb40e01c3a8e90850c7459d5f1afad06e`;
- private HF `submissions/v5/0.5975445721`;
- Actions artifact `9112337546`;
- organizer smoke passed.

## Rejected/diagnostic branches to remember

- Direct attribute likelihood shift: `0.523218903672764`, reject. Explicit attribute estimator features help; unconditional score shifts do not.
- Pretrained multilingual bi-encoder alone: `0.5318080650341337`, insufficient; supervised contrastive was useful.
- Nested category logistic: `0.5988060044248327`, not standalone best.
- First ruBERT teacher attempt failed integration before comparable OOF because of a stale helper call; it was not evidence against the model family.
- Infrastructure errors/OOM/API mismatches must never be interpreted as model-score evidence.

## Final decision

The `0.60` local engineering milestone is complete. The exact verified `0.6018115534` ZIP is the submission to upload to the E-CUP platform.

Public/Private leaderboard AP remains unknown until platform evaluation. Record it separately from strict local OOF, and do not relabel `0.6018115534` as leaderboard evidence.
