# E-CUP Matching — Canonical Project State

Updated: 2026-08-11

## Objective

Maximize E-CUP 2026 product-matching Macro AP while keeping validation honest for unseen products, submission runtime reproducible/offline-compatible, and competition artifacts private.

## Current state — read this first

- **Current honest development best:** `0.6018115534135564` strict OOF Macro AP.
- **Current verified production submission:** `v5-category-shrunk-hgb-equal-rank-fusion`.
- Competition ZIP: `ecup-v5-category-hgb-fusion-0.6018115534-submission.zip`.
- Competition ZIP SHA-256: `442769bd2c92d43730d7034fb91d8a83e596a8445ae3c3f887783890e90284d5`.
- Private HF: `Maksim123321/e-cup-2026-matching-private`, prefix `submissions/v5/0.6018115534`.
- Final production workflow: run `31526323018`, job `93895429369`.
- Actions artifact: ID `9116032675`, digest `fc6a72f63146df414c5ff4de4aef62a4568e516a12d465830492941348824a46`.
- Exact organizer-image offline smoke: **passed**.
- HGB joblib load inside organizer image before packaging: **passed**.
- Full repository tests after smoke: **230 passed, 1 warning**.
- Sealed gold: **unopened**, `0` rows scored.
- Public/private leaderboard score: **unknown until the exact verified ZIP is submitted to the platform**.

Do not regress this state to the old `0.5683065` explicit-only best, the `0.5975446` six-signal fallback, or the historical v2 hidden anchor when answering “current best”. They remain history/fallbacks only.

## Official metric and validation comparability

- Competition metric is `sklearn.metrics.average_precision_score` separately for each of the 20 official categories, then the unweighted mean.
- Local strict metric uses the same formula and requires exactly the official 20 categories and both classes.
- Human labels: `365,654` rows.
- Connected item components: `345,654`.
- Development rows: `285,210`.
- Sealed-gold rows: `80,444`.
- Five immutable development folds.
- Cross-split item overlap: `0`.
- Split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.

The component-disjoint split is deliberately stricter than random pair splitting: one item/component cannot occur in both meta-train and held fold. Local absolute AP may differ from Public/Private due to distribution shift, but deltas under this protocol are meaningful and the metric definition is organizer-compatible.

## Retained v5 ladder

| Stage | Strict OOF Macro AP | Decision |
|---|---:|---|
| audit baseline | `0.5315527708634168` | baseline |
| category specialists | `0.5476780661335778` | keep |
| weak specialists | `0.5514237338676234` | keep |
| sparse specialists | `0.5651306838802859` | keep signal |
| supervised contrastive | `0.5662217062664492` | keep signal |
| explicit attributes | `0.5683065131240066` | keep signal |
| four-signal equal-rank | `0.5870570848443828` | keep intermediate |
| + pair teacher | `0.5952697490140912` | keep signal |
| + typed explicit, six-signal equal-rank | `0.5975445721449741` | verified fallback |
| outer-cross-fitted global simplex | `0.5992720660193247` | keep intermediate |
| nested category logistic | `0.5988060044248327` | reject standalone, retain diversity |
| global-meta + logistic frozen 50/50 rank | `0.5995921709945611` | keep intermediate |
| fixed category-shrunk simplex | `0.60095424180184` | verified fallback, first honest `>0.60` |
| fixed HGB meta stack | `0.6006290884983169` | complementary source |
| **category-shrunk + HGB frozen 50/50 rank** | **`0.6018115534135564`** | **current dev + production best** |

## Current best architecture

The six underlying production signals are unchanged from the verified `0.5975446` package:

1. weak category specialist;
2. sparse TF-IDF specialist;
3. explicit per-key attribute specialist;
4. supervised contrastive item score;
5. pair-teacher score;
6. typed/canonicalized explicit specialist.

All six are converted target-free to percentile ranks.

Two complementary meta components are fit on development OOF evidence:

### A. Fixed category-shrunk simplex

- global nonnegative simplex optimized for Macro AP on outer-train;
- local simplex per official category on outer-train;
- fixed shrinkage prior `8000`, frozen before result:
  `(support * local + 8000 * global) / (support + 8000)`;
- strict OOF `0.60095424180184`.

### B. Fixed nonlinear HGB meta stack

- six percentile ranks + official category categorical feature;
- `HistGradientBoostingClassifier` with parameters frozen before result;
- no HPO on the observed OOF result;
- strict OOF `0.6006290884983169`.

### Final retained fusion

Frozen before its result was inspected:

`0.5 * percentile_rank(category_shrunk_score) + 0.5 * percentile_rank(hgb_score)`

Strict OOF: **`0.6018115534135564`**.

It improved category-shrunk alone on **all five outer folds**:

- fold 0: `0.600317954001536`;
- fold 1: `0.6073630562662657`;
- fold 2: `0.6122052716465903`;
- fold 3: `0.5973819202189384`;
- fold 4: `0.6105222735923926`.

Evidence run `31525549063`; metrics artifact `9114783508`; private HF `experiments/v5/category-hgb-fusion/79de99434912`.

## Why this OOF is honest

- Every target-fitted meta component predicts an outer fold using parameters fit without labels from that fold.
- Leakage tests explicitly mutate a held fold's labels and require its predictions/weights to remain unchanged.
- The category-logistic branch selected regularization by inner OOF only inside outer-train.
- Category-shrinkage prior `8000`, HGB hyperparameters, and final 50/50 fusion were frozen before inspecting their respective real OOF result.
- Full-development production refits are explicitly marked **not validation**.
- Sealed gold remains unopened; no gold item/label was used for feature learning, mining, meta fitting, or model choice.

## Verified production package

The final package reuses the byte-verified six-signal production models and changes only the retained final meta fusion/artifacts.

Final run `31526323018` verified:

1. exact CI/organizer sklearn version compatibility (`1.9.0`);
2. immutable fusion evidence `0.6018115534135564`;
3. deterministic full-development category/HGB production refits;
4. HGB joblib deserialization inside `odsai/ecup26-matching-baseline:1.0`;
5. base fallback inner ZIP SHA before patching;
6. final ZIP integrity and `<5GB` size;
7. full `run.py` execution in organizer image with `--network none`, read-only filesystem, 64-pair smoke;
8. output columns exactly `id1,id2,predict`, all finite, all 64 predictions nonconstant;
9. full tests: `230 passed, 1 warning`;
10. exact final ZIP upload to private HF;
11. exact same final package retained as GitHub Actions artifact.

Final ZIP size: `1,144,877,898` bytes.

## Verified fallbacks

### Category-shrunk only — `0.6009542418`

- ZIP: `ecup-v5-category-shrunk-0.6009542418-submission.zip`;
- SHA-256: `3a5341c42346727793ab8877ee6bc8f07e3ac4f18f97c32a9d39d76b5e0609c1`;
- HF: `submissions/v5/0.6009542418`;
- Actions artifact `9114889240`;
- organizer smoke passed; `225 passed, 1 warning`.

### Six-signal — `0.5975445721`

- ZIP: `ecup-v5-six-signal-0.5975445721-submission.zip`;
- SHA-256: `ee6fec40fe7e79095c33b5a2ed8a1c6cb40e01c3a8e90850c7459d5f1afad06e`;
- HF: `submissions/v5/0.5975445721`;
- Actions artifact `9112337546`;
- organizer smoke passed.

## Rejected/diagnostic branches worth remembering

- Direct attribute likelihood score shift: `0.523218903672764`, reject. Explicit estimator features are useful; unconditional score shifts are not.
- Pretrained bi-encoder alone: `0.5318080650341337`, insufficient. Supervised contrastive item-space is materially better.
- First ruBERT teacher run failed integration before metrics due stale helper call; it was not evidence against the model family.
- Nested category logistic `0.5988060044248327` is not standalone best but its diversity previously helped a frozen fusion.
- Do not interpret infrastructure failures/OOM/API mismatches as model-quality evidence.

## Memora policy

- Memora pin: `bc64ff745a9b2c0e6245e0137654f041fba0c155`.
- Local SQLite/TF-IDF only; LLM/graph/auto-capture disabled.
- Canonical sources include `PROJECT_STATE.md`, `EXPERIMENT_INDEX.md`, `DECISIONS.md`, `CURRENT.json`, plans/specs/results and `SAFE_METRICS.json`.
- Checkpoint only from GREEN repository state.
- Machine-readable `CURRENT.json` and `SAFE_METRICS.json` are mandatory memory sources so semantic retrieval cannot silently regress to an old narrative state.

## Next action

**Submit exactly** `ecup-v5-category-hgb-fusion-0.6018115534-submission.zip` from private HF `submissions/v5/0.6018115534` to the E-CUP platform.

After platform evaluation, record Public/Private score separately. Do not overwrite strict local OOF with leaderboard AP and do not claim local `0.6018115534` is the Public/Private score.
