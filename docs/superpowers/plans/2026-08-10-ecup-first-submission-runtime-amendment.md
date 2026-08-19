# E-CUP v1 Runtime Amendment

This amendment supersedes the CatBoost/RapidFuzz assumptions in `2026-08-10-ecup-first-submission.md` after Task 1 runtime inspection.

## Evidence-driven change

The exact organizer image `odsai/ecup26-matching-baseline:1.0` was pulled and probed on GitHub Actions. It contains:

- Python 3.12.3
- NumPy 2.2.3
- pandas 2.2.3
- scikit-learn 1.9.0
- joblib 1.4.2
- CatBoost: absent
- RapidFuzz: absent

The official image is the actual evaluation environment, so v1 must not assume CatBoost/RapidFuzz are importable offline.

## Revised v1 implementation

- Model: `sklearn.ensemble.HistGradientBoostingClassifier` behind a `ColumnTransformer`/`Pipeline`.
- Category: one-hot encoded with `OneHotEncoder(handle_unknown='ignore')`.
- String similarities: standard-library `difflib.SequenceMatcher` + explicit token/character Jaccard features.
- Structured signals: exact/containment, numbers, alphanumeric model codes, normalized quantities/units, flattened JSON attribute key/value agreement and conflicts.
- Validation: connected-component/item-disjoint 80/20 split, Macro Average Precision across 20 categories.
- Weighting: inverse category-pair frequency so training pressure better matches macro evaluation.
- Serialization: training runs *inside the exact organizer Docker image*, eliminating sklearn/joblib version mismatch.
- Packaging: official `metadata.json`, root `run.py`, bundled model/manifest/runtime Python modules.
- Final verification: ZIP is executed with `--network none` inside the exact organizer image on hidden-test-style parquet inputs.

## Why this is preferable for the first submission

The first submission is a correctness/runtime anchor, not the final quality ceiling. It has no missing wheel risk, no pairwise transformer latency, small model size, and a large runtime margin. Once a valid score and measured inference time exist, subsequent iterations can justify bundled libraries or neural artifacts by measured Macro-AP gain per runtime cost.
