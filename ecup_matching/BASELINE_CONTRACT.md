# Official E-CUP Matching Submission Contract

Verified on 2026-08-10 by downloading and inspecting the organizer archives on a GitHub Actions Ubuntu runner.

## Required archive metadata

The lightweight baseline contains `metadata.json` at archive root:

```json
{
    "image": "odsai/ecup26-matching-baseline:1.0",
    "entry_point": "python -u run.py"
}
```

Our submission keeps these values unchanged.

## Entrypoint CLI

Organizer `run.py` accepts exactly:

```text
--output_path <csv path>
--items_path <items parquet path>
--matches_path <matches parquet path>
```

The new `run.py` must accept the same underscore-style argument names and write the result to `--output_path`.

## Output schema

CSV with exactly these columns in input-pair order:

```text
id1,id2,predict
```

`predict` is a continuous numeric matching score/probability.

## Lightweight baseline archive tree

The official lightweight ZIP has files at archive root (plus ignorable macOS metadata):

```text
metadata.json
run.py
src/utils.py
baseline_logreg_l12.joblib
```

The full example additionally contains a local transformer under `models/cross-encoder-ms-marco-MiniLM-L12-v2/` and wraps the solution files in a `matching-baseline-submit/` directory. For our generated archive we use the lightweight/root layout because it is the minimal organizer-provided valid template and avoids ambiguous extra directory nesting.

## Runtime packages proven by official source

The organizer baseline imports and therefore the baseline image is expected to provide at least:

- pandas
- numpy
- joblib
- scikit-learn (`StandardScaler`, `LogisticRegression`, `MLPClassifier`, `Pipeline`)
- torch
- sentence-transformers
- transformers
- tqdm

No assumption is made that CatBoost, LightGBM or RapidFuzz are installed. First submission v1 therefore uses a scikit-learn model and standard-library text similarity code only. Later submissions may bundle additional compatible runtime artifacts after explicit container verification.

## Performance reference

The organizer full baseline performs pairwise transformer inference with max sequence length 256 and batch size 512. Our v1 performs only CPU structured/string feature extraction plus a compact sklearn classifier, giving substantial runtime headroom. Neural reranking can therefore be added later only where it improves the quality/runtime Pareto frontier.
