# E-CUP 2026 — Matching workspace

This directory is an isolated workspace for the E-CUP 2026 product-matching task. Existing Ansible files are not modified. Work is performed on branch `ecup-matching-2026`.

## Data destination

The workflow mirrors the organizer-provided parquet files to the private Hugging Face dataset repository:

`Maksim123321/e-cup-2026-matching-private`

The destination is explicitly created with `private=True` and `repo_type="dataset"`.

Files mirrored:

- `matches.parquet`
- `matches_llm.parquet`
- `items.parquet`
- `items_human.parquet`

Raw competition parquet files are never committed to this Git repository.

## Required GitHub Actions secret

The mirror job requires a Hugging Face User Access Token with permission to create/write repositories. Store it only as the repository Actions secret named exactly:

`HF_TOKEN`

Do not paste the token into source code, Git history, issue comments, workflow YAML, or chat.

To add it in GitHub:

1. Open `MakSoS1/Ansible_lab`.
2. Open **Settings**.
3. Open **Secrets and variables** → **Actions**.
4. Choose **New repository secret**.
5. Set the name to `HF_TOKEN`.
6. Paste a Hugging Face write-capable User Access Token as the secret value and save it.

The workflow checks that the secret is non-empty before downloading the multi-gigabyte full corpus.

## GitHub Actions workflow

Workflow: `.github/workflows/ecup-matching.yml`

It contains three jobs:

1. `tests` — network-free unit tests for the mirror and profiler.
2. `profile-human` — downloads only `matches.parquet` and `items_human.parquet`, produces aggregate-only `profile.json` and `profile.md`, then deletes raw files from the runner.
3. `mirror-hf` — downloads each official parquet file sequentially, uploads it to the private HF dataset, verifies its presence in the Hub, deletes the local copy, then continues with the next file.

The aggregate profile is stored as the temporary Actions artifact `ecup-human-profile` for 7 days. It intentionally contains no raw product names, attributes, or item IDs.

## Local unit tests

```bash
python -m pip install -r ecup_matching/requirements-ci.txt
python -m pytest ecup_matching/tests -q
```

The unit tests use synthetic data and fake network/API objects; they do not download competition data.

## Solution research

The full architecture comparison, ten candidate solution families, selected approach, validation design, runtime strategy, and ten-step iteration ladder are documented in:

`docs/superpowers/specs/2026-08-10-ecup-matching-design.md`

The executable implementation plan is documented in:

`docs/superpowers/plans/2026-08-10-ecup-matching.md`

### Selected family

The selected direction is a **noise-aware distilled hybrid cascade**:

1. deterministic product normalization preserving model numbers, dimensions, quantities, units and structured attributes;
2. item-disjoint validation that mimics unseen test items;
3. a fast lexical/numeric/attribute feature model;
4. a multilingual/Russian bi-encoder that encodes each unique item once;
5. source-aware weak-label training where human labels receive much higher weight than LLM labels;
6. hard-negative mining;
7. a compact distilled Cross-Encoder applied only to uncertain/hard pairs;
8. category-aware ranking features only where validation proves they help;
9. macro Average Precision and per-category AP as the primary quality metrics;
10. inference wall-clock time as a first-class acceptance criterion for every experiment.

## Competition-data handling warning

The competition rules should be treated as controlling. Keep the Hugging Face dataset private, do not commit raw organizer data, and review the publication/redistribution restrictions before making competition-specific solution material publicly visible. This repository itself is public, so keep that constraint in mind before merging or publishing additional model code, reports, weights, or data-derived artifacts.
