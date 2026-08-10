# E-CUP Matching — Iteration v2 Plan

Date: 2026-08-10
Status: in progress

## Objective

Beat the v1 fixed item-disjoint validation Macro AP of **0.49616548946964434** by independently transferring the strongest reproducible product-matching ideas visible in public E-CUP 2024 and later Kuper product-matching material.

## Fixed evaluation

- Human validation split must be exactly the v1 component/item-disjoint split.
- Validation rows: expected 73,131 from v1.
- Train/validation item overlap must remain zero.
- Headline metric: unweighted mean of 20 per-category Average Precision scores.
- A calibration subset is carved only from the v1 training components and is used for blend selection; fixed validation is not used to train base models.

## v2 ablations

### v2a — 2024 structured transfer

Human-only training with:

- canonical pair deduplication/conflict report;
- capped positive-component transitive closure with explicit negative veto;
- brand match/conflict;
- training-derived category-aware attribute importance;
- weighted attribute agreement/conflict;
- critical model/number/quantity contradiction count;
- hard-negative score.

### v2b — weak-label curriculum

Add `matches_llm.parquet` with soft/confidence weights:

- <=0.03 or >=0.97: 1.0;
- 0.03–0.15 / 0.85–0.97: 0.6;
- 0.15–0.30 / 0.70–0.85: 0.3;
- 0.30–0.70: excluded initially;
- human base weight: 10.0 before category balancing.

Weak examples conflicting with human labels or positive human identity components are removed.

### v2c — hard negatives

Prefer weak/human negatives with high lexical/attribute similarity plus explicit variant contradictions. Reweight/mine examples such as same brand/model family but different memory, size, volume, pack count or generation.

### v2d — compact GPU reranker

Fine-tune a small Russian/multilingual pair classifier on Lightning AI using human + selected weak examples. Max sequence length 256. Preserve soft targets for weak labels.

### v2e — hard-negative second stage

Mine high-scoring negative errors from v2d and run a short second-stage fine-tune emphasizing those examples.

### v2-final — blend

Blend structured v2c and neural v2d/v2e using a coefficient selected on the item-disjoint calibration subset. Apply structured contradiction penalty only if calibration improves. Evaluate retained candidate exactly on the fixed v1 validation and then refit final training weights without changing hyperparameters.

## Runtime budget

- Use CPU/GitHub Actions for deterministic preprocessing, structured ablations, packaging and organizer smoke tests.
- Use Lightning GPU only for transformer fine-tuning/scoring.
- Final inference must run offline in `odsai/ecup26-matching-baseline:1.0`.
- Target at least 25% headroom against the 13-minute private runtime limit.
- Final ZIP < 5 GiB.

## Security

- No raw parquet, weights, submission ZIP, SQLite DB or secret in public Git.
- Lightning credentials exist only as runtime secrets; never workflow inputs or committed values.
- Private durable model/submission artifacts go to `Maksim123321/e-cup-2026-matching-private`.

## Abort / acceptance criteria

- Reject any candidate with item leakage.
- Reject any neural candidate whose offline runtime does not fit the organizer budget.
- If final fixed-validation Macro AP does not exceed v1, keep v1 as best and document v2 as rejected; still preserve reproducible v2 metrics privately/publicly as appropriate.
