# E-CUP Matching — Iteration v4 Plan

Date: 2026-08-11
Status: in progress

## Hypothesis

The retained v3 neural branch leaves substantial quality on the table because it uses `rubert-tiny2`, only 180,000 human training rows and no neural weak-label curriculum. A stronger Russian BERT cross-encoder trained first on the complete leakage-safe human train, then continued with confidence-filtered LLM soft labels, then optionally refined with hard-negative replay should improve the unchanged item-disjoint Macro AP while the H100 organizer target can absorb the larger inference model.

## Fixed baseline

Retained v3:

- model: v2b structured anchor + `cointegrated/rubert-tiny2` stage-1 global blend;
- structured/neural weights: `0.55 / 0.45`;
- Macro AP: `0.5254642645846543`;
- validation rows: `73,131`;
- train/validation item overlap: `0`;
- immutable submission SHA-256: `b833ceb203f8cc7d87517257df8ee5e0a2590075db0ecd2932b8281950015660`;
- canonical private artifact: `submissions/v3/canonical/b833ceb203f8cc7d87517257df8ee5e0a2590075db0ecd2932b8281950015660/ecup-v3-submission.zip`.

v3 remains untouched until a fully verified v4 strictly beats it.

## Base model

Primary v4 neural model: `ai-forever/ruBert-base`.

Verified public model facts at design time:

- license: Apache-2.0;
- language: Russian;
- architecture: BERT encoder, 12 layers, hidden size 768, 12 attention heads;
- approximately 178M parameters;
- model repository approximately 718 MB.

The retained run must record and package an exact immutable Hugging Face revision rather than rely on moving `main`.

## Data and split

Private source: `Maksim123321/e-cup-2026-matching-private`.

Inputs:

- `matches.parquet` — authoritative human pairs;
- `items_human.parquet` — authoritative human item rows;
- `matches_llm.parquet` — LLM soft labels;
- `items.parquet` — full item universe;
- retained v2 structured validation predictions.

Fixed split:

- human outer train: 292,523 pairs before any curriculum transformation;
- validation: 73,131 pairs;
- item overlap: exactly 0;
- split implementation remains `fixed_v1_split` / connected-component item-disjoint protocol.

## v4a — full human stronger model

Train the stronger cross-encoder on the complete leakage-safe human training curriculum. Do not reproduce the v3 180k compaction.

Initial configuration:

- max length: 256;
- CUDA mixed precision;
- micro-batch: 4;
- gradient accumulation: 8;
- human source dominates all weighting;
- complete validation evaluated after training;
- raw neural, global v2b blend and shrinkage category blend are compared.

OOM fallback, in fixed order:

1. micro-batch 4 → 2;
2. micro-batch 2 → 1;
3. enable gradient checkpointing;
4. max length 256 → 192.

Never solve OOM by dropping human rows or changing validation.

## v4b — high-confidence weak curriculum

Warm-start from v4a.

Weak confidence policy:

- `target <= 0.03` or `>= 0.97`: weight 1.0;
- `0.03 < target <= 0.15` or `0.85 <= target < 0.97`: weight 0.6;
- `0.15 < target <= 0.30` or `0.70 <= target < 0.85`: weight 0.3;
- `(0.30, 0.70)`: excluded from direct supervision.

Safety:

- canonical pair direction;
- exact weak duplicate collapse;
- exact human conflicts removed;
- authoritative-positive-component false negatives removed;
- any weak pair touching a validation item removed;
- deterministic category/class-balanced weak sampling.

First retained weak cap: 600,000 rows. Human rows remain present and dominant. A later 800,000-row ablation is allowed only under a different immutable run ID after 600k results are safely stored.

## v4c — hard negatives with replay

Start from whichever of v4a/v4b has the better complete-validation Macro AP.

Mine difficult negatives from authoritative human negatives plus eligible high-confidence weak negatives using current neural score and available structured/neural disagreement signals.

Fine-tune on a deterministic replay mixture:

- 25% mined hard negatives;
- 25% positives/hard positives;
- 50% ordinary examples sampled from the parent curriculum.

Use a lower learning rate and short continuation. Reject v4c if it does not exceed its parent on the complete validation.

## Blend selection

For each stage evaluate:

1. neural score only;
2. deterministic global alpha grid against v2b structured scores;
3. category-specific alphas with shrinkage toward the global alpha.

No hard classification threshold is used. Final submission writes continuous scores.

## GPU backend

Primary: isolated home RTX 2060 SUPER through private `MakSoS1/gpu-dispatch`.

Before the production train, run a short fixed CUDA benchmark and record GPU, peak VRAM and examples/second. GitHub M1 MPS remains fallback/reference only.

The organizer target is H100 80 GB + 20 CPU + 200 GB RAM, so local CPU/M1 submission timing is not an intermediate model-selection criterion. Runtime is a hard gate after a quality winner exists.

## Artifact isolation

Every training execution stores a unique path:

- `experiments/v4/runs/<source-sha>/<run-id>/v4a/...`
- `experiments/v4/runs/<source-sha>/<run-id>/v4b/...`
- `experiments/v4/runs/<source-sha>/<run-id>/v4c/...`

No training run writes over another run.

Only a final verified winner may be promoted to:

- `submissions/v4/canonical/<submission-sha256>/ecup-v4-submission.zip`;
- `submissions/v4/canonical/<submission-sha256>/v4-package-metrics.json`.

## Primary metric

Unweighted mean of `sklearn.metrics.average_precision_score` over all 20 categories.

## Success criteria

Required for retained v4:

- exact validation rows = 73,131;
- validation overlap = 0;
- Macro AP > `0.5254642645846543`;
- target Macro AP >= 0.54;
- stretch Macro AP >= 0.55;
- all 20 category APs recorded;
- exact model revision recorded;
- actual NVIDIA CUDA training verified;
- final exact organizer-image offline run succeeds with network disabled;
- output rows/order/schema/range/finite checks pass;
- ZIP <5 GB and Docker image <15 GB;
- canonical private HF artifact checksum and presence verified;
- public tests, memory policy, Memora ingest and v4 checkpoint pass.

## Reject criteria

- any validation item leakage;
- any ambiguity that a run overwrote another run;
- final v4 score <= retained v3;
- any requirement to expose credentials or publicize private competition data;
- final package fails offline organizer execution or resource limits.

Individual v4b/v4c stages may be rejected while an earlier v4 stage remains the winner.
