# E-CUP Matching — Iteration v6 Runtime-Constrained Results

Date: 2026-08-11
Status: in progress

## Objective

v5 exceeded the local quality target but is too slow for the runtime-constrained production objective. v6 therefore uses a hard Pareto rule:

1. retain only architectures with honest strict component-disjoint OOF Macro AP `>= 0.6000`;
2. among passing architectures, minimize real end-to-end inference runtime;
3. keep the sealed gold split unopened.

The pure-quality v5 best remains `0.6018115534135564`. v6 is not claimed to improve that number; it is the runtime-constrained production line.

## Selected v6 quality point

**Strict outer OOF Macro AP:** `0.6006003614522999`

Selected architecture:

- weak category specialist;
- sparse TF-IDF specialist;
- explicit per-key specialist;
- supervised contrastive item score;
- typed/canonicalized explicit specialist;
- pair teacher evaluated only for the target-free top-disagreement `95%` of pairs within each official category;
- for remaining pairs, teacher is replaced by the unweighted mean of the five target-free percentile ranks;
- selected teacher scores are percentile-ranked only over selected teacher rows;
- final meta layer remains a full outer-cross-fitted category-shrunk simplex plus fixed HGB, fused as `0.5 * percentile_rank(category_shrunk) + 0.5 * percentile_rank(HGB)`.

Requested teacher coverage: `0.95`.
Actual development teacher fraction: `0.9500262964131693`.

Selection evidence:

- workflow run `31531141700`;
- job `93911179929`;
- source commit `fb15ec43a90c892c416acb2d10fe04cc126a4398`;
- private HF prefix `experiments/v6/teacher-gate/95/fb15ec43a90c`.

## Validation contract

- development rows: `285,210`;
- immutable five outer folds;
- component/item-disjoint validation;
- split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- sealed gold opened: **false**;
- sealed gold rows scored: **0**;
- strict metric: unweighted mean of `average_precision_score` over exactly the 20 official categories;
- gate selection uses only predictions and official category, never target labels;
- meta models are fit only on outer-train rows for each held fold;
- no post-result weight search is used for the retained formula.

## Runtime/quality exploration

### Remove expensive signals and refit meta

| Architecture | Strict OOF Macro AP | Decision |
|---|---:|---|
| structured only | `0.5808404005946962` | reject |
| no teacher | `0.5931387077244183` | reject |
| no contrastive | `0.5928725263319000` | reject |

Simple deletion cannot satisfy the `0.6000` quality gate.

### Target-free partial teacher

| Teacher coverage | Strict OOF Macro AP | Decision |
|---:|---:|---|
| 25% | `0.5929214688140778` | reject |
| 55% | `0.5966896566149946` | reject |
| 85% | `0.5999300791828578` | reject narrowly |
| **95%** | **`0.6006003614522999`** | **retain** |

### Teacher distillation

A leakage-safe HGB student was trained on teacher percentile ranks using only outer-train teacher values for each held fold.

Strict OOF: `0.5931935841654697` — reject.

### Student + real-teacher hybrid

| Real teacher coverage | Strict OOF Macro AP | Decision |
|---:|---:|---|
| 25% | `0.5955200995988962` | reject |
| 40% | `0.5964542084428204` | reject |
| 55% | `0.5974645385215002` | reject |
| 70% | `0.5985536258037350` | reject |
| 85% | `0.5998746122650258` | reject narrowly |

The hybrid did not preserve enough teacher ordering to justify replacing the simpler gate95 architecture.

## Production runtime implementation

The selected model keeps FP32 neural semantics and obtains speed from implementation changes rather than mixed precision, because the retained OOF margin over `0.6000` is small.

Implemented changes:

- target-free teacher gate avoids approximately 5% of pair-teacher calls;
- contrastive items are stable length-bucketed before tokenization to reduce padding;
- selected teacher pairs are stable length-bucketed before cross-encoder inference;
- CUDA batch sizes are scaled by detected VRAM;
- RTX-class 8 GiB default: contrastive `256`, teacher `96`;
- 20+ GiB default: contrastive `512`, teacher `192`;
- 60+ GiB default: contrastive `1024`, teacher `384`;
- CUDA OOM fallback halves batches safely;
- non-blocking CUDA transfers;
- PyTorch SDPA is requested where supported, with eager fallback;
- structured feature chunk size is `10,000` (pinned by test; see chunk-size note below);
- structured chunks are scored across worker processes inherited by `fork`;
- `difflib` ratios are shared between the legacy and typed structured passes;
- the contrastive and teacher text caches share one `ItemNorm` pass;
- phase-level runtime telemetry is emitted for load, structured, contrastive, gate, teacher, meta and write stages;
- no network is required at inference.

## Production equivalence safeguards

Tests verify that assembling a teacher signal from only selected teacher rows is numerically identical to the validation formula that receives a full teacher vector and then masks it. Changing unselected teacher values cannot change gated predictions.

The runtime gate no longer imports training-only `v5_meta_blend`; its signal order aliases the production-safe `v5_production.FINAL_SIGNAL_NAMES`. This removes the transitive training-only dependency that previously caused organizer-smoke import failure.

## Latest packaging checkpoint

Workflow run `31535674086`, source `4da50f66942472b8e8b70270cbeb00639930b6b5`, established the following before the repository documentation gate:

- selected v6 contract tests: **passed** (`36 passed`);
- production category/HGB refit: **passed**;
- production refit time: approximately `92.3s` on the GitHub-hosted CPU runner;
- peak production-refit RAM: approximately `0.736 GiB`;
- exact verified v5 six-signal base SHA: **passed**;
- v6 candidate ZIP creation/integrity: **passed**;
- candidate ZIP bytes: `1,143,630,143`;
- candidate SHA-256: `20c5f128e43c5303893301f012726381df06a4e20d027ea054acf36e0f6aae40`;
- exact organizer-image offline/read-only 64-row smoke: **passed**;
- output columns/order/finite/nonconstant checks: **passed**;
- CPU smoke phase times: load `6.696s`, structured `0.913s`, contrastive `7.864s`, gate `0.002s`, teacher `8.654s`, meta `0.008s`, write `0.003s`;
- CPU smoke total: `24.14s`;
- full repository suite after smoke: `264 passed, 1 failed, 91 warnings`.

The single full-suite failure was documentation policy only:

- stale `CURRENT.status=production_verified`, outside the allowed policy enum;
- experiment index used a bold-formatted v5 row that did not satisfy the literal policy lookup.

There was no model, inference or organizer-smoke failure at that point. The pre-policy candidate is not retained as final because the workflow correctly stopped before upload/release/artifact publication.

The canonical state has now been moved to v6 `in_progress`, a canonical v6 `PLAN.md` has been added, and the experiment index/handoff state have been corrected. A fresh full gate is required; no previous ZIP SHA is reused as final evidence.

## Final packaging target

Final archive target:

`ecup-v6-fast-gate95-0.6006003615-submission.zip`

The final archive must be rebuilt from a GREEN repository. Exact organizer smoke, full tests, SHA-256/provenance, private HF upload, GitHub artifact and exact-byte RTX 2060 full runtime must all refer to the rebuilt artifact.

## GPU verification contract

The authoritative runtime measurement uses self-hosted `ecup-rtx2060`:

- NVIDIA GeForce RTX 2060 SUPER;
- 8 GiB VRAM;
- exact organizer image;
- `--gpus all`;
- network disabled;
- exact release ZIP SHA verified before execution;
- benchmark samples plus a full reference `matches.parquet` run;
- per-phase v6 timing captured as an immutable artifact.

No CPU smoke duration or sample extrapolation is accepted as the final runtime figure.

## Leaderboard status

Public leaderboard AP: unknown.
Private leaderboard AP: unknown.

Local OOF must remain separate from leaderboard evidence.

## Measured runtime root cause (2026-08-12)

The platform rejected submissions with a timeout. Until now the only runtime
evidence was a 64-row CPU smoke, whose 24.14s is almost entirely fixed
overhead and therefore says nothing about per-pair cost. Profiling the real
structured path at scale identified the cause.

Host: 10-core Apple Silicon, synthetic Ozon-shaped items (mean name length
61 chars, 10 attribute keys), real feature builders and real fitted
estimators, 90,000 pairs / 125,999 items, production `chunk_size=10_000`.

| Structured path | Total | Per pair |
|---|---:|---:|
| as submitted (serial, no shared fuzzy cache) | `198.91s` | `2210.1us` |
| + shared fuzzy cache | `159.34s` | `1770.5us` |
| + 9 worker processes | `43.83s` | `487.0us` |

All three produce **bitwise identical** score vectors for all four structured
signals.

Projected onto the organizer host (20 cores, 19 workers, 80% efficiency,
`10_000`-pair chunks):

| Test set | Pairs | Budget | Structured before | Structured after |
|---|---:|---:|---:|---:|
| public | `115,000` | `360s` | `~254s` | `~22s` |
| private | `275,000` | `780s` | `~608s` | `~44s` |

The structured phase alone consumed roughly `71%` of the public budget and
`78%` of the private budget, single-threaded, before load, neural inference,
meta and write. That is the timeout. Nothing about the model was at fault:
the pipeline used one of twenty available CPU cores and recomputed the same
values several times.

### Prediction-preserving changes

1. **Shared `difflib` results across the legacy and typed passes.**
   `features.py` and `features_v2.py` are byte-identical between legacy pin
   `cb350b4e7ba6` and HEAD, and `textnorm` differs only in
   `extract_model_codes` and `extract_quantities`. Neither touches
   `ItemNorm.name` or `ItemNorm.name_tokens`, so both passes called
   `_ratio`/`_partial_ratio` with the same strings and recomputed all eight
   SequenceMatcher results per pair. The memo keys on the argument strings and
   on an implementation fingerprint, so a future legacy pin that changes
   `_ratio` stops sharing instead of returning the wrong value.
2. **Parallel structured chunks.** Chunk boundaries are unchanged; each chunk
   was already a pure function of its own pairs and `collect_chunked_scores`
   already reassembled by global offset. Workers inherit the loaded models
   through `fork`, so nothing large is pickled. Falls back to the serial path
   when fork is unavailable or the container refuses extra processes.
3. **Single-pass item selection.** `select_items_by_ids` rebuilt the requested
   Arrow value set on every record batch, making the scan quadratic in the
   number of requested ids. Measured `13.5x` faster on a 2,000,000-row item
   file selecting 200,000 ids.
4. **Shared `ItemNorm` between the contrastive and teacher text caches.** The
   two caches differ only in `max_chars` and the category prefix, but each
   normalized every item from scratch.

### Chunk size is not a free parameter

`predict_proba` runs float32 GEMM whose accumulation order depends on the row
count of the call, so changing `STRUCTURED_CHUNK_SIZE` shifts scores by a few
ULPs (observed `~3e-8`). Harmless for ranking, but it breaks byte-reproducible
packaging, so `10_000` is pinned by test and parallelism was implemented at
unchanged boundaries rather than by re-chunking.

### Batch dependence of the rank fusion

The final score is `0.5*percentile_rank(category_shrunk) + 0.5*percentile_rank(hgb)`
with ranks over the whole scored batch. Validation ranks over all `285,210`
development rows at once; the platform ranks over `~115,000` public rows and
`~275,000` private rows in separate runs, so the same pair can be ordered
differently. Simulated over 8 seeds with 20 categories of varying prevalence
and separability, holding rows, labels and raw signals fixed and changing only
what the rank transform saw:

| Fusion | Worst \|delta macro AP\| |
|---|---:|
| global percentile rank (current) | `0.000257` |
| per-category percentile rank | `0.002812` |

Per-category ranking is **worse**, because small per-category batches quantize
the rank transform. Do not "fix" the global rank fusion. Treat `~+/-0.0003` as
an irreducible local-to-leaderboard wobble, which matters because the retained
margin over `0.6000` is `0.0006`.
