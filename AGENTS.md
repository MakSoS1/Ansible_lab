# AGENTS.md — READ THIS FIRST

Mandatory entry point for **any agent** working on branch `ecup-matching-2026`. Do not code, train, tune or open validation labels until the required state files below are read.

## Project in one paragraph

We are solving ODS E-CUP 2026 Ozon product matching. Official metric is unweighted Macro Average Precision over 20 categories. Hidden evaluation contains new/unseen products, so item leakage and repeated holdout tuning are unacceptable. Competition data/models/submission artifacts are private; public Git contains reproducible code and source-backed handoff documentation only.

## Current verified state

- Branch: `ecup-matching-2026`; **do not modify/merge `main`** unless explicitly requested.
- Private data/artifact repo: `Maksim123321/e-cup-2026-matching-private`.
- Human pairs: `365,654`; LLM weak pairs: `>11M`.
- **Production/hidden fallback is v2**, hidden Macro AP `0.2583231811423486`.
- **v5 is the current development iteration; no v5 submission is retained yet.**
- v5 immutable validation: `285,210` development rows + `80,444` sealed-gold rows, five component-disjoint development folds, cross-split item overlap `0`.
- v5 split SHA-256: **`aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`**.
- Sealed gold is **unopened**: `gold_metric_opened=false`, `gold_rows_scored=0`.
- v5 audit baseline OOF: `0.5315527708634168`.
- Retained category-specialist base: `0.5476780661335778`.
- Leakage-safe weak specialists: `0.5514237338676234`.
- Strict meta-OOF combo: `0.559512531439709`.
- Strict train-only sparse TF-IDF specialists: `0.5651306838802859`.
- Supervised contrastive item-space stack: `0.5662217062664492`.
- **Current honest development best: explicit per-key attribute specialists `0.5683065131240066`**, run `31485990777`, source `cb350b4e7ba6bb4a6d283f91bae4d6ea13235d57`; every held fold improved.
- Stretch target `0.60`; current remaining gap `0.0316934868759934`.
- Field-aware weak ranking teacher run `31486298300` is still in progress at this snapshot; **do not invent a metric**.
- First ruBERT pair-teacher run `31485127564` is **not a model rejection**: it failed before predictions because the runner called `build_reranker_examples` without required `attribute_importance`.
- Home RTX 2060 SUPER is reachable only through private `MakSoS1/gpu-dispatch`; public source never owns the self-hosted runner.
- Hardened Memora pin: `bc64ff745a9b2c0e6245e0137654f041fba0c155`; local SQLite + TF-IDF only; graph/LLM/external embeddings/auto-capture disabled.

## Two meanings of best — never conflate them

1. **Production/leaderboard best:** v2, based on observed hidden score.
2. **Development best:** current v5 honest OOF on the immutable sealed protocol.

Old v3/v4 local improvements are real historical offline measurements but hidden results proved that the old holdout did not transfer monotonically. Never switch production fallback merely because a v5 dev score is larger numerically.

## Mandatory reading order

1. `ecup_matching/experiments/CURRENT.json`
2. `docs/agent-memory/PROJECT_STATE.md`
3. `docs/agent-memory/EXPERIMENT_INDEX.md`
4. `docs/agent-memory/DECISIONS.md`
5. `ecup_matching/experiments/v5/PLAN.md`
6. `ecup_matching/experiments/v5/RESULTS.md`
7. `ecup_matching/experiments/v5/SAFE_METRICS.json`
8. `docs/agent-memory/SECURITY.md`
9. `docs/agent-memory/ITERATION_PROTOCOL.md`
10. `ecup_matching/SOLUTION_RESEARCH.md`
11. `ecup_matching/BASELINE_CONTRACT.md`

Historical v3/v4 PLAN/RESULTS and design docs remain useful for lessons, but they do not redefine v5 selection or the production anchor.

## v5 validation invariants

- Never alter split SHA `aae58f...eb55b` to improve a metric.
- Never inspect/use sealed-gold labels during development.
- Do not encode/mine sealed-gold items as adaptation/hard-negative/weak-label data while developing.
- Every new signal must be evaluated on held components/items, not random pair leakage.
- Every stacking layer must remain genuinely OOF; a row cannot train a second-level model using its own target plus an in-sample base prediction.
- Freeze candidate, preprocessing, config and relevant artifact hashes **before** the one-shot sealed-gold evaluation.
- `0.60` means honest development OOF, not a repeatedly tuned holdout score.

## Retained / rejected v5 lessons

### KEEP

- category specialists `0.5476780661`;
- leakage-safe weak specialists `0.5514237339`;
- cross-fitted combo `0.5595125314`;
- strict train-only sparse TF-IDF `0.5651306839`;
- supervised contrastive item-space `0.5662217063`;
- explicit per-key attribute specialists **`0.5683065131`** current dev best.

### DO NOT REPEAT BLINDLY

- direct attribute log-likelihood score shift: `0.5232189037`, all folds regress — REJECT;
- fold-weighted specialists: small mean gain but folds 2/3 regress — diagnostic OOF input only;
- pretrained multilingual embeddings: only `+0.000255` vs audit — insufficient standalone;
- first ruBERT teacher: integration failure before metrics, not model REJECT.

Important distinction: **direct attribute likelihood shift failed, while explicit per-key attribute estimator features succeeded.** Do not conflate these two approaches.

## Debugging lessons that must survive handoff

- For >11M weak rows, use deterministic PyArrow streaming/bounded sampling; do not materialize the entire weak table in pandas before Transformer load.
- Perform CPU-heavy preparation before loading a large Transformer to avoid host-memory spikes.
- MPS physical batch 96 OOM is an infrastructure event; the successful contrastive run preserves effective batch 96 through microbatch 24 × gradient accumulation 4.
- TF-IDF tests validate unseen `transform`, symmetry, finite/bounded values; do not force a desired OOV ranking in unit tests.
- Read the exact failing test/stack before blaming the latest implementation; overlapping RED TDD cycles previously made a serializer commit look broken when the next deliberately missing module was the real failure.
- Heavy workflows need integration tests for composed helper calls; isolated helper tests did not catch the first ruBERT teacher's stale `build_reranker_examples` signature.
- Infrastructure/integration failures are not model scores.

## Persistent memory startup

Public source-backed files are canonical; Memora provides semantic retrieval/history.

If shell access and `HF_TOKEN` are available, restore latest verified SQLite checkpoint first:

```bash
python scripts/memory_bootstrap.py
```

Install/rebuild exact hardened runtime only through:

```bash
python tools/memora_hardened/install.py --prefix .agent-memory/runtime
```

MCP clients start Memora only through:

```bash
bash scripts/memora_mcp.sh
```

Never run arbitrary/unpinned Memora, Cloud Graph, cloud storage backends, OpenAI/OpenRouter embeddings/chat, or auto-capture for this project.

## Memora source contract

`memory_ingest.py` must ingest not only durable Markdown/PLAN/RESULTS but also:

- `ecup_matching/experiments/CURRENT.json`;
- `ecup_matching/experiments/v*/SAFE_METRICS.json`.

A regression test `ecup_matching/tests/test_memory_ingest_sources.py` enforces these machine-readable sources. They were missing before the 2026-08-11 memory audit.

## Mandatory iteration / handoff protocol

Before a new implementation:

1. Read all mandatory state sources above.
2. Preserve immutable split/gold rules.
3. Create/update PLAN with hypothesis, data, exact split, expected gain, runtime and abort criteria.
4. Use TDD/systematic debugging; distinguish RED-by-design from real regressions.

After every meaningful KEEP/REJECT/FAIL result:

1. Update `ecup_matching/experiments/v5/RESULTS.md` with exact run/source/metric/failure evidence.
2. Update `ecup_matching/experiments/v5/SAFE_METRICS.json`.
3. Update `ecup_matching/experiments/CURRENT.json` when current-best/status changes.
4. Update `docs/agent-memory/EXPERIMENT_INDEX.md`.
5. Update `docs/agent-memory/PROJECT_STATE.md` when current stage/next action changes.
6. Record durable rules/lessons in `docs/agent-memory/DECISIONS.md`.
7. **Only after the repository is GREEN**, run/verify:

```bash
python -m pytest ecup_matching/tests -q
python scripts/memory_policy.py
python scripts/memory_ingest.py
python scripts/memory_checkpoint.py --iteration v5
```

A handoff is not complete until full tests and memory policy pass, Memora ingest succeeds, SQLite integrity/secret checks pass, private HF checkpoint upload succeeds and remote verification succeeds.

### Prior checkpoint incident

Memora runs `31481012401` and `31482891498` failed **before ingest** because memory-triggering commits landed while the repository was intentionally RED during TDD. In `31482891498`, collection failed on missing `v5_weighted_specialists`; later GREEN code did not retroactively create the checkpoint.

**Never weaken the test gate.** If a memory update lands during RED TDD, force a new memory-triggering commit or manual dispatch after GREEN.

## Security invariants

- Never print/paste/store HF/GitHub/API tokens, passwords, private keys or credentials in Git, docs, Memora, artifacts or logs.
- `.agent-memory/`, `*.db`, raw competition parquet, models, OOF predictions and submission ZIPs are private.
- Public repository must never be attached directly to the home self-hosted runner; GPU execution goes only through private dispatcher and exact allowed source SHA.
- Historical canonical v3/v4 artifacts are immutable.
- Contest rules prohibit copying private/current participant solutions; research analogous public methods only.

## Where things live

- current machine-readable state: `ecup_matching/experiments/CURRENT.json`
- v5 plan/results/safe metrics: `ecup_matching/experiments/v5/`
- durable project memory: `docs/agent-memory/`
- ML implementation: `ecup_matching/ml/`
- submission runtime: `ecup_matching/submission/`
- organizer contract: `ecup_matching/BASELINE_CONTRACT.md`
- hardened Memora: `tools/memora_hardened/`
- private memory: HF `agent-memory/latest/` and `agent-memory/checkpoints/`.

When uncertain, preserve leakage resistance, reproducibility, immutable evidence, production-vs-development separation and this GREEN-only memory protocol.
