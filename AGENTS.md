# AGENTS.md — READ THIS FIRST

This file is the mandatory entry point for **any agent** working on branch `ecup-matching-2026`. Do not start coding or training until you have read the files listed below.

## Project in one paragraph

We are solving ODS E-CUP 2026 Ozon product matching. Input pairs contain product IDs; products have `name`, flattened JSON `attributes`, and one of 20 categories. Official metric is unweighted Macro Average Precision over categories. Hidden test contains new/unseen products, so validation must avoid item leakage. Competition data and submission/model artifacts are private; public Git contains only reproducible code/docs.

## Current verified state

- Branch: `ecup-matching-2026`; **do not modify or merge `main`** unless the user explicitly asks.
- Private data/artifact repo: `Maksim123321/e-cup-2026-matching-private`.
- Dataset mirrored: `matches.parquet`, `matches_llm.parquet`, `items.parquet`, `items_human.parquet` plus official baseline ZIPs.
- Current completed experiment: **v1 / `v1-structured-hgb`**.
- Human pairs: 365,654.
- Leakage-resistant split: 292,523 train / 73,131 validation / **0 overlapping item IDs**.
- v1 validation Macro AP: **0.4961654895**.
- v1 submit: private HF `submissions/v1/ecup-v1-submission.zip`.
- v1 package was executed offline in exact organizer image `odsai/ecup26-matching-baseline:1.0`.
- Hardened Memora memory is **operational and CI-verified**: upstream commit `bc64ff745a9b2c0e6245e0137654f041fba0c155`, MCP `1.29.0`, TF-IDF/local SQLite only, Graph/LLM/auto-capture disabled, private checkpoint under `agent-memory/latest/` in HF.
- Dedicated memory CI verified 28 repository tests, 51 pinned-upstream Memora tests, behavioral secret redaction, `0700/0600` permissions, SQLite integrity/secret scan, HF checkpoint, and hardened wheel upload.
- Next experiment: **v2 — filtered 11M LLM weak labels + confidence curriculum + hard-negative mining**, keeping the fast structured model as an anchor. Then prepare multilingual bi-encoder features for v3.

## Mandatory reading order

1. `docs/agent-memory/PROJECT_STATE.md`
2. `docs/agent-memory/EXPERIMENT_INDEX.md`
3. `docs/agent-memory/DECISIONS.md`
4. `docs/agent-memory/SECURITY.md`
5. `docs/agent-memory/ITERATION_PROTOCOL.md`
6. `ecup_matching/SOLUTION_RESEARCH.md`
7. Current experiment's `PLAN.md` / `RESULTS.md` when present.

## Persistent memory startup

Markdown above is canonical and must always be enough to recover the project. Memora adds semantic retrieval/history.

If shell access and `HF_TOKEN` are available, restore the latest verified SQLite checkpoint first:

```bash
python scripts/memory_bootstrap.py
```

To install/rebuild the exact hardened Memora runtime:

```bash
python tools/memora_hardened/install.py --prefix .agent-memory/runtime
```

A verified hardened wheel and runtime manifest are also preserved privately under `agent-memory/runtime/` in the HF dataset.

MCP clients must start Memora only through:

```bash
bash scripts/memora_mcp.sh
```

Never run an arbitrary/unpinned Memora installation. Never enable Cloud Graph, Cloudflare Pages/Worker, S3/R2/D1 Memora storage, OpenAI/OpenRouter embeddings/chat, or auto-capture for this project.

## Mandatory iteration protocol

Before v2/v3/etc. implementation:

1. Update `ecup_matching/experiments/CURRENT.json` to the new version with `status: "in_progress"`.
2. Create `ecup_matching/experiments/vN/PLAN.md` with hypothesis, exact data, split, features/model, runtime budget, expected metric movement, and abort criteria.
3. Use item-disjoint validation unless a documented experiment explicitly studies a different split.
4. Keep all raw data/models/submission ZIPs private; Git gets code and source-backed docs only.

After every training run that is kept as an experiment:

1. Write/update `ecup_matching/experiments/vN/RESULTS.md` with exact command/workflow, commit, data counts, Macro AP, all category APs, runtime, artifact paths, failures and conclusions.
2. Update `docs/agent-memory/EXPERIMENT_INDEX.md`.
3. Update `docs/agent-memory/PROJECT_STATE.md` if best model/current stage/next action changed.
4. Record durable architectural changes in `docs/agent-memory/DECISIONS.md`.
5. Set `CURRENT.json` status appropriately.
6. Run:

```bash
python scripts/memory_policy.py
python scripts/memory_ingest.py
python scripts/memory_checkpoint.py --iteration vN
```

A completed iteration is **not complete** if documentation policy or the private memory checkpoint fails. Normal CI also runs `memory_policy.py`, so future agents cannot silently mark a v2+ iteration complete without its required PLAN/RESULTS/state handoff.

## Security invariants

- Never print/paste/store `HF_TOKEN`, GitHub tokens, API keys, passwords, private keys or credentials in Git, experiment docs, Memora, artifacts or logs.
- `.agent-memory/`, `*.db`, models, submission ZIPs and competition parquet are not public artifacts.
- Hardened Memora is pinned to upstream commit `bc64ff745a9b2c0e6245e0137654f041fba0c155`, constrained to `mcp>=1,<2`, local SQLite, TF-IDF, LLM off, graph off, auto-capture off.
- Local memory directory must be mode `0700`; DB must be `0600`.
- Memory checkpoint scans for secrets and fails closed.
- Contest rules prohibit copying other participants' solutions and create publication risk. Research analogous public methods, not private/current participant code.

## Where things live

- Competition design/research: `ecup_matching/SOLUTION_RESEARCH.md`
- Organizer runtime contract: `ecup_matching/BASELINE_CONTRACT.md`
- ML implementation: `ecup_matching/ml/`
- Submission runtime: `ecup_matching/submission/`
- Experiments: `ecup_matching/experiments/`
- Superpowers specs/plans: `docs/superpowers/`
- Agent memory/state: `docs/agent-memory/`
- Hardened Memora tooling: `tools/memora_hardened/`
- Memory lifecycle scripts: `scripts/memory_*.py`
- Durable private memory: HF `agent-memory/latest/` and immutable `agent-memory/checkpoints/`.

When uncertain, preserve reproducibility, leakage resistance, private artifacts, and this memory protocol.