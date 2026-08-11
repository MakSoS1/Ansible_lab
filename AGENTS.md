# AGENTS.md — READ THIS FIRST

This file is the mandatory entry point for **any agent** working on branch `ecup-matching-2026`. Do not start coding or training until you have read the files listed below.

## Project in one paragraph

We are solving ODS E-CUP 2026 Ozon product matching. Input pairs contain product IDs; products have `name`, flattened JSON `attributes`, and one of 20 categories. Official metric is unweighted Macro Average Precision over categories. Hidden test contains new/unseen products, so validation must avoid item leakage. Competition data and submission/model artifacts are private; public Git contains only reproducible code/docs.

## Current verified state

- Branch: `ecup-matching-2026`; **do not modify or merge `main`** unless the user explicitly asks.
- Private data/artifact repo: `Maksim123321/e-cup-2026-matching-private`.
- Human pairs: 365,654.
- Leakage-resistant split: 292,523 train / 73,131 validation / **0 overlapping item IDs**.
- v1 Macro AP: `0.4961654895`.
- v2b structured weak-curriculum Macro AP: `0.5010008995`.
- v3 immutable fallback Macro AP: `0.5254642645846543`, v2b structured weight 0.55 + `cointegrated/rubert-tiny2` neural weight 0.45.
- **Current completed experiment and best retained candidate: v4.** It preserves the v3 learned models and uses 5-fold component-cross-fitted, shrinkage-regularized per-category neural blend alphas.
- **v4 honest cross-fitted Macro AP: `0.5276431099433088`**, absolute delta vs v3 `+0.0021788453586544243`.
- v4 deployable full-data coefficient fit: `0.5284493942551521`; do **not** present this larger value as the unbiased headline.
- v4 routing selection groups all validation candidate edges into 53,131 item-components and selected shrinkage prior `4000`.
- Canonical v4 ZIP: `submissions/v4/canonical/b29e4d9fb066810e22838eddf04887aba845b0141d503f5716db714000e35849/ecup-v4-submission.zip`, SHA-256 `b29e4d9fb066810e22838eddf04887aba845b0141d503f5716db714000e35849`, 109,185,879 bytes.
- Exact organizer-image offline v4 smoke: 1,000/1,000 real neural pairs, network disabled, valid ordered finite output, 1,000 unique scores, private canonical freeze verified.
- Canonical v3 fallback remains immutable at SHA-256 `b833ceb203f8cc7d87517257df8ee5e0a2590075db0ecd2932b8281950015660`.
- Home RTX 2060 SUPER is connected only through private `MakSoS1/gpu-dispatch`; public source never owns a self-hosted runner.
- A stronger pinned `ai-forever/ruBert-base` v4a/v4b/v4c ladder is implemented but **not retained and not the source of the v4 score**. Treat it only as a future v4.1/v5 ablation.
- Hardened Memora memory is operational and CI-verified: pinned upstream `bc64ff745a9b2c0e6245e0137654f041fba0c155`, MCP `1.29.0`, TF-IDF/local SQLite only, Graph/LLM/auto-capture disabled.

## Mandatory reading order

1. `docs/agent-memory/PROJECT_STATE.md`
2. `docs/agent-memory/EXPERIMENT_INDEX.md`
3. `docs/agent-memory/DECISIONS.md`
4. `docs/agent-memory/SECURITY.md`
5. `docs/agent-memory/ITERATION_PROTOCOL.md`
6. `ecup_matching/SOLUTION_RESEARCH.md`
7. `ecup_matching/experiments/v4/PLAN.md`
8. `ecup_matching/experiments/v4/RESULTS.md`

The original strong-reranker design/implementation documents remain useful historical context but do not redefine the retained v4 artifact:

- `docs/superpowers/specs/2026-08-11-ecup-v4-strong-reranker-design.md`
- `docs/superpowers/plans/2026-08-11-ecup-v4-strong-reranker.md`

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

MCP clients must start Memora only through:

```bash
bash scripts/memora_mcp.sh
```

Never run an arbitrary/unpinned Memora installation. Never enable Cloud Graph, Cloudflare Pages/Worker, S3/R2/D1 Memora storage, OpenAI/OpenRouter embeddings/chat, or auto-capture for this project.

## Mandatory iteration protocol

Before a new vN implementation:

1. Update `ecup_matching/experiments/CURRENT.json` to the new version with `status: "in_progress"`.
2. Create `ecup_matching/experiments/vN/PLAN.md` with hypothesis, exact data, split, features/model, runtime budget, expected metric movement and abort criteria.
3. Use item-disjoint validation unless a documented experiment explicitly studies a different split.
4. Keep all raw data/models/submission ZIPs private; Git gets code and source-backed docs only.

After every training/run result that is retained:

1. Update `ecup_matching/experiments/vN/RESULTS.md` with exact run/commit/data/metrics/artifact/failure evidence.
2. Update `docs/agent-memory/EXPERIMENT_INDEX.md`.
3. Update `docs/agent-memory/PROJECT_STATE.md` when best model/current stage/next action changes.
4. Record durable architectural changes in `docs/agent-memory/DECISIONS.md`.
5. Set `CURRENT.json` status appropriately.
6. Run:

```bash
python scripts/memory_policy.py
python scripts/memory_ingest.py
python scripts/memory_checkpoint.py --iteration vN
```

A completed iteration is **not complete** if documentation policy or the private memory checkpoint fails.

## Security invariants

- Never print/paste/store `HF_TOKEN`, GitHub tokens, API keys, passwords, private keys or credentials in Git, experiment docs, Memora, artifacts or logs.
- `.agent-memory/`, `*.db`, models, submission ZIPs and competition parquet are not public artifacts.
- Hardened Memora stays pinned/local-only with TF-IDF, LLM off, graph off and auto-capture off.
- Contest rules prohibit copying other participants' solutions; research analogous public methods, not private/current participant code.
- The public repository must never be attached directly to the home self-hosted runner. GPU execution goes only through the private dispatcher and exact allowed branch SHA.
- Canonical v3/v4 packages are immutable. A changed alpha/model/package requires a new immutable artifact and new experiment evidence.

## Where things live

- Competition design/research: `ecup_matching/SOLUTION_RESEARCH.md`
- Organizer runtime contract: `ecup_matching/BASELINE_CONTRACT.md`
- ML implementation: `ecup_matching/ml/`
- Submission runtime: `ecup_matching/submission/`
- Experiments: `ecup_matching/experiments/`
- Agent memory/state: `docs/agent-memory/`
- Hardened Memora tooling: `tools/memora_hardened/`
- Durable private memory: HF `agent-memory/latest/` and `agent-memory/checkpoints/`.

When uncertain, preserve reproducibility, leakage resistance, immutable retained artifacts and this memory protocol.