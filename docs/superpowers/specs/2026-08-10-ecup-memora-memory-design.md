# E-CUP Hardened Memora Agent Memory Design

## Goal

Make the `ecup-matching-2026` branch self-explanatory to a new agent with zero chat history, while preserving durable semantic memory between ephemeral agents/runners without publishing competition data, models, secrets, or the memory database.

## Architecture

Use Memora as a **local-only MCP memory engine** and private Hugging Face as the durable checkpoint store.

- Upstream Memora is pinned to commit `bc64ff745a9b2c0e6245e0137654f041fba0c155`.
- The repository stores only a deterministic hardening/installation layer, never an unpinned `pip install memora`.
- Each agent works with local SQLite under `.agent-memory/memories.db`.
- `scripts/memory_bootstrap.py` restores the latest verified database from private dataset `Maksim123321/e-cup-2026-matching-private`.
- `scripts/memory_ingest.py` synchronizes canonical Markdown project state and experiment documents into Memora.
- `scripts/memory_checkpoint.py` creates a SQLite backup, scans it for secret patterns, writes a SHA-256 manifest, and commits both checkpoint and `latest/` state to the private HF dataset.
- Markdown remains the universal fallback: agents that cannot use MCP must still understand the project by reading `AGENTS.md` plus `docs/agent-memory/*`.

## Agent bootstrap contract

Every agent must, in order:

1. Read root `AGENTS.md`.
2. Read `docs/agent-memory/PROJECT_STATE.md`, `EXPERIMENT_INDEX.md`, `DECISIONS.md`, `SECURITY.md`, and `ITERATION_PROTOCOL.md`.
3. If `HF_TOKEN` and a shell are available, run `python scripts/memory_bootstrap.py`.
4. If Memora is needed, install/start only through the hardened scripts; never invoke an arbitrary upstream install or Cloud Graph deployment.
5. Before a new ML iteration, create/update `ecup_matching/experiments/vN/PLAN.md` and `ecup_matching/experiments/CURRENT.json`.
6. After training, write `RESULTS.md`, update experiment index/current project state, ingest the docs, and run the checkpoint command.
7. A completed iteration is not considered complete if policy validation or memory checkpoint fails.

Small adapter files (`CLAUDE.md`, `GEMINI.md`, `.github/copilot-instructions.md`) point clients that use their own instruction convention back to the same root `AGENTS.md`; the project has one canonical agent contract rather than divergent instructions.

## Security profile

The E-CUP profile intentionally removes functionality rather than trying to safely expose unnecessary remote surfaces.

### Disabled attack surface

- Cloudflare Pages Graph API: not used and removed from the hardened build source.
- Cloudflare WebSocket/broadcast worker: not used and removed from the hardened build source.
- local interactive Graph UI: disabled; the safe launcher always passes `--no-graph`.
- OpenAI/OpenRouter embeddings and chat: disabled.
- S3/R2/D1 Memora storage: disabled for Memora itself. HF transfer is performed by repository-controlled bootstrap/checkpoint scripts instead.
- Memora auto-capture: disabled.

This makes the reported unauthenticated Cloudflare API, Graph XSS/CDN supply-chain, worker broadcast, and `sync.sh eval` paths unreachable in the supported E-CUP configuration.

### Hardened upstream patch

The deterministic patcher fails closed if the pinned source no longer matches expected anchors and makes these changes:

1. `mcp>=1.0.0` becomes `mcp>=1.0.0,<2`.
2. default embedding backend changes from `openai` to `tfidf`.
3. default `MEMORA_LLM_ENABLED` changes from true to false.
4. content validation redacts detected secrets before persistence.
5. metadata string leaves and tags are redacted before persistence.
6. `add_memories` batch content is routed through the same content validator.
7. local/cache directories are forced to mode `0700`; SQLite files are forced to `0600` after creation/open.
8. `memora-graph/` is removed from the hardened source tree and the packaged Graph HTML is replaced by an inert disabled page.

The safe launcher additionally strips network/cloud credential environment variables, forces local DB + TF-IDF + LLM off + auto-capture off, and always runs `memora-server --no-graph`.

### Defense in depth at checkpoint

Checkpointing is fail-closed:

- the DB must be a regular file, not a symlink;
- SQLite `PRAGMA integrity_check` must return `ok`;
- a consistent SQLite backup is created rather than copying a live database file;
- all text columns are scanned using the project secret detector;
- if a probable secret is detected, upload is refused;
- SHA-256 is recorded and validated during bootstrap;
- local directory mode is `0700`, DB mode is `0600`;
- no token value is written to Git, manifests, logs, or memory.

## Iteration memory model

Canonical memory sources are deliberately source-backed instead of free-form agent recollection:

- `AGENTS.md` — mandatory project operating contract;
- `PROJECT_STATE.md` — current best model, current stage, blockers, next action;
- `EXPERIMENT_INDEX.md` — one row per experiment with validation score, runtime, commit and private artifact location;
- `DECISIONS.md` — durable architectural decisions and reasons;
- `SECURITY.md` — security invariants and forbidden configurations;
- `ITERATION_PROTOCOL.md` — required lifecycle for v2+;
- each `ecup_matching/experiments/vN/PLAN.md` and `RESULTS.md`.

`memory_ingest.py` stores or updates one Memora record per canonical source file using `metadata.source_path` as the stable identity and project-scoped tags. The database is therefore reconstructable from Git documentation, while Memora provides semantic retrieval and history across sessions.

## CI enforcement

A dedicated workflow installs/hardens the pinned source on GitHub Actions and verifies:

- exact upstream commit;
- hardening patch anchors and resulting configuration;
- upstream Python test suite under `mcp<2`;
- additional E-CUP security tests;
- no `memora-graph` cloud deployment tree in hardened source;
- safe launch configuration;
- memory documentation policy;
- initial/updated Memora DB creation and integrity;
- private HF checkpoint upload and verification.

The existing general test workflow also runs repository-side memory policy tests, so future agents get a failing CI signal when they add a completed iteration without its required documentation.

## Current seed state

The initial memory checkpoint must include the already completed v1 experiment:

- version: `v1-structured-hgb`;
- human pairs: 365,654;
- item-disjoint validation Macro AP: `0.4961654895`;
- train rows: 292,523;
- validation rows: 73,131;
- item overlap: 0;
- submission: `submissions/v1/ecup-v1-submission.zip` in the private HF dataset;
- next planned iteration: filtered LLM weak labels + hard-negative mining, followed by multilingual bi-encoder work.

## Non-goals

- No Cloudflare Graph deployment.
- No public memory DB or model artifacts.
- No automatic capture of arbitrary shell/WebFetch output.
- No external LLM calls from the memory system.
- No attempt to make Memora a security boundary against a malicious local user with write access to the repository or process environment.
