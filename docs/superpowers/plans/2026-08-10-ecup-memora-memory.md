# Hardened Memora Agent Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install a pinned, hardened local-only Memora runtime and make every future E-CUP iteration self-documenting, semantically retrievable, and durably checkpointed to the existing private Hugging Face dataset.

**Architecture:** Keep public Git as the canonical source-backed Markdown state and code, with local Memora SQLite as the semantic index/history. Restore/checkpoint the SQLite DB through private HF. Build Memora reproducibly from pinned upstream commit `bc64ff745a9b2c0e6245e0137654f041fba0c155`, apply a fail-closed hardening transformation, and only launch through a local-only wrapper.

**Tech Stack:** Python 3.11+, SQLite, MCP 1.x, Memora MCP, huggingface_hub, pytest, GitHub Actions.

## Global Constraints

- Work only on branch `ecup-matching-2026`; do not merge into or alter `main`.
- Memora upstream commit is exactly `bc64ff745a9b2c0e6245e0137654f041fba0c155`.
- Python dependency must resolve `mcp>=1,<2`.
- Memora runs with `MEMORA_EMBEDDING_MODEL=tfidf`, `MEMORA_LLM_ENABLED=false`, `MEMORA_AUTO_CAPTURE=false`, and `--no-graph`.
- Memora Cloudflare Graph/Worker and S3/R2/D1 storage are unsupported in this E-CUP profile.
- Persistent DB/checkpoints live only in private `Maksim123321/e-cup-2026-matching-private`.
- Never commit competition parquet, model files, submission ZIPs, Memora DBs, credentials, or tokens to public Git.
- Local memory directory mode `0700`; SQLite DB mode `0600`.
- A future completed experiment is invalid until PLAN/RESULTS/index/state and private memory checkpoint are all updated.

---

### Task 1: Universal agent contract and canonical state

**Files:**
- Create: `AGENTS.md`
- Create: `CLAUDE.md`
- Create: `GEMINI.md`
- Create: `.github/copilot-instructions.md`
- Create: `docs/agent-memory/PROJECT_STATE.md`
- Create: `docs/agent-memory/EXPERIMENT_INDEX.md`
- Create: `docs/agent-memory/DECISIONS.md`
- Create: `docs/agent-memory/SECURITY.md`
- Create: `docs/agent-memory/ITERATION_PROTOCOL.md`
- Create: `ecup_matching/experiments/CURRENT.json`

**Interfaces:**
- Produces the source-backed documents consumed by `memory_ingest.py` and `memory_policy.py`.

- [ ] Write the universal onboarding files with current v1 state and mandatory startup/finalization sequence.
- [ ] Record v1 in experiment index and set `CURRENT.json` to completed v1 with next iteration v2.
- [ ] Ensure every client-specific instruction file redirects to `AGENTS.md` instead of duplicating policy.
- [ ] Commit.

### Task 2: RED tests for hardening and lifecycle

**Files:**
- Create: `ecup_matching/tests/test_memora_hardening.py`
- Create: `ecup_matching/tests/test_memory_policy.py`
- Create: `.github/workflows/ecup-memora-memory.yml`

**Interfaces:**
- Tests expect `tools.memora_hardened.harden`, `scripts.memory_common`, and `scripts.memory_policy` which do not exist yet.

- [ ] Add tests asserting pinned commit, `mcp<2`, local-only defaults, graph removal, 0700/0600 patching, secret redaction transformations, secret scanner, and v1 documentation policy.
- [ ] Add dedicated workflow that runs repository tests first.
- [ ] Push tests and verify the workflow fails specifically because implementation modules are missing.

### Task 3: Hardened Memora source transformer and installer

**Files:**
- Create: `tools/__init__.py`
- Create: `tools/memora_hardened/__init__.py`
- Create: `tools/memora_hardened/pin.json`
- Create: `tools/memora_hardened/constraints.txt`
- Create: `tools/memora_hardened/harden.py`
- Create: `tools/memora_hardened/install.py`
- Create: `tools/memora_hardened/verify_install.py`

**Interfaces:**
- `harden_tree(source: Path) -> dict[str, object]` applies fail-closed source transformations.
- `install.py --prefix PATH` clones exact upstream SHA, hardens it, creates venv, installs with constraints and builds a wheel.
- `verify_install.py` asserts effective safe defaults and filesystem modes.

- [ ] Implement exact-anchor transformations specified in the design.
- [ ] Remove `memora-graph/` from hardened source and replace Graph HTML with an inert local-only disabled page.
- [ ] Make patcher idempotence/error behavior explicit: unexpected/missing anchors are fatal.
- [ ] Install under a venv with `mcp>=1,<2` constraint.
- [ ] Run unit tests to GREEN.
- [ ] Commit.

### Task 4: Safe Memora launcher

**Files:**
- Create: `scripts/memora_safe_server.py`
- Create: `scripts/memora_mcp.sh`
- Create: `.mcp.json`

**Interfaces:**
- `build_safe_env(repo_root: Path, inherited: Mapping[str,str]) -> dict[str,str]` strips cloud/LLM credentials and forces local-only values.
- shell launcher executes `.agent-memory/runtime/venv/bin/memora-server --no-graph` through `memora_safe_server.py`.

- [ ] Test that unsafe env keys are removed and local safety values are forced.
- [ ] Implement launcher and fail if the hardened runtime has not been installed.
- [ ] Add repo MCP config with no embedded secrets.
- [ ] Run tests to GREEN.
- [ ] Commit.

### Task 5: HF bootstrap/checkpoint and source-backed ingestion

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/memory_common.py`
- Create: `scripts/memory_bootstrap.py`
- Create: `scripts/memory_ingest.py`
- Create: `scripts/memory_checkpoint.py`

**Interfaces:**
- `memory_common.scan_text_for_secrets(text) -> list[str]` is shared by checkpoint validation/tests.
- `memory_bootstrap.py` restores `agent-memory/latest/{memories.db,manifest.json}` and verifies SHA/integrity/modes.
- `memory_ingest.py` upserts canonical Markdown documents by `metadata.source_path` using Memora storage APIs.
- `memory_checkpoint.py --iteration vN` validates policy, creates consistent SQLite backup, scans DB, commits checkpoint+latest to private HF, and verifies remote files.

- [ ] Implement DB-path symlink/regular-file guards and chmod helpers.
- [ ] Implement bootstrap download and SHA/SQLite integrity verification.
- [ ] Implement source-backed Memora upsert for canonical docs and experiment PLAN/RESULTS.
- [ ] Implement fail-closed checkpoint secret scan and atomic HF `create_commit` upload.
- [ ] Run unit tests to GREEN.
- [ ] Commit.

### Task 6: Iteration documentation enforcement

**Files:**
- Create: `scripts/memory_policy.py`
- Modify: `.github/workflows/ecup-matching.yml`
- Modify: `.github/workflows/ecup-train-submit-v1.yml` only if needed to demonstrate the finalization hook without rerunning the v1 training job.

**Interfaces:**
- `validate_repository(repo_root: Path) -> list[str]` returns policy violations.
- CLI exits non-zero on any violation.

- [ ] Require valid `CURRENT.json`.
- [ ] For completed current iteration require PLAN (or grandfathered v1 design reference), RESULTS, experiment-index row, and project-state reference.
- [ ] Require all future `v2+` completed iterations to have both PLAN.md and RESULTS.md.
- [ ] Add policy validation to normal CI.
- [ ] Run tests to GREEN.
- [ ] Commit.

### Task 7: Install, run upstream tests, initialize persistent memory

**Files:**
- Modify: `.github/workflows/ecup-memora-memory.yml`

**Interfaces:**
- Workflow uses repository secret `HF_TOKEN` and existing private HF dataset.

- [ ] Clone pinned upstream on GitHub Actions.
- [ ] Harden and install runtime.
- [ ] Install upstream dev test dependencies under `mcp<2` and run the upstream Python test suite.
- [ ] Run E-CUP hardening verification.
- [ ] Bootstrap existing private memory if present, otherwise initialize a new local DB.
- [ ] Ingest `AGENTS.md`, canonical state, design/research, and v1 results.
- [ ] Checkpoint initial DB to private HF under versioned path and `agent-memory/latest/`.
- [ ] Upload hardened wheel/manifest to private HF `agent-memory/runtime/`.
- [ ] Cleanup runtime/DB from runner.
- [ ] Verify workflow is fully green.

### Task 8: Final verification and handoff

**Files:**
- Modify as needed only for verification-discovered defects.

- [ ] Run complete repository tests and verify zero failures.
- [ ] Verify dedicated Memora workflow successful.
- [ ] Verify private HF files listed by the workflow: latest DB, manifest, checkpoint, hardened wheel/manifest.
- [ ] Verify public Git tree contains no `.db`, model, submission ZIP, token, or raw competition files.
- [ ] Re-read `AGENTS.md` from the perspective of a zero-context agent and verify it has current v1 metrics, next action, startup, and finalization commands.
- [ ] Record the Memora integration as a durable decision/state update.
