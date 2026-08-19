# E-CUP Iteration Protocol — v2 and Later

Every future agent follows this protocol. It exists so a new agent can reconstruct not only the best score but **why** the project is in its current state.

## 1. Start

Read `AGENTS.md` and all `docs/agent-memory/*` files. Run `python scripts/memory_bootstrap.py` when HF access is available. Search Memora for the experiment topic if the MCP client supports it.

## 2. Declare the experiment before changing training behavior

Set `ecup_matching/experiments/CURRENT.json`:

```json
{
  "version": "vN",
  "status": "in_progress",
  "plan": "ecup_matching/experiments/vN/PLAN.md",
  "results": "ecup_matching/experiments/vN/RESULTS.md",
  "previous": "vN-1"
}
```

Create `PLAN.md` containing at minimum:

- hypothesis;
- exact train/validation data used;
- leakage controls;
- feature/model changes;
- weak-label weighting/sampling when relevant;
- primary metric and comparison baseline;
- runtime/memory budget;
- success and abort criteria;
- expected private artifact path.

## 3. Implement with tests

Use Superpowers/TDD for production code changes. Preserve the fixed item-disjoint human validation unless the experiment explicitly studies validation itself. Do not tune on a random pair split and report it as comparable to v1.

## 4. Run and retain evidence

For every retained run record:

- Git commit SHA and GitHub Actions run/job ID;
- data row/item counts;
- exact split;
- Macro AP;
- AP for all 20 categories;
- fit/feature/inference runtime;
- archive/model size where relevant;
- exact private HF paths;
- failures/warnings and unexpected behavior;
- comparison versus previous best.

## 5. Finish documentation

Write `RESULTS.md`. Update `EXPERIMENT_INDEX.md`. Update `PROJECT_STATE.md` if current best/stage/next action changes. Add durable architecture changes to `DECISIONS.md`.

Set CURRENT status to one of:

- `completed` — successful retained iteration with verified artifacts;
- `rejected` — retained negative result with documented reason;
- `blocked` — cannot complete due to explicit blocker.

Never mark `completed` before artifacts and memory checkpoint are verified.

## 6. Validate and checkpoint

Run in this order:

```bash
python scripts/memory_policy.py
python scripts/memory_ingest.py
python scripts/memory_checkpoint.py --iteration vN
```

The checkpoint is stored in private HF under both a versioned timestamped path and `agent-memory/latest/`. If secret scan, SQLite integrity, documentation policy, hash verification, or remote verification fails, the iteration remains incomplete.

## 7. Handoff

Before stopping, ensure `PROJECT_STATE.md` ends with an unambiguous immediate next action. A new agent should not need the previous chat to know what to do.

## v1 grandfathering

v1 predates this protocol and therefore has `RESULTS.md` but no per-iteration `PLAN.md`. Its design/implementation plans under `docs/superpowers/` are the accepted planning evidence. **v2 and later have no such exception.**
