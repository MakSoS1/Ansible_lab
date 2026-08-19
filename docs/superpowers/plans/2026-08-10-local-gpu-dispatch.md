# Local GPU Dispatch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the E-CUP branch to the user's RTX 2060 SUPER through a private, manually dispatched, WSL2-isolated self-hosted runner.

**Architecture:** A private `gpu-dispatch` repository owns the trusted workflow and runner registration. It checks out an exact allowed public commit and runs its Python code only inside an offline, read-only, capability-free CUDA container; Windows integration and host mounts are disabled in the WSL runner environment.

**Tech Stack:** GitHub Actions, GitHub self-hosted runner, Windows 11, WSL2 Ubuntu 24.04, systemd, Docker, NVIDIA CUDA, PyTorch, Python 3.12, pytest.

## Global Constraints

- Public source is fixed to `MakSoS1/Ansible_lab`, branch `ecup-matching-2026`.
- Dispatcher is private `MakSoS1/gpu-dispatch` and uses only `workflow_dispatch`.
- The public source container has no network, secrets, Docker socket, Windows mounts, or writable source/data/model-cache mounts.
- Raw parquet, models, checkpoints, and derived pair rows never enter public Git or GitHub artifacts.
- Full training must require CUDA and preserve the fixed item-disjoint validation contract.
- The RTX 2060 SUPER has 8 GiB VRAM; full profile starts at batch size 16 with gradient accumulation 8.

---

### Task 1: Trusted dispatcher contract

**Files:**
- Create: `gpu-dispatch/dispatch_contract.py`
- Create: `gpu-dispatch/tests/test_dispatch_contract.py`

**Interfaces:**
- Consumes: CLI strings `source_sha`, `profile`, optional metrics path.
- Produces: `validate_source_sha(str) -> str`, `validate_profile(str) -> str`, and `validate_metrics(Path) -> dict`.

- [ ] Write failing tests for a 40-character lowercase/uppercase hex SHA, rejection of refs/shell metacharacters, the three exact profiles, and the complete metrics schema.
- [ ] Run `pytest tests/test_dispatch_contract.py -q` and verify failure because the module does not exist.
- [ ] Implement strict full-match validation and fail-closed metrics validation.
- [ ] Run the test and verify all cases pass.
- [ ] Commit the dispatcher contract.

### Task 2: Trusted container launcher

**Files:**
- Create: `gpu-dispatch/run_job.py`
- Create: `gpu-dispatch/tests/test_run_job.py`
- Create: `gpu-dispatch/Dockerfile`

**Interfaces:**
- Consumes: validated SHA/profile plus trusted absolute roots.
- Produces: a list-form Docker command and a CLI that prepares source/data/cache/output and executes one bounded container.

- [ ] Write failing tests asserting `--network none`, `--read-only`, `--cap-drop ALL`, `no-new-privileges`, no Docker socket, read-only source/data/cache mounts, a size-limited output tmpfs copied only after success, and profile-specific fixed argv.
- [ ] Run the targeted tests and verify they fail.
- [ ] Implement command construction without `shell=True` and without accepting arbitrary command arguments.
- [ ] Implement exact branch ancestry verification with Git arguments passed as a list.
- [ ] Implement trusted downloads, offline model-cache preparation, timeout, metrics verification, and unconditional container removal.
- [ ] Run all dispatcher tests and verify pass.
- [ ] Build the trusted image and run a CUDA diagnostic.
- [ ] Commit launcher and image.

### Task 3: Private GitHub workflow

**Files:**
- Create: `gpu-dispatch/.github/workflows/ecup-gpu.yml`
- Create: `gpu-dispatch/tests/test_workflow_policy.py`
- Create: `gpu-dispatch/README.md`

**Interfaces:**
- Consumes: `workflow_dispatch` inputs `source_sha` and `profile`.
- Produces: a job routed only to `[self-hosted, Linux, X64, gpu, ecup]` and a documented CLI invocation.

- [ ] Write a failing static policy test that rejects `push`, `pull_request`, `pull_request_target`, write permissions, arbitrary inputs, unpinned action refs, and missing timeout/concurrency.
- [ ] Run the policy test and verify failure because the workflow is absent.
- [ ] Implement the manual-only workflow with `contents: read`, exact choice values, concurrency one, timeout, pinned checkout action, launcher call, summary, and always-cleanup.
- [ ] Document `gpu-check`, `smoke`, `train`, result paths, and cancellation.
- [ ] Run the entire dispatcher suite.
- [ ] Commit workflow and documentation.

### Task 4: WSL2 runner host

**Files:**
- Create on GPU host: `/etc/wsl.conf`
- Create on GPU host: `/etc/systemd/system/actions.runner.MakSoS1-gpu-dispatch.gpu-ecup.service`
- Create on GPU host: `/srv/github-gpu/{runner,data,cache,output}`

**Interfaces:**
- Consumes: the GitHub one-time runner registration token.
- Produces: a systemd-managed runner labeled `gpu,ecup` under user `gha-gpu`.

- [ ] Record current WSL, Docker, GPU, and runner state without reading personal files.
- [ ] Create system user `gha-gpu` and dedicated directories with least-privilege ownership/modes.
- [ ] Disable WSL automount and Windows interop, preserve systemd, shut down and restart the distro.
- [ ] Verify `/mnt/c` is absent and `.exe` interop is unavailable inside Ubuntu.
- [ ] Install current GitHub runner from the official release checksum and register it only to `MakSoS1/gpu-dispatch` with labels `gpu,ecup`.
- [ ] Install and start the systemd service.
- [ ] Verify the service runs as non-root and Docker/CUDA diagnostics succeed.

### Task 5: Publish and end-to-end verification

**Files:**
- Modify: `docs/agent-memory/DECISIONS.md`
- Modify: `docs/agent-memory/PROJECT_STATE.md`
- Modify: `ecup_matching/README.md`

**Interfaces:**
- Consumes: private dispatcher repository and online runner.
- Produces: reproducible invocation and verified smoke-run evidence.

- [ ] Create private `MakSoS1/gpu-dispatch` and push the tested trusted files over SSH.
- [ ] Add the durable GPU-dispatch decision and public documentation without exposing private paths or secrets.
- [ ] Run public E-CUP tests and `python scripts/memory_policy.py`.
- [ ] Dispatch `gpu-check` and verify CUDA/device output.
- [ ] Dispatch `smoke` against the exact head SHA of `ecup-matching-2026`.
- [ ] Verify zero validation overlap, 20 category AP values, finite Macro AP, local retained output, no public artifacts, and container cleanup.
- [ ] Report the exact invocation commands and remaining full-training runtime expectations.
