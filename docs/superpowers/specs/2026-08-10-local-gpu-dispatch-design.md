# Local GPU Dispatch Design

Date: 2026-08-10
Status: approved for implementation

## Goal

Run E-CUP neural training from the public `MakSoS1/Ansible_lab` branch `ecup-matching-2026` on the user's Windows RTX 2060 SUPER without attaching a self-hosted runner to the public repository and without exposing host files or credentials to public source code.

## Threat model

The design assumes the public repository, a selected public commit, or its Python code may be malicious. A compromise of the public repository must not directly schedule a job or modify the trusted runner workflow. Public code receives no GitHub/Hugging Face token, no SSH key, no Docker socket, no Windows filesystem mount, and no network access while executing.

The private dispatcher repository and the GitHub account controlling it remain trusted. A complete GitHub-account takeover can modify a private workflow and therefore remains outside the protection boundary of a normal self-hosted GitHub Actions runner. WSL2, Docker, and NVIDIA driver escapes are residual platform risks; they cannot be eliminated while sharing a consumer GPU with the Windows host.

## Architecture

### Public source repository

- Repository: `MakSoS1/Ansible_lab`.
- Allowed branch: `ecup-matching-2026` only.
- Every run names an exact 40-character commit SHA.
- The dispatcher verifies that the SHA is reachable from the allowed branch before execution.
- Public workflows never target the local runner.

### Private dispatcher repository

- Repository: `MakSoS1/gpu-dispatch` with private visibility.
- Trigger: `workflow_dispatch` only.
- Workflow permissions: `contents: read`.
- Inputs: exact source SHA and a fixed profile from `gpu-check`, `smoke`, or `train`.
- No arbitrary command, repository, branch, image, path, or shell argument is accepted.
- The workflow is the only GitHub repository registered with the local runner.

### Windows and WSL2 boundary

- Host: `maksi@10.78.211.199`, Windows 11 build 26200.
- GPU: NVIDIA GeForce RTX 2060 SUPER, 8 GiB.
- Runtime: dedicated `gha-gpu` user inside Ubuntu 24.04 on WSL2.
- WSL automount and Windows interop are disabled for the runner distro.
- The GitHub runner runs as a systemd service under `gha-gpu`, not as Windows administrator or WSL root.
- The runner has access only to its work directory, trusted cache, trusted data directory, and per-run output directory.

### Container boundary

The trusted dispatcher starts public code in a fresh CUDA/PyTorch container with:

- `--network none`;
- `--read-only` root filesystem;
- all Linux capabilities dropped;
- `no-new-privileges` enabled;
- no Docker socket;
- no host home or Windows mounts;
- source, data, and model cache mounted read-only;
- a 4 GiB size-limited `/output` tmpfs, copied to a fresh host directory only
  after successful container exit;
- bounded memory, process count, shared memory, and wall-clock duration;
- unconditional container removal.

The trusted image and Python dependency set are built by the private dispatcher, not by a Dockerfile from the public repository.

## Profiles

### `gpu-check`

Runs a fixed PyTorch CUDA diagnostic in the trusted image. It does not fetch public source or competition data.

### `smoke`

Fetches the exact public source SHA, prepares a small trusted data subset, and invokes `ecup_matching.ml.train_reranker_v2` with bounded row counts, short epochs, batch size 16, and gradient accumulation. It proves checkout, data mounting, CUDA training, metric validation, cleanup, and local result retention.

### `train`

Runs the existing full leakage-safe reranker pipeline using the four official parquet inputs, batch size 16, gradient accumulation 8, and the existing v2/v3 hard-negative logic. The 8 GiB GPU constraint is treated as authoritative. The trusted launcher may retry only a CUDA out-of-memory failure once with batch size 8 and gradient accumulation 16; it must not hide other failures.

## Data and outputs

- Organizer parquet files are downloaded by a trusted dispatcher step into a dedicated cache and verified non-empty before the untrusted container starts.
- Data is mounted read-only and is never uploaded to GitHub artifacts.
- The base model cache is populated by a trusted online preparation step, then mounted read-only while training runs offline.
- During execution, untrusted code can write only to the 4 GiB `/output` tmpfs.
- After successful exit, the trusted launcher copies that bounded output to
  `/srv/github-gpu/output/<run-id>` and validates metrics before reporting them.
- `metrics.json` and a short sanitized summary are printed to GitHub logs.
- Model weights, derived parquet, and checkpoints remain only on the GPU host until a separate trusted upload path is explicitly configured.
- Cleanup removes the public source checkout and container, but retains the named output directory.

## Failure handling

- Invalid profile/SHA/branch ancestry fails before source execution.
- Missing GPU, Docker, data, model cache, or expected output fails closed.
- A timeout stops and removes the container.
- Metrics must contain a finite Macro AP in `[0, 1]`, zero validation item overlap, and 20 per-category AP entries.
- Cleanup runs after success, failure, or cancellation.
- No retry occurs for validation, integrity, network-policy, or source errors.

## Verification

1. Unit tests cover input validation, command construction, mount policy, network policy, and metrics validation.
2. Static tests reject public-repository workflow triggers and unpinned action references.
3. Baseline E-CUP tests and `scripts/memory_policy.py` continue to pass.
4. Remote verification proves CUDA inside Docker, absence of `/mnt/c`, disabled Windows interop, non-root runner identity, and systemd service health.
5. End-to-end verification dispatches `gpu-check`, then `smoke`, and confirms sanitized logs plus local retained outputs.

## Invocation contract

From an authenticated GitHub CLI:

```bash
gh workflow run ecup-gpu.yml \
  --repo MakSoS1/gpu-dispatch \
  -f source_sha=<40-character-sha> \
  -f profile=gpu-check
```

The same command with `profile=smoke` or `profile=train` runs the corresponding bounded workload.
