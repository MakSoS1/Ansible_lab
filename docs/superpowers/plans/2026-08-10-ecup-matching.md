# E-CUP 2026 Matching Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated GitHub Actions workspace that mirrors the four E-CUP matching parquet files into a private Hugging Face dataset, profiles the human-labelled subset, and documents the solution research/iteration roadmap.

**Architecture:** `ecup_matching/hf_sync.py` owns source download, private Hub repo creation, upload, cleanup and verification. `ecup_matching/profile_human.py` owns local parquet profiling and emits JSON/Markdown aggregate reports. `.github/workflows/ecup-matching.yml` runs unit tests, profiling, and the private-HF mirror on branch `ecup-matching-2026`; no raw data is committed to Git.

**Tech Stack:** Python 3.11, `requests`, `huggingface_hub`, `pandas`, `pyarrow`, GitHub Actions, Hugging Face Hub.

## Global Constraints

- Modify only `MakSoS1/Ansible_lab` branch `ecup-matching-2026`.
- Do not modify existing Ansible files.
- Keep all competition code under `ecup_matching/` and its single dedicated workflow.
- Destination dataset ID is exactly `Maksim123321/e-cup-2026-matching-private`.
- Destination Hub repository must be private.
- Never store or print `HF_TOKEN`; read it only from environment / GitHub Actions Secrets.
- Never commit raw parquet files to Git.
- Download/upload one large file at a time and delete local copies after successful upload.
- Unit tests must not access the network.

---

## File map

- `ecup_matching/hf_sync.py` — resumable HTTP download, private dataset repo creation, upload, verification, cleanup.
- `ecup_matching/profile_human.py` — aggregate statistics for `matches.parquet` + `items_human.parquet`.
- `ecup_matching/tests/test_hf_sync.py` — network-free tests for mirror orchestration and safety checks.
- `ecup_matching/tests/test_profile_human.py` — synthetic-parquet tests for profiling logic.
- `ecup_matching/requirements-ci.txt` — pinned/compatible CI dependencies.
- `ecup_matching/README.md` — operator instructions, exact secret name, workflow outputs, solution roadmap link.
- `.github/workflows/ecup-matching.yml` — tests, human profile artifact, private HF mirror.
- `docs/superpowers/specs/2026-08-10-ecup-matching-design.md` — design and 10 solution families.

---

### Task 1: HF mirror core with tests

**Files:**
- Create: `ecup_matching/tests/test_hf_sync.py`
- Create: `ecup_matching/hf_sync.py`

**Interfaces:**
- `SourceFile(name: str, url: str)` immutable dataclass.
- `download_file(source: SourceFile, destination: pathlib.Path, session: requests.Session) -> int` returns downloaded byte count.
- `mirror_files(repo_id: str, token: str, workdir: pathlib.Path, sources: tuple[SourceFile, ...] = SOURCES, api=None, session=None) -> list[str]` returns verified Hub file names.
- `main() -> int` reads `HF_TOKEN`, optional `HF_REPO_ID`, and performs the mirror.

- [ ] **Step 1: Write failing tests**

Tests use fake `requests.Session` and fake HF API objects. Required cases:

```python
def test_mirror_requires_token(tmp_path):
    with pytest.raises(ValueError, match="HF_TOKEN"):
        mirror_files(REPO_ID, "", tmp_path, sources=())


def test_mirror_creates_private_dataset_and_cleans_local_files(tmp_path):
    api = FakeApi()
    session = FakeSession({"https://example.test/a": b"abc"})
    files = mirror_files(
        REPO_ID,
        "secret",
        tmp_path,
        sources=(SourceFile("a.parquet", "https://example.test/a"),),
        api=api,
        session=session,
    )
    assert files == ["a.parquet"]
    assert api.create_repo_calls[0]["private"] is True
    assert api.create_repo_calls[0]["repo_type"] == "dataset"
    assert not (tmp_path / "a.parquet").exists()


def test_zero_byte_download_is_rejected(tmp_path):
    session = FakeSession({"https://example.test/a": b""})
    with pytest.raises(RuntimeError, match="zero bytes"):
        download_file(SourceFile("a.parquet", "https://example.test/a"), tmp_path / "a.parquet", session)
```

- [ ] **Step 2: Run tests and confirm failure**

Run in CI/local checkout:

```bash
python -m pytest ecup_matching/tests/test_hf_sync.py -q
```

Expected: import/function failures because `hf_sync.py` is not implemented.

- [ ] **Step 3: Implement minimal safe mirror**

Implementation requirements:

- constants for the four official Yandex Object Storage URLs;
- 8 MiB streaming chunks;
- HTTP timeout `(15, 180)` and `raise_for_status()`;
- create private dataset repo with `exist_ok=True`;
- upload via `HfApi.upload_file`;
- after each upload call `list_repo_files` and ensure the name exists;
- delete the local file in a `finally` block after upload attempt if it exists;
- never log token values;
- `main()` exits with a clear error when `HF_TOKEN` is missing.

- [ ] **Step 4: Re-run tests**

```bash
python -m pytest ecup_matching/tests/test_hf_sync.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ecup_matching/hf_sync.py ecup_matching/tests/test_hf_sync.py
git commit -m "feat: add private HF dataset mirror"
```

---

### Task 2: Human data profiler with synthetic-parquet tests

**Files:**
- Create: `ecup_matching/tests/test_profile_human.py`
- Create: `ecup_matching/profile_human.py`

**Interfaces:**
- `build_profile(matches_path: Path, items_path: Path) -> dict`
- `render_markdown(profile: dict) -> str`
- CLI accepts `--matches`, `--items`, `--json-out`, `--md-out`.

- [ ] **Step 1: Write failing synthetic-data test**

Create tiny parquet fixtures with items from two categories and binary match targets. Assertions must cover:

```python
profile = build_profile(matches_path, items_path)
assert profile["matches"]["rows"] == 3
assert profile["matches"]["positive_rate"] == pytest.approx(2 / 3)
assert profile["items"]["rows"] == 4
assert profile["items"]["categories"] == 2
assert profile["pair_categories"]["same_category_rate"] == pytest.approx(1.0)
assert "category_breakdown" in profile
```

- [ ] **Step 2: Run and confirm failure**

```bash
python -m pytest ecup_matching/tests/test_profile_human.py -q
```

Expected: import/function failures.

- [ ] **Step 3: Implement profiler**

Profile must include:

- row counts;
- unique item IDs referenced by pairs;
- overall positive rate;
- target counts;
- category counts;
- same-category pair rate;
- per-category pair count and positive rate;
- name null rate, mean/median/p95 character length;
- attributes null rate, mean/median/p95 character length;
- duplicate normalized-name rate;
- number of distinct normalized names;
- pair text-length quantiles after joining names;
- a warning list when pairs reference missing items or cross categories.

No raw names, attributes, or item IDs are written to reports.

- [ ] **Step 4: Re-run tests**

```bash
python -m pytest ecup_matching/tests/test_profile_human.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ecup_matching/profile_human.py ecup_matching/tests/test_profile_human.py
git commit -m "feat: add aggregate human-data profiler"
```

---

### Task 3: CI dependencies and GitHub Actions orchestration

**Files:**
- Create: `ecup_matching/requirements-ci.txt`
- Create: `.github/workflows/ecup-matching.yml`

**Interfaces:**
- GitHub secret: `HF_TOKEN`.
- Optional workflow env: `HF_REPO_ID=Maksim123321/e-cup-2026-matching-private`.
- Artifact: `ecup-human-profile`, containing `profile.json` and `profile.md` only.

- [ ] **Step 1: Add dependencies**

`requirements-ci.txt` contains compatible bounds for `huggingface_hub`, `requests`, `pandas`, `pyarrow`, and `pytest`.

- [ ] **Step 2: Add `tests` job**

- checkout;
- setup Python 3.11;
- pip install `-r ecup_matching/requirements-ci.txt`;
- run `pytest ecup_matching/tests -q`.

- [ ] **Step 3: Add `profile-human` job**

After tests pass:

- download only `matches.parquet` and `items_human.parquet` from official URLs;
- run `profile_human.py`;
- upload only `profile.json` + `profile.md` via `actions/upload-artifact@v4`;
- delete parquet files in an `if: always()` cleanup step.

- [ ] **Step 4: Add `mirror-hf` job**

After tests pass:

- map `HF_TOKEN: ${{ secrets.HF_TOKEN }}` and `HF_REPO_ID` into environment;
- fail before downloading if token is empty;
- run `python -m ecup_matching.hf_sync`;
- remove temporary mirror directory in an `if: always()` cleanup step.

- [ ] **Step 5: Trigger scope**

Workflow runs on `workflow_dispatch` and pushes to branch `ecup-matching-2026` affecting this workflow or `ecup_matching/**`.

- [ ] **Step 6: Commit**

```bash
git add ecup_matching/requirements-ci.txt .github/workflows/ecup-matching.yml
git commit -m "ci: add E-CUP profile and HF mirror workflow"
```

---

### Task 4: Operator documentation

**Files:**
- Create: `ecup_matching/README.md`

- [ ] **Step 1: Document exact operational flow**

README must state:

1. Branch and isolation guarantee.
2. Destination private HF repo ID.
3. Required GitHub Actions secret is exactly `HF_TOKEN` and must be a Hugging Face user access token with write/create-repository permission.
4. How to add the secret in GitHub UI without exposing it in chat or Git history.
5. Which four source files are mirrored.
6. How to inspect workflow runs and the `ecup-human-profile` artifact.
7. Link to the design document containing the 10 solution families and selected architecture.
8. Explicit warning not to commit raw competition parquet files.

- [ ] **Step 2: Commit**

```bash
git add ecup_matching/README.md
git commit -m "docs: add E-CUP matching workspace guide"
```

---

### Task 5: Verification and first profiling feedback loop

**Files:** none unless a defect is discovered.

- [ ] **Step 1: Inspect GitHub Actions run**

Confirm `tests`, `profile-human`, and `mirror-hf` jobs are present.

- [ ] **Step 2: Verify tests**

All unit tests must pass before interpreting profile output.

- [ ] **Step 3: Read aggregate profile artifact**

Use the profile to update assumptions about category imbalance, text lengths, positive rates and attribute density. Do not expose raw rows.

- [ ] **Step 4: Verify private HF repository**

After `mirror-hf` succeeds, verify `Maksim123321/e-cup-2026-matching-private` contains exactly the four parquet source files plus any private dataset card created by the uploader.

- [ ] **Step 5: Start ML iteration 1**

Use the profile to parameterize the first modeling experiments in this order:

1. item-disjoint validation splitter;
2. normalized lexical/attribute features + CatBoost/LightGBM baseline;
3. off-the-shelf multilingual bi-encoder features;
4. fine-tuned contrastive bi-encoder;
5. weak-label weighting sweep;
6. hard-negative mining;
7. compact Cross-Encoder student;
8. uncertainty cascade;
9. teacher distillation;
10. runtime/quality Pareto selection.

## Self-review

- Spec coverage: data mirror, privacy, profiling, 10-solution research, selected architecture, verification and next ML steps are covered.
- Placeholder scan: no TBD/TODO placeholders are present.
- Interface consistency: names and repo IDs are consistent across tasks.
- Scope: the plan deliberately implements data plumbing/profiling first; model training follows as the next independently testable plan after the first real profile is available.
