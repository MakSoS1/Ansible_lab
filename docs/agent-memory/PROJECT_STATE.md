# E-CUP Matching — Canonical Project State

Updated: 2026-08-13
Current iteration: **v10 no-teacher faststack — COMPLETED, published to private HF, awaiting platform score**

## HARD distribution policy

**GitHub Releases/prereleases are forbidden for E-CUP.** All submission ZIPs and keeper manifests are stored and delivered only through the private Hugging Face dataset `Maksim123321/e-cup-2026-matching-private` under `submissions/<version>/...`.

Do not use `gh release create`, GitHub Release creation APIs/actions, or Release assets as submission storage. GitHub Actions artifacts are allowed only as transient CI/debug evidence. Private HF is the single canonical submission store.

## Objective

Maximize E-CUP 2026 product-matching Macro AP with honest unseen-product validation and an offline organizer-compatible submission that finishes safely inside runtime limits.

The owner reports v7 leaderboard at approximately `0.36`. That is external evidence only and was not used for row-level fitting. The requested region near `0.5` remains the external target; v10 has no measured leaderboard score yet.

## Current keeper

Architecture: **no_teacher + category-shrunk/HGB rank fusion + frozen target-free graph**, with the CPU structured branch overlapped with GPU contrastive inference. The pair cross-encoder teacher is absent from inference and is not packaged.

Exact archive:

- `ecup-v10-no-teacher-graph-0.5950413763-submission.zip`;
- `480249520` bytes;
- SHA-256 `6cebc276f45fc52247db054eb83d2a8110b25d4407cc34b0d5b148a4773c321d`;
- build run `31689478925`;
- source SHA `9de2bc83f878c87703c3290670f042bfdbb70dfc`;
- canonical private HF `submissions/v10/final/`;
- HF run `31693414226` verified both ZIP and `V10_KEEPER.json` after upload.

## Immutable validation protocol

- human labels `365654`;
- development rows `285210`;
- sealed gold rows `80444`;
- five component-disjoint folds;
- cross-split item/component overlap `0`;
- split SHA-256 `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- metric: unweighted mean of `average_precision_score` over exactly 20 categories;
- sealed gold opened `false`, rows scored `0`.

## v10 frozen quality evidence

| Candidate | Strict OOF | Graph OOF | Target-stress mean |
|---|---:|---:|---:|
| structured_only | `0.5808404006` | `0.5821464488` | `0.4355474106` |
| **no_teacher** | **`0.5931387077`** | **`0.5950413763`** | **`0.4496152683`** |
| no_contrastive | `0.5928725263` | `0.5978943607` | `0.4535367991` |

`no_contrastive` has the strongest local diagnostics but retains the pair teacher and was rejected for production runtime risk. v10 deliberately selects `no_teacher` because runtime architecture matters more than the small local delta.

Graph delta for selected v10 is `+0.0019026685699552`, positive on all `5/5` immutable folds. Target-stress is diagnostic only; `0.44961526826354` is not a claimed leaderboard score.

## Runtime evidence

Exact same archive SHA under `odsai/ecup26-matching-baseline:1.0` on NVIDIA GeForce RTX 2060 SUPER:

| Gate | Rows | Outer inference wall | Acceptance | Headroom | Result |
|---|---:|---:|---:|---:|---|
| public-size | `115000` | `173.842174445 s` | `330 s` | `156.157825555 s` | PASS |
| private-size | `275000` | **`391.608035937 s`** | `700 s` | **`308.391964063 s`** | **PASS** |

Private keeper run `31692817075`, artifact `9178292328`, returned code `0` and validated exact pair order, finite/nonconstant scores, `271964` unique scores and absence of a teacher checkpoint.

The earlier `<120/<250` runtime thresholds were an exploratory over-strict tuning target, not organizer rules and not the final keeper contract.

## Why v10 is faster

v9 still used a pair-scaled cross-encoder teacher. v10 removes it entirely and overlaps independent CPU structured scoring with GPU contrastive embedding/scoring. On the 275k keeper run, structured took `297.304 s` and text+contrastive `345.838 s`, but their critical path was `345.838 s` rather than their sum. `run.py` total was `364.64 s`, outer docker wall `391.608035937 s`.

## Hugging Face keeper

Private dataset: `Maksim123321/e-cup-2026-matching-private`.

- `submissions/v10/final/ecup-v10-no-teacher-graph-0.5950413763-submission.zip`;
- `submissions/v10/final/V10_KEEPER.json`.

Publication run `31693414226`: SUCCESS. Both files were SHA/contract-verified and relisted after upload. These HF objects are the canonical distribution copies. Historical GitHub Release copies are not part of the project contract and are being removed.

## Binding lessons

- Never create GitHub Releases or prereleases for E-CUP; publish submissions only to the private HF dataset.
- Infrastructure/runtime failures are not model scores.
- Production refit is not validation.
- Platform leaderboard, strict OOF and target-stress are separate evidence axes.
- Sealed gold is never opened to recover a leaderboard/runtime gap.
- Outside-container wall is authoritative for timeout safety.
- Small local metric gains do not justify restoring a pair-scaled inference stage without a fresh exact runtime proof.
- Continuous graph-rescored ranking scores do not need clipping to `[0,1]` unless the contract requires probabilities.
- Never mutate keeper bytes after the exact runtime gate and keep calling them the same keeper.

## Current files to read

1. `ecup_matching/experiments/CURRENT.json`
2. `ecup_matching/experiments/v10/PLAN.md`
3. `ecup_matching/experiments/v10/RESULTS.md`
4. `ecup_matching/experiments/v10/SAFE_METRICS.json`
5. `docs/agent-memory/EXPERIMENT_INDEX.md`
6. `docs/agent-memory/DECISIONS.md`
7. `docs/agent-memory/SECURITY.md`
8. `docs/agent-memory/ITERATION_PROTOCOL.md`

## Next action

Submit the exact v10 keeper from private HF to the competition platform. When it finishes scoring, record the measured leaderboard value separately without rewriting frozen local validation evidence.