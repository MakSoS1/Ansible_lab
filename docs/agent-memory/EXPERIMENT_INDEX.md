# E-CUP Matching — Experiment Index

Canonical short registry. Detailed rationale and immutable evidence live in `ecup_matching/experiments/v*/PLAN.md`, `RESULTS.md`, `SAFE_METRICS.json`, `VALIDATION_V2.json` where present, and private artifacts.

## Version summary

| Version | Status | Validation/evidence | Interpretation |
|---|---|---|---|
| v1 | historical | hidden `0.2345852292` | historical anchor |
| v2 | historical verified platform fallback | hidden `0.2583231811` | strongest early hidden anchor |
| v3 | historical | hidden canonical `0.2481015189` | historical |
| v4 | historical | hidden canonical `0.2531285195` | historical |
| v5 | completed quality-first | strict 5-fold OOF `0.6018115534` | best retained strict local quality |
| v6 | runtime reference | strict OOF `0.6006003615` | selective-teacher/runtime engineering family |
| v7 | platform-scored historical candidate | owner reports leaderboard `~0.36`; no honest strict 5-fold OOF was completed | high fold-0 diagnostic did not transfer |
| v8 | rejected runtime failure | graph/prevalence diagnostics useful; exact outer wall `820.784 s` on gate70 evidence | workflow runtime pass marker was invalid; platform timed out |
| **v9** | **in progress** | gate40 graph strict OOF `0.5970059311`; target-stress `0.4515676235`; exact package built | **leaderboard-adapted + corrected outer-wall runtime gate** |

Local OOF, target-stress diagnostics and platform leaderboard scores are distinct evidence axes and must never be relabeled as one another. Sealed gold remains unopened.

## Immutable validation facts

- human rows: `365,654`;
- development rows: `285,210`;
- sealed gold rows: `80,444`;
- connected item components: `345,654`;
- immutable development folds: `5`;
- cross-split item overlap: `0`;
- split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- metric: unweighted mean of sklearn `average_precision_score` over exactly 20 official categories;
- sealed gold opened: **false**;
- sealed gold rows scored: **0**.

## v5 retained quality reference

The v5 ladder established that heterogeneous lexical/attribute/neural signals combine well but exhibit diminishing returns:

| Step | Strict OOF Macro AP |
|---|---:|
| category specialists | `0.5476780661` |
| weak specialists | `0.5514237339` |
| sparse TF-IDF | `0.5651306839` |
| explicit per-key attrs | `0.5683065131` |
| supervised contrastive | `0.5662217063` |
| four-signal equal-rank | `0.5870570848` |
| + pair teacher | `0.5952697490` |
| + typed explicit | `0.5975445721` |
| category-shrunk simplex | `0.6009542418` |
| fixed HGB stack | `0.6006290885` |
| **50/50 category-shrunk + HGB rank fusion** | **`0.6018115534`** |

Verified v5 package SHA-256: `442769bd2c92d43730d7034fb91d8a83e596a8445ae3c3f887783890e90284d5`.

## v6 runtime family

v6 introduced target-free selective pair-teacher gating plus prediction-preserving runtime work. Retained reference was gate95 at strict OOF `0.6006003614522999`.

Important runtime improvements retained by v9 where applicable:

- unchanged structured chunk boundaries distributed through a fork worker pool;
- shared normalization/difflib work;
- single-pass item selection;
- deferred CUDA initialization;
- stable length bucketing and VRAM-aware batches;
- offline/local-files-only neural inference;
- import-closure packaging rather than hand-maintained file lists;
- CUDA FP16/autocast after measured near-rank identity;
- structured multiprocessing capped at `8` after empirical sweep.

## v7 platform anchor

v7 produced fold-0 diagnostics around `0.70238`, but a complete five-fold strict OOF was never established before submission. The owner now reports the successfully scored v7 leaderboard result at approximately **`0.36`**.

Interpretation:

- fold-0 research diagnostics were not a reliable estimate of platform performance;
- the leaderboard value is external evidence only and is not used as a training target;
- v9 stores it with `used_for_fitting=false` and uses it only to motivate distribution-shift-aware validation.

## v8 useful evidence and rejection

v8 established two useful findings despite being rejected as a submission:

1. Human positive prevalence and retrieval/LLM pair prevalence differ materially; the measured ratio used for deterministic stress is `0.566880890615799`.
2. A cheap target-free graph rescore is small but consistently positive when evaluated fold-locally.

v8 is rejected because the exact gate70 runtime evidence showed:

- inner `run.py` roughly `731.22 s`;
- true outside-container wall `820.784 s`;
- the old workflow nevertheless wrote a pass because it relied on timeout exit/output instead of the measured outer wall;
- the platform subsequently returned `Container did not finish in time` again.

Binding consequence: runtime success must be computed from outside-container wall time.

## v9 validation v2

Source validation run: `31639183423`.

Frozen graph config:

- reciprocal-best bonus `0.0`;
- reciprocal-top3 bonus `0.0`;
- endpoint-rank weight `0.02`;
- ambiguity penalty `0.01`.

| Candidate | Teacher fraction | Strict OOF | Fold-local graph OOF | Graph delta | Target-stress mean | Stress p05–p95 |
|---|---:|---:|---:|---:|---:|---:|
| gate25 | `0.2500227902` | `0.5947115591` | `0.5961903713` | `+0.0014788122` | `0.4507779206` | `0.4481787745–0.4534007374` |
| **gate40** | **`0.4000245433`** | **`0.5955054274`** | **`0.5970059311`** | **`+0.0015005037`** | **`0.4515676235`** | **`0.4489228671–0.4542054080`** |

All five folds have positive graph delta for both candidates. Gate40 was selected **before** the final runtime result because it dominates gate25 on strict OOF, graph OOF and target-stress mean. Gate25 remains the predeclared fallback if the exact runtime veto fires.

The owner-reported v7 leaderboard score `~0.36` is not used to fit these scores. The desired v9 leaderboard region near `0.5` is a target only; v9 measured leaderboard score is still unknown.

## v9 production/package evidence

Gate40 production refit:

- run `31639692541`;
- artifact `9158411928`;
- full development rows `285,210`;
- actual teacher fraction `~0.400025`;
- elapsed `74.4 s`;
- peak RAM `0.736 GiB`;
- private HF `experiments/v9/production/gate40/853a3925ac2b`;
- sealed gold untouched.

Exact candidate package:

- build run `31640050373`;
- file `ecup-v9-gate40-fp16-graph-0.5970059311-submission.zip`;
- bytes `1,251,659,961`;
- SHA-256 `925456cde1e47c50dc0141ce64bed5ef00d9f574152f285869ebea2db6935782`;
- release tag `ecup-v9-gate40-final-eb2bcf18d53e`;
- complete optimized runtime closure, FP16, cap8 and frozen graph metadata verified during build.

Gate25 production fallback is also prebuilt:

- run `31640425364`;
- artifact `9158679674`;
- strict OOF `0.5947115591000889`;
- graph OOF `0.5961903713277379`;
- target-stress mean `0.45077792061326727`.

## v9 exact runtime gate

Corrected private GPU run: `MakSoS1/gpu-dispatch` run `31640233511` on RTX 2060 SUPER.

Pass contract:

- exact final ZIP bytes;
- organizer image;
- exact `275,000` pair fixture;
- watchdog `720 s`;
- pass only if exit code `0`, output valid, and **outer wall `<=700.0 s`**;
- `700.001 s` fails;
- no threshold relaxation after observing a result.

At the time of this registry update the exact organizer run is still in progress, so v9 is not yet marked completed.

## Binding lessons

- infrastructure failures are not model-quality evidence;
- production refit is not validation;
- target-fitted layers require outer cross-fitting;
- leaderboard, strict OOF and stress diagnostics remain separate;
- sealed gold is never used to recover runtime/leaderboard gaps;
- high research-fold diagnostics cannot substitute for complete strict OOF;
- a smoke test is compatibility evidence, not runtime evidence;
- outer wall is authoritative for timeout safety;
- submission runtime closure is derived from imports, never maintained manually;
- mixed precision must be validated before retention;
- do not weaken a gate to publish an archive.

## Next gate

Finish run `31640233511`. Keep gate40 only if the exact outside-container wall is `<=700 s` and output validation passes. Otherwise build/package the already-predeclared gate25 fallback and run the same exact gate. Then finalize `v9/RESULTS.md`, `v9/SAFE_METRICS.json`, canonical state, full tests, memory policy, hardened Memora ingest/checkpoint, and hand off only the byte-verified keeper ZIP.
