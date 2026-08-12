# E-CUP Matching — Experiment Index

Canonical short registry. Detailed rationale and immutable evidence live under `ecup_matching/experiments/` and in the private artifacts.

## Version summary

| Version | Status | Validation/evidence | Interpretation |
|---|---|---|---|
| v1 | historical | hidden `0.2345852292` | historical anchor |
| v2 | historical verified platform fallback | hidden `0.2583231811` | strongest early hidden anchor |
| v3 | historical | hidden canonical `0.2481015189` | historical |
| v4 | historical | hidden canonical `0.2531285195` | historical |
| v5 | completed quality-first | strict 5-fold OOF `0.6018115534` | best retained strict local quality |
| v6 | runtime reference | strict OOF `0.6006003615` | selective-teacher/runtime engineering family |
| v7 | platform-scored historical candidate | owner reports leaderboard `~0.36`; strict 5-fold OOF was not completed | high fold-0 diagnostic did not transfer |
| v8 | rejected runtime failure | exact gate70 outer wall `820.784 s` | old workflow runtime pass marker was invalid; platform timed out |
| v9 | in progress | gate40 graph strict OOF `0.5970059311`; target-stress `0.4515676235`; exact package built | leaderboard-adapted + corrected outer-wall runtime gate |

Local OOF, target-stress diagnostics and platform leaderboard scores are separate evidence axes. Sealed gold remains unopened.

## Immutable validation facts

- human rows: `365,654`;
- development rows: `285,210`;
- sealed gold rows: `80,444`;
- connected item components: `345,654`;
- immutable development folds: `5`;
- cross-split item overlap: `0`;
- split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- metric: unweighted mean of sklearn `average_precision_score` over exactly 20 official categories;
- sealed gold opened: `false`;
- sealed gold rows scored: `0`.

## v5 retained quality reference

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
| 50/50 category-shrunk + HGB rank fusion | `0.6018115534` |

Verified v5 package SHA-256: `442769bd2c92d43730d7034fb91d8a83e596a8445ae3c3f887783890e90284d5`.

## v7 platform anchor

The owner reports v7 leaderboard `~0.36`. v7 also had fold-0 diagnostics around `0.70238`, but those were not honest five-fold strict OOF. v9 stores the leaderboard observation as external evidence with `used_for_fitting=false`.

## v8 rejection and retained lessons

v8 established a retrieval/human positive-prevalence ratio `0.566880890615799` and a small consistently positive target-free graph rescore. It is rejected as a submission because exact gate70 evidence showed inner `run.py` around `731.22 s` but true outside-container wall `820.784 s`; the old workflow nevertheless declared a pass. The platform then timed out again.

Binding consequence: outside-container wall is authoritative for timeout safety.

## v9 validation v2

Source run: `31639183423`.

Frozen graph config: reciprocal-best `0`, reciprocal-top3 `0`, endpoint-rank `0.02`, ambiguity penalty `0.01`.

| Candidate | Teacher fraction | Strict OOF | Fold-local graph OOF | Graph delta | Target-stress mean |
|---|---:|---:|---:|---:|---:|
| gate25 | `0.2500227902` | `0.5947115591` | `0.5961903713` | `+0.0014788122` | `0.4507779206` |
| gate40 | `0.4000245433` | `0.5955054274` | `0.5970059311` | `+0.0015005037` | `0.4515676235` |

All five folds have positive graph delta for both candidates. Gate40 was selected before final runtime because it dominates gate25 on strict OOF, graph OOF and target stress. Gate25 is the predeclared fallback.

Two zero/negligible-runtime meta experiments were rejected after held-out evaluation:

- fixed prevalence-weighted HGB: strict delta `-0.0000658754`, graph delta `-0.0000604780`, target-stress delta `-0.0000779612`;
- cross-fitted category-specific category/HGB fusion: strict delta `-0.0005091711`, graph delta `-0.0002712630`, target-stress delta `-0.0002826191`.

Neither is present in the keeper package.

## v9 production and package evidence

Gate40 production refit run `31639692541`:

- `285,210` development rows;
- actual teacher fraction `~0.400025`;
- elapsed `74.4 s`;
- peak RAM `0.736 GiB`;
- artifact `9158411928`;
- private HF `experiments/v9/production/gate40/853a3925ac2b`;
- sealed gold untouched.

Exact candidate package from run `31640050373`:

- `ecup-v9-gate40-fp16-graph-0.5970059311-submission.zip`;
- `1,251,659,961` bytes;
- SHA-256 `925456cde1e47c50dc0141ce64bed5ef00d9f574152f285869ebea2db6935782`;
- release tag `ecup-v9-gate40-final-eb2bcf18d53e`;
- optimized complete runtime closure, FP16, cap8 and frozen graph metadata verified.

Gate25 fallback refit is prebuilt in run `31640425364`, artifact `9158679674`.

## v9 runtime evidence

First corrected exact 275k gate40 run `MakSoS1/gpu-dispatch#31640233511` measured:

- exact package SHA above;
- organizer image;
- RTX 2060 SUPER;
- return code `0`;
- inner `run.py` total `567.23 s`;
- outside-container wall `637.82083456 s`;
- 275,000 output rows.

Its red workflow conclusion came only from an over-strict validator that required graph-rescored predictions to lie in `[0,1]`. A separate 1000-row diagnostic run `31641425359` proved exact columns, row count, ID order, finite predictions and nonconstant scores; the only failed predicate was the artificial `[0,1]` range check. No clipping was added because the official submission contract requires a continuous numeric score and clipping would introduce ties/change the validated ranking.

Final dual public/private exact-package gate is `MakSoS1/gpu-dispatch#31641656589`: the 115k public-size stage is already GREEN and the 275k private-size repeat is the final pending runtime verdict.

## Binding lessons

- infrastructure failures are not model-quality evidence;
- production refit is not validation;
- target-fitted layers require outer cross-fitting;
- leaderboard, strict OOF and stress diagnostics remain separate;
- sealed gold is never used to recover runtime/leaderboard gaps;
- high single-fold diagnostics cannot substitute for strict OOF;
- outside-container wall is authoritative for timeout safety;
- submission runtime closure is derived from imports, never hand-maintained;
- mixed precision must be validated before retention;
- do not weaken a gate to publish an archive.

## Next gate

Finish dual run `31641656589`. If both public and private stages pass, freeze the exact gate40 ZIP as v9. Then update final result/state files, run the full repository test suite plus memory policy, execute hardened Memora ingest/checkpoint for iteration v9, and hand off only the byte-verified keeper archive.
