# E-CUP Matching — Experiment Index

Canonical short registry. Detailed rationale and immutable evidence live under `ecup_matching/experiments/` and in private artifacts.

## Version summary

| Version | Status | Validation/evidence | Interpretation |
|---|---|---|---|
| v1 | historical | hidden `0.2345852292` | historical anchor |
| v2 | historical verified platform fallback | hidden `0.2583231811` | strongest early hidden anchor |
| v3 | historical | hidden canonical `0.2481015189` | historical |
| v4 | historical | hidden canonical `0.2531285195` | historical |
| v5 | completed quality-first | strict 5-fold OOF `0.6018115534` | best retained strict local quality reference |
| v6 | runtime reference | strict OOF `0.6006003615` | selective-teacher/runtime engineering family |
| v7 | platform-scored historical candidate | owner reports leaderboard `~0.36`; strict 5-fold OOF was not completed | external transfer anchor only |
| v8 | rejected runtime failure | exact gate70 outer wall `820.784 s` | old workflow runtime pass marker invalid; platform timed out |
| v9 | completed historical keeper | graph strict OOF `0.5970059311`; stress `0.4515676235`; compact local 275k `646.947 s` | platform timeout family; superseded by v10 runtime redesign |
| v10 | **completed current keeper** | graph strict OOF `0.5950413763`; stress `0.4496152683`; exact 275k `391.608 s` | no-teacher overlapped faststack, HF-published, awaiting platform score |

Local OOF, target-stress diagnostics, runtime evidence and platform leaderboard scores are separate evidence axes. Sealed gold remains unopened.

## Immutable validation facts

- human rows: `365654`;
- development rows: `285210`;
- sealed gold rows: `80444`;
- connected item components: `345654`;
- immutable development folds: `5`;
- cross-split item overlap: `0`;
- split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- metric: unweighted mean of sklearn `average_precision_score` over exactly 20 official categories;
- sealed gold opened: `false`;
- sealed gold rows scored: `0`.

## v10 selection evidence

| Candidate | Strict OOF | Graph OOF | Graph delta | Positive folds | Stress mean |
|---|---:|---:|---:|---:|---:|
| structured_only | `0.5808404006` | `0.5821464488` | `+0.0013060482` | `5/5` | `0.4355474106` |
| **no_teacher** | **`0.5931387077`** | **`0.5950413763`** | **`+0.0019026686`** | **`5/5`** | **`0.4496152683`** |
| no_contrastive | `0.5928725263` | `0.5978943607` | `+0.0050218344` | `5/5` | `0.4535367991` |

`no_contrastive` was rejected for production because it retains the pair cross-encoder teacher. v10 deliberately accepts a small local-metric tradeoff to remove the stage responsible for the worst pair-scaled runtime risk.

## v10 exact keeper

Build run `31689478925`:

- `ecup-v10-no-teacher-graph-0.5950413763-submission.zip`;
- `480249520` bytes;
- SHA-256 `6cebc276f45fc52247db054eb83d2a8110b25d4407cc34b0d5b148a4773c321d`;
- release tag `ecup-v10-faststack-9de2bc83f878`;
- source SHA `9de2bc83f878c87703c3290670f042bfdbb70dfc`;
- teacher checkpoint absent;
- CPU structured and GPU contrastive branches overlap.

## v10 runtime evidence

Exact same archive SHA, organizer image, RTX 2060 SUPER:

| Gate | Rows | Outer inference wall | Acceptance | Headroom | Result |
|---|---:|---:|---:|---:|---|
| public-size | `115000` | `173.842174445 s` | `330 s` | `156.157825555 s` | PASS |
| private-size | `275000` | `391.608035937 s` | `700 s` | `308.391964063 s` | **PASS** |

Private keeper gate `31692817075`, artifact `9178292328`: return code `0`, exact pair order valid, finite nonconstant scores, `271964` unique scores.

The earlier `<120/<250` experiment was an intentionally over-strict tuning target and is not the production acceptance contract or an organizer rule.

## v10 publication

HF run `31693414226`: **SUCCESS**. Verified remote paths:

- `submissions/v10/final/ecup-v10-no-teacher-graph-0.5950413763-submission.zip`;
- `submissions/v10/final/V10_KEEPER.json`.

## Historical v9 note

v9 validation-v2 selected gate40 at strict graph OOF `0.597005931143384` and target-stress `0.4515676235464289`. Its compact keeper reduced package size and passed local engineering gates, but the platform-timeout experience is the reason v10 removes pair-teacher inference rather than continuing to optimize the same multi-stage runtime.

## Binding lessons

- infrastructure/runtime failures are not model-quality evidence;
- production refit is not validation;
- leaderboard, strict OOF and target-stress remain separate axes;
- sealed gold is never used to recover runtime or leaderboard gaps;
- outside-container wall is authoritative for timeout safety;
- continuous ranking scores do not require clipping to `[0,1]` unless the competition contract requires probabilities;
- runtime architecture can dominate small local metric gains;
- never restore pair-teacher inference to v10 only to recover the `no_contrastive` local delta without a new end-to-end runtime proof.

## Next external measurement

Submit the exact v10 HF keeper. When the platform finishes, record the measured v10 leaderboard score separately without rewriting frozen OOF or target-stress evidence.
