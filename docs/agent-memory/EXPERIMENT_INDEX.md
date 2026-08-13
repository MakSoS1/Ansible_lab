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
| v7 | platform-scored historical candidate | owner reports leaderboard `~0.36`; strict 5-fold OOF was not completed | high fold-0 diagnostic did not transfer |
| v8 | rejected runtime failure | exact gate70 outer wall `820.784 s` | old workflow runtime pass marker invalid; platform timed out |
| v9 old 1.25 GB | rejected-for-submission runtime risk | graph strict OOF `0.5970059311`; target-stress `0.4515676235`; platform `Container did not finish in time` | retain only as historical algorithm evidence |
| **v9 compact** | **completed; published; awaiting platform score** | inherited graph OOF `0.5970059311`; Spearman vs old v9 `0.9999931452`; full E2E 115k/275k GREEN | **current exact keeper** |

Local OOF, compact-equivalence, target-stress diagnostics and platform leaderboard scores are separate evidence axes. Sealed gold remains unopened.

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

## v9 validation v2

Source run `31639183423`. Frozen graph config: reciprocal-best `0`, reciprocal-top3 `0`, endpoint-rank `0.02`, ambiguity penalty `0.01`.

| Candidate | Teacher fraction | Strict OOF | Fold-local graph OOF | Graph delta | Target-stress mean |
|---|---:|---:|---:|---:|---:|
| gate25 | `0.2500227902` | `0.5947115591` | `0.5961903713` | `+0.0014788122` | `0.4507779206` |
| gate40 | `0.4000245433` | `0.5955054274` | `0.5970059311` | `+0.0015005037` | `0.4515676235` |

Gate40 was selected before final runtime. The compact package stores the same production model family differently; the OOF/stress values above were **not remeasured after storage compaction** and remain algorithm-selection evidence.

## v9 compact package

Build run `31675196422`:

- `ecup-v9-compact-fp16storage-0.5970059311-submission.zip`;
- `596,925,132` bytes;
- SHA-256 `aabe663502b9dafe5b925347c3908d6bfe731045467aa85029da6255fbc78345`;
- release tag `ecup-v9-compact-6ba133ce25f7`;
- saved `654,734,829` bytes vs superseded 1.25 GB v9;
- teacher/contrastive floating safetensors stored FP16; integer tensors preserved; structured/meta models unchanged.

Private HF publication run `31677161875` verified both:

- `submissions/v9/compact/ecup-v9-compact-fp16storage-0.5970059311-submission.zip`;
- `submissions/v9/compact/V9_COMPACT_KEEPER.json`.

## Prediction-transfer evidence

RTX equivalence run `31675338174`, 20,000 fixed pairs:

- Spearman `0.9999931451822124`;
- Pearson `0.9999931722281602`;
- mean absolute delta `0.00016378653597360943`;
- p99 `0.0019011406844106082`;
- top-1% overlap `1.0`;
- top-5% and top-10% overlap `0.999`;
- output schema/order/finite-score checks pass.

This is near-identical ranking evidence, not exact numeric equality and not a new strict OOF score.

## End-to-end runtime evidence

Final run `MakSoS1/gpu-dispatch#31675903851` on RTX 2060 SUPER and organizer image. Timer starts before safe ZIP extraction and ends after output validation.

| Gate | Rows | Extraction | Inference | Total | Acceptance | Headroom | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| public-size | `115,000` | `5.234229078 s` | `287.592726358 s` | `293.569452608 s` | `330 s` | `36.430547392 s` | PASS |
| private-size | `275,000` | `5.714303003 s` | `640.236212537 s` | `646.947129008 s` | `700 s` | `53.052870992 s` | PASS |

Evidence artifact `9171929877`. Both returned code `0` with exact pair order and valid finite nonconstant scores.

## Repository verification

Compact verification run `31676442849`: **425 passed, 5 skipped**, `scripts/memory_policy.py` **OK**.

## Binding lessons

- infrastructure/runtime failures are not model-quality evidence;
- production refit is not validation;
- leaderboard, strict OOF, target-stress and compact-equivalence remain separate axes;
- sealed gold is never used to recover runtime/leaderboard gaps;
- organizer-like runtime should include ZIP extraction/setup where feasible;
- outside-container wall is authoritative for timeout safety;
- mixed-precision/storage changes require direct prediction/ranking evidence;
- continuous ranking scores need no clipping to `[0,1]` unless the contract requires probabilities;
- do not weaken a gate to publish an archive.

## Next external measurement

Submit the exact compact HF keeper. When the platform finishes, record the measured leaderboard score separately without rewriting strict OOF or target-stress history.
