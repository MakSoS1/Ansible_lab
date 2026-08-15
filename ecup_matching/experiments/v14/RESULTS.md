# E-CUP v14 — Results

Status: **submission-ready; runtime-verified; Public LB not yet measured**

## External anchors

| Version | Local comparable fold-0 Macro AP | Public LB |
|---|---:|---:|
| v12 | 0.7059297810308699 | **0.3798116204** |
| v13 B | 0.7086611385531062 | **0.3783781653** |
| v14 final | **0.7065769713851786** | pending |

The v13 result demonstrates a real local/external ranking inversion: v13 improved fold0 but lost `0.0014334551` Public LB to v12. v14 therefore uses v12 as the measured external parent and only adds a correction that survives item-disjoint cross-fit.

## Canonical validation recovery

The historical split was recovered from strict OOF evidence instead of recomputing a new split:

- historical split SHA: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`
- recovered row-map SHA: `00778edd7ed4581f8aedc143052d17d6fb86c55abfaee9fc6a169f72bb47b32f`
- source artifact ID: `9175469673`
- source run ID: `31680767570`
- development rows: `285210`
- sealed rows: `80444`
- current-data dev↔sealed item overlap: `0`
- current-data train↔held overlap across all five folds: `0`
- sealed gold opened: `false`
- sealed gold scored: `0`

## LLM weak-label audit

`matches_llm.parquet` contains `11,187,780` rows but has exact human-pair overlap `0`. Its positive/negative precision cannot be audited against the human truth set. Admission was therefore `false`; v14 new work uses `0` LLM-labelled rows.

## Residual research

### v1 scalar residual — rejected

- v12 fold0 reproduced exactly: `0.7059297810308699`
- diagnostic fold0: `0.7060368061079549`
- diagnostic delta: `+0.00010702507708482134`
- cross-fit side0 delta: `+0.00020317536350666998`
- cross-fit side1 delta: `+0.0003702048387999035`
- cross-fit mean: `+0.0002866901011532867`
- only `11/20` categories non-negative
- rejected by the frozen promotion gate

### v2 category-gated residual — evaluator bug found

The first v2 implementation correctly computed category admission evidence but its aggregate cross-fit prediction mistakenly applied opposite-half alpha to categories later rejected by the category gate. This made the aggregate gate inconsistent with the production decision rule.

The evaluator was fixed so rejected categories remain at the v12 base score. **No promotion threshold was relaxed.**

### v2 corrected — accepted

Probe run: `31882322590`

Evidence artifact: `9246360741`

- v12 fold0 anchor: `0.7059297810308699`
- v14 diagnostic fold0: **`0.7065769713851786`**
- diagnostic delta vs v12: **`+0.0006471903543086022`**
- cross-fit side0 delta: **`+0.000437006267165585`**
- cross-fit side1 delta: **`+0.000734831086673049`**
- cross-fit mean delta: **`+0.000585918676919317`**
- admitted categories: `6`
- full-fold categories non-negative vs v12: **`20/20`**
- accepted by the unchanged rule (`mean >= 0.0005`, both sides non-negative, >=3 admitted categories, full fold improves, >=17/20 non-negative categories)

Active production corrections:

- `Автотовары`: alpha `0.025`
- `Аптека`: alpha `0.025`
- `Бытовая техника`: alpha `0.10`
- `Мебель`: alpha `0.05`
- `Музыкальные инструменты`: alpha `0.075`
- `Спорт и отдых`: alpha `0.025`

All other categories preserve the v12 base score exactly.

## Final submission artifact

Packaging/runtime run: `31882572941`

Filename:

`ecup-v14-v12-category-gated-residual-submission.zip`

- bytes: **`663770301`**
- SHA-256: **`fcaace1a7f0e663b7c9b0b29ca78a768241c3b417b8f4d4a342f52874a29615e`**
- parent v12 archive SHA-256: `a189eb9eaf97ad74c323ef446759c4b42e392f09df8d65327f938b582d01dac1`
- exactly one `.safetensors` checkpoint
- no competition parquet/csv data inside the ZIP
- final ZIP integrity test: PASS

## Binding runtime Check on exact final bytes

Organizer image: `odsai/ecup26-matching-baseline:1.0`

- pair fixture: `1000`
- supplied items: `1999`
- ZIP extraction: `3.4397788900005253 s`
- wall time, including extraction + inference + output: **`28.810029840000425 s`**
- limit: `60 s`
- return code: `0`
- timed out: `false`
- output valid: `true`
- unique scores: `910`
- result: **PASS**

## Interpretation

This package has stronger local/cross-fit evidence than the v12 parent while retaining the measured-best v12 CrossEncoder as the dominant inference signal and remaining safely inside the runtime budget.

This does **not** prove a Public LB above v12 or above `0.5`; only a platform submission can measure that. v12 remains the best measured external anchor until v14 receives its own Public LB result.
