# E-CUP v14 — canonical validation recovery + safe residual plan

Status: **in progress** (final artifact runtime gate pending)

## Why v14 exists

v13 B improved the local fold-0 / Validation-v3 proxy but regressed externally:

- v12 fold-0 Macro AP: `0.7059297810308699`
- v12 Public LB: `0.3798116204`
- v13 B fold-0 Macro AP: `0.7086611385531062`
- v13 B Public LB: `0.3783781653`
- Public LB delta `v13 - v12 = -0.0014334551`

Therefore v13 is an external negative anchor: local near-neighbour ordering was inverted. v14 must not be another weak-sampling micro-tune selected on fold0 alone.

## Past failures incorporated into the design

1. **v9/v10:** strong local neural candidates were too expensive / unsafe for the runtime contract. v14 keeps a single v12-compatible CrossEncoder at inference and allows only a lightweight post-score.
2. **v11:** startup / item-scan behavior can dominate the time budget. Every promoted archive must pass the organizer-shaped supplied-item Check on the exact final bytes.
3. **v13:** Validation-v3/fold0 improved while Public LB declined. v14 records v13 as a measured negative external anchor and requires cross-fit evidence instead of accepting a tiny full-fold gain.
4. **Historical weak labels:** `matches_llm.parquet` has 11,187,780 rows but exact overlap with human-labelled pairs is `0`; its precision cannot be audited against human truth. v14 admits **zero** LLM-labelled rows.
5. **Split reproducibility:** recomputing the historical split in the current environment produced a different manifest hash. v14 recovers and pins the original row map from a historical strict OOF artifact instead of silently creating new folds.

## Immutable validation contract

Historical split SHA:

`aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`

Recovered canonical row-map SHA:

`00778edd7ed4581f8aedc143052d17d6fb86c55abfaee9fc6a169f72bb47b32f`

Source evidence:

- historical artifact ID: `9175469673`
- source run: `31680767570`
- source artifact: `ecup-v10-tiny-oof-31680767570`
- development rows: `285210`
- sealed rows: `80444`
- fold counts: `57042 / 57042 / 57043 / 57042 / 57041`
- current-data alignment: `0` dev↔sealed item overlap and `0` train↔held item overlap for each of the five folds
- sealed gold remains unopened and unscored

## Implementation stages

### Stage A — audit label streams

- Audit `matches_llm.parquet` against human truth before any use.
- Admission rule: if a controlled overlap does not exist or precision floors cannot be measured, do not use the stream.
- Result: overlap `0`, admission `false`; v14 is human-only.

### Stage B — new item-centric architecture research

Implement human-only item-centric alternatives:

- A0 pooled multi-vector item matcher;
- A1 pooled matcher + repeated **known human negatives only**;
- A2 one-encoder token MaxSim / late-interaction matcher;
- component closure may create positive rows only inside training components; unknown pairs are never invented as negatives;
- pair scoring is symmetric by construction.

Engineering findings are preserved even when the expensive model is not promoted:

- component closure originally failed for endpoints that occurred only as `id2`; fixed with endpoint-safe category lookup;
- split recomputation mismatch was replaced by the explicit recovered canonical row map;
- A2 strict five-fold was intentionally not completed after projected multi-hour GPU cost; this is **not** a quality rejection.

### Stage C — fast safe-parent residual

Use the best measured external parent, v12, unchanged as the neural base:

- exact v12 parent archive SHA: `a189eb9eaf97ad74c323ef446759c4b42e392f09df8d65327f938b582d01dac1`
- exact archive bytes: `663760211`
- v12 fold-0: `0.7059297810308699`
- v12 Public LB: `0.3798116204`

Fit a small structured/lexical residual on **human development labels only** using six symmetric features:

1. model-token Jaccard;
2. numeric-token Jaccard;
3. numeric conflict count;
4. model/SKU overlap indicator;
5. title-token Jaccard;
6. title token-count ratio.

The residual is category-specific logistic regression. The final blend is category-local rank blending, preserving the base CrossEncoder as the dominant signal.

### Stage D — cross-fit gate

The fold-0 held set is partitioned into two item/component-disjoint halves.

For each category:

1. select alpha on side 0 and evaluate that alpha on side 1;
2. select alpha on side 1 and evaluate that alpha on side 0;
3. admit the category only if both sides select a positive correction and both opposite-half deltas are non-negative;
4. rejected categories remain **exactly** at the v12 base score.

Promotion rule, frozen before the corrected evaluation:

- at least 3 admitted categories;
- both aggregate cross-fit deltas `>= 0`;
- mean aggregate cross-fit delta `>= 0.0005`;
- full fold-0 Macro AP improves over v12;
- at least 17/20 full-fold category AP deltas are non-negative.

A bug in the first v2 evaluator applied opposite-half alphas even to categories that had been rejected by the category gate. The evaluator was corrected so rejected categories stay at base; the promotion thresholds were **not changed**.

### Stage E — production packaging

If the corrected category-gated residual passes:

- reuse the exact byte-pinned v12 checkpoint/runtime as parent;
- add only `residual.json`, a lightweight residual runtime module, and a wrapper entrypoint;
- keep exactly one `.safetensors` checkpoint;
- no competition parquet/csv data inside the archive;
- no LLM rows;
- no sealed-gold score.

### Stage F — binding runtime gate

On the **exact final ZIP**:

- safely extract ZIP;
- build the organizer-shaped 1000-pair fixture and supplied-item subset;
- run organizer image `odsai/ecup26-matching-baseline:1.0`;
- timer includes ZIP extraction + inference + output generation;
- require wall `< 60 s`, return code `0`, valid `[id1,id2,predict]`, finite predictions and >10 unique scores.

Only after this gate may `ready_for_submission=true` be written.

## Acceptance and honesty rules

- Do not claim Public LB > 0.5 without a platform submission proving it.
- A local improvement is evidence, not a guaranteed leaderboard gain.
- v12 remains the external safety anchor until the new archive obtains its own Public LB.
- If a new residual fails cross-fit or runtime, fall back to exact v12 behavior rather than shipping an unverified regression.
