# E-CUP Matching — Experiment Index

Updated: **2026-08-15**. This index separates historical local metrics, external leaderboard evidence and runtime/package evidence.

| Version | Core idea | Local / proxy evidence | Public / platform evidence | Decision |
|---|---|---|---|---|
| v1 | structured lexical HGB | local `0.4961654895` on old holdout | `0.2345852292` | historical |
| v2 | structured + weak curriculum | local `0.5010008995` old holdout | `0.2583231811` | historical external anchor |
| v3 | v2 + tiny ruBERT blend | local `0.5254642646` old holdout | canonical `0.2481015189` | historical |
| v4 | cross-fitted category routing | OOF `0.5276431099` | `0.2531285195` | historical |
| v5 | six-signal / category-shrunk / HGB development ladder | strict OOF reached `0.6018115534` | no reliable comparable LB retained here | proved extra signals useful but inference became heavy |
| v6 | runtime engineering of structured stack | strict OOF `0.6006003615` | — | CPU bottleneck identified |
| v7 | one ruBERT CrossEncoder | fold0 ~`0.70238` | **`0.3655833314`** | reliable fast external anchor |
| v8 | hard-negative training | fold0 `0.6555648876`, worse than v7 | calibration submission not recorded | hard-negative policy rejected locally; useful spread anchor candidate |
| v9 | teacher/contrastive/structured/graph stack | OOF ~`0.597` | timeout family | too heavy |
| v10/v11 | parallelized structured/TF-IDF/graph runtime | OOF ~`0.595`; v11 local 115k `161.9 s`, 275k `379.2 s` | platform timeout | architecture closed; benchmark fixture was unrepresentative |
| v12 | v7 runtime + stronger weak supervision | fold0 `0.7059297810` | **`0.3798116204`** | best measured external parent |
| v13 | B/groupweak: preserve weak retrieval-anchor groups/topology | fold0 **`0.7086611386`**; frozen p05 **`0.5690974845`**, mean **`0.6869505675`** | **`0.3783781653`** | measured negative anchor: local ordering inverted vs v12 |
| v14 | v12 CrossEncoder + human-only category-gated lexical residual | fold0 **`0.7065769714`**; `Δ=+0.0006471904`; cross-fit mean **`+0.0005859187`**; 20/20 categories non-negative | pending | **submission-ready; exact final ZIP runtime-verified** |

## High-value research results

### Validation and distribution

- Immutable historical split: `285,210` development + `80,444` sealed gold, 5 component-disjoint folds, SHA `aae58f...eb55b`, gold unopened.
- v7 `0.70238 local -> 0.36558 LB`; v12 `0.70593 local -> 0.37981 LB`; v13 `0.70866 local -> 0.37838 LB`. v13 proves local near-neighbour ordering can invert externally.
- The original split row map was recovered from historical strict-OOF artifact `9175469673` / run `31680767570` and pinned by row-map SHA `00778edd7ed4581f8aedc143052d17d6fb86c55abfaee9fc6a169f72bb47b32f`.
- Current data alignment against that row map is zero dev↔sealed item overlap and zero train↔held item overlap in every fold.
- `matches_llm.parquet` contains `11,187,780` weak rows but exact human-pair overlap is `0`, so weak-label precision cannot be audited directly; v14 admits zero LLM-labelled rows.

### Runtime

- Historical v11 exact full-item Check forensic run `31789001358`: `60.033 s` timeout before valid output.
- v13 binding supplied-item Check: `26.1353473 s / 60 s`, valid output, return code 0.
- v14 final binding supplied-item Check on the exact ZIP: **`28.81002984 s / 60 s`**, valid output, return code 0, `910` unique scores.
- Heavy multi-model inference remains rejected; v14 keeps the single v12 ruBERT checkpoint and adds only a lightweight structured residual.

### v13 conclusion

- v13 B local fold0 `0.7086611385531062` was higher than v12 `0.7059297810308699`.
- Public LB `0.3783781653` was lower than v12 `0.3798116204` by `0.0014334551`.
- Therefore v13 is retained as a measured negative external anchor; Validation-v3/fold0 alone must not promote near-neighbour candidates.

### v14 causal ladder

- **LLM admission REJECT:** exact human overlap `0`; no measurable precision against human truth.
- **A2 MaxSim research retained but not quality-rejected:** canonical-split and endpoint-direction engineering bugs were fixed; full strict Transformer cycle was stopped because projected GPU cost was multi-hour, not because a quality metric failed.
- **Residual v1 REJECT:** diagnostic `+0.0001070`, cross-fit mean `+0.0002867`, only 11/20 categories non-negative.
- **Residual v2 initial evaluator REJECT was invalid as a promotion decision:** aggregate cross-fit incorrectly applied residual to categories later rejected by the category gate.
- **Residual v2 corrected ACCEPT:** unchanged gate passed with side deltas `+0.0004370063` / `+0.0007348311`, mean `+0.0005859187`, fold0 `0.7065769714`, six admitted categories and 20/20 full-fold categories non-negative.

## Exact v14 artifact identity

- `ecup-v14-v12-category-gated-residual-submission.zip`
- bytes `663770301`
- SHA-256 `fcaace1a7f0e663b7c9b0b29ca78a768241c3b417b8f4d4a342f52874a29615e`
- model SHA-256 `b137761de29dd17b5ac058bc51a4cd5d113f3531a1d60071a91a3ae058ac55e6`
- parent v12 archive SHA-256 `a189eb9eaf97ad74c323ef446759c4b42e392f09df8d65327f938b582d01dac1`
- corrected probe run `31882322590`, artifact `9246360741`
- packaging/runtime run `31882572941`

## Interpretation rule

The current best *observed external* result remains v12 `0.3798116204` until v14 receives its own platform score. v14 has stronger leakage-safe local/cross-fit evidence than its v12 parent and passes the final runtime gate, but this is not a claim of Public LB >0.5 or even >v12 before ODS measures it.
