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
| v12 | v7 runtime + stronger weak supervision | fold0 `0.7059297810` | **`0.3798116204`** | current observed Public-LB best |
| v13 | B/groupweak: preserve weak retrieval-anchor groups/topology | fold0 **`0.7086611386`**; frozen p05 **`0.5690974845`**, mean **`0.6869505675`** | **pending** | next Public-LB candidate; package verified |

## High-value research results

### Validation and distribution

- Immutable split: `285,210` development + `80,444` sealed gold, 5 component-disjoint folds, SHA `aae58f...eb55b`, gold unopened.
- v7 `0.70238 local -> 0.36558 LB`; v12 `0.70593 local -> 0.37981 LB`. Local score magnitude is not calibrated to LB.
- Canonical distribution run `31788445849`: `365,654` human rows, `11,187,780` weak rows; human prevalence ~`0.25677`, weak target mean ~`0.24356`. Retrieval degree/hardness, not prevalence alone, is the primary shift.
- Validation v3 uses human truth and deterministic category-local retrieval-hard stress; it is for candidate ordering, not LB prediction.

### Runtime

- Historical v11 exact full-item Check forensic run `31789001358`: `60.033 s` timeout before valid output.
- v13 binding supplied-item Check: `26.1353473 s / 60 s`, valid output, return code 0.
- v13 stricter full-item diagnostic: `60.0049954 s` timeout. Diagnostic-only under the current supplied-item subset contract.
- Graph transform runtime is cheap (~`1.29 s / 275k`) but graph was rejected for quality instability.

### v13 causal ladder

- **B / groupweak KEEP as next external candidate:** preserve complete retrieval-anchor groups/orientation; fold0 `0.7086611385531062`.
- **C2 / equal-exposure ListNet REJECT:** more complex group ranking did not beat B on the frozen diagnostics.
- Ambiguous all-soft/ListNet work remains research evidence, not a reason to alter the already packaged B candidate without a clean selection win.

## Exact v13 artifact identity

- `ecup-v13-groupweak-v7runtime-submission.zip`
- bytes `663760087`
- SHA-256 `f4b7aad36c8d293a3939d9fb2ce7f91cff1bd8381c870015b2f16ea65a17badb`
- production run `31828844182`
- packaging run `31829720888`
- HF roundtrip run `31843423348`
- packaging source commit `4e83294eb5f6c31c720f7cbb0220f0f4d0ee3cb1`

## Interpretation rule

The current best *observed external* result is v12. v13 B is only the next candidate until ODS returns a score. Never relabel fold0/p05/mean as Public LB, and never label a successful package/runtime test as a quality win.
