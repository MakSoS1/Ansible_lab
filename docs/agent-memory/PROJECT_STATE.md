# E-CUP Matching — Project State

Updated: **2026-08-15**

## Executive state

The project has moved far beyond the v5 snapshot that previously lived in Memora. The current observed Public-LB best is **v12 = `0.3798116204`**. A new **v13 B / groupweak** candidate has been built, organizer-checked and byte-for-byte round-trip verified in private Hugging Face, but has **not yet received a platform score**. It is therefore the next external calibration candidate, not a claimed final keeper.

## External leaderboard anchors

| Candidate | Comparable local diagnostic | Public LB | Meaning |
|---|---:|---:|---|
| v7 one-epoch | `0.7023802626` (`0.7023556010` original probe) | `0.3655833314` | first reliable single-CrossEncoder anchor |
| v12 weak-0.35 | `0.7059297810` | **`0.3798116204`** | current observed Public-LB best |
| v13 B groupweak | `0.7086611386` | pending | next candidate; retrieval topology preserved |

Correct v7→v12 arithmetic: `ΔLB=+0.0142282890`; comparable local delta `+0.0035495184`; local→LB gap improved by `0.0106787706`. Local values are not calibrated estimates of leaderboard AP.

## Immutable validation / safety state

- Human rows: `365,654`.
- Development rows: `285,210`.
- Sealed gold: `80,444` rows.
- Five component-disjoint development folds; cross-split item overlap `0`.
- Split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.
- Sealed gold remains unopened; `0` rows scored.

Validation v3 is a retrieval-hard stress proxy using human binary truth, category-local prevalence/hardness/degree strata and deterministic stress replicas. It is a directional selector, not a leaderboard-score calculator.

## Distribution research

Canonical measurement run `31788445849` measured all `365,654` human rows and `11,187,780` weak rows.

- human binary prevalence: `0.2567727961406138`;
- weak soft-target mean: `0.2435606726464766`;
- weak candidate lists have materially greater degree/hardness than human pairs.

Conclusion: the large local→LB gap is not explained by a simple class-balance shift. Candidate-list topology and confusable retrieval negatives are central.

## Runtime research

Historical v10/v11 local benchmarks were misleading because they used `pairs.head(N)`-style training fixtures and did not test the exact 1,000-row Check stage. A forensic rerun of historical v11 (`31789001358`) used the organizer image and full canonical 4.1-GB item universe; it timed out at `60.033 s` before valid output. Fixed startup/item scanning can therefore kill a submission even when large-N throughput appears acceptable.

The retained runtime rule is one CrossEncoder / one tokenizer / one checkpoint. Structured/TF-IDF/large multi-model inference branches stay out of the submission.

Graph post-processing was reopened independently and benchmarked at ~`1.29 s / 275k`, but was rejected on quality stability: no predeclared variant improved v7, v8 and v12 simultaneously.

## Why v13 B exists

Source audit found two problems in v7/v12 weak training:

1. weak rows were sampled individually, splitting original retrieval candidate lists;
2. canonical pair ordering discarded original retrieval-anchor orientation.

In addition, v12 excluded ambiguous weak targets in `0.30–0.70` entirely. v13 decomposed fixes into causal ablations rather than changing everything at once.

**B / groupweak** changes only retrieval topology/orientation: preserve `_retrieval_anchor` and complete candidate groups while keeping v12 weak exposure/model/runtime. Probe run `31791177120` achieved fold0 **`0.7086611385531062`**, +`0.0027313575222363` over v12.

Frozen Validation-v3 for the packaged B candidate: p05 **`0.5690974845`**, mean **`0.6869505675`**. Controlled C2/ListNet equal-exposure testing was rejected because it regressed the relevant diagnostics. More complicated objectives therefore did not automatically replace B.

## v13 submission evidence

Candidate: `v13b-groupweak`.

- production refit run: `31828844182`;
- packaging run: `31829720888`;
- private-HF upload/roundtrip run: `31843423348`;
- exact public source used by packager: `4e83294eb5f6c31c720f7cbb0220f0f4d0ee3cb1`;
- filename: `ecup-v13-groupweak-v7runtime-submission.zip`;
- size: `663,760,087` bytes;
- SHA-256: `f4b7aad36c8d293a3939d9fb2ce7f91cff1bd8381c870015b2f16ea65a17badb`;
- model SHA-256: `9ae7676f96818a367eb348f8648d503b56c86e3d0c62f665f030b4c29bcde0a5`;
- private HF path: `submissions/v13/candidates/b-groupweak/ecup-v13-groupweak-v7runtime-submission.zip`;
- runtime: single `ai-forever/ruBert-base` CrossEncoder, max length 256, inference batch 64;
- sealed gold unopened.

### Binding Check

Organizer-shaped supplied-item subset, 1,000 pairs / 1,999 materialized items:

- ZIP extraction `2.9943 s`;
- wall `26.1353473 s / 60 s`;
- return code `0`;
- output valid;
- `881` unique scores;
- acceptance: **PASS**.

### Stricter diagnostic

The same 1,000 pairs while scanning the entire canonical `items.parquet` (`4,104,103,411` bytes) reached `60.0049954 s` and timed out. The packaging workflow marks this explicitly diagnostic-only and still accepts the candidate because the closed-test contract supplies the relevant item subset. Do not erase this result; it is residual-risk evidence if the platform contract changes.

### Transport integrity

HF upload run `31843423348` uploaded the exact candidate, downloaded it back to `canonical.zip`, checked the exact byte count and SHA-256, and printed both `canonical.zip: OK` and `V13_CANDIDATE_HF_ROUNDTRIP_VERIFIED`. One-time credential material was deleted afterward.

## Source branch provenance

The current `ecup-matching-2026` tip was later repurposed for the one-time HF bridge and diverges from the 176-commit research/source line that contains `4e83294…`. Do not silently treat current branch-tip source as the source of the already built candidate. Artifact/runtime reproduction is bound to the exact packaging source commit.

## Immediate next action

Upload **exactly** the SHA-verified v13 B ZIP to the competition platform and record its Public LB. Until that score exists, v12 (`0.3798116204`) remains the best observed external anchor and v13 B remains a candidate.
