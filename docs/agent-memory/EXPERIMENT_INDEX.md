# E-CUP Matching — Experiment Index

Updated: **2026-08-16**. Local metrics, external leaderboard evidence and package/runtime evidence are kept separate.

| Version | Core idea | Comparable local evidence | Public / platform evidence | Decision |
|---|---|---:|---:|---|
| v7 | one RuBERT CrossEncoder | fold0 ~`0.70238` | `0.3655833314` | reliable external anchor |
| v12 | v7 runtime + stronger weak supervision | fold0 `0.7059297810` | **`0.3798116204`** | current observed Public-LB best |
| v13B | preserve weak retrieval groups/orientation | fold0 `0.7086611386` | `0.3783781653` | reject externally vs v12; proves local ranking mismatch |
| v14 A17 | retrieval-distilled token-cross | `0.5913742384` | — | reject |
| v14 A20 | Granite full-pair + typed features | `0.6077998852` | — | reject Granite parent |
| v14 A21 | human-only RuBERT + zero-safe typed residual | teacher `0.6974019210` → `0.6991600103` | — | retain residual evidence |
| v14 final | v12 + category-gated cross-fit residual | `0.7065769714`; cross-fit mean Δ `+0.0005859187` | pending | **submission-ready**, safer new calibration |
| v15 A0 | Granite title/category pair control | `0.5989106811` | — | reject production parent |
| v15 A4 | A0 + attrs + typed + category + macro balance | `0.6164130847`, ΔA0 `+0.0175024036` | — | fields/category/macro signal validated |
| v15 R1 | strong RuBERT teacher + fast typed/category residual | teacher `0.6974019210` → **`0.7014872395`**, Δ `+0.0040853186` | pending | **experimental submission-ready** |

## Validation contract

- `365,654` human rows.
- `285,210` development rows + `80,444` sealed gold.
- 5 component/item-disjoint development folds; cross-split item overlap `0`.
- Split SHA `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.
- Rowmap SHA `00778edd7ed4581f8aedc143052d17d6fb86c55abfaee9fc6a169f72bb47b32f`.
- Gold unopened, `0` rows scored.

## Key causal conclusions

1. v13B is the strongest warning against treating fold0 as an LB estimate: it beat v12 locally but lost externally.
2. Token-cross/retrieval distillation (A17) is not the route.
3. Granite is too weak as the main pair parent (A20/A0), but A0→A4 `+0.0175` shows that field-aware serialization, typed pair features, category specialization and macro balancing are useful ingredients.
4. A21 and R1 show the useful place for those ingredients: a zero-safe residual on top of a strong RuBERT pair model. R1 improves its human-only teacher by `+0.0040853`, about 2.32x A21's uplift.
5. The current production-transfer risk is explicit: R1 was trained/selected against the human-only outer-fold teacher, while the packaged v15 parent is the full-development v12 production teacher. External calibration is required before a five-fold OOF/refit investment.

## Exact submission artifacts

### v14

- `ecup-v14-v12-category-gated-residual-submission.zip`
- bytes `663770301`
- SHA-256 `fcaace1a7f0e663b7c9b0b29ca78a768241c3b417b8f4d4a342f52874a29615e`
- packaging run `31882572941`
- 1,000-row organizer-shaped Check `28.81002984 s / 60 s`, PASS

### v15

- `ecup_submission_v15.zip`
- bytes `665146742`
- SHA-256 `8623981f54e0cead65695ba2c44eaccd75230a626d65c4508ba64142e527b26b`
- R1 screen run `31939170747`
- packaging run `31940154217`
- 1,000-row organizer-image Check `24 s / 60 s`, PASS

## External calibration order

Submit v14 first, record Public LB, then submit v15 R1. Until those scores arrive, **v12 remains the best observed external result** and neither new candidate may be described as a `0.5` solution.
