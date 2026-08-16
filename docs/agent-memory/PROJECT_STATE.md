# E-CUP Matching — Project State

Updated: **2026-08-16**

## Executive state

The current observed Public-LB best remains **v12 = `0.379811620418641`**. v13B scored `0.37837816527590995`, so its higher local fold0 result did not transfer externally. This is now a hard project rule: fold0 is a causal screen, not a leaderboard estimator.

Two new exact submission candidates are built and runtime-gated:

1. **v14 — v12 category-gated cross-fit residual**: safer calibration candidate. Local fold0 `0.7065769713851786`, `+0.0006471903543086022` over v12; cross-fit mean delta `+0.000585918676919317`; six categories admitted, 20/20 nonnegative. Exact ZIP SHA `fcaace1a7f0e663b7c9b0b29ca78a768241c3b417b8f4d4a342f52874a29615e`. Organizer-shaped 1,000-row Check: `28.81002984 s / 60 s`, PASS.
2. **v15 R1 — v12 production RuBERT parent + fast typed/category residual**: experimental calibration candidate. R1 screen `0.7014872395264526` vs its human-only RuBERT teacher `0.6974019209735696`, delta `+0.004085318552883077`. Exact ZIP SHA `8623981f54e0cead65695ba2c44eaccd75230a626d65c4508ba64142e527b26b`. Organizer-image 1,000-row Check: `24 s / 60 s`, PASS.

The recommended external order is v14 first, then v15. Neither local result is evidence that Public LB has reached `0.5`; only the ODS leaderboard can resolve the remaining distribution gap.

## Immutable validation / safety state

- Human rows: `365,654`.
- Development rows: `285,210`.
- Sealed gold: `80,444` rows.
- Five component/item-disjoint folds; cross-split overlap `0`.
- Split SHA-256: `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.
- Canonical rowmap SHA-256: `00778edd7ed4581f8aedc143052d17d6fb86c55abfaee9fc6a169f72bb47b32f`.
- Sealed gold remains unopened; `0` rows scored.

## Completed research since v13

| Experiment | Fold0 Macro AP | Interpretation |
|---|---:|---|
| v14 A17 retrieval-distilled token-cross | `0.5913742383876135` | reject token-cross/retrieval-distillation route |
| v14 A20 Granite full-pair typed | `0.607799885174354` | Granite parent too weak |
| v14 A21 human-only RuBERT + typed residual | `0.6991600103234464` | positive residual signal, `+0.001758089349876868` vs teacher |
| v15 A0 Granite control | `0.5989106811324902` | confirms Granite is not a valid strong A0 parent |
| v15 A4 Granite + fields/typed/category/macro | `0.61641308474721` | `+0.017502403614719775` vs A0; new ingredients are useful even though parent is weak |
| v15 R1 RuBERT + fast typed/category residual | `0.7014872395264526` | `+0.004085318552883077` vs same human-only teacher; keep for external calibration |

The architectural conclusion is therefore **not** “replace RuBERT with Granite”. It is: preserve the strong pair CrossEncoder family and spend extra capacity on deterministic pair-level identity/conflict evidence and category-conditioned residual corrections.

## v14 artifact identity

- Candidate: `v14-v12-category-gated-residual`.
- Packaging run: `31882572941`.
- Bytes: `663770301`.
- SHA-256: `fcaace1a7f0e663b7c9b0b29ca78a768241c3b417b8f4d4a342f52874a29615e`.
- Parent: exact v12 weak035 production runtime.
- Single safetensors Transformer checkpoint plus tiny deterministic category residual.
- Check gate: PASS.

## v15 artifact identity

- Candidate: `v15-R1-fast-category-residual`.
- R1 screen run: `31939170747`.
- Packaging run: `31940154217`.
- Bytes: `665146742`.
- SHA-256: `8623981f54e0cead65695ba2c44eaccd75230a626d65c4508ba64142e527b26b`.
- Runtime: exact v12 production RuBERT checkpoint + tiny 19-feature category-conditioned residual head.
- No LLM labels, graph, HGB or TF-IDF at runtime.
- Check gate: PASS.
- Important caveat: the R1 residual was selected/trained on the human-only outer-fold teacher, then transferred to the full-development v12 production parent. Treat this as an experimental external calibration, not as a proven local improvement over the exact production parent.

## Next action

Submit the exact v14 ZIP and record Public LB. Then submit the exact v15 ZIP and record Public LB. If v15 wins externally, the next quality step is five-fold OOF residual training followed by a production refit that removes the current teacher-transfer mismatch. If it does not, retain the A0→A4 causal lesson but stop investing in the current residual transfer and search for a stronger full-pair teacher/weak-data policy.
