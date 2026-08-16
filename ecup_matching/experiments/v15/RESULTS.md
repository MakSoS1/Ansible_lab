# E-CUP v15 — retained results

Updated: 2026-08-16

## External anchors

- v7 Public LB: `0.3655833314`.
- v12 Public LB: `0.379811620418641` — current observed external champion.
- v12 diagnostic fold0 Macro AP: `0.70592978103087`.
- v13B Public LB: `0.37837816527590995` — rejected externally vs v12 despite higher local fold0.
- Target Public LB: `0.5`.

## Completed v14 evidence reviewed for the new plan

These remain v14 runs and are not relabelled as v15.

| Run | Fold0 Macro AP | Decision | Reason |
|---|---:|---|---|
| A17 `31902830418` | `0.5913742383876135` | REJECT | retrieval-distilled token-cross is far below the retained pair CrossEncoder anchor |
| A20 `31906787615` | `0.607799885174354` | REJECT | full-pair Granite + typed features is much weaker than RuBERT/v12 |
| A21 `31906735883` | `0.6991600103234464` | KEEP EVIDENCE | typed residual improves its human-only RuBERT teacher `0.6974019209735696` by `+0.001758089349876868`, but remains below v12 fold0 |

The separately cross-fit/category-gated v14 production residual is the retained v14 submission candidate: fold0 `0.7065769713851786`, delta vs v12 `+0.0006471903543086022`, cross-fit mean delta `+0.000585918676919317`, 20/20 categories nonnegative and six categories admitted. Final archive SHA-256: `fcaace1a7f0e663b7c9b0b29ca78a768241c3b417b8f4d4a342f52874a29615e`; exact 1,000-row organizer-shaped Check gate passed in `28.81002984 s`.

## v15 Granite causal controls

| Variant | Run | Fold0 Macro AP | Delta vs A0 | Interpretation |
|---|---:|---:|---:|---|
| A0 | `31936366096` | `0.5989106811324902` | — | title/category pair CE control; rejects Granite as the production parent |
| A4 | `31936375282` | `0.61641308474721` | `+0.017502403614719775` | attrs + typed pair features + category specialization + macro balancing are causally useful, but the Granite ceiling remains too low |

A0→A4 is the key causal result: the new field-aware ingredients help materially on the same backbone, while the backbone choice itself is the dominant failure mode.

## v15 R1 — strong RuBERT parent + fast typed/category residual

The primary v15 route was corrected to preserve the proven RuBERT pair family and add only pair-specific evidence that the CrossEncoder under-uses. R1 uses 19 cheap deterministic features (brand/model/numeric/title/attribute conflicts plus teacher-confidence interactions), category-specific experts, macro-balanced BCE + category ranking loss, and an exactly zero-safe residual floor.

- Successful R1 screen run: `31939170747`.
- Human-only RuBERT teacher fold0: `0.6974019209735696`.
- R1 fold0 Macro AP: **`0.7014872395264526`**.
- Delta vs same teacher: **`+0.004085318552883077`**.
- A21 residual uplift was `+0.001758089349876868`; R1 produces about 2.32x the uplift on the same teacher family.
- R1 is still `-0.00444254150441743` below the v12 fold0 diagnostic, so this is not a local quality champion.
- Labels: human only; LLM rows `0`; sealed gold unopened; cross-split item overlap `0`.

## v15 submission artifact

R1 is packaged as an experimental leaderboard candidate on top of the retained full-development v12 production teacher. This transfer is deliberately recorded as experimental because the residual was selected on the human-only outer-fold teacher, while the packaged parent is the stronger v12 production refit.

- Packaging run: `31940154217` — SUCCESS.
- File: `ecup_submission_v15.zip`.
- Bytes: `665146742`.
- SHA-256: `8623981f54e0cead65695ba2c44eaccd75230a626d65c4508ba64142e527b26b`.
- ZIP integrity: PASS.
- Organizer-image Check fixture: 1,000 development rows, exit code `0`, valid non-degenerate output, `24 s / 60 s` — PASS.
- Runtime shape: one production RuBERT pair checkpoint + one tiny typed/category residual head; no graph, HGB, TF-IDF or teacher ensemble at inference.
- Sealed gold remains unopened and unscored.

## Decision

For external calibration, submit v14 first (safer, cross-fit accepted directly over v12), then v15 R1 (larger architectural change, stronger residual uplift but less certain production-parent transfer). Neither local result justifies claiming Public LB `0.5`; only the competition leaderboard can calibrate the remaining distribution gap.
