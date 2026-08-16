# E-CUP v15 — retained results

Updated: 2026-08-16

## External anchors

- v7 Public LB: `0.3655833314`.
- v12 Public LB: `0.379811620418641` — current external champion.
- v12 diagnostic fold0 Macro AP: `0.70592978103087`.
- v13B Public LB: `0.37837816527590995`.

## Completed v14 evidence reviewed before v15 promotion

These remain v14 runs and are **not** relabelled as v15.

| Run | Fold0 Macro AP | Decision | Reason |
|---|---:|---|---|
| A17 `31902830418` | `0.5913742383876135` | REJECT | retrieval-distilled token-cross is far below the retained pair CrossEncoder anchor |
| A20 `31906787615` | `0.607799885174354` | REJECT | full-pair Granite + typed features is much weaker than RuBERT/v12 |
| A21 `31906735883` | `0.6991600103234464` | KEEP-AS-EVIDENCE | typed residual improves its human-only RuBERT teacher (`0.6974019209735696`) by about `+0.001758`, but remains below v12 fold0 |

All three retained metrics report sealed gold unopened, zero gold rows scored, zero cross-split item overlap and no LLM labels in the human-only runs.

## v15-A0 — Granite control

- GPU run: `31936366096`.
- Exact public source: `d8bdc9fad3aacd105d2ede1157ba453fccafdd1e`.
- Variant: A0, title/category pair CrossEncoder, no attrs/typed/category head, max length 128.
- Fold0 Macro AP: **`0.5989106811324902`**.
- Optimizer steps: `3209`.
- Elapsed: `655.418648470004 s` on the home RTX 2060 runner.
- Sealed gold: unopened / 0 rows scored.
- Decision: **REJECT as a production direction**. This confirms that Granite-97M is not a valid strong A0 parent for this task.

## Architecture correction after A0/A20

The primary v15 route is now a **frozen strong RuBERT pair teacher + zero-safe fast typed category residual** rather than a Granite replacement backbone. This preserves the externally proven v12 family and spends model capacity only on pair-specific conflicts that the CrossEncoder under-uses.

The R1 residual uses cheap brand/model/numeric/title/attribute set evidence, teacher-confidence interactions, a separate category-conditioned expert projection, macro-balanced BCE + category ranking loss, and an exactly zero-initialized residual projection. The residual is trained on fold0 outer-train only; the held component remains item-disjoint.

- R1 screen workflow: `E-CUP v15 R1 RuBERT fast residual`.
- R1 GPU run: `31936982360` (queued behind the already-running A4 control at the time of this update).
- A4 Granite control remains untouched and is allowed to finish; it is evidence, not the production parent.

No v15 submission is promoted from fold metrics alone. The actual submission packager uses the retained v12 full-development production teacher plus only a fold-safe residual, then executes an offline organizer-image Check gate before export.
