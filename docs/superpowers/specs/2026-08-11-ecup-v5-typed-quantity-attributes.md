# E-CUP v5 Typed Quantity / Attribute Evidence

Date: 2026-08-11

## Entry evidence

Current clean target-free development best is `0.5952697490140912` from equal global percentile ranks of weak, sparse, explicit, raw supervised-contrastive cosine and raw field-aware teacher2. Five-signal leave-one-out run `31494988017` showed every signal is useful; no removal improved the anchor.

Gap to `0.60` is `0.0047302509859088`.

## Problem

Current normalization recognizes mass/volume/length/count only. Important commerce quantities such as `128GB`, `5000mAh`, `65W`, `2.4GHz`, `15.6 inch` are not canonical quantity evidence. Storage tokens such as `128GB` are also explicitly excluded from model-code evidence, so a core electronics discriminator can disappear from both numeric and model features.

Explicit attribute features compare normalized leaf values by string equality. Equivalent values expressed with different units (`128 GB` vs `0.125 TB`) become false conflicts.

## Scope: conservative typed parser

Add unambiguous typed dimensions with canonical units:

- storage -> bytes (MB/GB/TB and Russian equivalents, decimal multipliers for unit equivalence inside the parser);
- battery capacity -> mAh (`mah`, `мач` where unambiguous);
- power -> W (`w`, `kw`, `вт`, `квт`);
- voltage -> V (`v`, `в` only when immediately attached/following a number);
- frequency -> Hz (`hz`, `khz`, `mhz`, `ghz`, Russian `гц`, `кгц`, `мгц`, `ггц`);
- display diagonal -> inch (`in`, `inch`, `inches`, `дюйм`, `дюйма`, `дюймов`, quote suffix where safely parseable).

Keep existing mass/volume/length/count behavior.

## Canonical attribute values

Introduce a deterministic `canonical_attribute_value()` representation. It must:

1. start from `clean_text`;
2. replace recognized typed quantities with canonical dimension/value markers;
3. preserve surrounding text, so `black 128GB` and `black 256GB` remain different;
4. make unit-equivalent whole values equal, e.g. `128 GB == 0.125 TB`;
5. avoid target labels or learned dictionaries.

Explicit leaf-value matching uses this canonical representation in the new code path. Key selection remains outer-train-only.

## First experiment

Retrain the same per-category explicit specialist architecture on all five outer folds using the expanded quantity normalization + canonical attribute values. No HGB hyperparameter changes are allowed in this experiment.

Compare:

- old explicit direct OOF `0.5683065131240066`;
- new typed-explicit direct OOF;
- fixed six-signal fusion = current five clean signals + new typed-explicit direct score.

The six-signal fusion is an equal global percentile-rank vote, target-free. Old explicit remains in the current five; adding typed-explicit tests whether the new representation supplies incremental evidence rather than merely replacing an already useful branch.

## Gates

Typed-explicit is retained as a new evidence source if direct AP improves old explicit or if the six-signal target-free fusion improves `0.5952697490140912` with minimum fold delta >= `-0.001`.

Milestone is reached only if strict-official aggregate dev OOF >= `0.60` without inspecting sealed-gold metrics/scores.
