# v17 — weak exposure, measured on the population we are scored on

Created: 2026-08-18.

## Why the previous line stopped paying

Four candidates now have both a local fold-0 number and a real Public LB score:

| candidate | local fold-0 | Public LB |
|---|---|---|
| v7 | `0.7023802626` | `0.3655833314` |
| v12 | `0.7059297810` | `0.3798116204` |
| v13b | `0.7086611386` | `0.3783781653` |
| v14 residual | `0.7065769714` | `0.3803270470` |

Local ranks them v13b > v14 > v12 > v7. The leaderboard ranks them
v14 > v12 > v13b > v7. The two orders disagree everywhere except v7, so local
fold-0 has been selecting on close to no information for three iterations.

The total external movement across all four is `+0.0147`, and `+0.0142` of it
came from one change: v7 -> v12, more weak supervision. Everything after that —
groupweak resampling, listwise, all-soft, category-gated residual,
field-aware residual, graph rescoring, late interaction, MoE, setwise — moved
the leaderboard by `0.002` or less, or moved it down.

## What the v17 audits established

Two measurements were run on the runner against the canonical parquet
(`run_v17_retrieval_audit`, run `32166532516`; `run_v17_population_audit`, run
`32167195585`).

**The two label sources describe disjoint products.**

| | pairs | items | prevalence / target mean | degree mean | degree = 1 |
|---|---|---|---|---|---|
| `matches.parquet` | `365,654` | `711,304` | `0.2567729614` | `1.028` | `97.72%` |
| `matches_llm.parquet` | `11,187,780` | `12,384,610` | `0.2435606726` | `1.807` | `74.31%` |

`shared_items = 0`. Not zero shared *pairs* — zero shared *items*. The earlier
conclusion that weak label quality was unverifiable was correct, but the reason
recorded for it was not: there is no common ground to verify against, and
there never was. Any plan that audits the LLM annotator against human truth on
overlapping rows is therefore impossible as specified.

**The universe partitions into three populations, and they look alike.**

`items.parquet` holds `13,397,761` items:

| population | items | share | name chars | attribute keys |
|---|---|---|---|---|
| human-labelled | `711,304` | `5.31%` | `57.68` | `12.608` |
| weak-labelled | `12,384,610` | `92.44%` | `55.17` | `12.062` |
| neither | `301,847` | `2.25%` | `54.13` | `10.806` |

Text and attribute density are within `4%` across populations, so the local /
leaderboard gap is not explained by the human slice being a cleaner or richer
kind of product. The human slice is a near-uniform per-category sample
(each category `4.5–6.5%` of the slice), consistent with a set built for a
macro-averaged metric.

**Per-category prevalence is what macro AP actually rests on.** In the human
table it ranges from `0.0726` (Аптека) to `0.4693` (Бытовая химия). The metric
weights those categories equally, so the low-prevalence ones cap the mean.

**The retrieval-shaped hypothesis is not supported by what is on disk.**
Neither table looks like a top-K candidate list: `1.03` and `1.81` edges per
item, `1.53` candidates per weak anchor. Building v17 around listwise or
setwise scoring over long candidate lists would have been a bet on a structure
that is not present in any data we hold.

## The one knob that was never turned

`v12`, `v13` and every workflow descended from them train with:

```text
--weak-presample-rows 1200000 --weak-final-rows 600000 --weak-epochs 0.35
```

That is `600,000 x 0.35 = 210,000` weak examples seen, out of `11,187,780`
available — **`1.88%` of the pool**, identical in every candidate since v12.
The lever with the only demonstrated external effect has never been moved.

## Hypothesis

**H1.** Leaderboard performance is limited by weak-supervision exposure, not by
architecture. Increasing weak exposure while holding the runtime shape fixed
moves the external score more than any architecture change tried so far.

H1 is falsifiable on the weak axis below, and cheap: the production path needs
no new training code, because `v12_production_entry.py --mode plain-v7` already
accepts the weak knobs and forces the frozen split.

## The measurement that was missing

Human fold-0 covers `5.31%` of the universe. `run_v17_weakscale_probe` adds a
second axis: an item-disjoint slice of the weak corpus, scored with Macro AP
over its categories, before and after the human phase.

The split takes whole connected components rather than rows, so no endpoint can
appear on both sides and no retrieval-anchor group is cut. Held labels are weak
targets, so this measures agreement with the LLM annotator on unseen items in
the population that covers `92.44%` of the universe. It is **not** a
leaderboard estimate and is never to be reported as one.

## Ladder

| step | weak rows | weak epochs | examples seen | vs baseline |
|---|---|---|---|---|
| x1 control | `600,000` | `0.35` | `210,000` | `1.0x` |
| scaled | `1,500,000` | `1.00` | `~1,425,000` | `~6.8x` |

The control exists to prove the new driver reproduces the known pipeline before
a multi-hour run depends on it, and to give the weak axis a baseline. Its
fold-0 anchor is **v12 `0.7059297810`**, not v13b `0.7086611386`: the driver
uses the ungrouped `_prepare_common_weak`, which `v12_production_entry` does
not patch, while v13b's number came from the grouped replacement installed by
`v13_groupweak_entry`.

## Selection rule, fixed before the results

Promote the scaled candidate to production refit only if **both** hold:

1. weak-holdout Macro AP improves over the control by more than `0.005`;
2. human fold-0 does not fall more than `0.005` below the control.

Rule 2 is deliberately permissive on the human axis, because that axis has
demonstrably mis-ranked three of four candidates. Rule 1 carries the decision.

If rule 1 fails, H1 is rejected and the exposure line closes with a recorded
measurement rather than another guess.

## Runtime

Unchanged from the accepted contract: one `ai-forever/ruBert-base` pair
CrossEncoder, `max_length=256`, inference batch `64`, single `.safetensors`
checkpoint, no graph, no second model. Exposure is a training-time change and
costs nothing at inference, so the organizer-shaped Check that v13/v14/v15
already passed applies unchanged and must be re-run on the exact final bytes.

## What this does not claim

Nothing here predicts a Public LB value. The gap between local `0.706` and
external `0.380` is measured but still unexplained: the populations match, the
prevalences are close, and the pair tables are not retrieval-shaped. H1 is a
bet on the one lever with external evidence behind it, not a derivation.
