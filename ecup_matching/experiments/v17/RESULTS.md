# v17 — results

## Audits

| run | what it measured | outcome |
|---|---|---|
| `32166532516` | pair shape, degree, prevalence, item overlap | `shared_items = 0` between human and weak pools |
| `32167195585` | three-population profile over 13,397,761 items | populations differ by `<4%` in text/attribute density |

Recorded in `PLAN.md`. The operative consequences: LLM label quality cannot be
audited against human truth on shared rows because no shared rows can exist,
and the local/leaderboard gap is not explained by population shift.

## Probe ladder

Driver `run_v17_weakscale_probe`, ungrouped weak preparation, frozen split
`aae58fb4...`, seed `2026`, `max_length=256`, human epochs `1.00`, weak holdout
`5%` by connected component.

| run | label | weak rows x epochs | weak examples seen | human fold-0 | weak holdout after weak | weak holdout after human | wall |
|---|---|---|---|---|---|---|---|
| `32169719512` | x1-control | `600,000 x 0.35` | `199,500` | `0.7017637364` | `0.6973930799` | `0.6565798751` | `70.0 min` |
| `32176864104` | x7-scaled | `1,500,000 x 1.00` | pending | pending | pending | pending | running |

Weak holdout: `30,000` rows over `53,668` items, `20` categories, train/held
item overlap `0`.

## Two findings from the control alone

### 1. Weak exposure is steeply leveraged, at least locally

The control is the v12 pipeline minus the `5%` of weak rows reserved for the
holdout: `199,500` examples instead of `210,000`. Everything else — seed,
split, human phase, learning rates, batch shape — is identical.

- control human fold-0 `0.7017637364`
- v12 reference `0.7059297810`
- **delta `-0.0041660447`**

A `5%` cut in weak exposure cost `0.0042` of fold-0. For scale, the entire
v7 -> v12 local improvement was `+0.0035`, and the v14 residual that shipped
was `+0.0006`. This is one observation and part of it may be run-to-run
variance, but it is a single-variable delta and it points the same way H1 does.

It also means the control is **not** a reproduction of v12 and must not be
quoted as one. The `-0.0042` is the price of the measurement, not a regression.

### 2. The human phase costs 0.041 on the weak population

On the item-disjoint weak holdout, the same model scores:

- after the weak phase: `0.6973930799`
- after the human phase: `0.6565798751`
- **delta `-0.0408132048`**

The final training phase, fitted to the `5.31%` of the universe that human
labelling covers, moves the model `0.041` Macro AP worse on unseen items from
the `92.44%` population — roughly ten times the effect of the exposure cut
above.

**This is ambiguous and must not be over-read.** The holdout is scored against
weak targets, so the metric measures agreement with the LLM annotator. The
human phase is *supposed* to move the model away from LLM agreement wherever
the LLM is wrong, and the competition scores against human-quality labels, not
LLM ones. So a drop here is consistent with two opposite stories:

- the human phase is correcting LLM bias, and the drop is the correction; or
- the human phase is overfitting a `5.31%` slice and losing the rest.

Nothing measured so far separates them. What makes it worth a controlled test
anyway is the magnitude: whichever story is true, the last phase of training is
the largest single influence on behaviour in the population that covers most of
the universe, and it has never been tuned — `--human-epochs 1.00` is as
unexamined as `--weak-epochs 0.35` was.

## Follow-up queued, not yet run

Vary the human phase (`--human-epochs`, or mixing weak rows into it) and watch
whether human fold-0 and the weak holdout trade off. If fold-0 holds while the
weak holdout recovers materially, that is a candidate worth an external slot.
This is queued behind the exposure ladder because the GPU is serial.

## Infrastructure notes

- Run `32168884723` died in `5m50s` on the D043 failure: the driver called the
  recomputing manifest builder and got `d1b31023...` instead of `aae58fb4...`.
  Fixed by importing the frozen loader directly; pinned by
  `test_v17_probe_uses_frozen_split.py`.
- gpu-dispatch artifact storage hit quota at `4.34 GiB` / 172 artifacts.
  Superseded v7-era artifacts and the already-submitted v14 parts were deleted,
  bringing it to `1.25 GiB`. `v12-final` and the never-submitted `v15-r1` parts
  were kept. GitHub recalculates usage every 6-12 h, so uploads may lag.
