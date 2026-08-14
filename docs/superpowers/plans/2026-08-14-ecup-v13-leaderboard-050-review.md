# v13 review — closing the local↔leaderboard gap toward 0.50

Review and amendment of the owner's v13 plan. Method and metrics only; private
transport paths, archive SHAs and token material stay in `gpu-dispatch`.

## 1. Where things actually stand

Two external anchors exist. Everything else is local.

| | local fold-0 | Public LB | gap |
|---|---:|---:|---:|
| v7 one-epoch | `0.7023802626` | `0.3655833314` | `0.3367969312` |
| v12 weak-0.35 | `0.7059297810` | `0.3798116204` | `0.3261181606` |

**Arithmetic correction.** The plan states the LB moved `+0.0198`. It moved
`+0.0142282890`. Local moved `+0.0035495184`, so the observed ratio is `4.0085`,
not `5.6`. The qualitative conclusion survives — local deltas are not calibrated
— but the number that will be quoted downstream should be right.

One genuinely encouraging fact nobody has written down: **the gap shrank by
`0.0106787706`** between v7 and v12. Local rose a little and the LB rose four
times as much. That is the signature of a change that helped the *test*
distribution more than the dev distribution — exactly what more weak supervision
should do if the weak pool is closer to retrieval than the human pool is. It is
one observation, not a trend, but it points the same way as the plan's main bet.

**The target is first place.** `0.50` is `+0.1201883796` from here, and the
current top is `0.48`, i.e. `+0.1001883796`. This is not "improve the model", it
is "beat everyone". That changes risk appetite: a top-15 finish advances to the
next stage, and top-15 is a very different engineering problem from first place.
The plan should state which one it is optimizing.

## 2. Timeout post-mortem

The plan's account is correct as far as it goes. Three additions.

### 2.1 The benchmark under-measured the thing that was over budget

Fixtures were `pairs.head(115000)` and `pairs.head(275000)` from training
`matches.parquet` plus `items_human.parquet`. The plan already flags that this is
not the closed test set. The sharper point is *why it biases in the dangerous
direction*: CPU cost in the structured/serialization branch scales with **unique
items**, not with pairs. Training pairs are grouped by connected component, so
the first N training rows touch far fewer distinct items than N real retrieval
pairs would. The fixture therefore made the branch look cheaper precisely where
it was expensive. Scanning `items_human.parquet` instead of the full item
universe compounds this, because `select_items_by_ids` cost scales with the file
being scanned.

### 2.2 The single-core inversion

"H100 is faster" is irrelevant to a CPU-bound Python branch. A desktop box
driving an RTX 2060 SUPER typically runs a much higher single-core clock than a
20-core server part. For pure-Python work the platform can be **slower per core**
than the local benchmark box even though it is far more capable overall.

v11 measured `161.9 s` at 115k and `379.2 s` at 275k, inside `360 s` / `780 s`.
Apply a plausible `1.7–2.0×` single-core penalty to the CPU-dominated part and
both land at or past the limit. No exotic explanation is needed.

### 2.3 The hypothesis nobody tested: it may have died at Check

v10/v11 had no exact 1000-row / 60 s Check gate — the plan says so itself. Check
is dominated by **fixed** cost: imports, model load, item-file scan. Those do not
shrink with pair count. A v11 archive carrying structured joblib, TF-IDF
vectorizers with large char vocabularies, per-category specialists, contrastive
and graph assets has to deserialize all of it before scoring a single pair.

v12 passes Check in `20.246 s` of `60 s` with one CrossEncoder. v11 never
measured Check at all.

**So "Container did not finish in time" for v10/v11 may never have been about
115k or 275k.** It may have been the first stage, on a thousand pairs. That
matters because it changes the lesson: the fatal property was not "structured
features are slow per pair", it was "the archive has too much to load". The plan
should verify this before concluding what is safe, since it is cheap to test and
the two diagnoses imply different design rules.

## 3. Why the validation lies — and the measurement nobody has made

The plan's four problems are well identified. Two corrections and one gap.

**Problem 3 is understated.** `v8_testlike.py` thresholds LLM soft labels at
`<0.05` / `>0.95` and calls the result ground truth. That does not merely add
noise — it *selects the easy tail*. Pairs the LLM was confident about are, by
construction, the ones a model finds easy. The proxy is hardest exactly where it
is emptiest. It cannot be repaired by reweighting; it needs real labels.

**The gap: per-category prevalence has still never been measured.** The constant
`0.566880890615799` appears hard-coded in at least five workflows as
`RETRIEVAL_PREVALENCE_RATIO`. It is an assumption, not an observation, and every
target-stress number produced since inherits it. Meanwhile the plan correctly
argues that a *single global* ratio is wrong for a macro-averaged metric.

The measurement is one cheap CPU pass and has never been run:

* positive rate per official category in `matches.parquet` (human);
* positive rate per official category in `matches_llm.parquet` at several
  confidence thresholds;
* unique-item degree distribution per category in both.

If human prevalence is materially higher than the weak pool's, that alone
accounts for much of a `0.33` AP gap without any model defect, and it tells you
what prevalence to resample dev to. Until it is measured, the stress apparatus is
calibrated to a guess.

### 3.1 The acceptance test in the plan is a coin flip

The plan gates Validation v3 on: *it must rank v12 above v7*.

Two anchors, `0.0142` apart. A proxy chosen at random passes that test half the
time. Worse, iterating on the proxy until it passes fits it to a single bit.

**This is the most important thing to fix in the plan**, and the fix is cheap.
There are more than a week and five submissions per day — over thirty
submissions, of which the project is currently using almost none. Spend some on
deliberately *spread* anchors rather than near-identical ones:

| anchor | why it is informative | exists already? |
|---|---|---|
| v8 hard-negative | local `0.6555649`, i.e. `-0.0468` — thirteen times the v7→v12 delta | yes, trained |
| v12 weak-0.6 | extends the weak-supervision axis past 0.35 | one training run |
| v6 gate95 | entirely different family: classical ensemble, honest 5-fold `0.6006` | needs rebuild only |
| two-seed v12 average | tests whether ensembling transfers at all | no training |

v8 is the single most valuable submission available. It is *known worse locally*
by a large margin. If the LB drops proportionally, the local metric has real
directional signal and Validation v3 has something to reproduce. If the LB rises,
the local metric is actively inverted and every selection made on it is suspect.
Either result is worth far more than another point near 0.38.

With six to eight spread anchors a proxy can be checked by rank correlation
instead of one comparison.

## 4. Assessment of the v13 plan

**Right, and worth keeping:**

* one CrossEncoder at inference — proven at `20.246 s` Check;
* complexity moves offline into a teacher — the correct structural response to a
  runtime-constrained scored metric;
* v8 rejected on 20/20 category regression — well evidenced, correctly closed;
* three-tier runtime gate including Check — fixes a real hole;
* multiple workload profiles, full item universe — fixes the fixture bias;
* HF-only distribution;
* controlled A→B→C→D→E ablations with one cause per step.

**Missing or wrong:**

1. Proxy acceptance is a coin flip (§3.1). Needs spread anchors and rank
   correlation.
2. Per-category prevalence never measured (§3); a guessed constant is load-bearing.
3. The Check-stage timeout hypothesis is untested (§2.3), so the rule "CPU-heavy
   branch is banned" may be the wrong generalization of the evidence.
4. `0.50` is first place; the plan does not say whether it is chasing #1 or a
   top-15 berth.
5. Sealed gold — `80,444` rows — is still unopened and unplanned. It cannot fix
   the LB gap because it is the same human distribution, but it is the only
   untouched honest holdout and the natural tiebreaker for the two final
   solutions. It should have an explicit one-shot plan rather than drifting.
6. Ordering: validation work has no feedback loop of its own, and the GPU is the
   bottleneck. The calibration campaign should run *in parallel* from day one,
   not after Tasks 1–4.
7. **Graph post-processing was closed for the wrong reason.** It was dropped with
   "structured/TF-IDF/graph runtime finally closed" — but graph post-processing
   over *pair scores* is not the structured branch. It reads no item text, builds
   no TF-IDF, loads no model: it is numpy over 275k rows and finishes in
   milliseconds. Conflating the two throws away the cheapest remaining lever
   (§5.2). The v9 reading of `+0.0015` was taken on the human-distribution proxy
   that is already known not to predict the LB, so it is not evidence of a small
   real effect.

## 5. Five candidates, ranked

`+0.1201883796` is needed. Tuning will not produce it.

### 5.1 Retrieval-realistic training data (agree — keep first)

The plan's #1, and correct. The one confirmed external improvement so far came
from raising weak supervision, and the gap shrank at the same time. The next step
is not more `weak_epochs` but a better weak set: pairs built by blocking so they
look like retrieval candidates, with per-category prevalence matched to what §3
measures.

### 5.2 Graph post-processing over pair scores (promote — plan has this closed)

Free at inference and uses information the per-pair model structurally cannot
see:

* **mutual best match** — a pair that is the top candidate for both endpoints is
  far more likely genuine than one ranked fifth for both;
* **endpoint degree** — an item with many near-equal candidates sits in an
  ambiguous cluster and should be shrunk;
* **transitivity** — high `s(a,b)` and `s(b,c)` with low `s(a,c)` is an
  inconsistency worth smoothing.

All target-free, all computed within category, all monotone-safe to fuse with the
existing percentile-rank machinery. Cost is milliseconds against a `60 s` Check
that currently uses `20.2 s`. This should be tried *before* any retraining
because it needs none.

### 5.3 Retrieval-group ranking loss (agree)

Train on anchor → positive + confusable negatives with a listwise objective.
Closer to AP than independent pair classification. The plan already has ranking
weight `0.25`; the missing part is that the groups must be real candidate lists.

### 5.4 Two-checkpoint ensemble (add — absent from the plan)

Two seeds of the same architecture, averaged. Typically `+0.005–0.015`, requires
no new ideas, and the runtime headroom exists — Check is at a third of its limit.
Worth one measurement to see whether ensembling transfers to the LB at all, which
is itself diagnostic.

### 5.5 Category-tail repair (agree, but sequence it last)

Macro AP weights a small category like a large one, so tail repair is high
leverage in principle. But without a proxy that predicts LB direction, tail work
cannot be validated. Do it after §3 is fixed.

## 6. Recommended sequence

**Track A — calibration, starts immediately, costs no GPU.**
Submit v8 hard-negative first. Then a second spread anchor. Build the LB-vs-proxy
table as results arrive.

**Track B — measurement, one CPU pass, starts immediately.**
Per-category prevalence and degree in human vs weak pools. Replace the hard-coded
`0.566880890615799` with measured per-category values, or delete the stress
apparatus that depends on it.

**Track C — free inference wins, no retraining.**
Graph post-processing on saved v12 predictions. Validate against Track A anchors,
not against fold-0.

**Track D — retraining, only after B reports.**
Retrieval-realistic weak set at measured prevalence, listwise ranking loss.

**Track E — Check-gate forensics, one run.**
Re-run the archived v11 archive against a 1000-row / 60 s Check on the full item
universe. Confirms or kills §2.3 and settles what the runtime rule should be.

## 7. What would falsify this

* If v8 hard-negative scores *above* `0.3798` on the LB, local fold-0 is
  anti-correlated with the leaderboard and every selection to date is suspect,
  including the choice of v12 over v7.
* If measured per-category prevalence in the human pool is close to the weak
  pool, prevalence is not the explanation and the gap is model generalization,
  which points at §5.1 and §5.3 rather than at resampling.
* If the archived v11 passes a real Check gate, then §2.3 is wrong, the timeouts
  really were about per-pair cost at 115k/275k, and the "no CPU branch" rule
  stands as written.

## 8. Binding constraints carried forward

* Sealed gold stays unopened until a single declared one-shot evaluation.
* Private HF only; no GitHub Releases, not even as temporary transport.
* One CrossEncoder in the archive; any new inference branch must pass the
  three-tier gate including Check on the full item universe.
* Local fold-0 is a development signal. It is not a leaderboard estimate and must
  never be written into a field that names the archive's validated quality.
