# Control Room UI — 2026-09-05

## Job

Reservoir engineer / jury must answer four questions in one screen:

1. What schedule do we submit?
2. Is it better than baseline on real OPM+CHDD NPV?
3. Are hard constraints OK (WLPR ≤ 500, SHA match)?
4. Where is `wells_schedule.inc`?

## Cut from the four mockups

- Duplicate KPI ribbons and five strategy cards + table + sidebar saying the same numbers
- Fake well A-17, John Doe, “Live / last 30 days”, simulator queue
- Pressure heatmap toggles, connectivity spaghetti, scenario planner templates
- Surrogate sold as the score

## Keep

- Winner NPV vs baseline
- Constraint badge
- OPM verification + SHA
- Plain-language why
- 2–3 candidate comparison (Baseline / CMA-ES / MAPPO)
- Simple well-group map
- Honest holdout tail, including failed top-3 recall

## Stack

Existing FastAPI + static HTML. Data from `submission/` artifacts, never demo names.
