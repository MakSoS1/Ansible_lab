# AIIJC Lighting — Scoreboard

Only measured results belong here.

## Competition leaderboard submissions

| Timestamp | Submission | Local OOF | Leaderboard Accuracy | Points | Notes |
|---|---|---:|---:|---:|---|
| 2026-09-10 01:40 | V1 `submission_best.csv` | 0.518 | 0.51 | 0.183 | User-confirmed competition result. Current external baseline. |

## Internal CV leaderboard

| Rank | Experiment | OOF Accuracy | Decision |
|---:|---|---:|---|
| 1 | V1 physics/DINOv2/CNN ensemble | 0.518 | baseline |
| 2 | SigLIP2 learned/prompt blend | 0.5113 | reject as standalone |
| 3 | DINOv3 ConvNeXt-Tiny blend | 0.4867 | reject |

Pending experiments are intentionally not ranked until their workflow finishes successfully.

## Target thresholds

- Points > 0.30 requires Accuracy > 0.58.
- "Good" in task statement: Accuracy ≥ 0.70.
- "Excellent" in task statement: Accuracy ≥ 0.90.
- Project stretch target requested by the user: Accuracy ≥ 0.95.

The 0.95 target is a goal, not a claimed result. Every improvement must be measured before being promoted to a submission.
