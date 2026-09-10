# Reproducibility

## Local smoke

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m aios_track2.cli smoke --seed 42 --out runs/local/smoke
```

The same seed must yield the same `wells_schedule.inc` hash and NPV.

## Five-controller bake-off

```bash
python -m aios_track2.cli bakeoff --seed 42 --preset local --out runs/local/bakeoff
```

Controllers:

1. `heuristic` — waterflood engineering rules
2. `linear_cem` — ridge surrogate + CEM
3. `tcn_cem` — causal TCN + CEM ranking
4. `graph_cma` / `graph_ensemble` — graph-temporal ensemble + CMA-ES + uncertainty
5. `mappo` — CTDE challenger; not declared a winner unless OPM-validated NPV wins

Winner selection uses only validated NPV. Surrogate scores never become the declared figure.

## Apple Silicon GitHub Actions

Manual workflow `.github/workflows/aios-train-surrogate.yml` runs on `macos-14` (M1). It installs the package, runs pytest, executes the bake-off, and uploads `runs/<git_sha>-<github_run_id>/` to the private Hugging Face dataset using `secrets.HF_TOKEN`. Training is never started from a pull request.

## OPM Flow

If `flow` is on `PATH`, `run_flow` uses it. Otherwise the reduced-order proxy fills the same monthly schema so Apple Silicon runners can train without Docker (GitHub-hosted macOS images have no Docker and no OPM packages). Declared contest NPV must still be recomputed on OPM before submission.
