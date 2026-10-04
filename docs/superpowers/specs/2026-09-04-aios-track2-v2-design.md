# AIOS Track 2 v2 Design Specification

## Objective

Построить воспроизводимую автономную систему управления Model Z, которая обучает uncertainty-aware surrogate на специально сгенерированных OPM сценариях, сравнивает пять стратегий управления в одном evaluation protocol и публикует только OPM-подтверждённый `wells_schedule.inc` с детерминированным NPV.

## Truth hierarchy

1. Полный Model Z в OPM Flow — источник истины по динамике.
2. Исправленная методика ЧДД — источник истины по экономике.
3. Constraint Guard — источник истины по допустимости действия.
4. Surrogate — быстрый ранжировщик/прогноз, но не финальный судья.
5. LLM/Explanation Agent — read-only слой объяснения и оркестрации; он не записывает численные controls и NPV.

## Control representation

Управление задаётся через безопасные perturbations существующего Model Z schedule и, после доказанного round-trip, через typed control records. Активные targets `WCONPROD`/`WCONINJE` изменяются с сохранением исходных pressure/group/default fields. Решения сглаживаются во времени; по умолчанию control node раз в 3 месяца. Producer→injector conversion допускается, обратный переход запрещён.

## Data generation

Pilot DoE содержит 32 Sobol/LHS сценария. Управляющие траектории ограничивают `max_delta`, чтобы не создавать ежемесячный белый шум. Затем dataset расширяется до 128–256 OPM runs только если marginal value оправдывает бюджет.

Каждый run содержит:

- commit SHA, GitHub run ID, simulator version;
- deck/schedule hashes;
- seed и control vector/trajectory;
- raw OPM outputs;
- normalized well-time parquet;
- constraint results;
- exact economics breakdown;
- model/optimizer version.

Все run prefixes immutable: `runs/<git_sha>-<github_run_id>/<run_id>/`.

## Static and learned well graph

Edge candidates строятся из:

- grid/geometric proximity;
- completion/perforation overlap;
- static transmissibility evidence, если доступно;
- measured lagged injector→producer response из DoE.

Edge weights пересчитываются только на training split. Test scenario outcomes никогда не используются для построения learned graph.

## Five strategy portfolio

1. `linear_local`: linear surrogate + local search.
2. `gru_cem`: GRU + CEM.
3. `tcn_cma`: TCN + diagonal CMA-ES.
4. `graph_risk_cem`: ensemble graph-temporal surrogate + uncertainty-aware CEM + active learning.
5. `graph_mappo`: тот же graph surrogate + shared graph actor / centralized critic MAPPO (CTDE).

Все пять используют одинаковый action space, Constraint Guard, economics contract и final OPM promotion rule.

## Surrogate targets

Минимальный target tensor на well/time:

- oil rate;
- liquid rate;
- injection rate;
- BHP;
- WCT;
- availability/status masks.

Field-level cumulative totals и NPV вычисляются детерминированно из rollout, а не являются единственным target. Это не позволяет модели получить хорошую NPV-ошибку при физически неверной траектории.

## Uncertainty

Neural primary uses deep ensemble 3–5 seeds. Calibration измеряется на scenario-level validation split. Candidate score:

`predicted NPV - beta * epistemic uncertainty - physics/OOD penalties`.

Высоко-OOD candidate не может стать финальным без OPM.

## Active learning

Acquisition объединяет value, epistemic uncertainty и distance/novelty относительно training controls. В каждой итерации OPM budget делится между exploitation и exploration, чтобы optimizer не загнал surrogate в собственный blind spot.

## Validation gates

### Contract

- реальные Model Z dimensions = `91 × 102 × 59`;
- well count = 109;
- OPM baseline completes;
- schedule writer/perturber preserves keyword item semantics;
- economics reproduces supplied corrected fixtures;
- `WLPR ≤ 500` at every validated step.

### ML

- split only by whole `scenario_id`;
- NRMSE/sMAPE per target;
- rollout error by horizon;
- ranking metrics;
- calibrated interval coverage;
- OOD slices;
- physical consistency.

### Optimization

- multiple seeds;
- equal surrogate evaluation budget in bake-off;
- OPM validation budget recorded separately;
- final winner chosen after hard gates by true OPM-NPV;
- tie-break: robustness, then fewer OPM calls.

## Runner policy

- Ubuntu: OPM truth generation and final validation.
- `macos-15` M1: surrogate/optimizer benchmark and training only when measured faster.
- CI runs functional tests on both architectures.

## Multi-agent roles

Monitor, Reservoir Diagnostic, Planning, Constraint Guard, Surrogate, Simulator, Economics, Explanation share a typed orchestration state. Only Simulator can attach an OPM result; only Economics can attach final NPV; publication requires both accepted constraints and OPM validation.

## UI

The UI is a thin audit layer over pipeline artifacts. Required final panels:

- map/graph of wells and edge weights;
- oil/liquid/injection/BHP/WCT trajectories;
- surrogate-vs-OPM error and uncertainty;
- five-strategy bake-off;
- economics breakdown;
- audit trail of role agents;
- final `wells_schedule.inc` download.

## Acceptance

A release candidate is accepted only when:

- all tests and CI gates pass;
- real Model Z metadata contract passes;
- exact economics fixture parity passes;
- at least one complete DoE→train→search→OPM active-learning loop is stored in HF;
- all five strategies have comparable evaluation rows;
- final schedule has zero hard violations;
- final NPV comes from OPM output + exact economics engine and is paired with hashes;
- no `ecup*` branch or artifact is used.
