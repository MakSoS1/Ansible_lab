# AIOS Track 2 — актуальный research и конкурсная стратегия v2

Дата среза: 2026-09-04.

## 1. Что на самом деле оценивает трек

Официальная страница AIOS формулирует Track 2 как задачу построения нейросетевой surrogate-модели, принятия решений на её прогнозах, генерации корректного `wells_schedule.inc` и максимизации NPV с обязательной повторной проверкой полной гидродинамической моделью.

Критерии оценки нельзя сводить к одной surrogate-ошибке. Организаторы отдельно проверяют:

1. NPV на полной гидродинамической модели;
2. улучшение относительно baseline;
3. сравнение с reference;
4. соблюдение технологических ограничений;
5. устойчивость решения;
6. адаптивность;
7. межскважинную согласованность;
8. для Track 2 — скорость и качество surrogate.

Источник: https://blueskyresearch.ru/hackathon_aios

Следствие: оптимизатор должен максимизировать не «красивый» surrogate reward, а вероятность получить высокий **повторяемый OPM-NPV без нарушения ограничений**.

## 2. Что подтверждают лекции организаторов

Расшифровки в `aios-track2/transcripts/cleaned/` задают важные инженерные ограничения:

- исходной выборки независимых управляющих сценариев нет — её нужно генерировать через DoE и симулятор;
- surrogate особенно опасен при экстраполяции, поэтому нужны uncertainty/OOD checks;
- хороший surrogate должен выдавать оценку уверенности;
- RL разумно обучать на surrogate, но финальные действия проверять симулятором;
- независимый агент на каждую скважину страдает от нестационарности, credit assignment и физического взаимовлияния; нужен CTDE/иерархия/общий граф;
- ключевые физические KPI: компенсация отбора закачкой, пластовое и забойное давление, запаздывающий отклик «нагнетательная → добывающая», обводнённость/прорыв воды;
- аквифер Model Z ограничен и не является источником постоянного давления;
- допускаются циклика и перевод добывающей скважины под закачку;
- конечное решение должно быть физически реализуемым, технологически корректным и экономически оправданным.

## 3. Публичные решения конкурентов

На 2026-09-04 публично индексируемых репозиториев команд AIOS с реализацией именно Model Z / `wells_schedule.inc` обнаружить не удалось. Поэтому сравнивать себя с выдуманными «решениями конкурентов» нельзя. Вместо этого ниже используются наиболее близкие опубликованные решения 2025–2026 и их открыто описанные архитектурные принципы.

## 4. Актуальные внешние ориентиры

### 4.1 Graph-based multi-agent RL, 2026

**Scalable and adaptive injection-production control in reservoirs via a multi-agent reinforcement learning approach**, Energy Reports, Vol. 15, 2026, DOI: 10.1016/j.egyr.2025.108983.

Полезные идеи:

- surrogate + graph-based unified agent;
- граф межскважинной связанности;
- адаптивная координация большого числа скважин;
- сравнение с PSO/GA по NPV и устойчивости.

Вывод для нас: граф нужен не как декоративный компонент UI, а как inductive bias для физически связанных скважин.

### 4.2 Spatiotemporal graph surrogate + continuous RL, 2026

**Spatiotemporal graph-based surrogate modeling and deep reinforcement learning for multi-layer injection-production decision in waterflood reservoirs**, Engineering Applications of Artificial Intelligence, indexed 2026, DOI: 10.1016/j.engappai.2026.115452.

Архитектура: GCN + Transformer + LSTM surrogate и TD3 control. Авторы отдельно возвращают найденную политику в физический симулятор. Публикация показывает, что ключевой выигрыш даёт сочетание пространственного графа, временной динамики и simulator re-validation.

Вывод: `graph_temporal + continuous optimizer/RL + simulator confirmation` — сильный современный шаблон, но для короткого хакатона TD3/MARL не должен быть единственным путём.

### 4.3 Multi-fidelity transfer, 2025

**A multi-fidelity transfer learning framework for efficient reservoir production optimization**, Petroleum Science, 2025, DOI: 10.1016/j.petsci.2025.02.014.

Coarse→fine обучение одной и той же физической модели может существенно сократить дорогие fine-grid расчёты. Но Model Y нельзя автоматически считать low-fidelity версией Model Z: перенос оправдан только после доказательства совместимости геологии, управления и отклика.

### 4.4 Active learning, 2025

Актуальные работы по waterflood optimization показывают пользу ensemble-surrogate + active learning: дорогие симуляции тратятся не равномерно, а на области с высоким ожидаемым value, uncertainty и novelty. Это лучше «128 случайных прогонов и больше никогда не спрашивать OPM».

## 5. Топ-5 реализаций для честного bake-off

| ID | Surrogate | Search | Роль |
|---|---|---|---|
| `linear_local` | Ridge/linear | локальные perturbations | sanity lower bound |
| `gru_cem` | GRU | CEM | дешёвый sequential baseline |
| `tcn_cma` | TCN | diagonal CMA-ES | параллелизуемый temporal challenger |
| `graph_risk_cem` | graph-temporal ensemble | uncertainty-aware CEM + active learning | **основной кандидат** |
| `graph_mappo` | graph-temporal ensemble | shared graph MAPPO/CTDE | MARL challenger |

Почему основной кандидат — `graph_risk_cem`: он использует физическую связанность и uncertainty, но не несёт всей нестабильности RL. MARL проходит в финальный контур только если его преимущество воспроизводится на нескольких seed и подтверждается OPM.

## 6. Единый evaluation protocol

### 6.1 Hard gates

Кандидат не ранжируется вообще, если:

- расписание синтаксически/семантически невалидно;
- `WLPR > 500 м³/сут`;
- превышены заданные pressure/infrastructure limits;
- недопустимая последовательность смены роли скважины;
- финальный NPV не подтверждён OPM;
- заявленный NPV расходится с детерминированным economics engine.

### 6.2 Dynamic fidelity

По полностью отложенным `scenario_id`:

- MAE, RMSE, NRMSE, sMAPE;
- ошибки WOPR/WLPR/WWIR/BHP/WCT;
- ошибка по горизонту авторегрессионного rollout;
- ошибка field totals и cumulative volumes.

### 6.3 Decision fidelity

Для оптимизации важнее не только точное значение, но и правильный порядок кандидатов:

- Spearman;
- Kendall τ;
- pairwise ranking accuracy;
- top-k recall;
- OPM NPV regret выбранного кандидата.

### 6.4 Uncertainty/OOD

- empirical coverage 50/80/90/95% intervals;
- sharpness;
- error-vs-uncertainty monotonicity;
- OOD slice по дальности от DoE;
- запрет auto-promotion высоко-OOD кандидатов без OPM.

### 6.5 Physics

- compensation deviation;
- pressure violation rate;
- watercut bounds/breakthrough;
- lagged injector→producer response;
- zero `WLPR` violations;
- conservation/residual checks там, где доступны нужные поля OPM.

### 6.6 Robustness and efficiency

- минимум 3 seed для neural/RL branches;
- sensitivity к ±5–10% управляющим perturbations;
- число полных OPM runs до лучшего подтверждённого NPV;
- wall-clock train/inference/search;
- CPU/MPS benchmark;
- конечный NPV после OPM остаётся главным критерием.

## 7. Active-learning loop

1. Sobol/LHS pilot: 32 физически допустимых гладких сценария.
2. Fit ensemble.
3. Сгенерировать большой pool кандидатов CEM/CMA/MAPPO.
4. Посчитать `acquisition = z(NPV) + 0.5*z(uncertainty) + 0.25*z(novelty)`.
5. OPM для top-value и top-uncertainty/novelty кандидатов.
6. Добавить immutable run в `MakSoS1/aios-track2-runs`.
7. Retrain/recalibrate.
8. Остановиться по marginal true-OPM improvement / simulation budget.

## 8. Почему OPM — Linux, а Apple M1 — не догма

На GitHub-hosted `macos-15` стандартный ARM64 runner — Apple M1, но это 3 CPU / 7 GB RAM. Он хорошо подходит для ARM/MPS PyTorch benchmarks, однако OPM предоставляет готовые Linux packages, а macOS обычно требует source build. Поэтому:

- OPM Flow и генерация truth data: Ubuntu;
- PyTorch surrogate/bake-off: Ubuntu и M1, затем выбирать runner по измеренному wall-clock;
- никаких предположений «M1 точно быстрее» без benchmark artifact.

## 9. Что исправлено относительно исходного плана

- добавлен fifth candidate `tcn_cma`, чтобы bake-off действительно был из пяти исполняемых стратегий;
- исправлена семантика `WCONPROD`: LRAT должен попадать в item 7, а не в первое свободное числовое поле;
- добавлены ranking/OOD/coverage/rollout/physics metrics;
- active learning использует novelty, а не только uncertainty;
- Model Y не используется как low-fidelity Model Z без доказанной совместимости;
- PyG/RLlib не являются обязательными зависимостями: graph message passing и shared-policy CTDE реализованы на PyTorch, что упрощает ARM64 CI;
- final winner выбирается только по OPM-validated outcome после hard gates.
