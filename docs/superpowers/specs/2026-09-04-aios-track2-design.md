# AIOS Track 2 Design Specification

## Objective

Создать воспроизводимую масштабируемую многоагентную систему для Model Z, которая генерирует допустимый `wells_schedule.inc`, максимизирует подтверждённый OPM Flow NPV и использует обучаемый surrogate для сокращения числа полных симуляций.

## Delivery boundary

Текущий публикационный коммит переносит очищенные исходные материалы, расшифровки и этот проектный пакет. Он создаёт только ручной workflow загрузки материалов в приватный Hugging Face Dataset. Симуляция, обучение и оптимизация не запускаются.

Будущая реализация выполняется по `docs/superpowers/plans/2026-09-04-aios-track2-implementation.md`.

## System architecture

Поток данных состоит из неизменяемого Model Z, детерминированного генератора расписаний, OPM Flow adapter, нормализованного сценарного хранилища, суррогатного ансамбля, оптимизатора, hard-constraint gate, экономического ядра и генератора финальной поставки. Ролевые агенты вызывают эти компоненты через типизированные интерфейсы. LLM участвует в объяснении и оркестрации, но не может обойти проверку ограничений или изменить вычисленный NPV.

## Public and private storage

Публичный GitHub содержит все технические материалы, модели, записи и расшифровки, кроме четырёх документов с персональными данными участников. WEBM хранятся через Git LFS. Приватный dataset `MakSoS1/aios-track2-runs` получает тот же очищенный комплект и в дальнейшем хранит данные экспериментов.

Каждый эксперимент сохраняется в неизменяемый каталог `runs/<git_sha>-<github_run_id>/`; повторный запуск создаёт новый каталог. Токены передаются только через GitHub Secrets и никогда не записываются в конфигурации или логи.

## Control representation

Решения задаются на квартальных узлах, интерполируются до разрешённого симулятором шага и группируются по графу взаимовлияния. Базовые действия: включение/отключение, коэффициент дебита/приёмистости, режим по давлению, перераспределение закачки, необратимый перевод добывающей скважины в нагнетательную. Constraint Guard обязан проверить тип скважины, диапазоны, инфраструктуру, давление, `WLPR ≤ 500` и допустимость последовательности переходов до записи include-файла.

## Dataset contract

Одна строка метаданных описывает целый сценарий: `scenario_id`, `seed`, `deck_sha256`, `schedule_sha256`, `simulator_version`, `status`, `runtime_seconds`, `npv_mrub`, `constraint_violations`, `github_run_url`. Динамические данные хранятся в Parquet с ключом `(scenario_id, date, well)` и полями управления, `WLPR`, `WOPR`, `WWIR`, `BHP`, `THP`, `WCT`, накопленными показателями и масками доступности.

Train/validation/test разделяются по `scenario_id`. Ни одна временная строка сценария не может оказаться в другой выборке.

## Surrogate candidates

Обязательные кандидаты: persistence/linear baseline, GRU/TCN и graph-temporal ensemble. Transformer-lite и residual-physics вариант являются stretch-кандидатами. Выход surrogate включает среднее и uncertainty по динамическим показателям; физически невозможные значения маскируются и штрафуются.

## Optimizer candidates

Обязательные кандидаты: baseline schedule, local perturbation, CEM и CMA-ES. MAPPO/QMIX является challenger и использует тот же Constraint Guard. Оптимизируемый surrogate score равен ожидаемому NPV минус uncertainty penalty; продвижение кандидата определяется только OPM-валидацией.

## Multi-agent roles

Monitor, Reservoir Diagnostic, Planning, Constraint Guard, Surrogate, Simulator, Economics и Explanation работают через единый orchestration state. Economics Agent является единственным источником NPV. Explanation Agent получает read-only audit trail.

## Validation gates

1. Baseline OPM output воспроизводится повторным запуском.
2. Исправленный ЧДД проходит поставляемые тесты и отдельные fixtures для стартов 1991/2007.
3. `wells_schedule.inc` проходит синтаксическую и семантическую проверку до OPM.
4. Surrogate оценивается на полностью отложенных сценариях.
5. Оптимизатор не может выбирать OOD-кандидаты без OPM-проверки.
6. Финальный NPV публикуется только вместе с OPM output hash и полным breakdown.
7. Все случайные генераторы получают явный seed.

## GitHub Actions policy

CI и OPM выполняются на Linux runner. Тяжёлое обучение запускается только ручным или явно разрешённым workflow из GitHub Actions и может оркестрировать Lightning через `LIGHTNING_API_KEY`/`LIGHTNING_USER_ID`. Результаты публикуются в HF через `HF_TOKEN`. Ни один workflow текущего публикационного коммита не запускает обучение.

## User interface

UI показывает карту/граф скважин, состояние по времени, действия агентов, surrogate uncertainty, сравнение surrogate/OPM, breakdown NPV и ссылку на сформированный `wells_schedule.inc`. Пользователь может запустить заранее определённый pipeline, но не редактирует результаты после запуска вручную.

## Acceptance criteria

- Полная поставка стартует одной документированной командой.
- Повторный запуск с тем же seed и артефактами даёт сопоставимый результат.
- Невалидное расписание не достигает OPM и не публикуется.
- Финальный schedule и NPV подтверждены OPM и точным экономическим ядром.
- HF run manifest связывает commit, Action run, dataset revision, model checkpoint и simulator output.
- Публичный репозиторий не содержит исключённых списков участников.
