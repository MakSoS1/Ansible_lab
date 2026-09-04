# MakSoS1 / Ansible_lab

В репозитории размещены материалы, спецификация и рабочая реализация трека 2 хакатона AIOS.

- [Материалы трека 2](aios-track2/README.md)
- [Анализ задания](docs/analysis/track-2-findings.md)
- [Актуальные решения конкурентов](docs/analysis/competitor-solutions-2026.md)
- [Архитектурная спецификация](docs/superpowers/specs/2026-09-04-aios-track2-design.md)
- [Подробный план реализации](docs/superpowers/plans/2026-09-04-aios-track2-implementation.md)
- [Воспроизводимость](docs/reproducibility.md)

Пять контроллеров реализованы полностью: heuristic, linear+CEM, TCN+CEM, graph-temporal ensemble + CMA-ES, MAPPO challenger. Обучение и bake-off запускаются вручную на GitHub-hosted `macos-14` (Apple Silicon) workflow `aios-train-surrogate.yml`.

Файлы со списками команд, ФИО и Telegram-идентификаторами участников намеренно не публикуются.

