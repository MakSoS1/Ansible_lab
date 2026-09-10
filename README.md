# MakSoS1 / Ansible_lab — AIOS Track 2

Ветка `aios-track2-v2`, PR https://github.com/MakSoS1/Ansible_lab/pull/8

## Что сдаём

- `submission/wells_schedule.inc` — расписание победителя MAPPO, независимо пересчитанное OPM Flow 2026.04 + ЧДД 7.0.2
- NPV **12 475.954558553085 млн ₽** (baseline 11 891.994046426242)
- max WLPR **62.55 ≤ 500**
- SHA `c5ff3221ac66dea460bbd638a589dc5c7f2dedeb1536b9f86b10fb2e3e030af3`
- Hugging Face: `Maksim123321/aios-track2-runs` / `runs/9ad40738134d-33925326455/final-mappo`

## Одна команда

```bash
python -m pip install -e '.[dev,api]'
python -m aios_track2.cli ui --submission submission
```

UI: http://127.0.0.1:8765 — рекомендуемый график, Δ к baseline, статус ограничений, скачивание `wells_schedule.inc`.

Docker:

```bash
docker build -t aios-track2 .
docker run --rm -p 8765:8765 aios-track2
```

Тесты: `PYTHONPATH=src pytest -q`

## Честные ограничения

- Сданное управление: **18D** (4 группы добычи + 2 группы закачки × узлы 2007/2016/2025), масштабирование существующих WCON. Это не полный поскважинный open/close / циклика / перевод.
- Код перевода, останова и циклики есть (`aios_track2.well_actions`) и покрыт тестами, но **не входит в сданное расписание**: нет отдельного OPM-подтверждения этих действий.
- Preregistered surrogate gate **не пройден**: top-3 recall = 2/3. Турнир финалистов всё равно авторизован по Spearman/pairwise/simple regret, а победитель выбран по OPM NPV.
- Хвост суррогата: P10 scenario×channel R² ≈ 0.80; худший канал FOPT сценария 29 имеет R² ≈ −16.17. Не прячем.
- PINCHREG: прогоны OPM с `--parsing-strictness=low`. Организатор на strict-парсере может получить другой NPV.

Ветки `ecup*` к этому треку не относятся.
