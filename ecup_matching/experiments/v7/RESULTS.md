# v7 Results

## Baselines retained

- strict quality reference: v5 category-shrunk HGB + equal-rank fusion = `0.6018115534135564`
- fast runtime reference: v6 gate95 = `0.6006003614522999`
- sealed gold opened: **no**

## 2026-08-12 — contract phase

Status: **IN PROGRESS**.

Evidence before implementation:
- retained final v5 teacher signal is teacher2 on `ai-forever/ruBert-base`;
- teacher2 used `max_length=128`, `max_steps=800`, and at most 100k weak rows per outer fold;
- v5 item text placed `[NUMERIC]` before `[ATTR]`, creating a context-allocation failure for canonical typed attributes;
- v7 RED CI run `31546090474` failed in the new targeted unit-test step before production modules existed.

Implemented after RED:
- isolated `v7_item_text.py` identity-first serializer; v5/v6 serializer unchanged;
- isolated `v7_teacher_contract.py` with 256-token minimum, full requested curriculum exposure contract, and explicit forbidden-item weak-pair filtering.

No metric has been claimed yet. Candidate A must complete five strict held-fold predictions before any OOF quality number is recorded.