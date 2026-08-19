# E-CUP v20.3 Modern Teachers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax.

**Goal:** Finish v20 with an empirically selected modern multilingual/Russian-capable teacher pair and statistically sound hierarchical admission while keeping the production RuBERT runtime unchanged.

**Architecture:** Extend the existing v20.2 bakeoff. Select two independent teacher families on fold-safe human audit data, calibrate generated labels using predicted-label, reason-by-label, category, and critical-family Wilson gates, then run the existing promotion and packaging ladder.

**Tech Stack:** Python, pandas/pyarrow, PyTorch/Transformers, llama.cpp CUDA12, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-19-ecup-v20-3-modern-teachers-hierarchical-admission-design.md`

## Global Constraints
- Keep sealed evaluation data untouched.
- Preserve the immutable split and one-checkpoint production runtime.
- Teacher peak VRAM must fit the configured 8 GB runner gate.
- Exact model revisions and quantized artifact hashes are recorded.
- Fail closed when teacher selection, admission, promotion, or package validation fails.

### Task 1: Hierarchical calibration
- [ ] Add a failing test proving one reason can contain both predicted labels when each label-specific bucket is reliable.
- [ ] Replace whole-reason gating with reason-by-predicted-label gating.
- [ ] Update row admission/reliability readers and run all v20 admission tests.

### Task 2: Teacher pair schema
- [ ] Add a regression test passing `score_pair()` output directly to `select_teacher_pair()`.
- [ ] Make the pair report expose the canonical `teachers` key while retaining model/revision/family metadata.
- [ ] Run teacher bakeoff tests.

### Task 3: Teacher runtime validation
- [ ] Validate Qwen3.5 and Gemma GGUF llama.cpp paths.
- [ ] Validate EuroLLM causal and FRED seq2seq paths.
- [ ] Keep Pollux in the Qwen family so it cannot pair with Qwen3.5.
- [ ] Run all source tests and py_compile.

### Task 4: Freeze executor
- [ ] Pin the new public code commit in the private executor.
- [ ] Add/update private regression tests for the exact source and teacher pool.
- [ ] Run private tests and py_compile.

### Task 5: Run the campaign
- [ ] Reuse or rebuild prepared data safely.
- [ ] Run the identical two-fold teacher bakeoff and select only an eligible independent pair.
- [ ] Run full calibration for the selected pair, prefilter candidate queues, label them, and require non-zero admitted train/proxy labels.
- [ ] Run the existing control/data/rationale/replay/scaled two-fold ladder.
- [ ] Refit production only after promotion gates pass.
- [ ] Build and exact-check the final submission archive before declaring it ready.
