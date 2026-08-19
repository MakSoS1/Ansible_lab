# E-CUP v20.3 Modern Teachers + Hierarchical Admission Design

## Goal
Improve v20 training data quality and leaderboard transfer without changing the production runtime: keep one RuBERT pair CrossEncoder checkpoint, but use a modern, empirically selected teacher pair to calibrate and label only high-confidence real-item candidates.

## Invariants
- Sealed gold remains unopened and unscored.
- Immutable development split SHA remains `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.
- Production runtime remains one `ai-forever/ruBert-base` pair CrossEncoder checkpoint, max length 256, one `.safetensors` model.
- Public-LB anchors remain v14 > v12 > v13B > v7 and the never-labelled proxy must reproduce this order before it is allowed to select candidates.
- Teacher models are training-time only.
- Every resolved model revision, quantized artifact revision/hash, backend, peak VRAM, throughput and prompt hash is recorded.
- Canonical teacher execution must fit <= 7.75 GiB peak VRAM on the RTX 2060 SUPER 8 GB runner.
- Yandex models are not canonical unless their license and quantized artifact provenance are explicitly approved; current v20.3 fails closed rather than silently using them.

## Teacher candidate pool
Run an empirical bakeoff on the same fold-safe human audit sample. Candidate pool:

1. `Qwen/Qwen3.5-4B`, Q4_K_M GGUF, family `qwen`.
2. `ai-forever/Pollux-4B-Judge`, quantized GGUF, family `qwen`; it is a Russian-oriented judge candidate but cannot pair with Qwen3.5 because the pair-family gate requires independence.
3. `google/gemma-4-E2B-it`, Q4_K_M GGUF, family `gemma4`.
4. `utter-project/EuroLLM-1.7B-Instruct`, Transformers causal, family `eurollm`.
5. `ai-forever/FRED-T5-1.7B`, Transformers seq2seq, family `fred-t5`.

Qwen3.6/3.8 are deliberately not canonical local teachers because current open checkpoints are far larger than the 8 GB runner envelope. The design prefers the strongest model that can be run reproducibly inside the actual hardware constraint, not the newest model name.

## Teacher bakeoff gates
Each candidate is evaluated on fold0 and fold1 human bakeoff rows with:
- JSON valid rate >= 0.98.
- coverage >= 0.70.
- positive precision Wilson LCB >= 0.94.
- negative precision Wilson LCB >= 0.97.
- critical-conflict precision Wilson LCB >= 0.95.
- peak VRAM <= 7.75 GiB.
- exact resolved revision and artifact provenance present.

Pair selection requires:
- both individual teachers eligible on both folds;
- distinct model families;
- identical binary verdict and identical teacher reason code;
- deterministic checker compatibility (except deterministic `OTHER`/`SPARSE_EVIDENCE` which may accept a more specific teacher reason);
- non-zero consensus coverage on both folds.

Ranking among eligible pairs is lexicographic by mean consensus precision, critical precision, coverage, then throughput. No model is selected by reputation alone.

## Hierarchical calibration
The old fine `category|reason|difficulty` Wilson gate is retained only as diagnostic evidence, not as the canonical admission gate. The canonical hierarchy is:

1. global predicted-label gate (`pred=0` and `pred=1` separately);
2. `reason_code × predicted_label` gate;
3. category aggregate gate;
4. critical-family aggregate gate for critical reasons.

This fixes the statistical defect where a semantic reason containing both MATCH and NON_MATCH rows was rejected wholesale, and avoids requiring every fine difficulty stratum to independently accumulate enough support.

For each row, hierarchical reliability is the minimum Wilson LCB among every gate that applies to that row. A row is admissible only if all applicable gates pass their frozen floors and support requirements.

## Two-stage labeling
Stage A labels only fold-safe human audit rows with every teacher candidate. It selects one pair and builds fold0/fold1 calibration policies.

Stage B filters train/proxy candidate queues using the intersection of both fold policies before expensive candidate teacher inference. Only candidates whose deterministic `reason_code` and category could pass both policies are sent to the selected teachers. Final generated labels must still pass actual teacher consensus and both fold policies after inference.

## Training ladder
The existing causal ladder remains unchanged:
- control;
- data-only;
- rationale;
- source-aware replay;
- scaled exact selected mode on fold0/fold1;
- two-fold confirmation;
- full development production refit;
- exact organizer runtime/package check.

Promotion still requires strict proxy improvement over control and over historical v14 on a proxy axis that reproduces Public-LB anchor order, while satisfying human/tail/category gates.

## Fail-closed behavior
No submission is built if any of the following occurs:
- no eligible independent teacher pair;
- no hierarchical reason×label group survives both folds;
- no generated train/proxy labels survive both policies;
- proxy does not reproduce v14 > v12 > v13B > v7;
- scaled confirmation fails either fold;
- sealed gold flags are not exactly false/zero;
- final archive runtime/check gate fails.
