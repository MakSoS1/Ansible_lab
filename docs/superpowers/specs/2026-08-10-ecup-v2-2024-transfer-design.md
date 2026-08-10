# E-CUP Matching v2 — 2024 Transfer + GPU Reranker Design

Date: 2026-08-10
Branch: `ecup-matching-2026`

## Goal

Build the next complete E-CUP 2026 submission by independently re-implementing the strongest reproducible ideas visible in public E-CUP 2024/product-matching material, testing them against the unchanged item-disjoint 2026 human validation, and using Lightning AI only for GPU work that materially benefits from it.

The v1 anchor remains `v1-structured-hgb` with Macro AP `0.49616548946964434`.

## Evidence transferred from 2024

Public 2024 material supports these transferable ideas:

1. clean duplicate/contradictory labels before fitting;
2. exploit positive connected components to derive safe transitive positives;
3. treat model numbers, size, volume, pack count and other numeric contradictions as first-class signals;
4. compare attributes with category-aware importance rather than one generic attribute similarity;
5. mine difficult negatives among very similar products instead of emphasizing random negatives;
6. combine structured product signals with a pairwise neural scorer/reranker;
7. use soft/weak labels selectively, not as if all pseudo-labels were equally reliable.

We will not copy a participant repository or weights. The implementation is original and only reproduces general methods described publicly.

## v2 architecture

### A. Fixed evaluation protocol

The v1 item-disjoint human validation remains unchanged. All headline v2 comparisons use the exact same split and macro mean of per-category `average_precision_score`.

Within the v1 training components, create a second item-disjoint calibration split for blend selection. The fixed validation is never used to train base models.

### B. Human label graph cleanup and augmentation

Canonicalize pair order `(min(id1,id2), max(id1,id2))`, remove exact duplicates, and detect conflicting labels.

Build connected components using human positive edges. Generate transitive positive pairs only inside positive components and only up to a deterministic per-component cap to avoid quadratic explosion. Never generate a positive if an explicit human negative exists for the same canonical pair.

Human negatives remain authoritative. Any weak-label pair contradicted by human labels or human positive-component identity is removed or down-weighted.

### C. Weak-label curriculum from `matches_llm.parquet`

Use continuous LLM targets with deterministic confidence buckets:

- `p <= 0.03` or `p >= 0.97`: weight 1.00;
- `0.03 < p <= 0.15` or `0.85 <= p < 0.97`: weight 0.60;
- `0.15 < p <= 0.30` or `0.70 <= p < 0.85`: weight 0.30;
- `0.30 < p < 0.70`: excluded from the first v2 fit.

Human examples receive weight 10 before category equalization. For models requiring hard labels, weak labels use `p >= 0.5`; for neural BCE training, preserve the original soft target.

To control runtime, sampling is category-balanced and deterministic. High-confidence weak positives are retained preferentially, while weak negatives are selected with a mixture of random negatives and structured hard-negative score.

### D. 2024-inspired structured features

Extend v1 with:

- explicit brand equality/conflict from common attribute-key aliases;
- category-aware attribute importance learned only on training data;
- weighted attribute agreement/conflict;
- critical numeric contradiction count rather than only a binary flag;
- exact normalized name + conflicting quantity/model indicators;
- hard-negative score combining name similarity with model/number/quantity conflicts.

The structured model stays organizer-safe (`sklearn` only at inference).

### E. GPU pair reranker

Train a compact multilingual/Russian pair classifier on Lightning AI. The preferred first candidate is a compact RuBERT/MiniLM-class encoder that fits comfortably on L4/A100 and whose tokenizer/model files can be bundled offline.

Input serialization:

`[A] <name> [ATTR] key=value ... [B] <name> [ATTR] key=value ...`

Attributes are deterministically truncated by training-derived category importance. Max sequence length starts at 256 because the organizer baseline already demonstrates this budget.

Training curriculum:

1. human train + capped transitive positives;
2. add sampled high-confidence weak labels with soft BCE;
3. mine false-positive/hard-negative pairs using the current reranker + structured score;
4. short second-stage fine-tune emphasizing hard negatives.

The GPU is used only for fine-tuning/scoring. CPU preprocessing, packaging and organizer-runtime verification stay on GitHub Actions.

### F. Final score

Produce two base scores on calibration/validation:

- structured v2 score;
- neural reranker score.

Select a global blend coefficient on the calibration split by macro AP. Permit category-specific residual coefficients only when each category has enough calibration positives/negatives and they improve calibration macro AP without pathological weights.

Final prediction is continuous and clipped to `[0,1]`. Structured contradiction signals are available to the blend so a semantically high neural score can be reduced for explicit model/size/pack conflicts.

### G. Submission/runtime

Package model/tokenizer locally in the submission ZIP. No network is allowed at inference. Use fp16/bf16 GPU inference when available and batched tokenizer/model scoring. The structured scorer remains a fallback and blend input.

The submission must pass the exact organizer image `odsai/ecup26-matching-baseline:1.0` with `--network none`, preserve pair order, emit exactly `id1,id2,predict`, and stay within archive/runtime limits with at least 25% private-runtime headroom.

## Lightning security

Lightning credentials are runtime secrets only. They must never be committed to Git, written to Markdown, printed in logs, embedded in commands stored in workflow YAML, or uploaded to Hugging Face.

Programmatic authentication uses process environment variables. GPU jobs receive only the minimum credentials needed to read the private HF dataset and write private v2 artifacts. If Lightning cannot be driven safely without exposing a credential, GPU execution stops rather than weakening this invariant.

## Experiment ladder inside v2

Record each ablation under `ecup_matching/experiments/v2/`:

1. `v2a`: human label cleanup + transitive positives + new structured features;
2. `v2b`: add confidence-weighted weak labels;
3. `v2c`: hard-negative reweighting/mining;
4. `v2d`: compact GPU reranker;
5. `v2e`: reranker hard-negative second stage;
6. `v2-final`: calibrated structured + reranker blend and final refit/package.

The retained v2 submission is whichever verified candidate has the best fixed-validation macro AP while satisfying runtime constraints.

## Success criteria

v2 is complete only if all of the following hold:

- fixed validation has zero item overlap;
- every retained ablation records macro AP and all 20 category APs;
- final fixed-validation Macro AP exceeds v1 `0.49616548946964434`, otherwise v1 remains the best model and v2 is documented as rejected;
- submission executes offline in the exact organizer image;
- runtime and archive size are measured;
- model/submission artifacts are stored only in private HF;
- experiment docs, project state and Memora checkpoint are updated;
- no secret appears in public Git, artifacts or logs.
