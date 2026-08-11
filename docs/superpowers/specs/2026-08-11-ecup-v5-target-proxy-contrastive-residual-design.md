# E-CUP 2026 Matching v5 — Target-Proxy Contrastive Residual

Date: 2026-08-11
Status: approved for implementation
Branch: `ecup-matching-2026`

## 1. Motivation and production evidence

Leaderboard evidence invalidates the old assumption that the fixed 73,131-row validation is a reliable proxy for the hidden distribution:

| Candidate | Old local Macro AP | Hidden Macro AP |
|---|---:|---:|
| v1 | 0.4961654895 | 0.23458522924335687 |
| v2 | 0.5010008995 | 0.2583231811423486 |
| v3 non-canonical | n/a | 0.2583231811423486 |
| v3 canonical | 0.5254642646 | 0.24810151893254498 |
| v4 canonical | 0.5276431099 OOF / 0.5284493943 full-fit | 0.2531285194869718 |

The production-best artifact is therefore v2, not v4. v3/v4 remain historical offline experiments.

The likely failure is not item-ID leakage but validation-distribution mismatch plus repeated adaptive model selection on one holdout. The existing component split guarantees zero train/validation item overlap but does not preserve category, target prevalence, lexical difficulty, contradiction regimes, missingness, or other pair-distribution structure.

## 2. Goal

Build v5 as a genuinely different model family whose changes are selected against a redesigned validation protocol and whose default behavior remains close to the hidden-best v2.

Primary success criterion:

- v5 must improve over v2 on leakage-safe repeated/grouped validation and on one frozen gold holdout.

Stretch criterion requested by the user:

- achieve Macro AP >= 0.600000 on the new honest frozen-gold validation protocol.

The stretch criterion must never be reached by fitting, tuning, thresholding, calibration, category-weight selection, or architecture selection on the frozen-gold labels. A score is called `honest_0p60` only if the complete candidate was frozen before the gold labels were scored.

No hidden-score guarantee is claimed from a local 0.60; the leaderboard remains the final external test.

## 3. Immutable fallback and anti-regression rule

`v2b-weak-curriculum` is the production anchor because it has the best observed hidden Macro AP: 0.2583231811423486.

v5 is a residual system:

```text
base_logit = logit(clamp(v2_score))
residual = f(new_features)
final_logit = base_logit + lambda * residual
final_score = sigmoid(final_logit)
```

The residual is regularized toward zero. A new representation does not get a fixed 40-50% blend weight merely because it improves a development holdout.

## 4. Validation redesign

### 4.1 Component construction

Build connected item components over authoritative human pair edges. Components are indivisible across any split; train/evaluation item overlap must remain exactly zero.

### 4.2 Difficulty descriptors

For every human pair compute split-only descriptors without using evaluation labels as model features:

- category;
- target class;
- component row count;
- normalized-name equality;
- title token Jaccard bin;
- char n-gram similarity bin;
- model-code overlap/conflict;
- numeric overlap/conflict;
- quantity overlap/conflict;
- attribute-key coverage bin;
- attribute missingness;
- title length bins;
- v2 hard-negative score bin.

### 4.3 Balanced grouped development folds

Reserve approximately 75-80% of components as development data. Assign components to five folds with an objective that minimizes divergence from the full development distribution across category, target prevalence, and difficulty bins while preserving component integrity.

For every candidate report:

- five fold Macro AP values;
- mean, median and standard deviation;
- per-category AP;
- per-fold `delta_vs_v2`;
- worst-fold delta;
- hard-slice deltas.

Selection should favor stable positive delta versus v2, not the largest single mean score.

### 4.4 Frozen gold holdout

Reserve approximately 20-25% of components before v5 model development. Its labels must not be used for:

- model-family selection;
- hyperparameter tuning;
- encoder selection;
- epoch selection;
- residual scale selection;
- weak-label weighting;
- category calibration;
- hard-negative curriculum design.

The gold set is evaluated only after a candidate/configuration hash is frozen. After one gold evaluation, that score is recorded as evidence and that exact holdout may not be reused as an unlimited tuning loop.

### 4.5 Adversarial slices

Always report v2-relative AP or ranking metrics for:

- high lexical similarity negatives;
- low lexical similarity positives;
- same normalized title with negative target;
- different normalized title with positive target;
- model/SKU conflicts;
- numeric/unit conflicts;
- quantity/pack-size conflicts;
- sparse-attribute pairs;
- attribute-rich pairs;
- weak categories identified in earlier iterations.

## 5. Target-proxy validation

When real unlabeled organizer candidate pairs are available to the local submission runtime, compute the same label-free pair descriptors on them.

Train a lightweight domain classifier on development-only rows:

```text
human-development pair -> 0
unlabeled target candidate pair -> 1
```

Use out-of-fold domain probabilities to measure covariate shift and derive clipped density-ratio weights for a supplemental target-weighted development metric.

Target-proxy metrics are diagnostic and model-selection evidence; they do not replace ordinary grouped Macro AP or the frozen gold holdout. If no representative unlabeled target-pair sample is available outside organizer execution, this component must degrade gracefully and be reported as unavailable rather than simulated.

## 6. New model family

### 6.1 Item-level representation

Replace the v3-style pair Cross-Encoder as the primary neural representation with an item-level encoder. Serialize each product with compact structured sections:

```text
[NAME] normalized name
[BRAND] normalized brand if available
[MODEL] model/SKU tokens
[NUMERIC] canonical numeric/unit facts
[ATTR] selected category-relevant key=value facts
```

Attributes are deterministically prioritized so long raw JSON tails do not dominate the token budget.

### 6.2 Contrastive training

Train an open-license Russian/multilingual encoder with authoritative human supervision plus selected weak-label data. Candidate encoders are compared only on development folds. The preferred first rung should fit the available RTX 2060 SUPER 8GB training environment and the organizer H100 inference environment.

Loss combines:

- supervised contrastive / metric learning on item identity evidence;
- gold pair BCE/ranking supervision through a compact pair head;
- soft weak-label loss for high-confidence teacher pairs;
- in-category ranking loss that explicitly orders positives above hard negatives.

### 6.3 Pair representation

For pair `(A, B)` build compact neural features such as:

- cosine similarity;
- normalized L1/L2 distance;
- learned projection similarity;
- summary statistics of `abs(E(A)-E(B))`;
- summary statistics of `E(A)*E(B)`.

Do not ship giant per-pair vectors to a slow Python loop if equivalent compact similarities are sufficient.

## 7. Weak-label curriculum

The prior v2 retained only about 300k weak pairs and largely hardened pseudo-label probabilities. v5 should stream a much larger target-like curriculum, initially 2-4 million rows if resource limits permit.

Rules:

- authoritative human labels dominate;
- retain weak probabilities as soft targets;
- exclude direct human conflicts;
- exclude development/frozen-gold item IDs from weak training data;
- stratify/balance by category and weak target;
- prefer weak pairs that resemble the target candidate distribution when a valid target-proxy sample exists;
- ambiguous weak rows may be used for mining/ranking rather than as hard truth.

A starting source-weight schedule may be evaluated on development folds, but the final values must be selected without frozen-gold labels.

## 8. Model-driven hard-example mining

After the first contrastive/residual candidate:

1. score a large streamed weak pool;
2. collect teacher/model disagreements;
3. collect v2/new-model ranking disagreements;
4. prioritize model/SKU, number, size, color, quantity and revision contradictions;
5. replay ordinary/easy examples to avoid collapsing onto pathological hard negatives;
6. retrain only if grouped-CV evidence improves consistently.

## 9. Residual ranker

The final compact pair ranker consumes:

- v2 score/logit;
- v2 structured features;
- compact neural similarity features;
- contradiction features;
- category indicator/interactions;
- optional target-proxy/domain features when they are valid at inference.

The ranker predicts a residual correction to v2 rather than an unconstrained replacement score.

Preferred loss is ranking-aware:

```text
L = w_gold * BCE_or_rank(gold)
  + w_weak * BCE(student, soft_teacher_target)
  + w_pair * pairwise_ranking_loss
  + w_resid * residual_regularization
```

The residual magnitude and any regularization hyperparameters are selected only from development folds.

## 10. Ablation ladder

Each rung must be a real ablation and must preserve an immutable v2 reference prediction:

1. `v5a-validation-audit`: redesigned split + exact v2 baseline only.
2. `v5b-structured-residual`: new residual ranker using structured signals only.
3. `v5c-contrastive-human`: add item-level contrastive encoder trained on human development data.
4. `v5d-contrastive-weak`: add multi-million soft weak-label curriculum.
5. `v5e-hard-mined`: add model-driven hard-example continuation.
6. optional `v5f-teacher-distilled`: strong Cross-Encoder used offline only as teacher if prior rungs justify it.

Never advance a rung because its absolute validation score is prettier. Advance only if its v2-relative behavior is stable across development folds.

## 11. Honest 0.60 protocol

A local score >= 0.60 is accepted as `honest_0p60` only when all conditions hold:

1. item overlap between candidate training data and frozen gold is zero;
2. no weak row touching a frozen-gold item is used for training;
3. split-generation code and gold component IDs are frozen before candidate training;
4. model configuration, checkpoint choice, residual scale and preprocessing hash are frozen before gold evaluation;
5. gold labels are read only by the evaluation step;
6. no category-specific post-hoc fitting is performed on gold predictions;
7. exact v2 and v5 predictions are both scored on the same gold rows;
8. the result includes per-category AP and component-bootstrap confidence interval;
9. the score can be reproduced from immutable artifacts.

If 0.60 cannot be reached under these constraints, the implementation must report the best honest score rather than relabel a tuned score as honest.

## 12. Retention gate

A v5 candidate may replace v2 only if:

- grouped development CV has positive mean delta vs v2;
- no unexplained catastrophic fold regression exists;
- frozen-gold Macro AP exceeds frozen-gold v2;
- target-weighted development metric does not strongly contradict the ordinary metric when available;
- adversarial slices show interpretable gains/tradeoffs;
- component bootstrap supports a non-fragile gain;
- organizer image runtime and submission contract pass with >=25% private-limit headroom where measurable.

The `0.60` stretch target is stronger than the replacement gate, but it does not override these anti-leakage requirements.

## 13. Artifact and experiment policy

Public repository stores only code, aggregate metrics, hashes, workflows and documentation. Raw competition rows, product text, IDs, labels, trained weights and submission archives remain private.

Every retained v5 artifact records:

- source commit SHA;
- split-manifest SHA;
- candidate configuration hash;
- model/encoder hashes;
- validation prediction hash;
- private artifact path;
- organizer smoke evidence.

v2 remains immutable fallback throughout v5 development.
