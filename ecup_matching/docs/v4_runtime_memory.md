# v4 bounded-memory preprocessing

The v4 GPU pipeline keeps the retained modelling recipe while bounding host RAM before transformer training.

- Structured weak-label presampling reads `matches_llm.parquet` in PyArrow batches instead of materializing the full weak table in pandas.
- Eligible-row sampling remains deterministic and is regression-tested against the retained in-memory pandas recipe, including sampled row order.
- The large `ai-forever/ruBert-base` checkpoint is loaded only after structured-anchor generation, v4 curriculum preparation, and structured-validation alignment complete.
- These changes address the host/WSL memory peak observed during the first full v4 CUDA run without changing the fixed item-disjoint validation split, random seed, weak-label thresholds, or staged v4a/v4b/v4c training recipe.
