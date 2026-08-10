# E-CUP Matching — Iteration v2 Results

Date: 2026-08-10
Status: model selection complete; final package benchmark/upload in progress

## Summary

v2 tested independently reimplemented ideas motivated by public E-CUP 2024/product-matching work against the unchanged E-CUP 2026 item-disjoint validation.

The retained candidate is **`v2b-weak-curriculum`** with Macro AP **0.5010008994958702**, improving the v1 anchor `0.49616548946964434` by **+0.004835410026225895** absolute Macro AP.

Validation remains exactly 73,131 human pairs with **0 train/validation item overlap**.

## Structured ablations

GitHub Actions run: `31421985279`
Job: `93564684371`
Training source commit: `3ad39ca999e0f5bcc5e289de823d498ea2868b49`
Private artifact prefix: `experiments/v2/structured/3ad39ca999e0/`

| Candidate | Change | Macro AP | Delta vs v1 | Decision |
|---|---|---:|---:|---|
| v1 anchor | original human-only structured HGB | 0.4961654895 | — | baseline |
| `v2a-human-2024-features` | human label cleanup + category-aware/brand/conflict/hard-negative structured features | **0.5006971263** | +0.0045316368 | keep idea |
| `v2b-weak-curriculum` | v2a + confidence-filtered weak LLM labels | **0.5010008995** | **+0.0048354100** | **selected** |
| `v2c-hard-negative-weighting` | v2b + naive static hard-negative sample-weight boost | 0.4957263069 | -0.0004391825 | reject |

The important result is that the 2024-inspired product-aware representation transferred positively, and carefully filtered weak labels added a smaller additional gain. Static hard-negative weighting did **not** transfer: hard negatives likely need model-driven mining/reranking rather than a fixed heuristic weight.

## Data behavior

- Human rows: 365,654 total.
- Fixed outer validation: 73,131 rows.
- Validation item leakage: 0.
- Human positive components discovered inside training: 77,515.
- Capped transitive positive closure added **0** new rows on this dataset under the implemented safety rules. Therefore the useful 2024 graph lesson here was chiefly cleanup/consistency checking, not synthetic closure expansion.
- Weak source rows examined: 11,187,780.
- Confidence-filtered presample: 450,000.
- Final weak sample: 300,000 rows.
- Unique items referenced by final weak sample: 559,153.
- Mid-confidence weak labels were excluded; human labels and human positive-component identity overrode weak labels.
- Weak pairs touching any fixed-validation item were excluded before training.

## v2b per-category Average Precision

| Category | AP |
|---|---:|
| Автотовары | 0.5169590733 |
| Аптека | 0.4457487205 |
| Бытовая техника | 0.6596753216 |
| Бытовая химия | 0.6815901925 |
| Галантерея и аксессуары | 0.3297527176 |
| Детские товары | 0.7497737327 |
| Дом и сад | 0.5056863202 |
| Канцелярские товары | 0.5328049590 |
| Красота и гигиена | 0.5831173877 |
| Мебель | 0.3712536237 |
| Музыкальные инструменты | 0.6299979892 |
| Обувь | 0.2773861937 |
| Одежда | 0.2679553906 |
| Продукты питания | 0.5500254443 |
| Спорт и отдых | 0.4542051903 |
| Строительство и ремонт | 0.4970972637 |
| Товары для животных | 0.6002689643 |
| Хобби и творчество | 0.7839105690 |
| Электроника | 0.2573188299 |
| Ювелирные изделия | 0.3254901061 |

The weakest remaining categories are still Electronics, Apparel, Footwear, Jewelry, Accessories and Furniture. These remain priority targets for a neural reranker/category-specialist iteration.

## What transferred from E-CUP 2024

The v2 implementation deliberately tested general methods rather than copying participant code:

1. canonical duplicate/conflict handling for pair labels;
2. positive-component consistency and safe transitive-closure analysis;
3. model/number/quantity contradiction features;
4. brand extraction and contradiction;
5. category-specific attribute importance learned only from training rows;
6. explicit hard-negative score for lexically similar but contradictory variants;
7. selective weak-label curriculum rather than treating every pseudo-label equally;
8. compact pairwise reranker design for a later GPU stage.

A subtle v1 issue was also found during TDD: Python `difflib.SequenceMatcher.ratio()` can be directional in ambiguous matching-block cases. v2 makes pair features symmetric without changing the reproducibility of v1.

## Lightning GPU reranker attempt

A compact `cointegrated/rubert-tiny2` pairwise reranker, soft-label BCE curriculum and model-driven hard-negative second stage were implemented and tested at the code-contract level.

Lightning authentication was passed via an ephemeral RSA-OAEP bridge so plaintext Lightning credentials were never committed, printed, stored in Hugging Face, or written to workflow inputs. Current Lightning SDK `2026.8.5` authenticated successfully and exposed the user's Teamspace.

GPU training itself did **not** start because the authenticated account currently exposes no reusable Studio through the SDK and returns HTTP 403 for `create_cloud_space` when attempting to create one. Therefore no GPU credits were consumed by the failed attempts and no neural score was mixed into the retained v2 candidate.

The GPU code remains ready for the next iteration once an accessible Studio exists. Until then, the reproducible structured v2b is the correct retained submission rather than inventing a neural result.

## Runtime / packaging

Structured training and evaluation run took approximately 953 seconds on GitHub Actions CPU, including feature construction for all ablations and weak-label selection.

Final organizer-image ZIP runtime benchmark and private `submissions/v2/` upload are completed by `.github/workflows/ecup-build-v2-submit.yml`; exact runtime/ZIP numbers are appended here after that verification gate completes.

## Conclusions

- **Accepted:** 2024-inspired product-aware structured feature transfer.
- **Accepted:** high/medium-confidence weak-label curriculum.
- **Rejected:** naive static hard-negative sample weighting.
- **Deferred for infrastructure, not model quality:** GPU cross-encoder/reranker and neural/structured blend.
- **Current best validation model:** `v2b-weak-curriculum`, Macro AP `0.5010008994958702`.
