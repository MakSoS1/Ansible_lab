# E-CUP v14 — new item-centric architecture plan

Updated: 2026-08-15

## Objective

Build a materially different product-matching architecture after v13 demonstrated that incremental improvement of the pair CrossEncoder/proxy pipeline no longer reliably translates to Public LB. Target Public LB is >0.50, but no local metric is allowed to be represented as a guaranteed leaderboard score.

## Historical failures carried forward

- v7 Public LB: `0.3655833314`.
- v12 Public LB: `0.3798116204` — best measured external anchor before v14.
- v13 B Public LB: `0.3783781653` despite fold0 `0.7086611385531062` and Validation-v3 p05 `0.5690974845479824` both improving on v12. This invalidates Validation-v3 as a near-neighbour promotion gate.
- v9/v10: local quality was not useful when inference/runtime was unsafe.
- v11: startup/item scan alone could trigger the 60-second Check timeout.
- v13 C2: adding ListNet to the same pair representation hurt under equal weak exposure.
- v13 teacher path assigned material weight to an LLM soft target before an independent human-truth audit; v14 treats that source as untrusted until proven otherwise.

## Architecture

v14 encodes each product independently with one shared `ai-forever/ruBert-base` backbone, then forms three normalized projected vectors:

1. global item representation;
2. title representation;
3. attributes/category representation.

A strictly symmetric pair head consumes cosine, absolute difference and element-wise product for each channel plus deterministic symmetric lexical/model/numeric conflict features. Directional `A||B` concatenation is forbidden.

Initial max length is `128`; projected dimension is `192`. Inference encodes each referenced item once and reuses item vectors for every pair.

## Label policy

### v14-A: human only

- Human labels are authoritative.
- Positive training components are built only from human positive edges within the training fold.
- Any positive component containing an explicit human negative contradiction is quarantined from transitive expansion.
- A1 hard negatives are selected only from existing human-labelled negative rows. Unknown pairs are never invented as negatives.
- No historical LLM row enters A0/A1.

### LLM quarantine

The exact historical LLM label stream must first be aligned to human truth and audited by category, class and conflict stratum. Self-reported confidence never admits a label.

Frozen admission floors:

- positive precision >= `0.985`;
- negative precision >= `0.995`;
- admitted category precision >= `0.970`;
- critical-conflict precision >= `0.950`;
- sufficient human overlap is mandatory.

Missing evidence means `reject`, not `assume safe`.

## Experiment ladder

1. **A0** — human-only item architecture + component closure + symmetric pair head.
2. **A1** — A0 plus repeated hard human negatives, still no LLM.
3. **B** — train-time setwise/component teacher only if A1 justifies it.
4. **C** — audited LLM supervision only if the independent audit passes all frozen floors.
5. **D** — same-architecture seed/weight soup only when strict folds show complementary gain and runtime remains one model.

## Validation-v4

Validation-v3 remains frozen as historical evidence. Validation-v4 requires:

- exact immutable split SHA `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`;
- all five component-disjoint development folds;
- exactly `285210` OOF development rows with no duplicate indexes;
- sealed `80444` gold rows unscored;
- all 20 categories;
- category-level deltas and deterministic bootstrap stability;
- external experiment-level anchor sanity preserving `v12 > v13B > v7`.

External leaderboard values are never row labels and Validation-v4 is not fit to predict leaderboard score numerically.

## Runtime gate

Final inference contains one Transformer backbone, one tokenizer and one item-matching checkpoint. No LLM, graph engine, HGB, TF-IDF or second Transformer runs in submission inference.

Binding internal promotion target for the organizer-shaped fixture:

- 1000 input pairs;
- only referenced supplied items;
- valid ordered `id1,id2,predict`;
- continuous scores;
- wall <= `50 s` to retain >=10 s headroom below the 60 s organizer limit.

The full-item scan remains a conservative diagnostic and is not conflated with the binding supplied-item fixture.

## Production and artifact

Only after strict evidence selects an immutable config:

1. refit on all 285210 development rows without scoring sealed gold;
2. build the offline archive once;
3. verify exact bytes and SHA-256;
4. run organizer-shaped runtime Check;
5. upload the exact archive to private Hugging Face only;
6. re-download and require byte-identical SHA;
7. update Memora and this experiment record with every accepted and rejected run.

No GitHub Release is used for final transport.
