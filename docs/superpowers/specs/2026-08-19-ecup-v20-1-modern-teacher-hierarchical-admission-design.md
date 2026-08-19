# E-CUP v20.1 Modern Teacher + Hierarchical Admission Design

**Status:** approved by user on 2026-08-19

## Goal
Finish v20 as a stronger submission pipeline without changing the production runtime family: keep one RuBERT pair CrossEncoder at inference, but improve the quality and causal validity of generated supervision by selecting modern Russian-capable teachers empirically and replacing statistically underpowered fine-stratum Wilson admission with hierarchical calibration.

## Constraints
- Production runtime remains one `ai-forever/ruBert-base` pair CrossEncoder, max length 256, one final safetensors checkpoint, no network.
- RTX 2060 SUPER 8 GB is the canonical research GPU.
- Immutable split SHA remains `aae58fb40f7cd481995bfa46b8bc5602134ad8779efb939a68a0ea0fbabeb55b`.
- Sealed 80,444-row gold is never opened or scored.
- Generated candidates must exclude every human-labelled item; the proxy must additionally exclude every historical weak-labelled item.
- Public anchor order used to validate the proxy remains `v14 > v12 > v13B > v7`.
- Teacher outputs are training-only; teacher weights never enter the final ZIP.
- A candidate cannot promote unless it beats the calibrated v14 proxy anchor and passes all existing human/proxy/tail/category gates.

## Why v20.0 needs this revision
1. The old teachers (`Granite-3.3-2B-Instruct` and `SmolLM2-1.7B-Instruct`) are not strong choices for Russian product cards.
2. The v20.0 admission key `category × reason × difficulty` fragments support. At a 95% Wilson lower-bound floor of 0.995, even 100/100 correct examples are insufficient, so many useful groups are mathematically unable to pass despite perfect observed precision.
3. The old pipeline labels the combined audit + candidate queue before proving which teacher pair and which groups are reliable, wasting GPU inference.

## Teacher candidate pool
The candidate pool is deliberately heterogeneous and open/commercially usable:

1. `Qwen/Qwen3.5-4B`
   - Apache-2.0.
   - 2026-generation model, 201-language coverage.
   - Primary modern multilingual candidate.
   - Must run quantized on 8 GB; full precision is not assumed to fit.

2. `google/gemma-4-E2B-it`
   - Apache-2.0.
   - 2026-generation model, 140+ languages.
   - Uses an official/traceable Q4 path for the 8 GB GPU.

3. `utter-project/EuroLLM-1.7B-Instruct`
   - Apache-2.0.
   - Explicit Russian support and independent Llama-family teacher.
   - Full-precision fallback/reference because it fits the GPU.

4. `ai-forever/FRED-T5-1.7B`
   - Apache-2.0, Russian-only control.
   - Not eligible as a production teacher unless it passes the same JSON-validity and precision gates; its purpose is to measure whether Russian-only pretraining adds useful conflict signal.

`YandexGPT-5-Lite-8B-instruct` is excluded from the canonical pipeline despite strong Russian capability because its custom license defines models created from its outputs as derivative works and therefore adds an avoidable downstream licensing dependency.

Qwen3.6 27B / 35B-A3B and Gemma 4 models larger than E2B are excluded from the 8 GB canonical bakeoff because total model storage/VRAM is not compatible with the runner budget. No official open-weight Qwen3.7/3.8 candidate is assumed.

## D3A: teacher bakeoff on human audit only
For each outer fold, reuse the component-disjoint human audit slice and do not expose generated candidates yet.

Each teacher labels a deterministic, stratified audit subset capped at 4,000 rows per fold, with explicit coverage of:
- positive identity matches;
- negative model conflicts;
- capacity/size/pack conflicts;
- accessory/main-product confusion;
- sparse/other cases;
- categories with historically weak local→LB transfer.

Every teacher emits the same normalized `TeacherDecision` schema. Backend-specific raw text is retained separately; the normalized record contains exact model ID, resolved revision, backend, quantization, prompt SHA, latency and peak VRAM metadata.

## Teacher scoring and pair selection
Per teacher compute:
- valid JSON rate;
- coverage rate (non-UNCERTAIN normalized decisions);
- positive precision and 95% Wilson LCB;
- negative precision and 95% Wilson LCB;
- critical-conflict precision and 95% Wilson LCB;
- reason-code agreement with the deterministic checker;
- macro precision across categories/reasons;
- median rows/sec and peak VRAM.

Hard teacher eligibility:
- JSON valid rate >= 0.98;
- coverage >= 0.70;
- positive precision LCB >= 0.94;
- negative precision LCB >= 0.97;
- critical-conflict precision LCB >= 0.95;
- peak VRAM <= 7.75 GiB;
- no audit/human item leakage.

Pair selection is not simply “two best scores.” Candidate pairs must be different model families and exact revisions. Among eligible pairs, rank by consensus audit precision first, then critical precision, then coverage, then throughput. The selected pair must have two-fold evidence; if no pair is eligible, generated-supervision axes fail closed and v20 falls back to the historical control instead of packaging a speculative model.

## D3B: hierarchical calibration
Calibration no longer admits the ultra-fine `category × reason × difficulty` bucket directly.

For two-teacher consensus rows, compute independent Wilson gates at these levels:
1. predicted-label gate: MATCH and NON_MATCH precision;
2. reason-family gate: one gate per reason code, pooling categories/difficulties;
3. category gate: pooled category reliability;
4. critical-family gate: pooled rows whose reason is in the critical conflict set.

A candidate row is admissible only when:
- both selected teachers agree on verdict and reason code;
- neither emits UNCERTAIN;
- deterministic checker does not contradict the reason;
- predicted-label gate passes;
- its reason-family gate passes;
- its category gate passes;
- critical-family gate passes when applicable.

Frozen Wilson floors remain strict:
- MATCH LCB >= 0.985;
- NON_MATCH LCB >= 0.995;
- category LCB >= 0.970;
- critical-family LCB >= 0.950.

Support is now pooled where the reliability claim is actually made. The implementation must record both observed precision and support; no lowering of floors is allowed to make a teacher pass.

## Staged candidate labeling
After D3A/D3B:
1. Build target-free generated candidates as in v20.0.
2. Pre-filter candidates to reason/category groups whose audit gates can possibly pass.
3. Label only this bounded candidate subset with the selected pair.
4. Apply the fold0 and fold1 hierarchical policies separately.
5. Production generated labels are the intersection of both fold policies, with reliability equal to the minimum supporting empirical reliability across folds.

This reduces LLM calls while increasing the expected precision of accepted labels.

## Model experiments
The causal model ladder is unchanged:
- `control`: historical human + quality-aware historical weak.
- `data-only`: control + admitted generated labels.
- `rationale`: data-only + existing reason auxiliary heads/loss.
- `replay-data` or `replay-rationale`: source-aware anti-forgetting replay for whichever stage-1 model wins.
- scaled confirmation: exact selected mode repeated at the 3M→1.5M × 1.0 weak scale on folds 0 and 1.

No new mechanism is silently enabled during scaling.

## Promotion
Keep existing gates:
- proxy gain strictly > 0.005 vs control;
- human delta >= -0.003 per fold;
- audited-tail delta >= -0.02;
- worst-category delta >= -0.04;
- selected candidate proxy score strictly exceeds v14 on the same calibrated proxy;
- both scaled folds pass and mean human delta >= 0.

Only then run D10 production refit and exact organizer ZIP check.

## Failure policy
- Teacher OOM/unsupported backend: that teacher is marked ineligible; other candidates continue.
- JSON/schema failure: row invalid; high invalid rate disqualifies teacher.
- No eligible pair: generated supervision fails closed; do not pretend v20 improved.
- No admitted hierarchical groups: fail generated axes closed.
- Proxy anchor ordering not reproduced: proxy axis is not promotable.
- Gold remains sealed in all failure modes.

## Deliverables
- updated v20 policy and hierarchical admission tests;
- teacher candidate/bakeoff selection modules;
- backend-agnostic normalized teacher inference contract;
- private runner campaign that performs bakeoff → calibration → staged candidate labeling → causal ladder → two-fold scaled confirmation → production/package;
- final `ecup-v20-*.zip` only when all gates pass;
- manifests containing exact teacher model IDs/revisions/backend/quantization, audit metrics, admission policy hashes, source SHA and final archive SHA-256.
