# E-CUP Matching — v13 results

Updated: 2026-08-15
Status: **B/groupweak submission-ready for next Public-LB test; external score pending**

## External context

- v7 Public LB: `0.3655833314`.
- v12 Public LB: `0.3798116204`.
- Correct delta: `+0.0142282890`.
- v12 remains the best observed external result until v13 is actually scored.

## Distribution measurement — KEEP as research evidence

Run `31788445849` measured canonical human/weak pools.

- human rows `365,654`;
- weak rows `11,187,780`;
- human binary prevalence `0.2567727961406138`;
- weak soft-target mean `0.2435606726464766`.

The main shift is retrieval structure/degree/hardness, not simple prevalence. Historical single global `RETRIEVAL_PREVALENCE_RATIO` is not a binding validation target.

## Historical v11 Check forensic — confirms runtime failure mode

Run `31789001358`, exact historical archive, organizer image, 1,000 spread pairs and full canonical `items.parquet` (`4,104,103,411` bytes): wall reached `60.033 s`, timed out, no valid output. This validates the rule that fixed startup/item scanning can be fatal.

## Graph post-processing — REJECT

Predeclared graph variants were evaluated on aligned v7/v8/v12 fold0 predictions. Best `mutual_conservative` deltas:

- v7 `-0.00018494985048111978`;
- v8 `+0.00010880404848689906`;
- v12 `-0.00014802779228761942`.

Runtime was only ~`1.29 s / 275k`, so rejection is based on quality stability, not speed.

## B / groupweak — KEEP as next external candidate

B preserves complete original weak retrieval-anchor groups and orientation while keeping v12 weak exposure, human phase and single-CrossEncoder runtime unchanged.

Probe run `31791177120`:

- v13 B fold0 **`0.7086611385531062`**;
- v12 fold0 `0.7059297810308699`;
- delta **`+0.0027313575222363`**.

Frozen Validation-v3 for the packaged B candidate:

- p05 **`0.5690974845`**;
- mean **`0.6869505675`**.

C2/ListNet equal-exposure testing was rejected because it did not beat B on the frozen diagnostics. Later all-soft/listwise work remains research evidence; it did not replace B as the next Public-LB candidate.

## Production/package evidence

Production refit run: `31828844182`.

Package run: `31829720888`.

Exact source used by packager: `4e83294eb5f6c31c720f7cbb0220f0f4d0ee3cb1`.

Model/runtime metadata:

- base model `ai-forever/ruBert-base`;
- base revision `43be4261797042e172adf7476c558734f3cbb2a0`;
- development/training rows `285,210`;
- weak epochs `0.35`;
- human epochs `1.0`;
- max length `256`;
- inference batch size `64`;
- one checkpoint;
- sealed gold unopened.

Artifact:

- filename `ecup-v13-groupweak-v7runtime-submission.zip`;
- bytes `663760087`;
- SHA-256 `f4b7aad36c8d293a3939d9fb2ce7f91cff1bd8381c870015b2f16ea65a17badb`;
- model SHA-256 `9ae7676f96818a367eb348f8648d503b56c86e3d0c62f665f030b4c29bcde0a5`.

## Binding organizer-shaped Check — PASS

Supplied-item subset fixture:

- 1,000 pairs;
- 1,999 unique/materialized items;
- item fixture `615,549` bytes;
- ZIP extraction `2.9942763 s`;
- wall `26.1353473 s`;
- limit `60 s`;
- return code `0`;
- output valid;
- `881` unique scores;
- runtime acceptance `accepted=true`.

Internal inference completed in ~`9.48 s`; the reported wall time includes extraction/container/output validation.

## Conservative full-item diagnostic — timeout, non-binding

The packaging workflow also ran the same 1,000-pair fixture while exposing the entire canonical item universe (`4,104,103,411` bytes). It reached `60.0049954 s`, timed out and produced no valid output.

This test is explicitly marked `full_item_stress_is_diagnostic_only=true` and `stress_semantics=stricter than closed-test subset contract`. It must remain documented as residual risk but does not overturn the binding subset Check acceptance.

## Private-HF publication — PASS

Upload/round-trip run `31843423348`:

- private repo `Maksim123321/e-cup-2026-matching-private`;
- path `submissions/v13/candidates/b-groupweak/ecup-v13-groupweak-v7runtime-submission.zip`;
- exact expected bytes/SHA checked before upload;
- candidate downloaded back as `canonical.zip`;
- `canonical.zip: OK`;
- `V13_CANDIDATE_HF_ROUNDTRIP_VERIFIED`;
- one-time credential material destroyed after the transfer.

## Final interpretation

The archive is **submission-ready for the next Public-LB test**. It has strong packaging, runtime and transport-integrity evidence. It is **not** yet a strict final keeper and no Public-LB quality gain may be claimed before ODS returns the actual score.
