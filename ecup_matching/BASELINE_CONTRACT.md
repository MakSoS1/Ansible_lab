# Official E-CUP Matching Submission Contract

Updated: 2026-08-15.

## Archive metadata and CLI

The organizer baseline uses `metadata.json` and the offline image `odsai/ecup26-matching-baseline:1.0`. Entrypoint accepts:

```text
--output_path <csv path>
--items_path <items parquet path>
--matches_path <matches parquet path>
```

Output must preserve input pair order and contain exactly:

```text
id1,id2,predict
```

`predict` is continuous numeric score/probability. Submission inference must not require network access.

## Current v13 runtime contract

The retained v13 submission runtime is deliberately simple:

- one `ai-forever/ruBert-base` pair CrossEncoder;
- one tokenizer;
- one `.safetensors` checkpoint;
- max sequence length `256`;
- inference batch size `64` with existing safe fallback behavior;
- no structured/TF-IDF/graph/second-model inference branch.

The exact packaged candidate was built from source commit `4e83294eb5f6c31c720f7cbb0220f0f4d0ee3cb1` in packaging run `31829720888`.

## Check semantics — do not conflate two tests

### Binding organizer-shaped supplied-item Check

For the v13 B candidate, the Check fixture contains 1,000 pairs and only the 1,999 referenced items, matching the supplied-item subset contract used by the packaging gate.

Evidence:

- ZIP extraction `2.9942763 s`;
- total wall `26.1353473 s / 60 s`;
- return code `0`;
- output valid;
- 1,000 rows in order;
- 881 unique prediction values;
- `accepted=true`.

This is the binding runtime acceptance used to mark the candidate ready for private-HF publication.

### Conservative full-item diagnostic

A separate diagnostic exposes the complete canonical `items.parquet` (`4,104,103,411` bytes) to the same 1,000-pair workload. It timed out at `60.0049954 s` and produced no valid output.

The workflow explicitly marks this `full_item_stress_is_diagnostic_only=true` and describes it as stricter than the closed-test subset contract. Keep this result as residual-risk evidence if organizer semantics ever change; do not report it as the binding Check result.

## Artifact identity

Current next Public-LB candidate:

- `ecup-v13-groupweak-v7runtime-submission.zip`;
- `663760087` bytes;
- SHA-256 `f4b7aad36c8d293a3939d9fb2ce7f91cff1bd8381c870015b2f16ea65a17badb`.

Private-HF upload run `31843423348` downloaded the exact candidate back and revalidated size and SHA (`canonical.zip: OK`). Therefore upload/storage corruption has been explicitly checked.

## Runtime lesson from v11

Historical v11 had no exact Check gate and used unrepresentative `pairs.head(N)` training fixtures. Forensic run `31789001358` showed that a full-item Check can hit `60.033 s` before valid output. Startup/item-scan cost is therefore a first-class failure mode. Future inference additions are rejected by default unless they independently clear the Check contract with substantial headroom.
