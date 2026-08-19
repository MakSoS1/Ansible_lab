# Hardened Memora Security Policy for E-CUP

## Supported profile

Only this configuration is supported:

- upstream commit `bc64ff745a9b2c0e6245e0137654f041fba0c155`;
- hardened build produced by `tools/memora_hardened/`;
- `mcp>=1,<2`;
- local SQLite only;
- `MEMORA_EMBEDDING_MODEL=tfidf`;
- `MEMORA_LLM_ENABLED=false`;
- `MEMORA_AUTO_CAPTURE=false`;
- `memora-server --no-graph` via `scripts/memora_mcp.sh`;
- durable checkpoint only via private HF scripts.

## Explicitly unsupported / disabled

Do not enable or deploy:

- Cloudflare Pages Memora Graph/API;
- Cloudflare Worker broadcast/WebSocket;
- local interactive graph UI;
- Memora D1/S3/R2 backend;
- OpenAI/OpenRouter embeddings, chat, dedup or rewrite;
- arbitrary external LLM backends;
- automatic capture of shell/test/web output;
- server binding to `0.0.0.0`.

These surfaces are unnecessary for E-CUP and correspond to the highest-risk findings in the upstream audit: unauthenticated cloud API, stored XSS/CDN dependencies, unauthenticated worker broadcast, and `.mcp.json`/`eval` sync path.

## Hardening performed during install

The source transformer must fail if its exact expected upstream anchors do not match. It:

1. constrains `mcp` to `<2`;
2. changes embedding default to `tfidf`;
3. changes LLM default to disabled;
4. redacts detected secrets in validated content before persistence;
5. recursively redacts metadata strings and redacts tags;
6. routes batch-add content through the same validator;
7. applies directory mode `0700` and SQLite mode `0600`;
8. deletes `memora-graph/` from the hardened source;
9. replaces packaged graph HTML with an inert disabled page.

The safe launcher strips inherited external-service credentials and storage configuration before exec.

## Secret policy

Never persist:

- HF/GitHub/API tokens;
- OpenAI/OpenRouter/Anthropic keys;
- AWS/R2/Cloudflare credentials;
- passwords;
- private keys;
- bearer tokens;
- Slack-style tokens;
- payment-card-like values caught by the configured scanner.

Memora input redaction is one layer. `memory_checkpoint.py` is a second fail-closed layer: it scans the consistent SQLite backup before upload. A probable secret means no checkpoint is uploaded.

## Filesystem policy

- `.agent-memory/`: mode `0700`.
- `.agent-memory/memories.db`: mode `0600`.
- cache/runtime state must not be committed.
- bootstrap rejects a database destination that is a symlink.
- checkpoint rejects a source DB that is not a regular file or that fails `PRAGMA integrity_check`.

## Public/private boundary

Public Git may contain source code, aggregate metrics, research summaries and reproducibility instructions. It must not contain raw competition parquet, model weights, submission ZIPs, persistent memory DBs or secrets.

Private HF stores binary artifacts and memory checkpoints. HF access token is obtained only from environment / GitHub Actions secret `HF_TOKEN`; code must never print the token.

## Updating Memora

Do not silently change the pinned commit. A future update requires:

1. explicit new commit pin;
2. re-review of security-sensitive files;
3. hardening-anchor update;
4. upstream + E-CUP security tests;
5. new runtime manifest and checkpoint verification.
