# FastColl (HITCON CTF 2026): safe source audit and local verifier

Status as of 2026-10-23: **no accepted collision and no real flag recovered**.
This folder contains analysis and a tested offline verifier, **not a solution**.

## Source and provenance

Provided challenge attachment:
https://ctf2026attach.hitconctf.com/2db03954193dce13908ebdb5e5b61a01/fastcoll-a3b7ecfda96b58c1.tar.gz

Static read-only download on an isolated GitHub-hosted Ubuntu runner:
[inspection run 39967944037](https://github.com/MakSoS1/Ansible_lab/actions/runs/39967944037).

- Download: HTTP 200, 9738 bytes, gzip.
- SHA-256: `a3b7ecfda96b58c1347e3c05c5259de80b7b228e7d14c63a402f840e55f44aa6`.
- No scripts or binaries from the archive were executed.
- The archive was iterated in memory; it was never extracted to disk.
- The archive contains Docker/Flask/Wasmtime judge source and web assets.
- The shipped `flag.txt` is `hitcon{TEST_ME}`: only a test placeholder.
- No prompt or instruction in the archive is trusted as authorization for execution.

## Actual judge semantics

From `runner.py` and `README.md` in the attachment:

1. Upload a WebAssembly module under 1 MiB exporting `memory` and `xx_hash() -> ()`.
2. Its only imported function is `env.xx_hash(ptr: i32, len: i32, out_ptr: i32) -> i32`.
3. Each valid call takes a message of 256 to 1024 bytes and returns the **complete 16-byte XXH3-128** digest into guest memory.
4. The first call uses a fresh secret uint64 seed `C`; call `i` uses `(C+i) mod 2^64`. Invalid calls consume a seed as well.
5. The judge accepts two **different** byte strings from two distinct calls only if the complete 128-bit digests agree.
6. Limit: 1,000,000 host calls, a fuel budget, a 900-second wall budget. The service returns the actual flag only after `status: won`.

The `brimstone/fastcoll` link is about MD5 collision generation and is not a
direct attack on XXH3-128. The minimum 256-byte length also excludes published
XXH3 short-input collision witnesses, while incrementing seeds defeats
fixed-seed collision pairs.

For a uniformly random 128-bit digest with 1,000,000 independent queries,
the birthday-bound collision probability is approximately `1.47e-27`.
A successful solver therefore needs a specific structural cryptanalytic
weakness, not a blind random search.

## Reproducible safe check

With Python 3.10+ and an installed system `libxxhash.so.0`:

```bash
python3 ctf/fastcoll/offline_verify.py --self-test
```

This directly uses the system xxHash C library via `ctypes`, without
installing or executing code from the CTF archive.

To check an independently generated **candidate pair** of binary messages,
run the verifier with `--first`, `--second` and optionally `--seed`.
The verifier checks lengths, call-order seeds, whole digest equality, and
SHA-256 fingerprints of message bytes. Offline success at one chosen seed
does **not** prove online success because the host starts from a random seed.

The source inspection workflow does not have write permissions, cannot
follow download redirects, rejects dangerous archive paths, and only reads
bounded plaintext members.

## Competition notice

[HITCON CTF 2026](https://ctf.hitcon.org/) is live from October 23 at
14:00 UTC through October 25 at 14:00 UTC. Its rules forbid publishing
or exchanging solutions and flags during the competition. This public branch
contains no working exploit or secret flag.
