# HITCON CTF 2026 — SPI TPM Free Trial

**Status: REVERSE ENGINEERED IN PART; REAL FLAG NOT RECOVERED.**

The provided archive contains a sample hitcon{REDACTED-2} flag, and our local isolated test uses hitcon{LOCAL_FAKE_FLAG_FOR_TESTS}. Neither is a competition solution.

## Validated chain of evidence

- HITCON archive SHA-256: e15a21887cc925818a5bdb0e6b669028bc30c94bbfc1fe0a06f4c5a4c543ad94. Thirteen archive members audited, no unsafe paths, links, or non-regular files.
- The challenge was booted in Docker with network disabled, read-only filesystem, no Linux capabilities, no-new-privileges, bounded resources, fake flag, and no access to GitHub secrets.
- Guest banner: SPI-TPM reverse service / budget 1024 / queries 27 / ready.
- Local test: get_flag, genkey, encrypt 2 1 41424344, decrypt using the returned ciphertext, budget, quit. Result: plaintext 41424344; budget 734 queries 23.
- Ghidra 12.1.3 / PyGhidra 3.2.0 decompilation performed on verified challenge x86-64 firmware, ARM64 service, ARM64 kernel client and DMA driver. Ghidra ZIP SHA-256: 93a5d11a9ad510622acaaf908c556a7b9b764d338e78a7567f3689bf5081fd54.
- Firmware SHA-256: 1bd76fdd713d7c1003e27e418813e0444c3a2a780bf895836f093048b5cc7a40.
- GitHub-hosted Actions VM resolves spi-tpm-rev.chal.hitconctf.com to 35.236.129.118, but TCP connection to 1337 has timed out repeatedly. This prevents reading the **real** sealed flag remotely; remote service status is not proven.

## Protocol and technical findings

An ARM64 service manages a SPI cryptography accelerator implemented in a custom QEMU firmware shared library. It generates a random key in slot 1, reads and seals /flag, erases/unlinks the plaintext, then drops UID/GID to nobody (65534) and all privileges before accepting user commands.

    get_flag
    genkey
    encrypt KEY COST HEX
    decrypt HEX
    budget
    quit

The flag response is a sealed STP1 version-2 ciphertext, not plaintext. The 40-byte STP1 header includes a 32-bit nonce, a 32-bit cost, a 32-bit plaintext length, a 32-bit key-routing fingerprint, and a 16-byte authentication tag. Fields use little-endian integer representation. Ciphertext payload is padded to a 4-byte boundary.

The firmware encryption uses a public reversible scrambling step, a secret-key-dependent 32-bit Feistel network whose round count equals COST, and a keyed authentication tag protecting ciphertext and header. The service allows 27 queries and 1024 total budget units.

The flag encryption is performed with COST = 0x85fb3 = 548787 *before* the interactive service starts. Decrypt charges COST * (ceil(length / 4) + 256). Our local test flag had length 34, so ordinary decrypt would cost 548787 * (9 + 256) = 145428555, far beyond 1024. Modifying the cost field in the returned ciphertext invalidates the keyed authentication tag.

**Decompiler caution:** an earlier Ghidra pseudocode expression indicated a zero plaintext word in one encryption subexpression. The verified x86-64 machine code loads the actual plaintext word at 0x29c2; this was a decompilation artifact, NOT a proven cryptographic vulnerability.

## Supply chain and prompt injection precautions

All archive contents, including descriptions, comments and scripts, were treated as untrusted bytes. No instructions embedded in the challenge were executed on their own authority. Downloads are limited to the user-provided exact HTTPS host/path, no redirects, bounded size and time, and verified against the archive SHA-256.

The challenge-provided Dockerfile and docker-compose.yml were NOT used. Instead, our independently written Dockerfile uses the official Arch base image and normal package signature verification. The challenge-provided QEMU and run.sh are executed ONLY inside our heavily restricted Docker sandbox with networking entirely disabled and a fake flag mounted read-only.

Workflows explicitly use contents: read and are confined to the separate ctf/spi-tpm-2026 branch. The main branch has not been modified.

## Reproduction and logs

- [Safe archive audit](https://github.com/MakSoS1/Ansible_lab/actions/runs/39968025860): passed.
- [Static reverse engineering via official Ghidra](https://github.com/MakSoS1/Ansible_lab/actions/runs/39970780963): passed.
- [Exact firmware x86 disassembly](https://github.com/MakSoS1/Ansible_lab/actions/runs/39971511357): passed.
- [Sandboxed ARM64 guest full local encrypt/decrypt exchange](https://github.com/MakSoS1/Ansible_lab/actions/runs/39971743862): passed.
- [Remote TCP connectivity check](https://github.com/MakSoS1/Ansible_lab/actions/runs/39968920562): TCP timed out.

Workflows: .github/workflows/spi-tpm-2026.yml; spi-tpm-static.yml; spi-tpm-pyghidra.yml; spi-tpm-asm.yml; spi-tpm-local.yml.

## What is not solved

The actual flag has not been retrieved. To solve, restore TCP connectivity to the official competition service, complete its documented scrypt PoW, find and prove a locally workable exploit or cryptanalytic shortcut that bypasses the cost bound, then obtain a valid non-redacted hitcon{...} result from the remote service.

**Do not report this work as a completed flag solve.**
