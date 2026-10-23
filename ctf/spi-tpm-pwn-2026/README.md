# HITCON CTF 2026 — SPI TPM (Pwn)

## Status — unsolved; actual Pwn flag not recovered

This is not the SPI TPM Free Trial (Reverse Engineering) challenge.
The Free Trial flag previously submitted does not belong to this Pwn task.

Challenge: SPI TPM, author giver, HITCON 2026
Official service: spi-tpm-pwn.chal.hitconctf.com:1337
Archive: https://ctf2026attach.hitconctf.com/d72ad1da709d0aca3b881f11a651420a/spi-tpm-pwn-db1d27f883a2e5e8.tar.gz
SHA256: db1d27f883a2e5e8f73c2b1c6b4b77e0ebfe14642676cce6cadfca4c9d9e173d

## Verified research

- [Archive inspection](https://github.com/MakSoS1/Ansible_lab/actions/runs/39996041268): audited 12 members, no binaries executed on runner
- [Rootfs static analysis](https://github.com/MakSoS1/Ansible_lab/actions/runs/39996164574): ARM64 kernel, initramfs, client, DMA modules and user helpers
- [Local sandbox boot](https://github.com/MakSoS1/Ansible_lab/actions/runs/39996780348): synthetic fake flag, network-disabled, read-only Docker, all Linux caps dropped
- [ARM64 privilege-drop / kernel module decompilation](https://github.com/MakSoS1/Ansible_lab/actions/runs/39997194943)
- [Corrected QEMU DMA/SRAM/SPI static decompilation](https://github.com/MakSoS1/Ansible_lab/actions/runs/39999976209)
- [Official Pwn TCP connectivity and successful scrypt PoW](https://github.com/MakSoS1/Ansible_lab/actions/runs/40000421264): real challenge shell started after blank payload

Local guest shell: UID/GID 65534, no effective/ambient Linux capabilities, NoNewPrivs=1.
Root-owned /flag has file mode 0400 and cannot be read as UID 65534.
The /dev/spi-tpm0 device node is intentionally chowned to UID 65534 and chmod 0600
before dropping privileges, allowing the user to issue legitimate SPI ioctl calls.
The unprivileged shell inherits only file descriptors 0, 1, and 2.

Local standalone SPI test passed: SPI_TPM_TEST_PASS key=1 ciphertext_len=64.
All archive binaries were executed only in isolated, network-disabled Docker.

## Attack surface

The client driver accepts kernel ioctls 0xc0184301 (genkey),
0xc0384302 (encrypt) and 0xc0384303 (decrypt).
It copies user input, validates length ranges, allocates two 0x2004-byte
kernel TX/RX buffers and uses a synchronous SPI transfer.

The custom QEMU emulator has a 3-channel spi-tpm-dma MMIO controller,
8,192-byte MMIO SRAM pool and STM32H7 SPI peripheral.
The emulator reads up to 256 guest DMA scatter-gather descriptors,
validates aggregate chunk size, handles TX and RX SG copies and
supports delayed RX/MDMA staged transfers.
No reliable arbitrary memory read/write primitive has yet been demonstrated.

Custom QEMU ELF SHA256:
6291b5b075f7e0cedcff4e46cb135f0b8d6e93d94c5973784c034c16eba662bb

Ghidra PIE image addresses require +0x100000 compared to objdump:
- QEMU DMA start: 0x3ae5b0
- DMA completion callback: 0x3af000
- TX descriptor handling: 0x3af1e0
- RX/MDMA transfer: 0x3af350
- STM32H7 delayed-transfer callback: 0x3b0850
- STM32H7 SPI start/stop: 0x3b0c10

Potential bug areas include asynchronous DMA cancellation, descriptor ownership
and lifetime, deferred RX-buffer state and MDMA length validation.
They are hypotheses requiring verification with the synthetic flag,
not confirmed exploits.

## Reproduce

- ctf/spi-tpm-pwn-2026/connect.py: target-allowlisted proof-of-work client
- .github/workflows/spi-tpm-pwn-guest.yml: offline Docker boot and controlled PTY
- .github/workflows/spi-tpm-pwn-ghidra.yml: ARM64 guest module decompilation
- .github/workflows/spi-tpm-pwn-qemu-ghidra.yml: corrected custom QEMU decompilation
- .github/workflows/spi-tpm-pwn-initramfs.yml: trusted static inspection

No changes to main; explicit contents:read GitHub workflow permissions.
Archive downloads limited to supplied CTF HTTPS endpoint, no redirects and
SHA256 verified. The supplied Dockerfile is not run (it disables
package signature checks); an independently written signed-package
Arch container is used instead.

## Remaining

Develop, locally test and reproduce a reliable privilege escalation to
read root-only /flag, then deploy the exploit against the official
spi-tpm-pwn.chal.hitconctf.com:1337 service. Capture and verify the
non-redacted flag for this exact challenge.

**A working exploit and the Pwn flag have NOT been obtained yet.**
