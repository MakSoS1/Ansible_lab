# E-CUP v10 distilled-fast implementation plan

1. Add test-first contracts for soft-target construction and student-only submission closure. Confirm RED before production code exists.
2. Implement `v10_distill.py` and v10 student training helpers; confirm targeted tests GREEN.
3. Benchmark pinned tiny student on RTX 2060 SUPER in the organizer image across sequence lengths/batches. Reject configurations that cannot plausibly meet the 250 s private-size gate.
4. Build honest five-fold component-disjoint OOF for the viable configuration. Never score sealed gold.
5. Add only fold-safe distillation / training-only weak-hard-negative augmentation and retain an addition only if complete OOF/target-stress improves.
6. Freeze quality configuration before production refit.
7. Refit selected student on all development rows, build minimal student-only ZIP, and assert forbidden heavyweight v8/v9 components are absent.
8. Run exact ZIP end-to-end on 115k and 275k, timing extraction + inference + output validation. Hard reject private wall >250 s.
9. Run full repository tests and memory policy.
10. Publish exact keeper + manifest to private HF under `submissions/v10/final/`, verify remote listing/SHA, update canonical docs, and run hardened Memora checkpoint.
