# V15 architecture pivot — durable memory note

Updated: 2026-08-16

The v15 pivot is approved and implementation has started. The canonical detailed design is `docs/superpowers/specs/2026-08-16-ecup-v15-field-aware-distillation-design.md` and the task-by-task implementation plan is `docs/superpowers/plans/2026-08-16-ecup-v15-field-aware-distillation.md`.

Key decision: the pure item-centric/LateInteraction v14 family remains valid research and existing jobs such as A17 are allowed to finish, but it is no longer the default path to the `0.50` Public-LB objective unless it produces exceptional new evidence. The primary v15 path is one field-aware full pair CrossEncoder with explicit deterministic attribute evidence and stronger offline supervision/distillation. The final runtime stays one model.

Repository rule: `Ansible_lab` is the canonical architecture/research/Memora source. `gpu-dispatch` is an execution plane bound to immutable public source SHAs and job manifests; private executor docs must not silently redefine the architecture.
