"""Compatibility hook for immutable explicit split executors.

The historical v14/v15 private GPU executors replace ``_build_immutable_manifest``
with a fail-closed implementation backed by the canonical row map before calling
it.  Keeping the unpatched public hook fail-closed prevents accidental fallback
to a regenerated split while preserving exact-source reproducibility.
"""

from __future__ import annotations


def _build_immutable_manifest(*args, **kwargs):
    raise RuntimeError(
        "immutable manifest builder is not installed; executor must install the "
        "verified canonical row-map adapter before training"
    )
