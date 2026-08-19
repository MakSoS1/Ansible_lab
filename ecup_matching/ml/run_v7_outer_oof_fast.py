from __future__ import annotations

from . import run_v7_outer_oof as base


def _load_model_no_checkpoint(model_path: str, *, last_n_layers: int, device: str):
    """Load the exact v7 model with checkpoint recomputation disabled.

    RTX 2060 benchmark 31548592806 measured 106.81 examples/s at physical
    batch 32 versus 84.02 examples/s with gradient checkpointing, while peak
    allocated VRAM remained ~4.37 GB on the 8 GiB card. The scoring path is
    unchanged; this only removes training-time activation recomputation.
    """
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    base.configure_trainable_layers(model, last_n_layers=last_n_layers)
    return model.to(device)


def main() -> int:
    base._load_model = _load_model_no_checkpoint
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())