#!/usr/bin/env python3
import os
from pathlib import Path

from huggingface_hub import HfApi


def main():
    token = os.environ["HF_TOKEN"]
    github_env = Path(os.environ["GITHUB_ENV"])
    owner = HfApi(token=token).whoami()["name"]
    dataset_id = f"{owner}/aios-track2-runs"
    with github_env.open("a", encoding="utf-8") as stream:
        stream.write(f"HF_DATASET_ID={dataset_id}\n")
    print(f"HF dataset target: {dataset_id}")


if __name__ == "__main__":
    main()
