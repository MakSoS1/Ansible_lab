"""``python -m aios_track2.serve`` starts the control room on verified submission artifacts."""

from __future__ import annotations

from .cli import main as cli_main
import sys


def main() -> None:
    sys.argv = [sys.argv[0], "ui", *sys.argv[1:]]
    cli_main()


if __name__ == "__main__":
    main()
