import sys
from pathlib import Path

from aios_track2.deck import parse_deck_text
from aios_track2.opm import FlowRequest, run_flow


def test_parse_deck_text_finds_dimensions_and_wells() -> None:
    text = """RUNSPEC
DIMENS
 3 2 1 /
SCHEDULE
WELSPECS
 'P1' 'G' 1 1 1* 'OIL' /
 'I1' 'G' 3 2 1* 'WATER' /
/
END
"""
    metadata = parse_deck_text(text)
    assert metadata.dimensions == (3, 2, 1)
    assert [(w.name, w.i, w.j, w.phase) for w in metadata.wells] == [
        ("I1", 3, 2, "WATER"),
        ("P1", 1, 1, "OIL"),
    ]


def test_opm_adapter_hashes_output_and_never_uses_shell(tmp_path: Path) -> None:
    fake = tmp_path / "fake_flow.py"
    fake.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "out=Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)\n"
        "(out/'CASE.UNSMRY').write_bytes(b'fixture')\n"
        "print('OPM FLOW FIXTURE OK')\n",
        encoding="utf-8",
    )
    deck = tmp_path / "CASE.DATA"
    deck.write_text("RUNSPEC\nEND\n", encoding="utf-8")
    result = run_flow(
        FlowRequest(
            deck=deck,
            output_dir=tmp_path / "out",
            executable=(sys.executable, str(fake)),
            timeout_seconds=10,
        )
    )
    assert result.status == "success"
    assert len(result.stdout_sha256) == 64
    assert any(p.name == "CASE.UNSMRY" for p in result.output_files)
