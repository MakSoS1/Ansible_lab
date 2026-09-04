from __future__ import annotations

FIELD_SUMMARY_VECTORS = (
    "FOPR",
    "FWPR",
    "FWIR",
    "FLPR",
    "FOPT",
    "FWPT",
    "FWIT",
    "FLPT",
    "FPR",
)

WELL_SUMMARY_VECTORS = (
    "WOPR",
    "WWPR",
    "WWIR",
    "WLPR",
    "WOPT",
    "WWPT",
    "WWIT",
    "WLPT",
    "WBHP",
    "WTHP",
    "WWCT",
)


def build_training_summary() -> str:
    """Return SUMMARY-section contents used for real OPM surrogate training.

    Field vectors need no selector. An empty selector record for a well vector
    asks OPM/ECLIPSE to emit that vector for all declared wells. TIME and YEARS
    are explicitly terminated so the following mnemonic cannot be parsed as
    data for the preceding keyword. This changes telemetry only; it does not
    alter reservoir physics or the SCHEDULE.
    """
    lines = [
        "-- AIOS training telemetry: output requests only; no physics/control changes",
        "TIME",
        "/",
        "YEARS",
        "/",
        *FIELD_SUMMARY_VECTORS,
    ]
    for keyword in WELL_SUMMARY_VECTORS:
        lines.extend((keyword, "/"))
    return "\n".join(lines) + "\n"
