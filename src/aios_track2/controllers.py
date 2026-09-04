from __future__ import annotations

from datetime import date

from aios_track2.deck import WellGraph
from aios_track2.schedule import ConstraintSet, Control, Schedule, WellRole, project_schedule


def heuristic_schedule(
    graph: WellGraph,
    dates: tuple[date, ...] | None = None,
    liquid: float = 90.0,
    injection: float = 110.0,
) -> Schedule:
    dates = dates or (date(2007, 1, 1), date(2007, 4, 1), date(2007, 7, 1), date(2007, 10, 1))
    controls: list[Control] = []
    for knot_index, knot in enumerate(dates):
        for well in graph.wells:
            injector = well.phase.upper() in {"WATER", "INJ"} or well.name.startswith("I")
            # Keep a small unused reserve: every 8th producer is shut to cut water-cut risk.
            shut = (not injector) and (int(well.name[-1]) % 8 == 0 if well.name[-1].isdigit() else False)
            if injector:
                factor = 1.05 if knot_index else 1.0
                controls.append(
                    Control(
                        date=knot,
                        well=well.name,
                        status="OPEN",
                        role=WellRole.INJECTOR,
                        wwir=injection * factor,
                    )
                )
            else:
                factor = 0.92 if knot.month >= 7 else 1.0
                controls.append(
                    Control(
                        date=knot,
                        well=well.name,
                        status="SHUT" if shut else "OPEN",
                        role=WellRole.PRODUCER,
                        wlpr=0.0 if shut else liquid * factor,
                    )
                )
    projected = project_schedule(
        Schedule(controls=tuple(controls)),
        ConstraintSet(known_wells=frozenset(well.name for well in graph.wells)),
    )
    if not projected.accepted:
        raise RuntimeError(projected.violations)
    return projected.projected
