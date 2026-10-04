from api.ws import _register_player, assign_coach, public_players


def _session():
    return {
        "dance_pack": {"duration_ms": 1000, "weights": [], "events": {"events": []}},
        "coach_count": 2,
        "coach_tracks": [
            {"coach_index": 0, "frames": [], "weights": [], "events": []},
            {"coach_index": 1, "frames": [], "weights": [], "events": []},
        ],
        "reference_loaded": True,
        "player_engines": {},
        "players": {},
        "assignments": {},
        "next_player_slot": 0,
        "state": "calibrated",
    }


def test_multiple_players_get_stable_separate_slots():
    session = _session()
    p0 = _register_player(session, "p17", 0.91)
    p1 = _register_player(session, "p42", 0.88)
    assert p0["slot"] == 0
    assert p1["slot"] == 1
    assert session["assignments"]["p17"] == 0
    assert session["assignments"]["p42"] == 1

    # Seeing the same person again must not allocate another player.
    again = _register_player(session, "p17", 0.95)
    assert again["slot"] == 0
    assert len(session["players"]) == 2


def test_each_player_can_select_a_different_reference_coach():
    session = _session()
    _register_player(session, "p0", 0.9)
    _register_player(session, "p1", 0.9)

    assign_coach(session, "p0", 1)
    assign_coach(session, "p1", 0)

    players = public_players(session)
    assert players[0]["coach_index"] == 1
    assert players[1]["coach_index"] == 0
