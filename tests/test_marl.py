from aios_track2.marl import ReservoirEnv, SpySurrogate


def test_reward_uses_npv_delta_and_constraint_cost() -> None:
    env = ReservoirEnv()
    _, reward, _, _, info = env.step(env.safe_action())
    assert reward == info["npv_delta"] - info["constraint_cost"] - info["uncertainty_cost"]


def test_invalid_action_is_projected_before_surrogate_call() -> None:
    spy = SpySurrogate()
    env = ReservoirEnv(surrogate=spy)
    env.step(env.action_with_wlpr(700.0))
    assert spy.last_batch is not None
    assert spy.last_batch.controls.max_wlpr <= 500.0
