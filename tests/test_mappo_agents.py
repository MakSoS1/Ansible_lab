import torch

from aios_track2.agents import AgentOrchestrator, DecisionState
from aios_track2.marl import SharedGraphMAPPO


def test_shared_graph_mappo_shapes_and_loss() -> None:
    adjacency = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    model = SharedGraphMAPPO(obs_dim=5, action_dim=2, adjacency=adjacency, hidden=16)
    obs = torch.randn(4, 3, 5)
    action, logp, value = model.act(obs)
    assert action.shape == (4, 3, 2)
    assert logp.shape == (4, 3)
    assert value.shape == (4,)
    loss = model.ppo_loss(
        obs,
        action.detach(),
        logp.detach(),
        torch.randn(4),
        torch.randn(4),
    )
    loss.backward()
    assert torch.isfinite(loss)


def test_orchestrator_cannot_publish_without_opm_validation() -> None:
    orchestrator = AgentOrchestrator()
    state = DecisionState(candidate_id="c1", surrogate_npv=12.5, uncertainty=0.2)
    state = orchestrator.plan(state)
    state = orchestrator.guard(state, accepted=True)
    state = orchestrator.record_opm(state, opm_npv=None)
    assert state.publishable is False
    state = orchestrator.record_opm(state, opm_npv=11.9)
    assert state.publishable is True
    assert state.opm_npv == 11.9


def test_guard_blocks_opm_promotion_for_invalid_candidate() -> None:
    orchestrator = AgentOrchestrator()
    state = DecisionState(candidate_id="bad", surrogate_npv=99.0, uncertainty=0.01)
    state = orchestrator.guard(state, accepted=False)
    state = orchestrator.record_opm(state, opm_npv=99.0)
    assert state.publishable is False
    assert state.status == "rejected"
