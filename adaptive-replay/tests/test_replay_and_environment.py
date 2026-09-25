import numpy as np
import pytest

from src.environment import GoalSwitchGridWorld
from src.replay_buffer import ReplayBuffer
from src.train import Config, train


def transition(step: int) -> tuple[np.ndarray, int, float, np.ndarray, bool, int]:
    state = np.array([step, 0], dtype=np.float32)
    return state, 0, -0.01, state, False, step


def test_fifo_evicts_oldest_transition() -> None:
    buffer = ReplayBuffer(2, 2, "fifo", np.random.default_rng(0))
    for step in (1, 2, 3):
        buffer.add(*transition(step))
    assert [item["step"] for item in buffer.items] == [3, 2]
    assert buffer.eviction_count == 1


def test_uniform_reservoir_stays_bounded() -> None:
    buffer = ReplayBuffer(2, 2, "uniform", np.random.default_rng(0))
    for step in range(1, 20):
        buffer.add(*transition(step))
    assert len(buffer) == 2
    assert buffer.total_seen == 19


def test_aer_uses_selective_eviction_after_detection() -> None:
    buffer = ReplayBuffer(2, 2, "aer", np.random.default_rng(0))
    buffer.add(*transition(1))
    buffer.add(*transition(2))
    buffer.detected_change = True
    buffer.add(*transition(3))
    assert buffer.eviction_count == 1
    assert buffer.selective_eviction_count == 1


def test_dynamics_detector_catches_action_remapping() -> None:
    buffer = ReplayBuffer(600, 2, "aer", np.random.default_rng(0))
    directions = [np.array([-0.125, 0.0]), np.array([0.125, 0.0]), np.array([0.0, -0.125]), np.array([0.0, 0.125])]
    step = 0
    for _ in range(100):
        for action, delta in enumerate(directions):
            step += 1
            state = np.array([0.5, 0.5], dtype=np.float32)
            buffer.add(state, action, -0.01, state + delta, False, step)
    for _ in range(25):
        step += 1
        state = np.array([0.5, 0.5], dtype=np.float32)
        buffer.add(state, 0, -0.01, state + directions[3], False, step)
    assert buffer.detected_change
    assert buffer.detection_source == "dynamics"


def test_goal_switch_changes_goal_location() -> None:
    env = GoalSwitchGridWorld(change_type="goal_switch")
    env.set_phase("b")
    assert env.goal == env.goal_b


def test_action_flip_changes_action_effect() -> None:
    env = GoalSwitchGridWorld(change_type="action_flip")
    env.set_phase("b")
    state, _, _ = env.step(0)
    assert np.allclose(state, np.array([1.0, 0.125], dtype=np.float32))


def test_invalid_capacity_is_rejected_before_training() -> None:
    config = Config(method="aer", seed=0, switch_step=100, buffer_capacity=100)
    with pytest.raises(ValueError, match="buffer-capacity"):
        train(config)
