from __future__ import annotations

import numpy as np


class GoalSwitchGridWorld:
    CHANGE_TYPES = {"goal_switch", "action_flip", "none"}

    def __init__(self, size: int = 9, max_episode_steps: int = 80, change_type: str = "goal_switch") -> None:
        if change_type not in self.CHANGE_TYPES:
            raise ValueError(f"change_type must be one of {sorted(self.CHANGE_TYPES)}")
        self.size = size
        self.max_episode_steps = max_episode_steps
        self.change_type = change_type
        self.start = (size - 1, 0)
        self.goal_a = (0, size - 1)
        self.goal_b = (size - 1, size - 1)
        self.position = self.start
        self.goal = self.goal_a
        self.phase_b = False
        self.steps = 0

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def action_dim(self) -> int:
        return 4

    def set_phase(self, phase: str) -> None:
        if phase not in {"a", "b"}:
            raise ValueError("phase must be 'a' or 'b'")
        self.phase_b = phase == "b"
        if self.change_type == "goal_switch":
            self.goal = self.goal_b if self.phase_b else self.goal_a

    def reset(self) -> np.ndarray:
        self.position = self.start
        self.steps = 0
        return self._observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        if self.phase_b and self.change_type == "action_flip":
            action = 3 - action
        row, col = self.position
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        d_row, d_col = moves[action]
        self.position = (
            int(np.clip(row + d_row, 0, self.size - 1)),
            int(np.clip(col + d_col, 0, self.size - 1)),
        )
        self.steps += 1
        reached_goal = self.position == self.goal
        done = reached_goal or self.steps >= self.max_episode_steps
        reward = 1.0 if reached_goal else -0.01
        return self._observation(), reward, done

    def _observation(self) -> np.ndarray:
        return np.asarray(self.position, dtype=np.float32) / (self.size - 1)
