from __future__ import annotations

from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, method: str, rng: np.random.Generator) -> None:
        if method not in {"uniform", "fifo", "aer"}:
            raise ValueError("method must be uniform, fifo, or aer")
        self.capacity = capacity
        self.state_dim = state_dim
        self.method = method
        self.rng = rng
        self.items: list[dict[str, object]] = []
        self.write_index = 0
        self.total_seen = 0
        self.td_history: deque[float] = deque(maxlen=500)
        self.detected_change = False
        self.detected_at: int | None = None

    def __len__(self) -> int:
        return len(self.items)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        step: int,
    ) -> None:
        item = {
            "state": state.copy(), "action": action, "reward": reward,
            "next_state": next_state.copy(), "done": done, "step": step, "td_error": 1.0,
        }
        self.total_seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        if self.method == "uniform":
            index = int(self.rng.integers(0, self.total_seen))
            if index >= self.capacity:
                return
        elif self.method == "aer" and self.detected_change:
            index = self._eviction_index(step)
        else:
            index = self.write_index
            self.write_index = (self.write_index + 1) % self.capacity
        self.items[index] = item

    def sample(self, batch_size: int, step: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
        if len(self.items) < batch_size:
            raise ValueError("not enough transitions to sample")
        probabilities = self._probabilities(step)
        indices = self.rng.choice(len(self.items), size=batch_size, replace=False, p=probabilities)
        selected = [self.items[int(index)] for index in indices]
        batch = {
            "states": np.stack([item["state"] for item in selected]),
            "actions": np.asarray([item["action"] for item in selected]),
            "rewards": np.asarray([item["reward"] for item in selected], dtype=np.float32),
            "next_states": np.stack([item["next_state"] for item in selected]),
            "dones": np.asarray([item["done"] for item in selected], dtype=np.float32),
        }
        return batch, indices

    def update_td_errors(self, indices: np.ndarray, errors: np.ndarray, step: int) -> None:
        average_error = float(np.mean(errors))
        self.td_history.append(average_error)
        for index, error in zip(indices, errors):
            self.items[int(index)]["td_error"] = float(error)
        if self.method == "aer":
            self._detect_change(step)

    def _detect_change(self, step: int) -> None:
        if self.detected_change or len(self.td_history) < self.td_history.maxlen:
            return
        errors = np.asarray(self.td_history)
        recent = errors[-100:]
        history = errors[:-100]
        threshold = float(history.mean() + 2.5 * (history.std() + 1e-6))
        if float(recent.mean()) > threshold:
            self.detected_change = True
            self.detected_at = step

    def _probabilities(self, step: int) -> np.ndarray | None:
        if self.method != "aer" or not self.detected_change:
            return None
        ages = np.asarray([step - int(item["step"]) for item in self.items], dtype=np.float64)
        td_errors = np.asarray([float(item["td_error"]) for item in self.items], dtype=np.float64)
        recency = np.exp(-ages / 10_000.0)
        utility = np.sqrt(td_errors + 1e-6)
        scores = 0.80 * recency + 0.20 * utility + 1e-6
        return scores / scores.sum()

    def _eviction_index(self, step: int) -> int:
        ages = np.asarray([step - int(item["step"]) for item in self.items], dtype=np.float64)
        td_errors = np.asarray([float(item["td_error"]) for item in self.items], dtype=np.float64)
        relevance = 0.80 * np.exp(-ages / 10_000.0) + 0.20 * np.sqrt(td_errors + 1e-6)
        return int(np.argmin(relevance))

    def phase_fraction(self, switch_step: int) -> float:
        if not self.items:
            return 0.0
        return float(np.mean([int(item["step"]) >= switch_step for item in self.items]))
