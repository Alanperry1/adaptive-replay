from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.dqn import DQNAgent
from src.environment import GoalSwitchGridWorld
from src.replay_buffer import ReplayBuffer


@dataclass
class Config:
    method: str
    seed: int
    total_steps: int = 120_000
    switch_step: int = 60_000
    buffer_capacity: int = 20_000
    change_type: str = "goal_switch"
    results_dir: str = "results"
    batch_size: int = 64
    learning_starts: int = 1_000
    train_frequency: int = 4
    target_frequency: int = 1_000
    learning_rate: float = 1e-3
    gamma: float = 0.99


def epsilon(step: int, total_steps: int) -> float:
    progress = min(step / (0.50 * total_steps), 1.0)
    return 1.0 + progress * (0.05 - 1.0)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_csv(path: Path, rows: list[dict[str, float | int]], fields: list[str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def train(config: Config) -> Path:
    if config.buffer_capacity >= config.switch_step:
        raise ValueError("--buffer-capacity must be smaller than --switch-step for this sudden-change experiment")
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GoalSwitchGridWorld(change_type=config.change_type)
    agent = DQNAgent(env.state_dim, env.action_dim, config.learning_rate, config.gamma, device)
    buffer = ReplayBuffer(config.buffer_capacity, env.state_dim, config.method, np.random.default_rng(config.seed))
    state = env.reset()
    episode_return = 0.0
    episode_rows: list[dict[str, float | int]] = []
    update_rows: list[dict[str, float | int]] = []

    for step in range(1, config.total_steps + 1):
        if step == config.switch_step and config.change_type != "none":
            env.set_phase("b")
        action = agent.act(state, epsilon(step, config.total_steps))
        next_state, reward, done = env.step(action)
        buffer.add(state, action, reward, next_state, done, step)
        state = next_state
        episode_return += reward

        if step >= config.learning_starts and step % config.train_frequency == 0:
            batch, indices = buffer.sample(config.batch_size, step)
            loss, errors = agent.update(batch)
            mean_td_error = float(np.mean(errors))
            buffer.update_td_errors(indices, errors, step)
            update_rows.append({
                "step": step,
                "mean_td_error": mean_td_error,
                "loss": loss,
                "change_detected": int(buffer.detected_change),
                "detected_at": buffer.detected_at or -1,
                "detection_source": buffer.detection_source,
                "switch_step": config.switch_step,
                "buffer_evictions": buffer.eviction_count,
                "selective_evictions": buffer.selective_eviction_count,
            })
        if step % config.target_frequency == 0:
            agent.sync_target()
        if done:
            episode_rows.append({
                "step": step,
                "episode_return": episode_return,
                "phase_b_buffer_fraction": buffer.phase_fraction(config.switch_step),
                "change_detected": int(buffer.detected_change),
                "detected_at": buffer.detected_at or -1,
                "detection_source": buffer.detection_source,
                "switch_step": config.switch_step,
                "buffer_evictions": buffer.eviction_count,
                "selective_evictions": buffer.selective_eviction_count,
            })
            state = env.reset()
            episode_return = 0.0

    output_dir = Path(config.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{config.method}_seed{config.seed}"
    episodes_path = output_dir / f"{stem}_episodes.csv"
    updates_path = output_dir / f"{stem}_updates.csv"
    write_csv(episodes_path, episode_rows, [
        "step", "episode_return", "phase_b_buffer_fraction", "change_detected", "detected_at", "detection_source", "switch_step", "buffer_evictions", "selective_evictions",
    ])
    write_csv(updates_path, update_rows, [
        "step", "mean_td_error", "loss", "change_detected", "detected_at", "detection_source", "switch_step", "buffer_evictions", "selective_evictions",
    ])
    return episodes_path


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["uniform", "fifo", "aer"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=120_000)
    parser.add_argument("--switch-step", type=int, default=60_000)
    parser.add_argument("--buffer-capacity", type=int, default=20_000)
    parser.add_argument("--change-type", choices=sorted(GoalSwitchGridWorld.CHANGE_TYPES), default="goal_switch")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    return Config(
        method=args.method,
        seed=args.seed,
        total_steps=args.total_steps,
        switch_step=args.switch_step,
        buffer_capacity=args.buffer_capacity,
        change_type=args.change_type,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    path = train(parse_args())
    print(f"saved {path}")
