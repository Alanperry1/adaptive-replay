from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {"uniform": "Uniform reservoir", "fifo": "FIFO", "aer": "AER"}
METHOD_COLORS = {"uniform": "#4C78A8", "fifo": "#F58518", "aer": "#54A24B"}


def load_csv(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return np.atleast_1d(data)


def method_from_path(path: Path) -> str:
    return path.name.split("_seed", maxsplit=1)[0]


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode="same")


def interpolated_runs(runs: list[np.ndarray], field: str, points: int = 500) -> tuple[np.ndarray, np.ndarray]:
    max_step = max(float(run["step"][-1]) for run in runs)
    grid = np.linspace(0, max_step, points)
    values = []
    for run in runs:
        steps = run["step"].astype(float)
        series = run[field].astype(float)
        values.append(np.interp(grid, steps, series, left=series[0], right=series[-1]))
    return grid, np.stack(values)


def mean_and_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    sem = values.std(axis=0, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.zeros_like(mean)
    return mean, sem


def switch_step(runs: list[np.ndarray]) -> int:
    return int(np.median([int(run["switch_step"][0]) for run in runs]))


def plot_reward(grouped: dict[str, list[np.ndarray]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    switches = []
    for method, runs in grouped.items():
        steps, values = interpolated_runs(runs, "episode_return")
        smoothed = np.stack([rolling_mean(run, 15) for run in values])
        mean, sem = mean_and_sem(smoothed)
        axis.plot(steps, mean, label=METHOD_LABELS.get(method, method), color=METHOD_COLORS.get(method))
        axis.fill_between(steps, mean - sem, mean + sem, color=METHOD_COLORS.get(method), alpha=0.2)
        switches.append(switch_step(runs))
    axis.axvline(int(np.median(switches)), color="black", linestyle="--", label="Goal switch")
    axis.set(xlabel="Environment step", ylabel="Episode return (15-episode moving average)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "reward_over_time.png", dpi=220)
    plt.close(figure)


def recovery_steps(run: np.ndarray) -> float:
    steps = run["step"].astype(float)
    rewards = rolling_mean(run["episode_return"].astype(float), 15)
    switch = int(run["switch_step"][0])
    after = np.flatnonzero(steps >= switch)
    if len(after) == 0:
        return float("nan")
    post_rewards = rewards[after]
    final_window = max(1, len(post_rewards) // 5)
    threshold = 0.8 * float(post_rewards[-final_window:].mean())
    recovered = after[np.flatnonzero(post_rewards >= threshold)]
    if len(recovered) == 0:
        return float("nan")
    return max(0.0, steps[recovered[0]] - switch)


def plot_recovery(grouped: dict[str, list[np.ndarray]], output: Path) -> None:
    methods = list(grouped)
    values = [np.asarray([recovery_steps(run) for run in grouped[method]], dtype=float) for method in methods]
    means = [float(np.nanmean(value)) for value in values]
    errors = [float(np.nanstd(value, ddof=1) / np.sqrt(np.isfinite(value).sum())) if np.isfinite(value).sum() > 1 else 0.0 for value in values]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar([METHOD_LABELS.get(method, method) for method in methods], means, yerr=errors,
             color=[METHOD_COLORS.get(method) for method in methods], capsize=5)
    axis.set(ylabel="Steps to 80% of final phase-B reward")
    figure.tight_layout()
    figure.savefig(output / "recovery_time.png", dpi=220)
    plt.close(figure)


def plot_buffer_composition(grouped: dict[str, list[np.ndarray]], output: Path) -> None:
    figure, axes = plt.subplots(len(grouped), 1, figsize=(9, 3.2 * len(grouped)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, (method, runs) in zip(axes, grouped.items()):
        steps, values = interpolated_runs(runs, "phase_b_buffer_fraction")
        phase_b, sem = mean_and_sem(values)
        switch = switch_step(runs)
        axis.fill_between(steps, 0, 1 - phase_b, color="#BDBDBD", alpha=0.85, label="Phase-A memories")
        axis.fill_between(steps, 1 - phase_b, 1, color=METHOD_COLORS.get(method), alpha=0.85, label="Phase-B memories")
        axis.plot(steps, phase_b, color="black", linewidth=1.2)
        axis.fill_between(steps, np.clip(phase_b - sem, 0, 1), np.clip(phase_b + sem, 0, 1), color="black", alpha=0.12)
        axis.axvline(switch, color="black", linestyle="--")
        axis.set(ylim=(0, 1), ylabel="Buffer fraction", title=METHOD_LABELS.get(method, method))
        axis.legend(loc="upper left")
    axes[-1].set_xlabel("Environment step")
    figure.tight_layout()
    figure.savefig(output / "buffer_composition.png", dpi=220)
    plt.close(figure)


def plot_detector(updates: list[np.ndarray], output: Path) -> None:
    if not updates:
        return
    figure, axis = plt.subplots(figsize=(9, 5))
    switch = switch_step(updates)
    for run in updates:
        steps = run["step"].astype(float)
        td_errors = rolling_mean(run["mean_td_error"].astype(float), 100)
        axis.plot(steps, td_errors, color=METHOD_COLORS["aer"], alpha=0.22, linewidth=0.8)
    steps, values = interpolated_runs(updates, "mean_td_error")
    mean, sem = mean_and_sem(np.stack([rolling_mean(run, 100) for run in values]))
    axis.plot(steps, mean, color=METHOD_COLORS["aer"], linewidth=2.2, label="AER mean TD error")
    axis.fill_between(steps, mean - sem, mean + sem, color=METHOD_COLORS["aer"], alpha=0.2)
    axis.axvline(switch, color="black", linestyle="--", label="True switch")
    detections = [int(run["detected_at"][-1]) for run in updates if int(run["detected_at"][-1]) >= 0]
    for index, detection in enumerate(detections):
        axis.axvline(detection, color="#D62728", linestyle=":", alpha=0.65,
                    label="Detected switch" if index == 0 else None)
    axis.set(xlabel="Environment step", ylabel="Mean absolute TD error")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "change_detector_trace.png", dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    episode_paths = sorted(results_dir.glob("*_seed*_episodes.csv"))
    if not episode_paths:
        raise FileNotFoundError("no episode result CSV files found")
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for path in episode_paths:
        grouped[method_from_path(path)].append(load_csv(path))
    plot_reward(grouped, output)
    plot_recovery(grouped, output)
    plot_buffer_composition(grouped, output)
    aer_updates = [load_csv(path) for path in sorted(results_dir.glob("aer_seed*_updates.csv"))]
    plot_detector(aer_updates, output)


if __name__ == "__main__":
    main()
