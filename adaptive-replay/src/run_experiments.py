from __future__ import annotations

import argparse
from pathlib import Path

from src.train import Config, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--change-types", nargs="+", choices=["goal_switch", "action_flip", "none"], default=["goal_switch", "action_flip", "none"])
    parser.add_argument("--total-steps", type=int, default=10_000)
    parser.add_argument("--switch-step", type=int, default=5_000)
    parser.add_argument("--buffer-capacity", type=int, default=2_000)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    for change_type in args.change_types:
        for method in ("uniform", "fifo", "aer"):
            for seed in args.seeds:
                results_dir = Path(args.results_dir) / change_type
                config = Config(
                    method=method,
                    seed=seed,
                    total_steps=args.total_steps,
                    switch_step=args.switch_step,
                    buffer_capacity=args.buffer_capacity,
                    change_type=change_type,
                    results_dir=str(results_dir),
                )
                print(f"running change={change_type} method={method} seed={seed}")
                train(config)


if __name__ == "__main__":
    main()
