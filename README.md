# Learning to Forget Your Ex

## In very simple words

A robot learns where a cookie is hidden in a grid. Then the cookie suddenly moves. Old memories of the first cookie location can make the robot slow to find the new one.

This project tests whether a robot can notice the change and give less attention to old, unhelpful memories without deleting everything it learned.

## Project goal

A DQN agent learns in a 9×9 GridWorld. It compares three replay-memory strategies after a sudden environment shift:

| Method | Behavior |
| --- | --- |
| `uniform` | Keeps a random sample of all past memories. |
| `fifo` | Forgets the oldest memory first. |
| `aer` | Detects reward-prediction or action-outcome shifts, then favors recent/useful memories and evicts stale low-score ones. |

The main question is whether AER adapts faster without simply throwing every old memory away.

## Project files

```text
adaptive-replay/
├── requirements.txt
├── src/
│   ├── environment.py       # GridWorld scenarios
│   ├── replay_buffer.py     # Uniform, FIFO, and AER memory logic
│   ├── train.py             # One training run
│   ├── run_experiments.py   # Multi-seed, multi-scenario runs
│   └── plot_results.py      # Paper figures
└── tests/
    └── test_replay_and_environment.py
```

## Setup

```bash
cd adaptive-replay
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Scenarios

- `goal_switch`: the goal moves to a new location.
- `action_flip`: the meaning of every action reverses; AER uses action-outcome surprise to detect it.
- `none`: no change occurs; this is the detector false-alarm control.

For sudden-change comparisons, `--buffer-capacity` must be smaller than `--switch-step`, so the buffer fills before the change.

## Run a quick comparison

```bash
for method in uniform fifo aer; do
  python -m src.train --method "$method" --seed 0 \
    --total-steps 10000 --switch-step 5000 --buffer-capacity 2000 \
    --change-type goal_switch
 done
python -m src.plot_results
```

## Run the full validation suite

This runs five seeds across both sudden changes and the no-change control:

```bash
python -m src.run_experiments
```

Results are written to `results/<scenario>/`. Create figures for a scenario with:

```bash
python -m src.plot_results --results-dir results/goal_switch --output-dir figures/goal_switch
```

Run unit tests with:

```bash
pytest
```

Use at least five seeds per method before drawing research conclusions. AER may still fail, false-alarm, or over-forget; those outcomes are part of the research result.
