# Learning to Forget Your Ex

## In very simple words

A robot learns where a cookie is hidden in a grid. Then the cookie suddenly moves. Old memories of the first cookie location can make the robot slow to find the new one.

This project asks whether the robot can notice the change and give less attention to old, unhelpful memories without deleting everything it learned.

## Project goal

This is a small reinforcement-learning experiment for:

> **Learning to Forget Your Ex: Adaptive Experience Replay for Nonstationary Reinforcement Learning**

A DQN agent learns in a 9×9 GridWorld. At the middle of training, the goal moves abruptly from the upper-right corner to the lower-right corner. The agent is not told that this happens.

> Does change-aware, selective replay help an agent adapt faster than ordinary replay after a sudden change?

## Replay methods

| Method | Behavior |
| --- | --- |
| `uniform` | Keeps a random sample of all past memories using reservoir sampling. |
| `fifo` | Forgets the oldest memory first. |
| `aer` | Detects a TD-error shift, then favors recent/useful memories and evicts stale low-score ones. |

A memory is stored in RAM as a transition:

```python
{
    "state": [row, column],
    "action": 0,
    "reward": -0.01,
    "next_state": [row, column],
    "done": False,
    "step": 4250,
    "td_error": 0.18,
}
```

## Project files

```text
adaptive-replay/
├── requirements.txt
└── src/
    ├── environment.py       # GridWorld and sudden goal switch
    ├── dqn.py               # Neural network and DQN learning update
    ├── replay_buffer.py     # Uniform, FIFO, and AER memory logic
    ├── train.py             # Runs training and writes CSV logs
    └── plot_results.py      # Creates paper figures from logs
```

## Setup

```bash
cd adaptive-replay
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The core experiment runs on a laptop CPU. For sudden-change comparisons, `--buffer-capacity` must be smaller than `--switch-step`, so the buffer is full before the environment changes.

## Run experiments

A quick, valid comparison uses a 2,000-entry buffer:

```bash
for method in uniform fifo aer; do
  python -m src.train --method "$method" --seed 0 \
    --total-steps 10000 --switch-step 5000 --buffer-capacity 2000
done
python -m src.plot_results
```

Each run writes episode and update CSV logs under `results/`. Plotting creates:

- `figures/reward_over_time.png`
- `figures/recovery_time.png`
- `figures/buffer_composition.png`
- `figures/change_detector_trace.png`

## Limits

This is a controlled toy environment. AER uses a hand-designed relevance score, and TD-error detection can false-alarm or miss mild changes. Run at least five seeds per method before drawing research conclusions.
