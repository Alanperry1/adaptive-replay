# Learning to Forget Your Ex

## In very simple words

A robot learns where a cookie is hidden in a grid. Then the cookie suddenly moves. Old memories of the first cookie location can make the robot slow to find the new one.

This project asks whether the robot can notice the change and give less attention to old, unhelpful memories without deleting everything it learned.

## Project goal

This is a small reinforcement-learning experiment for:

> **Learning to Forget Your Ex: Adaptive Experience Replay for Nonstationary Reinforcement Learning**

A DQN agent learns in a 9×9 GridWorld. At the middle of training, the goal moves abruptly from the upper-right corner to the lower-right corner. The agent is not told that this happens.

The research question is:

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
    "action": 0,          # up, down, left, or right
    "reward": -0.01,
    "next_state": [row, column],
    "done": False,
    "step": 4250,
    "td_error": 0.18,
}
```

## Project files

```text
src/
├── environment.py       # GridWorld and sudden goal switch
├── dqn.py               # Neural network and DQN learning update
├── replay_buffer.py     # Uniform, FIFO, and AER memory logic
├── train.py             # Runs training and writes CSV logs
└── plot_results.py      # Creates paper figures from logs
```

## Setup

From the `adaptive-replay` directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

The core experiment runs on a laptop CPU.

## Run an experiment

Start with a short run:

```bash
python -m src.train --method aer --seed 0 --total-steps 10000 --switch-step 5000
```

Run all methods over five seeds:

```bash
for method in uniform fifo aer; do
  for seed in 0 1 2 3 4; do
    python -m src.train --method "$method" --seed "$seed"
  done
done
```

Each run saves:

- `results/<method>_seed<seed>_episodes.csv`: reward and buffer-composition data.
- `results/<method>_seed<seed>_updates.csv`: TD errors, loss, and detection data.

## Create figures

```bash
python -m src.plot_results
```

This produces:

- `figures/reward_over_time.png`: reward curves with confidence bands and the true switch point.
- `figures/recovery_time.png`: steps needed to recover 80% of final phase-B reward.
- `figures/buffer_composition.png`: old versus new memories held in each buffer.
- `figures/change_detector_trace.png`: AER TD-error trace with true and detected switch points.

## What to compare

Use the same seed, switch step, total training steps, and hyperparameters for every method. The main outcome is recovery time after the goal moves. Also compare post-change reward, final reward, detector delay, and buffer composition.

## Limits

This is a controlled toy environment. AER uses a hand-designed relevance score, and TD-error detection can false-alarm or miss mild changes. It is a first experiment, not evidence that the method works in every changing environment.

## Ethics

The project uses only simulation; it has no people or personal data. For reproducibility, record package versions, commands, seeds, hardware, and raw CSV outputs. In real systems, adaptive forgetting would need safety checks so rare but important safety knowledge is not discarded.
