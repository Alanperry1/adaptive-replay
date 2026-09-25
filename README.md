# Learning to Forget Your Ex

## First: the five-year-old explanation

Imagine a little robot learning to find a cookie in a big square room.

At first, the cookie is in the top-right corner. The robot tries walking up, down, left, and right. Each time it gets the cookie, it remembers what helped. Soon it becomes very good at finding that cookie.

Then, one day, somebody moves the cookie to a different corner. Nothing warns the robot. The robot keeps remembering lots of old trips to the old cookie spot, so it may waste time going where the cookie **used to be**.

Our project teaches the robot to notice: "Hmm, my old memories are not helping right now." It should not erase every memory immediately. Instead, it should use the new memories more and slowly stop paying attention to old memories that no longer help.

That is the whole idea:

> When the world suddenly changes, can a robot learn which old memories to stop using?

## The real explanation

This repository is a small, laptop-friendly reinforcement-learning experiment for the paper idea:

> **Learning to Forget Your Ex: Adaptive Experience Replay for Nonstationary Reinforcement Learning**

The agent learns in an environment that changes abruptly once. The goal is to test whether a replay buffer that detects the change and selectively de-emphasizes stale experience can adapt more quickly than ordinary replay buffers.

The current implementation deliberately starts small and interpretable:

- A 9 by 9 GridWorld environment.
- A DQN agent implemented in PyTorch.
- One abrupt change: the reward goal moves from the upper-right corner to the lower-right corner.
- Three replay strategies: uniform reservoir replay, FIFO replay, and Adaptive Experience Replay (AER).

No human data, external data, or GPU is required for the core experiment.

## Research question and hypothesis

**Question:** Can an RL agent selectively reduce reliance on outdated experiences after a sudden environmental change, while retaining potentially useful experience, and therefore recover faster?

**Hypothesis:** AER will obtain higher reward during the post-change recovery period and reach competent performance in the new environment in fewer steps than uniform reservoir replay or FIFO replay.

## What happens in one run

1. The agent begins in the lower-left of the grid.
2. Before the switch, it receives `+1` for reaching the upper-right goal and `-0.01` for every other step.
3. It stores transitions in a replay buffer. A transition is a small memory: state, action, reward, next state, and whether the episode ended.
4. At `--switch-step` (default: 60,000), the goal instantly moves to the lower-right.
5. The agent is not told that a new phase began. It must infer that something changed from its learning errors.
6. Training continues, and the project records episode reward, replay-buffer composition, and change-detection status.

## Replay strategies

| Method | What it remembers | Why it is a useful baseline |
| --- | --- | --- |
| `uniform` | A random, representative sample from all experience seen so far | Keeps old and new experience balanced over the whole run |
| `fifo` | The newest experiences; old entries are overwritten first | Adapts through blind forgetting, without detecting a change |
| `aer` | Newer and useful experiences after a detected shift | Tests change-aware selective forgetting |

`uniform` uses reservoir sampling so it remains distinct from FIFO. A conventional fixed-size ring buffer sampled uniformly is effectively FIFO storage plus uniform sampling, which would not be a meaningful separate baseline here.

## How AER works in this prototype

AER is an interpretable heuristic baseline, not yet a learned neural memory policy.

### 1. Detect a possible sudden change

During DQN updates, the agent calculates a temporal-difference (TD) error. This is the gap between what the agent predicted would happen and what learning says should have happened.

If recent average TD error becomes substantially larger than the recent historical distribution, AER marks a possible environment change:

\[
\text{change if } \bar{\delta}_{recent} > \mu_{history} + 2.5(\sigma_{history} + 10^{-6})
\]

This detector needs 500 recorded update-level TD-error values before it can trigger. It is intentionally simple so its successes and failures are easy to study.

### 2. Prefer relevant replay after detection

Once a shift is detected, AER gives each memory a score based on:

- **Recency:** newer samples receive a larger score.
- **Learning utility:** samples with nonzero TD error still receive some weight.

\[
\text{score}_i = 0.80 \cdot e^{-\text{age}_i / 10000}
                 + 0.20 \cdot \sqrt{|\delta_i| + 10^{-6}}
\]

The score controls both sampling and eviction. Higher-scoring memories are replayed more frequently. When the buffer is full, the lowest-scoring memory is replaced by the next transition.

This is **selective forgetting** rather than clearing the entire buffer: old data can survive when it retains enough learning utility, but stale data is strongly penalized by age after a detected shift.

## Repository map

```text
adaptive-replay/
├── requirements.txt            # Python packages
├── README.md                   # This guide
├── results/                    # Created by training; CSV logs per run
├── figures/                    # Created by plotting
└── src/
    ├── environment.py          # Goal-switch GridWorld
    ├── dqn.py                  # Q-network and DQN update
    ├── replay_buffer.py        # Uniform, FIFO, and AER buffer logic
    ├── train.py                # Training loop and metric logging
    └── plot_results.py         # Mean reward plot across seeds
```

## Installation

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

The default project uses CPU unless PyTorch detects a supported CUDA GPU.

## Running experiments

Run this command from the `adaptive-replay` directory. Start with one short debugging run, then use the full defaults.

```bash
python -m src.train --method aer --seed 0 --total-steps 10000 --switch-step 5000
```

Run the full default AER experiment:

```bash
python -m src.train --method aer --seed 0
```

Run the three core methods across five seeds:

```bash
for method in uniform fifo aer; do
  for seed in 0 1 2 3 4; do
    python -m src.train --method "$method" --seed "$seed"
  done
done
```

Each run writes two files: `results/aer_seed0_episodes.csv` and `results/aer_seed0_updates.csv`.

## Plotting the result

After collecting at least one result CSV for each method:

```bash
python -m src.plot_results
```

This saves `figures/reward_over_time.png`, `figures/recovery_time.png`, `figures/buffer_composition.png`, and (when AER logs exist) `figures/change_detector_trace.png`.

The key visual is the period immediately after the goal moves. A better adaptive method should lose less reward and climb back to strong performance sooner.

## Recorded metrics

Training produces two result CSVs for every run. The episode file contains one row per completed episode; the update file contains one row per DQN update.

| Column | Meaning |
| --- | --- |
| `step` | Global environment step at the end of the episode |
| `episode_return` | Total reward earned in the episode |
| `phase_b_buffer_fraction` | Fraction of stored transitions collected at or after the true switch time; analysis only |
| `change_detected` | Whether AER has triggered its change detector |
| `detected_at` | Global step of first detection, or `-1` if no detection occurred |
| `mean_td_error` and `loss` | Present in the update file; used for the change-detector trace |

The code never supplies the phase label to the agent. `phase_b_buffer_fraction` is logged only for later analysis of whether a method memory changed as expected.

## How to evaluate the hypothesis

For each method, use the same seeds, switch time, number of steps, and hyperparameters. Report the mean and uncertainty across at least five seeds.

Main outcomes:

1. **Pre-change performance:** all methods should learn goal A first.
2. **Detection delay:** for AER, `detected_at - switch_step` when detection occurs after the switch.
3. **Recovery time:** steps after the switch until an agent returns to a fixed level of phase-B performance, such as 80% of its final phase-B reward.
4. **Post-change reward:** reward accumulated during a fixed window after the switch.
5. **Final performance:** average reward near the end of phase B.
6. **Memory behavior:** how quickly `phase_b_buffer_fraction` rises after the switch.

Do not rely on a single random seed. A seed can be unusually lucky or unlucky in reinforcement learning.

## Suggested experimental progression

1. Run one short AER experiment to confirm the script, CSV output, and plot pipeline work.
2. Run one full experiment each for `uniform`, `fifo`, and `aer` with seed 0.
3. Inspect the learning curves and detection time. Fix obvious implementation or parameter problems before scaling up.
4. Run five seeds per method.
5. Compare recovery time and post-change reward.
6. Vary one condition at a time: buffer capacity, switch time, detector threshold, or the recency and utility weights.
7. Add a second sudden change type, such as reversed rewards or altered action mapping, only after the goal-switch result is reliable.

## Important limitations

- This is a toy GridWorld, not evidence that the method will work in all nonstationary environments.
- The detector is based only on TD errors and may false-alarm during normal exploration or miss a mild change.
- The AER score is hand-designed. A stronger research extension would learn the retention score from adaptation outcomes.
- The current environment has no walls and exposes the agent position directly. It is intentionally simple for controlled debugging.
- Sudden one-time change is the target setting. This code does not yet study gradual, recurring, or adversarially chosen changes.

## Reproducibility and ethics

This experiment uses only simulated environments and has no human subjects or personal data. For a report or paper, save the exact command, package versions, random seeds, hardware details, and raw CSV logs.

Selective forgetting can become safety-relevant in real systems: a deployed agent must not discard rare but important safety knowledge merely because it appears stale. This project does not deploy an agent and should be interpreted as a controlled research prototype.

## Next research extensions

- Replace the hand-designed relevance score with a learned retention network.
- Measure false-positive and false-negative change detection rates directly.
- Test abrupt action-remapping and reward-reversal tasks.
- Test an A → B → A sequence to see whether retaining a small useful memory preserves fast re-adaptation.
- Use a more demanding control environment after the GridWorld result is stable.

## Citation

This is a project scaffold, not a published method. If you turn it into a paper, replace this section with your author information, experiment date, and references to related work on experience replay, prioritized replay, nonstationary RL, and catastrophic forgetting.
