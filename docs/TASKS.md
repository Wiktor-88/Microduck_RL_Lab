
# Microduck RL Lab — Project Tasks

This document tracks completed work, required improvements and planned tasks

## 1. Completed Work

### Project Setup
- [x] Initialize the GitHub repository
- [x] Create the modular Python project structure
- [x] Define the research question and project objectives
- [x] Prepare initial README and mathematical documentation

### Environment
- [x] Implement a simplified Microduck model in MuJoCo (`environments/factory.py`)
- [x] Define basic observations and a 14-dimensional action interface
- [x] Implement an initial reward function (`environments/rewards.py`)
- [x] Implement environment reset and episode termination logic

### Agents and Evaluation
- [x] Implement a random baseline agent (`agents/baseline.py`)
- [x] Implement basic evaluation metrics (`evaluation/metrics.py`)
- [x] Prepare a basic evaluation script (`scripts/evaluate.py`)
- [x] Perform preliminary random vs. zero-target baseline experiments (reported in checkpoint materials)

### Infrastructure
- [x] Prepare the Python environment and project dependencies
- [x] Configure an initial Slurm smoke test
- [x] Record a successful PPO GPU test on Eden using Isaac-Cartpole

---

## 2. Required Improvements

### High Priority — Before PPO Training

- [ ] **Validate the robot model**
  - Compare the current simplified XML model with official Microduck specifications
  - Resolve differences between model masses, limits and configuration files
  - Select one consistent simulation backend

- [ ] **Improve the reward function**
  - Reward tracking of commanded linear and angular velocities
  - Reconsider the constant survival bonus
  - Keep penalties for instability and unnecessary actions
  - Verify that reward maximization encourages actual locomotion

- [ ] **Improve episode handling**
  - Separate `terminated` (fall) from `truncated` (time limit)
  - Support correct value bootstrapping during PPO training
  - Prepare independent environment resets for parallel simulation

- [ ] **Fix neural network implementation**
  - Correct the `forward()` method in `agents/networks.py`
  - Verify model inputs and outputs
  - Avoid duplicating PPO components already available in RSL-RL

### Medium Priority — Evaluation and Documentation

- [ ] Extend `evaluation/metrics.py` with:
  - Fall rate.
  - Episode duration.
  - Velocity tracking RMSE.
  - Forward displacement.

- [ ] Implement automated tests:
  - `tests/test_environments.py`
  - `tests/test_rewards.py`
  - `tests/test_metrics.py`

- [ ] Implement plotting utilities in `analysis/plots.py`.
- [ ] Integrate checkpoint experiment files (`run_experiment.py`, CSV, JSON and demo).
- [ ] Complete the mathematical documentation of the environment.
- [ ] Fix missing documentation links in README.
- [ ] Update `.gitignore` to exclude unnecessary training outputs.
- [ ] Confirm repository access for the instructor.

---

## 3. New Development Tasks

### Stage 1 — PPO Integration

- [ ] Implement PPO training integration (`training/trainer.py`).
- [ ] Prepare the training entry point (`scripts/train.py`).
- [ ] Configure PPO hyperparameters.
- [ ] Prepare the Microduck GPU training script (`scripts/slurm/train.sbatch`).
- [ ] Run the first Microduck PPO smoke test on Eden.
- [ ] Train and save a nominal PPO policy.
- [ ] Load and evaluate the trained policy.

**Milestone:** A trained PPO policy can control Microduck in simulation.

### Stage 2 — Domain Randomization

- [ ] Implement randomization in `environments/randomization.py`.
- [ ] Introduce selected physical parameter variations, such as:
  - Ground friction.
  - Robot mass.
  - Actuator parameters.
  - External disturbances.
- [ ] Define and document randomization ranges.
- [ ] Prepare nominal and randomized training configurations.

**Milestone:** Two PPO training variants are available: with and without domain randomization.

### Stage 3 — Experiments and Analysis

- [ ] Train PPO policies without domain randomization.
- [ ] Train PPO policies with domain randomization.
- [ ] Use matched training budgets and multiple independent seeds.
- [ ] Evaluate both variants under identical test conditions.
- [ ] Test performance under unseen physical disturbances.
- [ ] Compare tracking error, fall rate and other metrics.
- [ ] Generate plots and summary tables.
- [ ] Analyze results and limitations.

**Milestone:** Experimental evidence addressing the research question.

### Stage 4 — Final Deliverables

- [ ] Complete the technical documentation.
- [ ] Prepare the final LaTeX report (maximum 3 pages).
- [ ] Prepare the final presentation.
- [ ] Demonstrate the trained Microduck policies.
- [ ] Summarize conclusions and possible future improvements.

---

## 4. Suggested Work Division

| Workstream | Responsibility | Main Deliverable |
|---|---|---|
| A — Environment / MDP | Robot model, observations, actions, rewards | Validated environment |
| B — Evaluation | Metrics, testing, baseline experiments | Reproducible evaluation |
| C — PPO / Infrastructure | Training, Eden, checkpoints | Trained PPO policy |
| D — Domain Randomization / Analysis | Randomization, experiments, plots | Robustness comparison |

**Note:** The final assignment of team members to workstreams is still to be confirmed.

---

## 5. Development Order

1. Validate and correct the environment.
2. Complete the evaluation pipeline.
3. Train the first nominal PPO policy.
4. Implement domain randomization.
5. Run controlled comparative experiments.
6. Analyze results and prepare the final deliverables.

**Current priority:** Validate the Microduck environment and prepare the first reproducible PPO training experiment.
