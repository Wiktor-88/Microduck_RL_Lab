# Microduck Reinforcement Learning Lab

#### This project demonstrates Framework of Deep Reinforcement Learning for varius purposes described further in more deatails. Yet the main goal is to answear the hypothesis:
 
**How does domain randomization affect the robustness and generalization of PPO-based locomotion policies under unseen physical disturbances?**

We would check if the robots can handle differnet envoirnoments such as different labyrinyhs with many obstacles.

The project is devided into 6 main branches:
- Envirnoment 
- RL learning
- Evaluation
- Infrastructure
- Experiment
- Report and Presentation


Below is shown full tree-files for this project:
```text
Microduck-RL/
│
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
│
├── configs/
│   ├── environments/
│   ├── training/
│   └── experiments/
│
├── src/
│   └── ducklab/
│       │
│       ├── __init__.py
│       │
│       ├── environments/
│       │   ├── __init__.py
│       │   ├── factory.py
│       │   ├── robot.py
│       │   ├── rewards.py
│       │   └── randomization.py
│       │
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── baseline.py
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   └── trainer.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── evaluator.py
│       │   └── metrics.py
│       │
│       ├── analysis/
│       │   ├── __init__.py
│       │   └── plots.py
│       │
│       └── infrastructure/
│           ├── __init__.py
│           └── logger.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── slurm/
│       ├── smoke.sbatch
│       └── train.sbatch
│
├── tests/
│   ├── test_environments.py
│   ├── test_rewards.py
│   └── test_metrics.py
│
├── docs/
│   ├── architecture.md
│   └── experiments.md
│
├── reports/
│   └── figures/
│
└── results/
```


The requiremenst for this project are specified in **requirements.txt** 