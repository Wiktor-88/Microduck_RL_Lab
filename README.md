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
microduck-rl-lab/
│
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
│
├── configs/
│   ├── experiments/
│   │   ├── smoke_test.yaml
│   │   ├── ppo_baseline.yaml
│   │   ├── ppo_friction.yaml
│   │   └── ppo_pushes.yaml
│   ├── environments/
│   │   └── single_duck.yaml
│   └── evaluation/
│       └── default.yaml
│
├── src/
│   └── ducklab/
│       ├── cli/
│       │   ├── train.py
│       │   ├── evaluate.py
│       │   ├── compare.py
│       │   └── export.py
│       │
│       ├── core/
│       │   ├── config.py
│       │   ├── types.py
│       │   └── paths.py
│       │
│       ├── environments/
│       │   ├── factory.py
│       │   ├── upstream_adapter.py
│       │   └── disturbances.py
│       │
│       ├── agents/
│       │   ├── policy.py
│       │   ├── random_policy.py
│       │   └── trained_policy.py
│       │
│       ├── training/
│       │   ├── trainer.py
│       │   └── ppo_runner.py
│       │
│       ├── evaluation/
│       │   ├── evaluator.py
│       │   ├── metrics.py
│       │   └── scenarios.py
│       │
│       ├── analysis/
│       │   ├── data_loader.py
│       │   ├── statistics.py
│       │   └── plots.py
│       │
│       ├── infrastructure/
│       │   ├── tracking.py
│       │   └── artifacts.py
│       │
│       └── deployment/
│           ├── onnx_exporter.py
│           └── validator.py
│
├── scripts/
│   └── slurm/
│       ├── smoke.sbatch
│       ├── train.sbatch
│       └── evaluate.sbatch
│
├── tests/
│   ├── test_config.py
│   ├── test_environment.py
│   ├── test_rewards.py
│   └── test_metrics.py
│
├── reports/
│   ├── report.tex
│   └── figures/
│
├── docs/
│   ├── architecture.md
│   ├── mdp.md
│   ├── experiments.md
│   └── checkpoint.md
│
└── results/
    └── .gitkeep

```


The requiremenst for this project are specified in **requirements.txt** 