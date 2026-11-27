#!/bin/bash
#SBATCH -p student
#SBATCH -A 22-stud-2627-z
#SBATCH --job-name=isaac_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=isaac_test_%j.out

set -e
export PYTHONUNBUFFERED=1

module load uv/0.9.5
ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/IsaacLab}"
cd "$ISAACLAB_DIR"

echo "=== NODE & GPU ==="
hostname
nvidia-smi

echo "=== ISAAC LAB + NEWTON PPO SMOKE TEST ==="
uv run --no-sync isaaclab train \
  --rl_library rsl_rl \
  --task Isaac-Cartpole \
  --num_envs 4096 \
  --max_iterations 100 \
  --logger tensorboard \
  physics=newton_mjwarp
