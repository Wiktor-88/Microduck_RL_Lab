# Experiment & Command Log (`eden` + MuJoCo)

## 1. Cluster Setup & Commands (`eden`)

- **Login node:** `eden` (Slurm partition: `-p student`, account: `-A 22-stud-2627-z`).
- **Isaac Lab 3.0 installation (`stud-1`, CPU-only):** installed on shared network storage at `/mnt/evafs/faculty/home/dwilk/IsaacLab` (branch `release/3.0.0`, module `uv/0.9.5`).
- **Installed packages:** `torch 2.11.0+cu130`, `warp-lang 1.17.0`, `newton 1.6.1`, `mujoco-warp 3.12.0`, `rsl-rl-lib 5.5.1`.

### Commands Used

```bash
# Check cluster availability and queue status
sfree
squeue -u $USER
squeue -j <JOBID> --start

# Interactive session on CPU node stud-1 (for package installation)
srun -p student -A 22-stud-2627-z --cpus-per-task=8 --mem=32G --time=01:00:00 --pty bash

# Install Isaac Lab 3.0 on stud-1 (shared /mnt/evafs)
module load uv/0.9.5
git clone https://github.com/isaac-sim/IsaacLab.git --branch release/3.0.0
cd IsaacLab
export UV_CACHE_DIR=$PWD/.uv-cache
export UV_HTTP_TIMEOUT=300
export UV_CONCURRENT_DOWNLOADS=4
uv sync
rm -rf $PWD/.uv-cache

# Submit GPU smoke test from eden
sbatch scripts/slurm/smoke.sbatch
```

---

## 2. Runs & Observed Results

### Cluster `eden` — Job `1990656` (Verified GPU Smoke Test)
- Script: `scripts/slurm/smoke.sbatch` (`Isaac-Cartpole`, `4096` envs, `physics=newton_mjwarp`, `rsl_rl` PPO, `100` iterations).
- Node: `stud-3` (`NVIDIA RTX PRO 6000 Blackwell`, `97887 MiB`, Driver `615.71.09`, CUDA `13.4`).
- Timing: `6,553,600` steps in `100.14 s` (`0.14–0.15 s/iter`, peak `474,454 steps/s`).
- Metrics: `Mean reward: 0.13 -> 4.94`, `Mean episode length: 12.78 -> 300.00` (`cart_out_of_bounds: 0.0000`).
- Cluster log path: `/mnt/evafs/faculty/home/dwilk/IsaacLab/logs/rsl_rl/cartpole/2026-11-26_10-35-09`.
