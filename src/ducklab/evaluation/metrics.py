import numpy as np


def summarize_episodes(returns: list[float], lengths: list[int], distances: list[float]) -> dict:
    return {
        "num_episodes": len(returns),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_survival_steps": float(np.mean(lengths)),
        "std_survival_steps": float(np.std(lengths)),
        "mean_distance_m": float(np.mean(distances)),
        "max_distance_m": float(np.max(distances)),
    }