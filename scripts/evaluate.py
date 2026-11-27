import json
from pathlib import Path
import torch
import numpy as np

from ducklab.agents.baseline import RandomAgent
from ducklab.environments.factory import MockEnv
from ducklab.evaluation.metrics import summarize_episodes



def main():
    torch.manual_seed(1)
    n_parell = 500
    env = MockEnv(n_parell)

    agent = RandomAgent(env.obs_dim, action_dim=env.action_dim)

    obs = env.reset()

    active = torch.ones(n_parell, dtype=torch.bool)

    ep_returns = torch.zeros(n_parell)
    ep_len = torch.zeros(n_parell)
    ep_dist = torch.zeros(n_parell)

    while active.any():
        with torch.no_grad():
            actions = agent.act(obs)
        obs, rewards, dones, info = env.step(actions)

        ep_returns[active] += rewards[active]
        ep_len[active] += 1
        died = active & dones 
        ep_dist[died] = info["pos_x"][died]

        active = active & (~died)

    returns_list = ep_returns.tolist()
    lengths_list = ep_len.tolist()
    distances_list = ep_dist.tolist()

    summary = summarize_episodes(returns_list, lengths_list, distances_list)

    
    print("Evaluation complete! Summary:")
    print(json.dumps(summary, indent=2))



if __name__ == "__main__":
    main()