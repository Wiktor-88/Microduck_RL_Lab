from typing import Any

from ducklab.agents.base import BaseAgent
import torch
from torch import nn

class PrimitiveNet(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, 
                hidden_layers: list[int] #How many neurons in each hidden layer 
                ):
        super().__init__()

        #lowk do wywalenia do pomocniczej 
        layers = []
        n_in = state_dim
        for n_out in hidden_layers:
            layers.append(nn.Linear(n_in, n_out))
            layers.append(nn.ReLU())
            n_in = n_out
        layers.append(nn.Linear(n_in, action_dim))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)


        def Forward(self, state: torch.Tensor) -> torch.Tensor:
            return self.net(state)
