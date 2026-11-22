from base import BaseAgent
import torch


class RandomAgent(BaseAgent):

    def __init__(self, in_dim: int, action_dim: int):
        super().__init__(in_dim, action_dim)

    def act(self, observed: torch.Tensor) -> torch.Tensor:
        batch_size = observed.shape[0] #Ile robocików na raz puszcamy 

        #Chemy losowe jakies 
        return torch.empty(batch_size, self.action_dim).uniform_(-1, 1)

