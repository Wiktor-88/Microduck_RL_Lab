from abc import ABC, abstractmethod
import torch 
from torch import nn

class BaseAgent(ABC, nn.Module):
    '''
    Just a blueprint for every agent.
    '''

    def __init__(self, in_dim: int, action_dim: int):
        super().__init__()

        self.in_dim = in_dim
        self.action_dim = action_dim

    @abstractmethod
    def act(self, observed: torch.Tensor) -> torch.Tensor:
        '''
        Dajesz stan dostajesz akcję
        '''
        pass

