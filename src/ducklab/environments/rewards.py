import torch 


def simple_reward(
        velocity_x: torch.Tensor, #we want our robot to go forward
        actions: torch.Tensor,
        fallen: torch.Tensor, #falling is bad
        alive_bonus: float = 0.2, #We are happy the robot is still alive
        falling_penalty: float = 5.0, #We dont like it falling
        energy_weight: float = 0.05
        ) -> torch.Tensor:

    '''A very basic reward function 
    gives positive points to things we like (going forward, managing to mantain balance)
    and penalizing behaviour we'd like to unlearn: rough movements, stumbling'''

    forward_reward = 1.5 * velocity_x
    energy_penalty = energy_weight * torch.sum(actions**2, dim=1)
    termination_penalty = falling_penalty * fallen.float()

    reward = forward_reward+ alive_bonus - energy_penalty -termination_penalty

    return reward