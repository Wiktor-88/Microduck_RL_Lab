# 1. MDP, Observations and Rewards
## Microduck problem, agent and envirnoment
In this project **agent** is trying to learn 



# 2. Function approximation and neural netowrks
## Function approximation in Mircoduck problem
FIrst we remind the value of the state is detrmined by:
$$v^{\pi}(s) = \mathbb{E}_{\pi}[G_t | S_t = s]$$
where $G_t = \sum_{i=0}^{\infty} \gamma^i R_{t+1 + i}$ and $s \in \mathcal{S}$

Function $v^{\pi}$ answers the question **what expected return we will recive if we start in state $\mathbf{s}$ and then working with policy $\mathbf{\pi}$** 

In a tabluar reprezetation each state get its own number:
$$\hat{v}(s_i) = w_i \qquad i \in [N]$$
where $N = |\mathcal{S}|$

The vector $w \in \mathbb{R}^{N}$. Changing one state does not interfere with other states.

However that representations has no use in Microduck problem. Velocitie and postion of the nitwits and orientation of the robot are continous things not discreate. So in a mathematical model we cannot use finite tabel of every possible configuration.

That is why we replace it with a **function**:
$$\hat{v}(s, w) \approx v^{\pi}(s) \qquad w \in \mathbb{R}^p$$
Here $s$ is an argument of the function, $w$ includes all the parameters used in learning and $\hat{v}(s, w)$ is approximation of the state value.

We choose a family of functions:
$$\mathcal{F} = \{s \rightarrow \hat{v}(s, w) \; | \; \; w \in \mathbb{R}^p  \}$$

The process of learning is finding specific parameters insinde this family. Of course we do not know whether the real function $v^{\pi}$ are included in this family.


# Policy Gradnient and Actor-Critic

# Math behind PPO

# Practical PPO

# Domain Randomization and evaluation