# 1. MDP, Observations and Rewards
## Microduck problem, agent and envirnoment

### Microduck problem
In this project **agent** is trying to learn how to control Microduck to make the robot listen to command how to move with stability.

We can assume that the movement of the robot is combined of:
$$c_t^{vel} =
\begin{bmatrix}
v_{x,t}^{*}\\
v_{y,t}^{*}\\
\omega_{z,t}^{*}
\end{bmatrix},
$$
where $v_{i,t}^{*}$ is a velocity in one of the axis $i \in \{x, y, z \}$. 
More detailed describtion of the coordinate system will be included later when we will describe commands and observations.

But we have to docus on the difference between **command given to robot** and **action made by agent**:
$$C_t^{vel} \neq A_t$$
For exmaple command to move forward defines expected result, while agent have to choose the movement of the nitwits that will lead to that result.

In our setup, the agent–environment boundary lies at the level of the joint control decision output. In detailed the differences are shown in a table:
| Element | Who | Why |
|---|---|---|
| What action will be taken based on the information we currently have | Agent | This is the mechanism that our agent will be learning |
| Action selection rule learning mechanism | Part of the agent's learning system | Changes how future decisions are made |
| Conversion of actions into controller targets  | Environment | Determines how decisions are physically executed |
| Robot body, ground, gravity, and contacts | Environment | Determine the consequences of actions |
| Measurements, noise, and delays | Environment | Define the information provided to the agent |
| Motion command generation | Task environment | Specifies what motion is required at a given moment |
| Reward calculation and termination conditions | Environment | Defines behavior evaluation and episode boundaries |


We assume the time is discreate:  $$t \in \{0, 1, 2, \dots \}$$

One step includes:
1. Agent get observation $O_t$ which has available information about robot and task
2. Action $A_t$ is chosen
3. Environment performs that task and changes its state
4. Agent get another observation $O_{t + 1}$ and the reward $R_{t+1}$

So finally we get a sequence:
$$O_0, A_0, R_1, O_1, A_1, R_2, \dots$$
In typical MDP there should be states $S_t$, but we will use $S_t$ for a full state and $O_t$ for informations the agent got


### MDP
Formally MDP is decribed by:
$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, K, p_0, \gamma)$$
where:
| Symbol | Meaning |
|---|---|
| $\mathcal{S}$ | Full environment state space, including task termination |
| $\mathcal{A}$ | Agent action space |
| $K$ | Joint transition and reward law |
| $p_0$ | Initial state distribution |
| $\gamma$ | Discount factor for future rewards |

But this is version for bounded set of states ($|\mathcal{S}| < \infty$). We will need a version for continous sets.

The state should satisfy Markov's property:
$$\mathbb{P}(S_{t+1} \in B, R_{t+1} \in C | S_{0:t}, A_{0:t}) = \mathbb{P}(S_{t+1} \in B, R_{t+1} \in C |S_t, A_t)$$
where $S_{0:t}$ and $A_{0:t}$ are whole history from moment $t_0 = 0$ to $t$  
$B, C$ are sets of possible states and rewards

So we will use more accurate states which will mean more than position and velocity:
$$
S_t=\left(X_t,\ Z_t,\ C_t,\Theta_t,\ N_t\right)
$$
In this model:
- $X_t$ — robot's physical state and necessary simulator state
- $Z_t$ — memory of controllers, measurements, and reward functions
- $C_t$ — current commands and the state of their change mechanism
- $\Theta_t$ — current physics and measurement parameters
- $N_t$ — counters and active schedule settings affecting environment behavior

This is a semantic breakdown, not yet a list of specific fields or a declaration of the vector dimension.

For example, two robots with identical positions and velocities may have different targets stored in the control delay buffer. Their subsequent movement may differ despite receiving the same new action. Omitting the buffer would make it impossible to justify the Markov property.

Another difference is in defintion of the probability of next movement. For finite state sets we define:
$$
p(s',r\mid s,a)
=
\mathbb{P}(S_{t+1}=s',R_{t+1}=r
\mid S_t=s,A_t=a)
$$
But for continous set of states we will use probabilioty kernel:
$$
K(B\times C\mid s,a)
=
\Pr(S_{t+1}\in B,R_{t+1}\in C
\mid S_t=s,A_t=a).
$$
which satisfies:
$$\forall s, a \qquad K(\mathcal{S} \times \mathbb{R} \mid s, a) = 1$$

In a continius set of state the probabilty of being in exactly one state may be $0$. We don't have to define the probability density(PDF)

In the case of Microduck, $K$ describes the consequences of a complete environment step: executing control actions, physics evolution, memory updates, and the corresponding random draws. The simulator implements sampling from this transition. Knowing the explicit formula for $K$ is not necessary to gather experience.

Using that kernel we can find the probability of movement:
$$
P(B\mid s,a)=K(B\times\mathbb R\mid s,a),
$$
and expected reward:
$$
\bar r(s,a)
=
\mathbb E[R_{t+1}\mid S_t=s,A_t=a]
=
\int_{\mathcal S\times\mathbb R}
r\,K(ds',dr\mid s,a).
$$
So we have the reward got in one step $R_{t+1}$ and its expected value $\bar{r}(s, a)$


Begining of the trajectory is described with:
$$S_0 \sim p_0$$
A reset does not always mean an identical configuration. It can sample both the initial state and the environment parameters.

The actual end of a task can be represented by an absorbing state $s$:
$$
\mathbb{P}(S_{t+1} = s \mid S_t = s) = 1, \quad R_{t+1} = 0
$$
No further rewards belonging to the same trajectory occur. The reset initiates a new one.

For the termination time step $T$, the return is given by:
$$
G_t = \sum_{k=t+1}^{T} \gamma^{k-t-1} R_k
$$

If the task can run indefinitely, we standardly assume $0 \le \gamma < 1$, along with conditions ensuring the existence of the expected return.

**The limit of the simulated episode length does not have to be a true task terminal.** When it is merely a technical cutoff, further movement would still hold value. Therefore, we will treat termination and truncation separately.


The difference between MDP 





# 2. Function approximation and neural netowrks
## Function approximation in Mircoduck problem
FIrst we remind the value of the state is detrmined by:
$$v^{\pi}(s) = \mathbb{E}_{\pi}[G_t | S_t = s]$$


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

# PPO

# Domain Randomization and evaluation