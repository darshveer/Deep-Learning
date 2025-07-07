# AlphaGo
**Use:** Primarily made only for the game Go.

## Components
### 1. Policy Network

- A deep CNN trained to predict expert moves.

- Trained using Supervised learning (SL) from human expert positions, followed by policy gradient into Reinforcement Learning(RL) to determine the Value Network through Self Play.
- Given a board position, the CNN gives a rollout (simulates the position till the game end using lightweight policy) and outputs the probabilities of the next moves $ p_\sigma (a|s) $ using stochastic gradient **ascent**:
$$ \Delta \sigma \propto \frac{\partial \log p_\sigma (a|s)}{\partial \sigma} $$

### 2. Reinforcement Learning of Policy Networks

- Start with same initialisation of weights of SL for RL:
$$ \rho = \sigma $$
- Randomly choose from a pool of opponents with a previous iteration of the policy network and self-play with the current iteration, $p_\rho$.
- Use a reward function from the current state at time $T$ given by:
$$r(s_t) = 
\begin{cases}
0, & \text{if } t < T \\
 1, & \text{if } t = T \text{ and win} \\
 -1, & \text{if } t = T \text{ and lose}
\end{cases}

\\[1em]

z_t = r(s_T) \in \{-1, +1\}

\\[1em]
$$ 

- Now update weights using stochastic gradient ascent, so as to maximise the ptrobability of winning:
$$ \Delta \rho \propto \frac{\partial \log p_\rho (a_t|s_t)}{\partial \rho} \cdot z_t $$
- We use log to determine the likelihop of choosing an action $a_t$ given a state $s_t$.

### 3. Reinforcement Learning of Value Networks

-  Estimate a value function to predict the outcome from a position $s$ of games played by using policy $p$ for bith players:
$$ v^p(s) = \mathbb E[z_t \ | \ s_t = s, a_{t ... T} \sim p] $$
- We approximate this with the RL policy (as it it the strongest policy we have) with weights $\theta$ to get as close to the optimal value $v^*(s)$ as possible:
$$ v_\theta (s) \approx v^{P_\rho}(s) \approx v^*(s) $$
- This gives a single value which can then be trained using MSE and stochastic gradient descent:
$$ \Delta \theta \propto \frac{\partial v_\theta (s)}{\partial \theta} \cdot (z - v_\theta (s)) $$

### 4. Monte-Carlo Tree Search (MCTS):
- MCTS is a decision-making algorithm used in games (and other problems) where the number of possible outcomes is huge — too big to  search exhaustively.

    i. **Selection**:
    - Each node has an edge $(s,a)$ with an action value $Q(s,a)$, a visit count $N(s,a)$, and a prior probability $P(s,a)$, and the tree is traverse from the initial game state or too node.
    - At each time step $t$ of the simulation, an action $a_t$ is selected from the state $s_t$:
    $$ a_t = \argmax_a (Q(s_t,a) + u(s_t, a)) $$
    - This is to form an Upper Confidence Bound (UCB) on the state $s$ which has been visited $N(s)$ times,such that:
    $$ u(s_t, a) =c \cdot \sqrt{N(s)} \cdot \frac{P(s, a)}{1 + N(s, a)} $$
    - Thus, this balances:
        - *Exploration:* If $Q(s,a)$ is high (known to be a good edge), increase $a_t$.
        - *Exploitation:* If $N(s,a)$ is low (rarely tried edge), increase $a_t$.

    ii. **Expansion**:
    - When you reach a leaf node (a state that hasn’t been fully explored), you add a new child node corresponding to one of the possible moves from this state.
    - Here, an expansion threshold $n_\text{thresh}$ is used to determine whether to expand a node or not. Its value is set to 40. Hence, a node will create child nodes only after $N(s) = 40$.

    iii. **Evaluation**:
    - Uses rollouts (simulated games with a fast policy) or a value network to estimate the outcome of the game from this node.
    - We get a leaf evaluation for the leaf node $s_L$ as:
    $$ V(s_L) = (1 - \lambda)v_\theta (s_L) + \lambda z_L  $$
    - The $z_L$ is the outcome fo the random rollout till the terminal step $T$ and is quick, but $v_\theta (s_L)$ is slow and is estimated through parameters $\theta$ of the value network.
    
    iv. **Backpropagation**:
    - Use the evaluation result (win/loss/draw or estimated value) to update all nodes along the path back to the root:
    $$ Q(s,a) \leftarrow Q(s,a) + \alpha (z - Q(s,a)) $$
    - Update win counts, visit counts and estimates.

# AlphaZero
- Combines the policy network and nvalue network into one network and generalizes for multiple games.
- Takes no human input, it only works on self-play and the only inputs are the gamr board and the rules.

# MCT Self-Refine (MCTSr):
- 