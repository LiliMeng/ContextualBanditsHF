
### **Explanation of the Full Setup**

1. **Dataset**:
   - Simulates a realistic recommendation environment with user and item feature vectors and sparse, noisy rewards.

2. **Baselines**:
   - **Epsilon-Greedy**: Simple exploration-exploitation trade-off.
   - **UCB**: Optimistic approach based on confidence bounds.

3. **HybridLinearBandit**:
   - Dynamically switches between AR and RM feedback based on entropy.

4. **Evaluation**:
   - Compares the cumulative rewards of all methods over multiple runs.
   - Includes error bars to account for variability.


### **Experiment Setup**

#### **Dataset Construction**

To evaluate the proposed **HybridLinearBandit** method, we constructed a synthetic dataset simulating a recommendation environment. The dataset includes:

- **Contextual Features**:
  - Each context \(\mathbf{x}_t \in \mathbb{R}^d\) represents the feature vector for the current interaction.
  - Actions \(a \in \mathcal{A}\) are associated with feature vectors \(\mathbf{x}_{t,a} \in \mathbb{R}^d\).

- **Reward Generation**:
  - The true reward \(r_t\) for an action is modeled as:
    \[
    r_t = \mathbf{x}_{t,a}^\top \theta^* + \epsilon_t
    \]
    where \(\theta^* \in \mathbb{R}^d\) is a latent parameter vector, and \(\epsilon_t\) is Gaussian noise \(\mathcal{N}(0, \sigma^2)\) to simulate real-world uncertainty.
  - Rewards are normalized to lie within the range \([0, 1]\).

- **Non-Linear Interactions**:
  - We introduce non-linear dependencies by applying \(\tanh\) transformations to user-item interactions, capturing more complex patterns.

- **Sparse Rewards**:
  - To simulate practical sparsity in rewards, only a subset of user-item interactions provides meaningful rewards, controlled by a sparsity mask.

#### **Baselines and Methods**

We compare our **HybridLinearBandit** approach against the following baselines:

- **Epsilon-Greedy**:
  - Selects actions randomly with probability \(\epsilon = 0.1\) and otherwise exploits the best-known action.

- **UCB (Upper Confidence Bound)**:
  - Balances exploration and exploitation by selecting actions with the highest upper confidence bound.

- **Action Recommendation (AR)**:
  - Always requests human feedback to directly guide the action selection.

- **Reward Manipulation (RM)**:
  - Always requests human feedback to refine the observed reward signal.

- **Hybrid Feedback (Ours)**:
  - Dynamically switches between AR and RM based on the model’s uncertainty, leveraging entropy.

#### **Experiment Settings**

- **Rounds and Runs**:
  - Each experiment runs for \(T = 1000\) rounds, with \(5\) independent trials to account for variability.

- **Hyperparameters**:
  - Initial entropy threshold \(\tau_{\text{init}} = 0.5\).
  - Maximum entropy threshold \(\tau_{\text{max}} = 1.5\).
  - Learning rate \(\alpha = 0.1\).
  - Regularization parameter \(\lambda = 0.01\).

- **Evaluation Metrics**:
  - **Cumulative Reward**: Measures the total reward achieved over \(T\) rounds.
  - **Cumulative Regret**: The gap between the optimal reward and the agent’s reward:
    \[
    \text{Regret}_T = \sum_{t=1}^T \left( r_t^* - r_t \right)
    \]
  - **Feedback Cost**: Tracks the frequency and type of human feedback requested.

