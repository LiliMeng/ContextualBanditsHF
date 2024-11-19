import numpy as np
import random

def generate_complex_dataset(num_users=1000, num_items=500, dim=10, noise_level=0.2):
    """
    Generate a synthetic dataset for contextual bandit experiments.
    
    Parameters:
    - num_users: Number of users.
    - num_items: Number of items.
    - dim: Dimensionality of feature vectors.
    - noise_level: Noise level for reward generation.
    
    Returns:
    - user_features: User feature matrix.
    - item_features: Item feature matrix.
    - true_rewards: True reward matrix.
    """
    np.random.seed(42)
    
    user_features = np.random.rand(num_users, dim)
    item_features = np.random.rand(num_items, dim)
    
    interaction_effects = np.tanh(np.dot(user_features, item_features.T))
    sparse_mask = np.random.binomial(1, 0.2, size=interaction_effects.shape)
    sparse_rewards = interaction_effects * sparse_mask
    noisy_rewards = sparse_rewards + noise_level * np.random.randn(*sparse_rewards.shape)
    
    true_rewards = np.clip(noisy_rewards, 0, 1)
    return user_features, item_features, true_rewards

class HybridLinearBandit:
    def __init__(self, dim, num_arms, alpha=0.1, regularization=0.01, initial_entropy_threshold=0.5):
        self.theta = np.zeros(dim)
        self.num_arms = num_arms
        self.arm_features = np.random.rand(num_arms, dim)
        self.alpha = alpha
        self.regularization = regularization
        self.initial_entropy_threshold = initial_entropy_threshold
        self.current_entropy_threshold = initial_entropy_threshold

    def select_arm(self):
        scores = np.dot(self.arm_features, self.theta)
        probabilities = np.exp(scores) / np.sum(np.exp(scores))
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-9))

        if entropy > self.current_entropy_threshold:
            return np.random.choice(self.num_arms)
        return np.argmax(scores)

    def update(self, chosen_arm, reward, context, is_human_feedback=False):
        weight = 2.0 if is_human_feedback else 1.0
        gradient = (reward - np.dot(self.theta, context)) * context
        self.theta += self.alpha * weight * gradient
        self.theta -= self.regularization * self.theta

    def adjust_entropy_threshold(self, round_num, T, max_threshold=1.5):
        progress = round_num / T
        self.current_entropy_threshold = self.initial_entropy_threshold + progress * (max_threshold - self.initial_entropy_threshold)

    def hybrid_feedback(self, entropy):
        if entropy > self.current_entropy_threshold:
            return 'AR'
        return 'RM'

class EpsilonGreedy:
    def __init__(self, num_arms, epsilon=0.1):
        """
        Initialize Epsilon-Greedy bandit.
        Parameters:
        - num_arms: Number of available actions (arms).
        - epsilon: Probability of exploring a random action.
        """
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.q_values = np.zeros(num_arms)  # Estimated rewards for each arm
        self.arm_counts = np.zeros(num_arms)  # Number of times each arm is chosen

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_arms)  # Exploration
        return np.argmax(self.q_values)  # Exploitation

    def update(self, chosen_arm, reward):
        """
        Update the estimated rewards (q_values) based on the observed reward.
        """
        self.arm_counts[chosen_arm] += 1
        n = self.arm_counts[chosen_arm]
        self.q_values[chosen_arm] += (reward - self.q_values[chosen_arm]) / n

class UCB:
    def __init__(self, num_arms, confidence=2):
        """
        Initialize UCB bandit.
        Parameters:
        - num_arms: Number of available actions (arms).
        - confidence: Exploration parameter controlling uncertainty bounds.
        """
        self.num_arms = num_arms
        self.confidence = confidence
        self.q_values = np.zeros(num_arms)  # Estimated rewards for each arm
        self.arm_counts = np.zeros(num_arms) + 1e-6  # Avoid division by zero

    def select_arm(self):
        total_counts = np.sum(self.arm_counts)
        ucb_values = self.q_values + self.confidence * np.sqrt(np.log(total_counts) / self.arm_counts)
        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        """
        Update the estimated rewards (q_values) based on the observed reward.
        """
        self.arm_counts[chosen_arm] += 1
        n = self.arm_counts[chosen_arm]
        self.q_values[chosen_arm] += (reward - self.q_values[chosen_arm]) / n

def run_experiment_hybrid(bandit, context_features, item_features, true_rewards, T=1000, fixed_feedback=None):
    num_users, num_items = context_features.shape[0], item_features.shape[0]
    cumulative_regret = 0
    total_reward = 0

    for t in range(T):
        user_id = random.randint(0, num_users - 1)
        chosen_arm = bandit.select_arm()
        true_reward = true_rewards[user_id, chosen_arm]

        if fixed_feedback:
            feedback_type = fixed_feedback
        elif isinstance(bandit, HybridLinearBandit):
            scores = np.dot(bandit.arm_features, bandit.theta)
            probabilities = np.exp(scores) / np.sum(np.exp(scores))
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-9))
            feedback_type = bandit.hybrid_feedback(entropy)
        else:
            feedback_type = None

        if feedback_type == 'AR':
            feedback = true_reward + np.random.normal(0, 0.05)
            is_human_feedback = True
        elif feedback_type == 'RM':
            feedback = true_reward
            is_human_feedback = True
        else:
            feedback = true_reward
            is_human_feedback = False

        regret = max(true_rewards[user_id]) - true_reward
        cumulative_regret += regret
        total_reward += feedback

        if isinstance(bandit, HybridLinearBandit):
            bandit.update(chosen_arm, feedback, item_features[chosen_arm], is_human_feedback=is_human_feedback)
            bandit.adjust_entropy_threshold(t, T)
        else:
            bandit.update(chosen_arm, feedback)

    return cumulative_regret, total_reward


if __name__ == "__main__":
    user_features, item_features, true_rewards = generate_complex_dataset()

    T = 1000  # Number of rounds
    num_items = item_features.shape[0]  # Total number of items
    num_runs = 5  # Number of experiment runs for error bars

    methods = {
        "Epsilon-Greedy": lambda: EpsilonGreedy(num_items),
        "UCB": lambda: UCB(num_items),
        "Action Recommendation": lambda: HybridLinearBandit(dim=user_features.shape[1], num_arms=num_items),
        "Reward Manipulation": lambda: HybridLinearBandit(dim=user_features.shape[1], num_arms=num_items),
        "Hybrid Feedback": lambda: HybridLinearBandit(dim=user_features.shape[1], num_arms=num_items)
    }

    cumulative_reward_results = {method: [] for method in methods}
    cumulative_regret_results = {method: [] for method in methods}
    feedback_cost_results = {method: [] for method in methods}

    for _ in range(num_runs):
        for name, bandit_factory in methods.items():
            bandit = bandit_factory()
            feedback_cost = 0

            if name == "Action Recommendation":
                cumulative_regret, total_reward = run_experiment_hybrid(bandit, user_features, item_features, true_rewards, T, fixed_feedback='AR')
                feedback_cost = T  # AR requests feedback every round
            elif name == "Reward Manipulation":
                cumulative_regret, total_reward = run_experiment_hybrid(bandit, user_features, item_features, true_rewards, T, fixed_feedback='RM')
                feedback_cost = T  # RM requests feedback every round
            else:
                cumulative_regret, total_reward = run_experiment_hybrid(bandit, user_features, item_features, true_rewards, T)
                
                # Only calculate feedback cost if the method supports hybrid feedback
                if hasattr(bandit, 'hybrid_feedback'):
                    feedback_cost = sum(
                        1 for _ in range(T) if bandit.hybrid_feedback(bandit.current_entropy_threshold) in ['AR', 'RM']
                    )

            cumulative_reward_results[name].append(total_reward)
            cumulative_regret_results[name].append(cumulative_regret)
            feedback_cost_results[name].append(feedback_cost)

    # Calculate mean and standard deviation for error bars
    def calculate_stats(results):
        return {method: (np.mean(values), np.std(values)) for method, values in results.items()}

    reward_stats = calculate_stats(cumulative_reward_results)
    regret_stats = calculate_stats(cumulative_regret_results)
    feedback_cost_stats = calculate_stats(feedback_cost_results)

    # Plot cumulative rewards
    plt.figure(figsize=(12, 8))
    plt.bar(reward_stats.keys(), [v[0] for v in reward_stats.values()], yerr=[v[1] for v in reward_stats.values()], capsize=5, alpha=0.7)
    plt.ylabel('Cumulative Rewards')
    plt.xlabel('Methods')
    plt.title('Comparison of Cumulative Rewards Across Methods')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Plot cumulative regret
    plt.figure(figsize=(12, 8))
    plt.bar(regret_stats.keys(), [v[0] for v in regret_stats.values()], yerr=[v[1] for v in regret_stats.values()], capsize=5, alpha=0.7)
    plt.ylabel('Cumulative Regret')
    plt.xlabel('Methods')
    plt.title('Comparison of Cumulative Regret Across Methods')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Plot feedback cost
    plt.figure(figsize=(12, 8))
    plt.bar(feedback_cost_stats.keys(), [v[0] for v in feedback_cost_stats.values()], yerr=[v[1] for v in feedback_cost_stats.values()], capsize=5, alpha=0.7)
    plt.ylabel('Feedback Cost')
    plt.xlabel('Methods')
    plt.title('Comparison of Feedback Cost Across Methods')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()
