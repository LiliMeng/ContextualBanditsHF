

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

