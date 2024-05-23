import numpy as np

class MultiArmedBandit:
    def __init__(self, num_arms, reward_distributions):
        self.num_arms = num_arms
        self.reward_distributions = reward_distributions

    def pull_arm(self, arm):
        return np.random.normal(self.reward_distributions[arm][0], self.reward_distributions[arm][1])

# Example usage:
# reward_distributions = [(mean1, std1), (mean2, std2), ..., (meanK, stdK)]
reward_distributions = [(0.2, 0.05), (0.5, 0.1), (0.7, 0.15), (0.3, 0.07), (0.4, 0.12)]
num_arms = len(reward_distributions)
bandit = MultiArmedBandit(num_arms, reward_distributions)
