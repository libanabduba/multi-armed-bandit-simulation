import numpy as np

class Agent:
    def __init__(self, num_arms, epsilon):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.arm_counts = np.zeros(num_arms)
        self.arm_rewards = np.zeros(num_arms)

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_arms)
        else:
            return np.argmax(self.arm_rewards / (self.arm_counts + 1e-10))

    def update(self, arm, reward):
        self.arm_counts[arm] += 1
        self.arm_rewards[arm] += reward

# Example usage:
# agent = Agent(num_arms, epsilon=0.1)
