import numpy as np
import matplotlib.pyplot as plt
from multi_armed_bandit import MultiArmedBandit
from agent import Agent

def run_simulation(num_arms, reward_distributions, epsilon, num_steps):
    bandit = MultiArmedBandit(num_arms, reward_distributions)
    agent = Agent(num_arms, epsilon)
    total_rewards = np.zeros(num_steps)
    
    for step in range(num_steps):
        arm = agent.select_arm()
        reward = bandit.pull_arm(arm)
        agent.update(arm, reward)
        total_rewards[step] = reward

    cumulative_rewards = np.cumsum(total_rewards)
    return cumulative_rewards

if __name__ == '__main__':
    reward_distributions = [(0.2, 0.05), (0.5, 0.1), (0.7, 0.15), (0.3, 0.07), (0.4, 0.12)]
    num_arms = len(reward_distributions)
    epsilon = 0.1
    num_steps = 1000

    cumulative_rewards = run_simulation(num_arms, reward_distributions, epsilon, num_steps)

    plt.plot(cumulative_rewards)
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.title('Epsilon-Greedy Multi-Armed Bandit')
    plt.savefig('multi_armed_bandit.png')
    plt.show()
