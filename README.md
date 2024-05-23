# Multi-Armed Bandit Simulation

This repository contains an implementation of the Single-State Multi-Armed Bandit problem using an epsilon-greedy algorithm. The objective is to maximize the total cumulative reward received over a fixed number of time steps by selecting the best arms to pull.

## Problem Definition

### Environment
- The environment is a single-state, multi-armed bandit problem with a single state.
- There are multiple arms (actions) available for the agent to choose from, typically denoted as K arms (e.g., K = 10 arms).

### Agent
- The agent's goal is to maximize the total reward it accumulates over a fixed number of time steps by selecting the best arms to pull.
- The agent must balance exploration (trying new arms to find better rewards) and exploitation (choosing arms with known high rewards).

### Actions
- The agent can take one of K actions at each time step, where each action corresponds to pulling one of the K arms.
- Once an arm is pulled, the agent receives a reward and observes the outcome.

### Rewards
- Each arm has a reward distribution that determines the reward received when the arm is pulled.
- The rewards are stochastic (random) and follow a specific distribution (e.g., normal distribution, Bernoulli distribution).
- The mean reward and variance of each arm's distribution may be unknown to the agent initially.

### Objective
- The agent's objective is to maximize the total cumulative reward it receives over a fixed number of time steps N.
- The agent must learn which arms provide the highest expected rewards while also exploring less-known arms.

## Implementation

The implementation consists of three main components:

1. **Environment**: Simulates the multi-armed bandit environment.
2. **Agent**: Implements the epsilon-greedy strategy.
3. **Simulation**: Runs the simulation and evaluates the agent's performance.

### Files

- `multi_armed_bandit.py`: Defines the environment for the multi-armed bandit problem.
- `agent.py`: Implements the epsilon-greedy strategy for the agent.
- `run_simulation.py`: Runs the simulation and evaluates the agent's performance.

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- Gymnasium

## Installation

1. Clone the repository:
   
   git clone https://github.com/libanabduba/multi-armed-bandit-simulation.git
   cd multi-armed-bandit-simulation

2. Create and activate a virtual environment:

    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install the required packages:

    pip install -r requirements.txt
 

## Running the Simulation

    python run_simulation.py




