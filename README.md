Reinforcement Learning – DQN & PPO on Custom FrozenLake

This project implements Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) to solve a custom FrozenLake environment provided as part of the course
“Introduction to Deep Reinforcement Learning” at the Technical University of Munich (TUM).

The task was to train an agent to find the optimal path to a treasure while avoiding terminal failure states, using only self-implemented deep reinforcement learning algorithms.

📌 Task Overview

The environment is a modified FrozenLake grid world (Gym-based) with:

7 state variables

2 for the agent’s position

5 affecting the reward of entering a state

4 deterministic actions: up, down, left, right

Terminal states:

3 positive reward states (treasures)

multiple negative reward states (lake breakpoints)

The reward structure is non-uniform and path-dependent, meaning not all optimal paths yield the same total reward, which makes exploration and credit assignment challenging (see reward heatmap in the task slides).

🎯 Objectives (from the exam task)

Implement DQN and PPO from scratch

Train an agent to reach a treasure without falling into terminal failure states

Compare both algorithms with respect to:

convergence speed

stability

final performance

Analyse the influence of hyperparameters and architecture choices

The examination consisted of:

Algorithm implementation (DQN & PPO)

Presentation + Q&A

Technical report

🧠 Algorithms Implemented
DQN (Deep Q-Network)

Value-based method using a neural network to approximate Q-values

Trained via double dqn, dueling networks, prioritized replay and temporal-difference learning

PPO (Proximal Policy Optimization)

Policy-gradient method

Uses clipped objective to ensure stable updates

Advantage estimation for variance reduction

Both agents were evaluated on their average episode reward and success rate.
