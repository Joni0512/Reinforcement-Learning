import os
import random
import numpy as np
import scipy.signal
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Dense, Input
from keras.models import Model
from environment import CustomFrozenLake  # Your custom env

# Module configuration and hyperparameters 
BASE_SEED = 1          # starting seed
N_SEEDS = 1           # code runs over N_SEEDS seeds
train_episodes = 500
test_episodes = 50
actors = 5

agent_hyperparameters = {
    "state_size": 7,
    "action_size": 4,
    "gamma": 0.99,          # Discount factor for future rewards
    "lam": 0.9,             # Lambda parameter for GAE-Lambda
    "clip_ratio": 0.2,      # PPO clipping parameter epsilon
    "actors": actors,       # Number of parallel actors
    "max_timesteps": None,  # Set by environment
    "actor_lr": 2.82e-4,    # Learning rate for policy network
    "critic_lr": 2e-4,      # Learning rate for value network
}

# Output directory setup
wd = os.getcwd()
save_folder = os.path.join(wd, "save_folder")
os.makedirs(save_folder, exist_ok=True)


# Utilities
def set_global_seed(seed):
    """
    Set seeds across Python, NumPy, and TensorFlow for reproducibility.
    Also attempts to enable deterministic TF ops (if available) and clears Keras session to avoid leftover graph/state between runs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism(True)
    except Exception:
        pass
    tf.keras.backend.clear_session()

def discounted_cumulative_sums(x, discount):
    """
    Compute discounted cumulative sums of a sequence.

    Args:
        x (np.ndarray): Input sequence to discount
        discount (float): Discount factor to apply

    Returns:
        np.ndarray: Discounted cumulative sums
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

# PPO Agent
class PPO_Agent:
    """
    Implementation of Proximal Policy Optimization (PPO) algorithm.
    """

    def __init__(self, hp):
        """
        Initialize the PPO agent with networks and buffers.

        Args:
            hp (dict): Hyperparameters for the agent including state_size,
                      action_size, gamma, lambda, learning rates
        """
        self.__dict__.update(hp)
        self.entropy_coeff = 0.01
        self.train_policy_epochs = 5
        self.train_value_function_epochs = 5
        self.minibatch_size = 32
        self.buffer = {k: [] for k in ["states", "actions", "rewards", "logits", "dones", "values"]}

        self.actor  = self._build_mlp((self.state_size,), self.action_size, [64, 64], logits=True)
        self.critic = self._build_mlp((self.state_size,), 1,            [64, 64], logits=False)
        self.policy_optimizer = tf.keras.optimizers.Adam(self.actor_lr)
        self.value_optimizer  = tf.keras.optimizers.Adam(self.critic_lr)

    def _build_mlp(self, input_shape, output_size, units, logits):
        """
        Build a multi-layer perceptron network.

        Args:
            input_shape (tuple): Shape of input layer
            output_size (int): Number of output units
            units (list): List of hidden layer sizes
            logits (bool): Whether to use logit outputs
        """
        x = Input(shape=input_shape)
        h = x
        for u in units:
            h = Dense(u, activation='relu')(h)
        act = None if logits else 'linear'
        out = Dense(output_size, activation=act)(h)
        return Model(inputs=x, outputs=out)

    def select_action(self, state):
        """
        Select an action based on the current policy.

        Args:
            state: Current environment state

        Returns:
            tuple: (policy logits, selected action)
        """
        s = np.asarray(state, dtype=np.float32).reshape(1, -1)
        logits = self.actor(s)
        probs  = tf.nn.softmax(logits)
        action = tf.random.categorical(tf.math.log(probs), 1)
        return logits.numpy().squeeze(), int(action.numpy().squeeze())

    def record(self, s, a, r, logit, done, v):
        """
        Store a transition in the experience buffer.

        Args:
            s: State
            a: Action
            r: Reward
            logit: Policy logits
            done: Episode termination flag
            v: Value estimate
        """
        self.buffer["states"].append(np.asarray(s, dtype=np.float32))
        self.buffer["actions"].append(int(a))
        self.buffer["rewards"].append(float(r))
        self.buffer["logits"].append(np.asarray(logit, dtype=np.float32))
        self.buffer["dones"].append(bool(done))
        self.buffer["values"].append(float(v))

    def _log_probs(self, logits, actions):
        """
        Compute log probabilities of actions.

        Args:
            logits: Raw policy network outputs
            actions: Selected actions
            
        Returns:
            tf.Tensor: Log probabilities of the actions
        """
        log_probs = tf.nn.log_softmax(logits)
        one_hot   = tf.one_hot(actions, self.action_size)
        return tf.reduce_sum(one_hot * log_probs, axis=1)

    def calc_advantage(self, last_value=0.0, last_done=True):
        """
        Calculate advantages using Generalized Advantage Estimation (GAE).

        Args:
            last_value: Value estimate for final state if episode didn't end
            last_done: Whether episode terminated naturally
        """
        r = np.array(self.buffer["rewards"], dtype=np.float32)
        v = np.array(self.buffer["values"],  dtype=np.float32)
        d = np.array(self.buffer["dones"],   dtype=np.float32)
        T = len(r)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        next_v = last_value
        next_not_done = 0.0 if last_done else 1.0
        for t in reversed(range(T)):
            v_tp1   = next_v if t == T - 1 else v[t + 1]
            not_done = next_not_done if t == T - 1 else 1.0 - d[t]
            delta = r[t] + self.gamma * v_tp1 * not_done - v[t]
            gae   = delta + self.gamma * self.lam * not_done * gae
            adv[t] = gae
        self.buffer["advantages"] = adv
        self.buffer["returns"]    = adv + v

    def train_policy(self):
        """
        Update policy network using PPO clipped objective.

        Updates policy using batched gradient descent on PPO loss
        with clipping and entropy regularization.
        """
        buf = {k: np.array(self.buffer[k]) for k in self.buffer}
        adv = buf["advantages"]
        buf["advantages"] = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)

        N = len(buf["states"])
        idxs = np.arange(N)
        for _ in range(self.train_policy_epochs):
            np.random.shuffle(idxs)
            for i in range(0, N, self.minibatch_size):
                mb = idxs[i:i+self.minibatch_size]
                with tf.GradientTape() as tape:
                    new_logits = self.actor(buf["states"][mb])
                    old_logp   = self._log_probs(buf["logits"][mb], buf["actions"][mb])
                    new_logp   = self._log_probs(new_logits,      buf["actions"][mb])
                    ratio      = tf.exp(new_logp - old_logp)
                    unclipped  = ratio * buf["advantages"][mb]
                    clipped    = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * buf["advantages"][mb]
                    entropy    = -tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(new_logits) * tf.nn.log_softmax(new_logits), axis=1))
                    loss       = -tf.reduce_mean(tf.minimum(unclipped, clipped)) - self.entropy_coeff * entropy
                grads = tape.gradient(loss, self.actor.trainable_variables)
                self.policy_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

    def train_value_function(self):
        """
        Train critic network to predict state values.

        Updates value function by minimizing MSE between
        predicted values and computed returns.
        """
        states  = np.array(self.buffer["states"],  dtype=np.float32)
        returns = np.array(self.buffer["returns"], dtype=np.float32)
        N = len(states)
        idxs = np.arange(N)
        for _ in range(self.train_value_function_epochs):
            np.random.shuffle(idxs)
            for i in range(0, N, self.minibatch_size):
                mb = idxs[i:i+self.minibatch_size]
                with tf.GradientTape() as tape:
                    values = tf.squeeze(self.critic(states[mb]), axis=1)
                    loss   = tf.reduce_mean((returns[mb] - values) ** 2)
                grads = tape.gradient(loss, self.critic.trainable_variables)
                self.value_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

    def reset_buffer(self):
        """
        Clear on-policy buffer after an update cycle to start fresh.
        """
        self.buffer = {k: [] for k in self.buffer}


# Environment Wrapper
class GymEnvironment:
    """
    Environment wrapper for CustomFrozenLake.
    """
    
    def __init__(self, save_path, render=False, seed=None):
        """
        Initialize environment wrapper.

        Args:
            save_path: Directory for saving results
            render: Whether to render environment
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.env = CustomFrozenLake(slippery=False, render=render)
        self.max_timesteps = self.env.max_timesteps
        self.save_path = save_path
        try:
            self.env.action_space.seed(seed)
            self.env.observation_space.seed(seed)
        except Exception:
            pass

    def runPPO(self, agent, no_episodes, training=False):
        """
        Run episodes using PPO agent.

        Args:
            agent: PPO agent instance
            no_episodes: Number of episodes to run
            training: Whether to collect training data

        Returns:
            tuple: (rewards array, episode count, goal counts)
        """
        rew = np.zeros((agent.actors, no_episodes), dtype=np.float32)
        goal_counts = {'goal_5': 0, 'goal_6': 0}

        for episode in range(no_episodes):
            for n in range(agent.actors):
                tot_rew = 0.0
                ep_seed = None if self.seed is None else (int(self.seed) + episode * agent.actors + n)
                state = self.env.reset(seed=ep_seed) if hasattr(self.env, 'reset') else self.env.reset()
                last_state = state
                timed_out = True

                for t in range(agent.max_timesteps):
                    logit, action = agent.select_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    tot_rew += reward

                    if reward == 5:
                        goal_counts['goal_5'] += 1
                    elif reward == 6:
                        goal_counts['goal_6'] += 1

                    if training:
                        value = float(agent.critic(np.expand_dims(state, axis=0)))
                        agent.record(state, action, reward, logit, done, value)

                    state = next_state
                    last_state = state
                    if done:
                        timed_out = False
                        break

                if training:
                    last_value = float(agent.critic(np.expand_dims(last_state, axis=0))) if timed_out else 0.0
                    last_done  = not timed_out
                    agent.calc_advantage(last_value, last_done)

                rew[n, episode] = tot_rew

            if training:
                agent.train_policy()
                agent.train_value_function()
                agent.reset_buffer()

            # Optional code to print score per episodes to monitor training progress
            avg_reward = float(np.mean(rew[:, episode]))
            print(f"episode: {episode+1}/{no_episodes} | score: {avg_reward:.2f}")

        return rew[:, :episode+1], episode, goal_counts


# Runner over many seeds
def run_one_seed(seed_value: int):
    """
    Run full training and evaluation cycle for one seed.

    Args:
        seed_value: Random seed for reproducibility

    Returns:
        dict: Training and testing metrics for this seed
    """
    set_global_seed(seed_value)

    env = GymEnvironment(save_folder, render=False, seed=seed_value)
    agent_hyperparameters["max_timesteps"] = env.max_timesteps
    agent = PPO_Agent(agent_hyperparameters)

    # Train
    rew_train, _, goal_counts_train = env.runPPO(agent, train_episodes, training=True)

    # Test
    agent.actors = 1
    test_env = GymEnvironment(save_folder, render=False, seed=seed_value)
    rew_test, _, goal_counts_test = test_env.runPPO(agent, test_episodes, training=False)

    avg_train   = np.mean(rew_train, axis=0)   # (train_episodes,)
    test_series = rew_test[0]                  # (test_episodes,)
    test_return = float(np.sum(test_series))   # <-- ranking metric = score
    test_goals  = int(goal_counts_test['goal_5'] + goal_counts_test['goal_6'])

    return {
        "seed": seed_value,
        "avg_train": avg_train,
        "test_series": test_series,
        "test_return": test_return,   # for ranking
        "test_goals": test_goals,     # for summary table
        "goal5": int(goal_counts_test['goal_5']),
        "goal6": int(goal_counts_test['goal_6']),
    }


# Main: run multiple seeds, plot top-5 by score, print table
if __name__ == "__main__":
    all_runs = []
    for s in range(BASE_SEED, BASE_SEED + N_SEEDS):
        print(f"\n=== Running seed {s} ===")
        d = run_one_seed(s)
        all_runs.append(d)
        print(f"Seed {s}: test_return={d['test_return']:.1f}, "
              f"left goals={d['goal6']}, right goals={d['goal5']}")

    # Rank by testing score (sum of rewards); left-goal seeds will dominate
    all_runs.sort(key=lambda d: d["test_return"], reverse=True)
    top5 = all_runs[:5]

    # Plot: for each of top5 seeds, show Training + Testing with boundary 
    plt.figure(figsize=(11, 5))
    for d in top5:
        avg_train   = d["avg_train"]
        test_series = d["test_series"]
        x_train = np.arange(1, len(avg_train) + 1)
        x_test  = np.arange(len(avg_train) + 1, len(avg_train) + len(test_series) + 1)
        # continuous look: draw both segments in same color
        line, = plt.plot(x_train, avg_train, linewidth=1.5,
                         label=f"Seed {d['seed']} (score={d['test_return']:.1f}, L={d['goal6']}, R={d['goal5']})")
        plt.plot(x_test,  test_series, linewidth=1.5, color=line.get_color())

    # Train/Test boundary
    boundary = len(top5[0]["avg_train"]) if top5 else train_episodes
    plt.axvline(x=boundary, linestyle="--", color="gray", alpha=0.85, label="Train/Test boundary")

    plt.title(f"PPO — Top 5 Seeds by Test Score (Training + Testing)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    out_plot = os.path.join(save_folder, f"top5_by_score_train_plus_test_{BASE_SEED}-{BASE_SEED+N_SEEDS-1}.png")
    plt.savefig(out_plot, bbox_inches="tight", dpi=150)
    plt.show()
    print(f"Saved top-5 (by score) training+testing plot to: {out_plot}")

    # ---------- Summary table: % of seeds by total goals during testing ----------
    more_than_45 = sum(1 for d in all_runs if d["test_goals"] > 45)
    between_20_45 = sum(1 for d in all_runs if 20 <= d["test_goals"] <= 45)
    less_than_20  = sum(1 for d in all_runs if d["test_goals"] < 20)

    total = float(N_SEEDS)
    summary = pd.DataFrame({
        "Category": [
            f"> 45 goals (of {test_episodes})",
            f"20–45 goals (of {test_episodes})",
            f"< 20 goals (of {test_episodes})"
        ],
        "Count": [more_than_45, between_20_45, less_than_20],
        "Percentage": [
            100.0 * more_than_45 / total,
            100.0 * between_20_45 / total,
            100.0 * less_than_20  / total
        ]
    })

    print("\n=== Summary over seeds ===")
    print(summary.to_string(index=False, formatters={"Percentage": "{:.1f}%".format}))

    out_csv = os.path.join(save_folder, f"summary_goals_{BASE_SEED}-{BASE_SEED+N_SEEDS-1}.csv")
    summary.to_csv(out_csv, index=False)
    print(f"Saved summary CSV to: {out_csv}")

    # Also save per-seed metrics in case you want to inspect
    per_seed = pd.DataFrame([{
        "seed": d["seed"],
        "test_return": d["test_return"],
        "test_goals": d["test_goals"],
        "goal_left": d["goal6"],
        "goal_right": d["goal5"],
    } for d in all_runs]).sort_values("test_return", ascending=False)
    per_seed_path = os.path.join(save_folder, f"per_seed_metrics_{BASE_SEED}-{BASE_SEED+N_SEEDS-1}.csv")
    per_seed.to_csv(per_seed_path, index=False)
    print(f"Saved per-seed metrics to: {per_seed_path}")
