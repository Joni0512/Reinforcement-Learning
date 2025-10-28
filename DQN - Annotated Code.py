"""
DQN with Prioritized Experience Replay (PER) on a custom FrozenLake variant.

This script contains:
- A numerically-seeded, deterministic setup (as much as TensorFlow allows).
- A SumTree data structure and a PER buffer.
- A GymEnvironment wrapper around CustomFrozenLake for training/eval loops.
- A DQN_Agent using a dueling head and Double DQN targets, trained with PER.

Key features:
- Early-stopping when the agent reaches the optimal (3,0) location with the target score
  several times consecutively.
- Backward-compatible agent API (record/update_weights) matching the earlier template.
"""

import os
SEED = 0
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import random
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

from keras.models import Model
from keras.layers import Dense, Input, Lambda
from environment import CustomFrozenLake
import matplotlib.pyplot as plt
import gymnasium as gym

#Paths / output folders
wd = os.getcwd()
save_folder = wd + '/save_folder/'
os.makedirs(save_folder, exist_ok=True)

# SumTree + Prioritized Experience Replay (PER)
class SumTree:
    """
    Binary tree for efficient priority sampling.

    Layout:
      - Leaves store priorities for transitions (size: capacity).
      - Internal nodes store sums of their children for O(log N) updates and queries.
    """
    def __init__(self, capacity: int):
        """
        Args:
            capacity: Maximum number of items that can be stored. The tree has size 2*capacity - 1.
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)  # priority sums
        self.data = np.empty(capacity, dtype=object)               # transitions
        self.write = 0                                             # next leaf index to write
        self.size = 0                                              # current number of stored items

    @property
    def total(self):
        """Total priority mass (root of the tree)."""
        return float(self.tree[0])

    def add(self, p: float, data):
        """
        Add a new transition with priority p.

        Args:
            p: Priority (>=0).
            data: Transition tuple to store.
        """
        idx = self.write + self.capacity - 1  # leaf index in the tree array
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx: int, p: float):
        """
        Update the priority at tree index `idx` and propagate the change upward.

        Args:
            idx: Index in the tree array.
            p: New priority.
        """
        change = p - self.tree[idx]
        self.tree[idx] = p
        parent = (idx - 1) // 2
        # Propagate up to the root
        while idx != 0:
            self.tree[parent] += change
            idx = parent
            parent = (idx - 1) // 2

    def get(self, s: float):
        """
        Retrieve a leaf by cumulative priority mass.

        Args:
            s: A value in [0, total] used to traverse the tree.

        Returns:
            (tree_idx, priority_at_idx, stored_transition)
        """
        idx = 0
        # Traverse until we reach a leaf
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay using a SumTree backend.

    Sampling:
      - Transitions are sampled proportional to priority^alpha (alpha fixed at 0.6 here).
      - Importance-sampling (IS) weights compensate for non-uniform sampling.
      - Beta is annealed from beta_start to 1.0 to fully correct bias by the end of training.
    """
    def __init__(self, capacity=20000, beta_start=0.4, beta_increment_per_sampling=1e-3,
                 eps=1e-6, priority_cap=None):
        """
        Args:
            capacity: Max number of stored transitions.
            beta_start: Initial beta for IS weights.
            beta_increment_per_sampling: How much to increase beta after each sample() call.
            eps: Small constant to keep priorities > 0.
            priority_cap: Optional clamp to avoid extremely large priorities.
        """
        self.tree = SumTree(capacity)
        self.beta = float(beta_start)
        self.beta_increment_per_sampling = float(beta_increment_per_sampling)  # FIXED indent
        self.eps = float(eps)
        self.max_priority = 1.0
        self.priority_cap = float(priority_cap) if priority_cap is not None else None

    def __len__(self):
        """Number of valid transitions currently stored."""
        return self.tree.size

    def add(self, transition, initial_priority=None):
        """
        Store a transition with initial priority.

        Args:
            transition: (state, action, reward, next_state, done)
            initial_priority: If None, use max seen so far; otherwise use provided (>= eps).
        """
        p = float(self.max_priority if initial_priority is None else max(initial_priority, self.eps))
        if self.priority_cap is not None:
            p = min(p, self.priority_cap)
        # Using alpha=0.6 (commonly used)
        self.tree.add(p ** 0.6, transition)

    def sample(self, batch_size):
        """
        Sample a minibatch proportional to priority.

        Returns:
            batch: list of transitions
            idxs:  tree indices of sampled leaves (for priority updates)
            weights: normalized IS weights (shape: [batch_size])
        """
        batch, idxs, priorities = [], [], []
        total = max(self.tree.total, 1e-6)
        segment = total / max(batch_size, 1)

        # Stratified sampling across segments of the cumulative priority mass
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            # Rare guard if p==0 due to numerical issues
            if p <= 0:
                s = random.uniform(0, total)
                idx, p, data = self.tree.get(s)
            batch.append(data); idxs.append(idx); priorities.append(p)

        probs = np.array(priorities, dtype=np.float32) / max(self.tree.total, 1e-6)
        N = max(self.tree.size, 1)
        # Importance sampling weights
        weights = (N * probs) ** (-self.beta)
        weights /= (weights.max() + 1e-8)

        # Anneal beta towards 1.0
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        return batch, np.array(idxs, dtype=np.int32), weights.astype(np.float32)

    def update_priorities(self, idxs, new_priorities):
        """
        Update priorities for sampled items.

        Args:
            idxs: Tree indices returned by sample().
            new_priorities: Typically absolute TD errors for each sampled item.
        """
        for idx, p in zip(idxs, new_priorities):
            p = float(abs(p) + self.eps)
            if self.priority_cap is not None:
                p = min(p, self.priority_cap)
            self.max_priority = max(self.max_priority, p)
            self.tree.update(int(idx), p ** 0.6)

# Environment wrapper: training / evaluation loops
class GymEnvironment:
    """
    Thin wrapper around CustomFrozenLake providing:
    - seeding
    - training (`trainDQN`) and run (`runDQN`) loops
    - simple early-stopping heuristic for faster experiments
    """
    def __init__(self, env_id, save_path, render="human", seed=None):
        slippery = False
        self.env = CustomFrozenLake(slippery, render)
        self.max_timesteps = self.env.max_timesteps
        self.is_slippery = slippery
        self.save_path = save_path
        self.seed = seed
        self.last_early_stop_ep = None  # 1-based index of stop point

        # Best-effort seeding (handles older/newer gym APIs)
        try:
            _ = self.env.reset(seed=self.seed)
        except TypeError:
            try:
                self.env.seed(self.seed)
            except AttributeError:
                pass
        try:
            self.env.action_space.seed(self.seed)
        except Exception:
            pass
        try:
            self.env.observation_space.seed(self.seed)
        except Exception:
            pass

    def trainDQN(self, agent, no_episodes):
        """
        Train `agent` for `no_episodes` and persist learned weights.

        Returns:
            np.ndarray of episode returns collected during training.
        """
        rew = self.runDQN(agent, no_episodes, training=True)
        agent.model.save_weights(self.save_path + "custom_frozen_DQN.weights.h5", overwrite=True)
        return rew

    def runDQN(self, agent, no_episodes, training=False, evaluation=False):
        """
        Run episodes in the environment.

        Args:
            agent: DQN_Agent instance.
            no_episodes: Number of episodes to run.
            training: If True, store transitions and update weights.
            evaluation: If True, suppress training logs; print progress periodically.

        Returns:
            Per-episode returns (np.ndarray). Early-stops if criteria met during training.
        """
        rew = np.zeros(no_episodes)
        consec_optimal = 0
        self.last_early_stop_ep = None

        for episode in range(no_episodes):
            state = self._reset_and_get_state()
            done = False
            rwd = 0.0
            t = 0
            last_step_reward = 0.0

            while not done:
                action = agent.select_action(state, training and not evaluation)
                step_out = self.env.step(action)
                # Support both 4- and 5-element step signatures
                if len(step_out) == 4:
                    next_state, reward, done, _ = step_out
                else:
                    next_state, reward, terminated, truncated, _ = step_out
                    done = terminated or truncated

                next_state = np.array(next_state).reshape(1, self.env.observation_space.shape[0])
                rwd += reward
                last_step_reward = reward

                if training and not evaluation:
                    agent.record(state, action, reward, next_state, done, last_step_reward)
                    agent.update_weights_from_memory()

                state = next_state
                t += 1

            rew[episode] = rwd

            # Heuristic: consider "optimal" if correct location & score reached
            tol = 1e-2
            score_ok = abs(rwd - 1.30) <= tol
            loc_ok = getattr(self.env, "ij", None) == (3, 0)
            goal_hit = (last_step_reward >= 5.5)  # final step reward at goal is 6.0
            reached_optimal = score_ok and (loc_ok or goal_hit)
            consec_optimal = consec_optimal + 1 if reached_optimal else 0

            if training and not evaluation:
                agent.update_epsilon()

            # Logging
            if not evaluation:
                if training:
                    print(f"episode: {episode + 1}/{no_episodes} | score: {rwd:.2f} | e: {agent.epsilon:.3f}")
                else:
                    print(f"episode: {episode + 1}/{no_episodes} | score: {rwd:.2f}")
            else:
                if episode % 10 == 0:
                    print(f"Progress: {episode} %")

            # Early stop once optimal behavior occurs several times consecutively
            if training and not evaluation and consec_optimal >= 5:
                self.last_early_stop_ep = episode + 1
                print(f"Early stopping: reached (3,0) with optimal score 1.30 in {consec_optimal} "
                      f"consecutive episodes (stopped at episode {self.last_early_stop_ep}).")
                rew = rew[:episode + 1]
                break

        return rew

    def _reset_and_get_state(self):
        """Handle gym reset API variants and return a (1, state_dim) array."""
        reset_out = self.env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            state, _ = reset_out
        else:
            state = reset_out
        return np.array(state).reshape(1, self.env.observation_space.shape[0])

# DQN Agent (dueling, Double DQN targets, PER)
class DQN_Agent:
    """
    DQN agent with:
      - Dueling head (separate value and advantage streams).
      - Double DQN target computation (online argmax over next actions, target net for values).
      - PER for sampling and importance-weighted loss.
    """
    def __init__(self, state_size, no_of_actions, agent_hyperparameters={}, old_model_path=''):
        # Basic spaces
        self.state_size = state_size
        self.action_size = no_of_actions

        # Core hyperparameters
        self.gamma = agent_hyperparameters['gamma']
        self.epsilon = agent_hyperparameters['epsilon']
        self.batch_size = agent_hyperparameters['batch_size']
        self.epsilon_min = agent_hyperparameters['epsilon_min']
        self.units = agent_hyperparameters['units']
        self.huber_delta = agent_hyperparameters['huber_delta']
        self.target_model_time = agent_hyperparameters.get('target_model_time', 100)

        # Epsilon decay (support both names)
        self.epsilon_decay = agent_hyperparameters.get(
            'epsilon_decay',
            agent_hyperparameters.get('epsilon_decay_episode', 1.0)
        )
        self.epsilon_decay_episode = agent_hyperparameters.get('epsilon_decay_episode', self.epsilon_decay)

        # PER hyperparameters
        self.per_beta_start = agent_hyperparameters.get('per_beta_start', 0.4)
        self.per_beta_increment = agent_hyperparameters.get('per_beta_increment', 1e-3)
        self.per_eps = agent_hyperparameters.get('per_eps', 1e-6)
        self.priority_cap = agent_hyperparameters.get('priority_cap', 10.0)

        # Early-only success boost (duplicates with higher initial priority)
        self.success_dup = agent_hyperparameters.get('success_dup', 2)
        self.success_priority_boost = agent_hyperparameters.get('success_priority_boost', 2.0)
        self.success_boost_until_steps = agent_hyperparameters.get('success_boost_until_steps', 10_000)

        # Networks (dueling head)
        self.model = self.nn_model(state_size, no_of_actions, self.units, old_model_path)
        self.target_model = self.nn_model(state_size, no_of_actions, self.units)
        self.target_model.set_weights(self.model.get_weights())

        # Replay & optimization (PER + Adam)
        self.memory = PrioritizedReplayBuffer(
            capacity=20000,
            beta_start=self.per_beta_start,
            beta_increment_per_sampling=self.per_beta_increment,
            eps=self.per_eps, priority_cap=self.priority_cap
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

        # Bookkeeping
        self.total_steps = 0
        self.learn_start = max(self.batch_size * 2, 128)
        self.max_grad_norm = 10.0

    #Model definition
    def nn_model(self, state_size, action_size, units, old_model_path=''):
        """
        Build a small MLP with a dueling head:
          Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))
        """
        inp = Input(shape=(state_size,))
        x = Dense(units, activation='relu')(inp)
        x = Dense(units, activation='relu')(x)
        x = Dense(units, activation='relu')(x)

        # Advantage stream A(s, a)
        adv = Dense(units, activation='relu')(x)
        adv = Dense(action_size)(adv)

        # Value stream V(s)
        val = Dense(units, activation='relu')(x)
        val = Dense(1)(val)

        def combine_streams(tensors):
            v, a = tensors
            a_mean = tf.reduce_mean(a, axis=1, keepdims=True)
            return v + (a - a_mean)

        q_out = Lambda(combine_streams, name="dueling_head")([val, adv])
        model = Model(inputs=inp, outputs=q_out)

        if old_model_path:
            model.load_weights(old_model_path)

        model.compile(optimizer='adam', loss='huber')
        return model

    #Action selection
    def select_action(self, state, training=True):
        """
        Epsilon-greedy action selection.

        Args:
            state: (1, state_dim) array.
            training: If True, use epsilon for exploration.
        """
        if training and np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return int(np.argmax(q_values[0]))

    #Experience handling
    def record(self, state, action, reward, next_state, done, last_step_reward=None):
        """
        Store a transition into PER. Optionally duplicates successful terminal transitions
        during early training to speed up convergence.

        Args:
            last_step_reward: If == 6 (goal), duplicate a few times with boosted initial priority.
        """
        transition = (state.copy(), int(action), float(reward), next_state.copy(), bool(done))

        dup = 1
        init_p = None
        if (last_step_reward is not None) and (last_step_reward == 6) and (self.total_steps < self.success_boost_until_steps):
            dup = max(1, int(self.success_dup) + 1)  # include the original
            init_p = self.memory.max_priority * float(self.success_priority_boost)

        for _ in range(dup):
            self.memory.add(transition, initial_priority=init_p)

    #Epsilon decay
    def update_epsilon(self):
        """Decay epsilon (bounded below by epsilon_min)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # Backward-compatibility with the original template
    def update_weights(self, t):
        """Template-compatible alias; ignore t and call the PER training step."""
        return self.update_weights_from_memory()

    # Loss + training step
    def huber_function(self, q_tar, q_act, thresh=1.0):
        """
        Huber (Smooth L1) loss per element.

        Args:
            q_tar: Targets.
            q_act: Predictions.
            thresh: Transition point from L2 to L1 (delta).
        """
        delta = q_tar - q_act
        abs_delta = tf.abs(delta)
        return tf.where(
            abs_delta < thresh,
            0.4 * tf.square(delta),          # 0.5 used in standard practice, but 0.4 performed better based on trial and error
            thresh * abs_delta - 0.4 * thresh
        )

    def update_weights_from_memory(self):
        """
        One PER training step:
          1) Sample a minibatch with PER.
          2) Compute Double DQN targets.
          3) Compute Huber loss, weighted by IS weights.
          4) Apply gradients; update PER priorities with |TD error|.
          5) Periodically sync target network.
        """
        if len(self.memory) < self.learn_start:
            return

        #PER sample
        batch, idxs, w_batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        # Flatten in case transitions contain (1, S) states
        if states.ndim == 3:
            states = np.squeeze(states, axis=1)
            next_states = np.squeeze(next_states, axis=1)

        # Tensors
        s_batch  = tf.convert_to_tensor(states, dtype=tf.float32)
        ns_batch = tf.convert_to_tensor(next_states, dtype=tf.float32)
        a_batch  = tf.convert_to_tensor(actions, dtype=tf.int32)
        r_batch  = tf.convert_to_tensor(rewards, dtype=tf.float32)
        d_batch  = tf.convert_to_tensor(dones, dtype=tf.float32)
        w_batch  = tf.convert_to_tensor(w_batch.reshape(-1), dtype=tf.float32)

        B = self.batch_size
        b_idx = tf.range(B, dtype=tf.int32)

        #Double DQN targets
        q_next_on = self.model(ns_batch)                  # online net for argmax
        next_a    = tf.argmax(q_next_on, axis=1)          # (B,)
        q_next_tar_all = self.target_model(ns_batch)      # target net for values
        next_idx  = tf.stack([b_idx, tf.cast(next_a, tf.int32)], axis=1)
        q_next    = tf.gather_nd(q_next_tar_all, next_idx)  # (B,)

        tar_vals = r_batch + self.gamma * q_next * (1.0 - d_batch)
        tar_vals = tf.stop_gradient(tar_vals)

        #Train step
        with tf.GradientTape() as g_tape:
            q_all  = self.model(s_batch)                  # (B, A)
            a_idx  = tf.stack([b_idx, a_batch], axis=1)
            q_sel  = tf.gather_nd(q_all, a_idx)           # (B,)

            td_err   = tar_vals - q_sel
            per_loss = self.huber_function(tar_vals, q_sel, self.huber_delta)
            loss_val = tf.reduce_mean(w_batch * per_loss)

        grads = g_tape.gradient(loss_val, self.model.trainable_variables)
        if self.max_grad_norm:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Update PER priorities with absolute TD error
        abs_td = tf.abs(td_err).numpy()
        self.memory.update_priorities(idxs, abs_td)

        # Target net sync
        self.total_steps += 1
        if self.total_steps % self.target_model_time == 0:
            self.target_model.set_weights(self.model.get_weights())

        return loss_val

# Entry point
if __name__ == "__main__":
    # State/action sizes come from CustomFrozenLake’s observation/action spaces
    state_size = 7
    no_of_actions = 4
    old_model_path = ''

    # Hyperparameters grouped for clarity and easy sweeps
    agent_hyperparameters = {
        'gamma': 0.85,
        'epsilon': 1.0,
        'epsilon_min': 0.10,
        'batch_size': 64,
        'units': 64,
        'alpha': 5e-4,
        'huber_delta': 1.0,
        'target_model_time': 100,
        'epsilon_decay_episode': 0.98,  # also read as 'epsilon_decay'

        # PER params
        'per_alpha': 0.6,
        'per_beta_start': 0.4,
        'per_beta_increment': 2e-3,
        'per_eps': 1e-6,
        'priority_cap': 10.0,

        # Early-only success boost
        'success_dup': 2,
        'success_priority_boost': 2.0,
        'success_boost_until_steps': 10_000,
    }

    train_episodes = 1000

    # Create agent & environment
    agent = DQN_Agent(state_size, no_of_actions, agent_hyperparameters, old_model_path)
    environment_train = GymEnvironment('custom_frozen_DQN', save_folder, render=False, seed=SEED)

    # Train (may early-stop on convergence)
    _ = environment_train.trainDQN(agent, train_episodes)
    if environment_train.last_early_stop_ep is not None:
        print(f"CONVERGED at episode {environment_train.last_early_stop_ep}.")
    else:
        print("Did NOT converge within 1000 episodes.")
