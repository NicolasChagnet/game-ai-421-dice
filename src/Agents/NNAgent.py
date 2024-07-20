from ..Dice421.Player import Player
from collections import defaultdict
import numpy as np


class NNPlayer(Player):
    """Agent built using the Gymnasium Blackjack agent as reference."""

    def __init__(
        self,
        env,
        learning_rate,
        initial_epsilon,
        epsilon_decay,
        final_epsilon,
        discount_factor=0.95,
        name="NNPlayer",
    ):
        super().__init__(env, name)
        self.q_values = defaultdict(lambda: np.zeros((2, 2, 2)))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_next_action(self, state):
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        state_hashed = tuple(state.values())
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            q_vals = self.q_values[state_hashed]
            action = np.unravel_index(np.argmax(q_vals), q_vals.shape)
            return action

    def learn(self, state, action, state_next, reward, done, info):
        """Updates the Q-value of an action."""
        state_hashed = tuple(state.values())
        state_next_hashed = tuple(state_next.values())
        # print(self.q_values)
        future_q_value = (not done) * np.max(self.q_values[state_next_hashed])
        temporal_difference = reward + self.discount_factor * future_q_value - self.q_values[state_hashed][action]

        self.q_values[state_hashed][action] = self.q_values[state_hashed][action] + self.lr * temporal_difference
        self.training_error.append(temporal_difference)

    def set_qvalues(self, q_values):
        self.q_values = q_values

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def reset(self) -> None:
        pass

    def save_model(self, model_prefix: str = None):
        pass
