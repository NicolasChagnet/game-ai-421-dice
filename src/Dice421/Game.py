from .Dice import Dice
from gymnasium import Env, spaces, error
import numpy as np
import itertools as it
from .logger import log, log_enable, log_disable

MAX_ROUNDS = 100
PASS = (0, 0, 0)
THROW_ALL = (1, 1, 1)
BASIC_ACTIONS = list(set(it.product((0, 1), (0, 1), (0, 1))))


LOSS_REWARD = -1
IMPROVEMENT_REWARD = 1
DRAW_REWARD = 0.5
WIN_GAME_REWARD = 100


class Dice421Env(Env):

    def __init__(self, seed=None):

        log.debug("Game initialization...")
        super(Dice421Env, self).__init__()

        # Main data of the game
        self.seed = seed  # Seed for random generators
        self.players = [None, None]  # Player instances
        self.scores = [0, 0]  # Scores of players
        self.dice = [Dice(seed=self.seed), Dice(seed=self.seed)]  # Dice instances
        self.gen = np.random.default_rng(seed=self.seed)

        # Required variables for Gymnasium
        self.metadata = {"render.modes": ["console"]}
        # Possible actions by the player
        # self.action_space = spaces.Discrete(len(BASIC_ACTIONS), seed=self.seed)  # Actions
        self.action_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2)), seed=self.seed)
        # Description of the states
        self.observation_space = spaces.Dict(
            {
                # Player's combination
                "player_comb": spaces.Box(low=0, high=6, shape=(3,)),
                # Opponents's combination
                "opp_comb": spaces.Box(low=0, high=6, shape=(3,)),
                # Current round number
                "round_nb": spaces.Discrete(MAX_ROUNDS),
                # Maximum number of throws allowed
                "max_throws": spaces.Discrete(3),
                # Current number of throws
                "current_throws": spaces.Discrete(3),
                # State of the round
                "state_round": spaces.Discrete(2),
                # Player score
                "player_score": spaces.Discrete(MAX_ROUNDS * 8),
                # Opponent's score
                "opp_score": spaces.Discrete(MAX_ROUNDS * 8),
            }
        )

        # Internal variables describing the events of the game
        self.__winner_round = -1  # Who won the last round
        self.__is_game_over = False  # Is the game over
        self.__winner_game = -1  # Who won the game
        self.__current_player = self.gen.integers(
            low=0, high=2
        )  # Who is currently playing (random for the first round)
        self.__state_round = 0  # Is the first or second player of the round currently playing
        self.__number_round = 0  # Running counter of rounds
        self.__current_combinations = [None, None]  # Combinations of the dice
        self.__max_throws = 3  # How many throws are allowed for the player
        self.__current_throw = 0  # Counter of throws by current player

        # Useful logging variable
        self.scores_history_ = [(0, 0)]  # History of scores in each round

    def reset_round(self):
        log.debug("Resetting round...")
        # At the beginning of each round, reset the variables
        # Set the first player to the winner of the previous round
        self.__current_player = max(self.__winner_round, self.gen.integers(low=0, high=2))
        self.__current_combinations = [None, None]
        self.__max_throws = 3
        self.__current_throw = 0

    def reset(self):
        log.debug("Resetting game...")
        # At the beginning of the game, reset all variables
        self.scores = [0, 0]
        self.dice = [Dice(seed=self.seed), Dice(seed=self.seed)]

        # self.action_space = spaces.Discrete(len(BASIC_ACTIONS), seed=self.seed)
        self.action_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2)), seed=self.seed)
        self.observation_space = spaces.Dict(
            {
                # Player's combination
                "player_comb": spaces.Box(low=0, high=6, shape=(3,)),
                # Opponents's combination
                "opp_comb": spaces.Box(low=0, high=6, shape=(3,)),
                # Current round number
                "round_nb": spaces.Discrete(MAX_ROUNDS),
                # Maximum number of throws allowed
                "max_throws": spaces.Discrete(3),
                # Current number of throws
                "current_throws": spaces.Discrete(3),
                # State of the round
                "state_round": spaces.Discrete(2),
                # Player score
                "player_score": spaces.Discrete(MAX_ROUNDS * 8),
                # Opponent's score
                "opp_score": spaces.Discrete(MAX_ROUNDS * 8),
            }
        )

        self.__winner_round = -1
        self.__state_round = 0
        self.__number_round = 0
        self.__is_game_over = False
        self.__winner_game = -1

        self.scores_history_ = [(0, 0)]

    def is_game_done(self):
        """Checks whether one player has won by score or whether the cutoff amount of rounds has been reached."""
        return abs(self.scores[0] - self.scores[1]) >= 21 or self.__number_round >= MAX_ROUNDS

    def get_combinations_values(self):
        """Returns the current combinations in the round as integers (0 if the player has not played yet)."""
        return [
            combination.get_value() if combination is not None else 0 for combination in self.__current_combinations
        ]

    def get_observation_space(self):
        """Build observation space out of the variables."""
        player_combination = self.__current_combinations[self.__current_player]
        other_combination = self.__current_combinations[1 - self.__current_player]
        return {
            "player_comb": tuple(player_combination.get_values()) if player_combination is not None else (0, 0, 0),
            "opp_comb": tuple(other_combination.get_values()) if other_combination is not None else (0, 0, 0),
            "round_nb": self.__number_round,
            "max_throws": self.__max_throws,
            "current_throws": self.__current_throw,
            "state_round": self.__state_round,
            "player_score": self.scores[self.__current_player],
            "opp_score": self.scores[1 - self.__current_player],
        }

    def get_reward(self, new_combination):
        """Returns the reward from the previous action.
        The reward is a sum of points depending on various conditions:
        - `-1` point if the new combination is worse than the previous one or if it loses to the other player's combination
        - `0.5` point if the new combination only equalizes to the other player's combination
        - `1` point if the new combination is an improvement on the previous one
        - `2` point if the new combination beats the current combination of the other player
        - `N` points if the new combination leads the player to win the round
        - `100` points if the new combination leads the player to win the game
        """
        old_combination, other_combination = tuple(self.__current_combinations)
        factor_improvement = int((old_combination is not None) and (new_combination > old_combination))
        factor_equalizing = int((other_combination is not None) and (new_combination == other_combination))
        factor_beating = int((other_combination is not None) and (new_combination > other_combination))
        factor_losing = int(
            ((old_combination is not None) and (new_combination < old_combination))
            or ((other_combination is not None) and (new_combination < other_combination))
        )
        factor_round_winning = int(self.get_win_round_estimation(new_combination))
        factor_game_winning = int(self.get_win_estimation(new_combination))
        reward = (
            factor_losing * LOSS_REWARD
            + factor_equalizing * DRAW_REWARD
            + factor_improvement * IMPROVEMENT_REWARD
            + factor_beating * IMPROVEMENT_REWARD * 2
            + factor_round_winning * new_combination.get_points()
            + factor_game_winning * WIN_GAME_REWARD
        )
        log.debug(
            f"Reward matrix: {[factor_losing, factor_equalizing, factor_improvement, factor_beating, factor_round_winning,factor_game_winning]}"
        )
        log.debug(f"Reward for action: {reward}")
        return reward

    def compute_new_score_and_winner_round(self):
        """Checks which combination is higher and computes new scores. Sets the winner of the round."""
        if self.__current_combinations[0] > self.__current_combinations[1]:
            self.scores[0] += self.__current_combinations[0].get_points()
            self.__winner_round = 0
        elif self.__current_combinations[0] < self.__current_combinations[1]:
            self.scores[1] += self.__current_combinations[1].get_points()
            self.__winner_round = 1
        else:
            self.__winner_round = -1

    def get_win_round_estimation(self, combination):
        """Checks whether a combination leads the player to win the round (must be in a position to end the round)."""
        end_round = self.__state_round == 1
        is_winner_round = (
            (self.__current_combinations[0] is not None)
            and (self.__current_combinations[1] is not None)
            and (combination > self.__current_combinations[1 - self.__current_player])
        )
        return end_round and is_winner_round

    def get_win_estimation(self, combination):
        leads_winner_game = (
            combination.get_points() + self.scores[self.__current_player] - self.scores[1 - self.__current_player]
            >= 21
        )
        return self.get_win_round_estimation(combination) and leads_winner_game

    def step(self, action):
        # Gets the new combination from the player's action
        new_combination = self.dice[self.__current_player].throw_dice(action)
        other_combination = self.__current_combinations[self.__current_player]
        log.debug(
            f"Player {self.players[self.__current_player].get_name()}'s new combination: {new_combination.get_value()} vs {other_combination.get_value() if other_combination is not None else 000}"
        )
        # Computes the reward from the player's action
        reward = self.get_reward(new_combination)
        # Updates combinations
        self.__current_combinations[self.__current_player] = new_combination
        return (
            self.get_observation_space(),
            reward,
            self.get_win_estimation(new_combination),
            {},
        )

    def run(self, player1, player2, render=False, player1_learn=True, player2_learn=True):
        if render:
            log_enable()
        else:
            log_disable()
        log.info("Starting the game...")
        # Define the players
        self.players = [player1, player2]
        player1.reset()
        player2.reset()
        # Reset the game
        self.reset()
        # Main game loop
        while not self.is_game_done():
            log.info(f"Starting round number {self.__number_round}")
            # Each loop is a new round
            self.reset_round()

            # Start with the player
            log.info(f"Player {self.players[self.__current_player].get_name()} has {self.__max_throws} throws")
            for i in range(self.__max_throws):
                self.__current_throw = i
                log.debug(f"Throw number {self.__current_throw + 1}")
                # The first throw is always of all dice!
                current_observation = self.get_observation_space().copy()
                if i == 0:
                    action = THROW_ALL
                else:
                    action = self.players[self.__current_player].get_next_action(current_observation)

                log.debug(f"Player {self.players[self.__current_player].get_name()}'s action: {action}")
                # Execute the action
                output_step = self.step(action)
                # We call the hook for the player
                if i > 0 and player1_learn:
                    self.players[self.__current_player].learn(current_observation, action, *output_step)
                log.info(f"Combinations are {self.get_combinations_values()}")
                # If the first player of the round passes, defined the max number of throws
                if action == PASS:
                    log.debug(f"Player {self.players[self.__current_player].get_name()} passed!")
                    self.__max_throws = self.__current_throw
                    break

            # Change player
            self.__current_player = 1 - self.__current_player
            self.__state_round = 1 - self.__state_round

            # Second player
            log.info(f"Player {self.__current_player} has {self.__max_throws} throws")
            for i in range(self.__max_throws):
                self.__current_throw = i
                log.debug(f"Throw number {self.__current_throw + 1}")
                # The first throw is always all dice!
                current_observation = self.get_observation_space().copy()
                if i == 0:
                    action = THROW_ALL
                else:
                    action = self.players[self.__current_player].get_next_action(self.get_observation_space())

                log.debug(f"Player {self.players[self.__current_player].get_name()}'s action: {action}")
                # Execute the action
                output_step = self.step(action)
                if i > 0 and player2_learn:
                    self.players[self.__current_player].learn(current_observation, action, *output_step)
                log.info(f"Combinations are {self.get_combinations_values()}")
                # If the second player of the round passes, stop the loop
                if action == PASS:
                    log.debug(f"Player {self.players[self.__current_player].get_name()} passed!")
                    break
            # Once both players have finished, compute the winner of the round and update the scores
            _ = self.compute_new_score_and_winner_round()
            self.scores_history_ += [self.scores]
            log.info(f"The current scores are {self.scores}")
            # Go to the next round (if there is a tie, do not increment)
            if self.__winner_round >= 0:
                log.info(f"The winner of the round is {self.players[self.__winner_round].get_name()}")
                self.__number_round += 1

        self.__is_game_over = 1
        log.info("Game finished!")
        if self.__number_round >= MAX_ROUNDS:
            log.info("Maximum number of rounds reached!")
        else:
            self.__winner_game = self.scores.index(max(self.scores))
            log.info(f"The winner of the game is {self.players[self.__winner_game].name}")

        return {
            "winner": self.__winner_game,
            "final_scores": self.scores,
            "number_rounds": self.__number_round,
            "scores_history": self.scores_history_,
        }

    def render(self, mode="console", close=False):
        if mode == "console":
            log.info(f"Round {self.__number_round}")
            log.info(f"Combinations: {self.__current_combinations}")
            print(f"Score: {self.scores}")
        else:
            raise error.UnsupportedMode()

    def close(self):
        pass
