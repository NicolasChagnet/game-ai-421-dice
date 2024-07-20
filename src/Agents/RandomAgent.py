from ..Dice421.Player import Player
import numpy as np


class RandomPlayer(Player):
    def __init__(self, env="Dice421Env", name="RandomPlayer"):
        super().__init__(env, name)

    def get_next_action(self, state):
        # Just randomly sample the possible actions
        return self.env.action_space.sample()

    def reset(self) -> None:
        pass

    def save_model(self, model_prefix: str = None):
        pass
