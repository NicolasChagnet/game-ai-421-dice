from ..Dice421.Player import Player
import numpy as np
from ..Dice421.logger import log
from ..Dice421.Game import BASIC_ACTIONS, PASS
import time


def parse_input(input_val):
    try:
        action = tuple(int(x) for x in input_val)
        print(f"Action chosen: {action}")
        if action not in BASIC_ACTIONS:
            return PASS
        return action
    except ValueError:
        return PASS


class ManualPlayer(Player):
    def __init__(self, env, name="ManualPlayer"):
        super().__init__(env, name)

    def get_next_action(self, state):

        time.sleep(2)
        action = parse_input(input("Which dice will you throw?"))
        return action

    def reset(self) -> None:
        pass

    def save_model(self, model_prefix: str = None):
        pass
