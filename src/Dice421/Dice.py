import numpy as np
from .Combination import Combination


class Die:
    def __init__(self, seed=None):
        # Random generator
        self.gen = np.random.default_rng(seed=seed)
        # Value of the die
        self.value = 0

    def throw_die(self):
        self.value = self.gen.integers(low=1, high=7)

    def get_value(self):
        return self.value


class Dice:
    def __init__(self, seed=None):
        self.dice = [Die() for _ in range(3)]
        self.values = [0, 0, 0]

    def throw_dice(self, dices_to_throw):
        """Throws specific dice.

        Args:
            dices_to_throw (list(bool)): Which dice to throw
        """
        for i, die in enumerate(self.dice):
            # Only throw the requested dice
            if dices_to_throw[i]:
                die.throw_die()
        _ = self.store_values()
        return Combination(self.values.tolist())

    def sort_dice(self):
        values_idx_sorted = np.flip(np.argsort(self.values))
        self.values = self.values[values_idx_sorted]
        # Dice are not stored in a numpy array
        self.dice = [self.dice[i] for i in values_idx_sorted]

    def store_values(self):
        self.values = np.array([die.get_value() for die in self.dice])
        self.sort_dice()

    def get_combination(self):
        return Combination(self.values)
