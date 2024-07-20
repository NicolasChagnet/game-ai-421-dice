from .utils import list_to_number
import numpy as np

list_points = {
    421: (8, 0),
    111: (7, 1),
    611: (6, 2),
    666: (6, 3),
    511: (5, 4),
    555: (5, 5),
    411: (4, 6),
    444: (4, 7),
    311: (3, 8),
    333: (3, 9),
    211: (2, 10),
    222: (2, 11),
    654: (2, 12),
    543: (2, 13),
    432: (2, 14),
    321: (2, 15),
}
sequences = [654, 543, 432, 321]


class Combination:
    def __init__(self, values_dice):
        # First sort the values of the dice from higher to lower
        self.values = values_dice.copy()
        self.value = list_to_number(self.values)
        self.points, self.order = list_points.get(self.value, (1, -1))

    def get_points(self):
        return self.points

    def get_values(self):
        return self.values

    def get_value(self):
        return self.value

    def __eq__(self, other):
        # Equality is easily defined as equal combinations.
        return self.value == other.value

    def __lt__(self, other):
        # We remove the equal case first as it's the most bothersome
        if self == other:
            return False
        # If the points are different, the order is simply given by the points
        if self.points != other.points:
            return self.points < other.points
        # If the points are equal, we must check the various cases

        # In the case of no special combination, the larger number wins
        if self.points == 1:
            return self.value < other.value
        else:
            return self.order > other.order

    def __gt__(self, other):
        return not (self < other or self == other)

    def __ge__(self, other):
        return self > other or self == other

    def __le__(self, other):
        return self < other or self == other
