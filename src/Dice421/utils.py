import numpy as np


def list_to_number(dice_values):
    """Given a list of integers, returns a concatenated integer.

    Args:
        dice_values (list(int)): List of integers.

    Returns:
        int: Decimal representation of the list of integers.
    """
    return int("".join([str(x) for x in dice_values]))


def number_to_list(number):
    """Given an integer, returns a list of its digits.

    Args:
        number (int): Number to split.

    Returns:
        list(int): List of digits.
    """
    return [int(x) for x in list(str(number))]


def add_one_mod_two(number):
    return (number + 1) % 2
