from abc import ABC, abstractmethod


class Player(ABC):
    """Class used to build Players"""

    def __init__(self, env, name="Player"):
        self.name = name
        self.env = env

    def get_name(self):
        return self.name

    @abstractmethod
    def get_next_action(self, state):
        pass

    def learn(self, state, action, state_next, reward, done, info) -> None:
        pass

    def save_model(self, model_prefix: str = None):
        raise NotImplementedError()

    def load_model(self, model_prefix: str = None):
        raise NotImplementedError()

    def reset(self, starting: int = 1) -> None:
        pass
