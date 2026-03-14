from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self):
        self.batch = False
        self.info = {}

    @abstractmethod
    def sample_action(self, state):
        pass

    @abstractmethod
    def update(self, state, action, reward, next_state, done_mask):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def get_parameters(self):
        pass
    
    @abstractmethod
    def add_to_replay(self, state, action, reward, next_state, done_mask):
        pass

    def on_episode_end(self):
        pass