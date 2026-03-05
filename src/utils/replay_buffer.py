from abc import ABC, abstractmethod
import torch
import numpy as np


class ExperienceReplayBuffer(ABC):

    def __init__(
        self,
        capacity: int,
        seed: int,
        state_dim: tuple,
        action_dim: tuple,
        device = None
    ):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity
        self.is_full = False
        self.position = 0

        self.cast = lambda x: x

        self.random = np.random.default_rng(seed)
        
        self.state_dim = state_dim
        self.action_dim = action_dim

        self._sampleable = False

        self.state_buffer = None
        self.action_buffer = None
        self.reward_buffer = None
        self.next_state_buffer = None
        self.done_buffer = None
        self.init_buffer()

    @property
    def sampleable(self):
        return self._sampleable

    @abstractmethod
    def init_buffer(self):
        pass

    def push(self, state, action, reward, next_state, done):
        reward = np.array([reward], dtype=np.float32)
        done = np.array([done])

        state = self.cast(state)
        action = self.cast(action)
        reward = self.cast(reward)
        next_state = self.cast(next_state)
        done = self.cast(done)

        self.state_buffer[self.position] = state
        self.action_buffer[self.position] = action
        self.reward_buffer[self.position] = reward
        self.next_state_buffer[self.position] = next_state
        self.done_buffer[self.position] = done

        if self.position >= self.capacity - 1:
            self.is_full = True
        self.position = (self.position + 1) % self.capacity
        self._sampleable = False

    def is_sampleable(self, batch_size):
        if self.position < batch_size and not self.sampleable:
            return False
        elif not self._sampleable:
            self._sampleable = True

        return self.sampleable
    
    def sample(self, batch_size):
        if not self.is_sampleable(batch_size):
            return None, None, None, None, None
        
        if self.is_full:
            indices = self.random.integers(low=0, high=len(self), size=batch_size)
        else:
            indices = self.random.integers(low=0, high=self.position, size=batch_size)
        
        state = self.state_buffer[indices, :]
        action = self.action_buffer[indices, :]
        reward = self.reward_buffer[indices, :]
        next_state = self.next_state_buffer[indices, :]
        done = self.done_buffer[indices, :]

        return state, action, reward, next_state, done
    
    def __len__(self):
        return self.position if self.is_full else self.position
    

class NumpyReplayBuffer(ExperienceReplayBuffer):
    def __init__(
        self,
        capacity: int,
        seed: int,
        state_dim: tuple,
        action_dim: tuple,
        device = None,
        state_dtype = np.float32,
        action_dtype = np.float32,
    ):
        self._state_dtype = state_dtype
        self._action_dtype = action_dtype
        super().__init__(capacity, seed, state_dim, action_dim, None)


    def init_buffer(self):
        self.state_buffer = np.zeros((self.capacity, *self.state_dim), dtype=self._state_dtype)
        self.action_buffer = np.zeros((self.capacity, self.action_dim), dtype=self._action_dtype)
        self.reward_buffer = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((self.capacity, *self.state_dim), dtype=self._state_dtype)
        self.done_buffer = np.zeros((self.capacity, 1), dtype=bool)

    def __len__(self):
        return self.position if self.is_full else self.position
            

class TorchReplayBuffer(ExperienceReplayBuffer):
    def __init__(
        self,
        capacity: int,
        seed: int,
        state_dim: tuple,
        action_dim: tuple,
        device = None,
    ):
        super().__init__(capacity, seed, state_dim, action_dim, device)
        self.cast = torch.from_numpy
    
    def init_buffer(self):
        self.state_buffer = torch.FloatTensor(self.capacity, *self.state_dim).to(self.device)
        self.action_buffer = torch.FloatTensor(self.capacity, self.action_dim).to(self.device)
        self.reward_buffer = torch.FloatTensor(self.capacity, 1).to(self.device)
        self.next_state_buffer = torch.FloatTensor(self.capacity, *self.state_dim).to(self.device)
        self.done_buffer = torch.FloatTensor(self.capacity, 1).to(self.device)