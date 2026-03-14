from abc import ABC, abstractmethod
import numpy as np


class ExplorationSchedule(ABC):
    """Callable that maps a training step → a scalar (epsilon or sigma)."""

    @abstractmethod
    def __call__(self, step: int) -> float:
        raise NotImplementedError


class ConstantSchedule(ExplorationSchedule):
    """Returns the same value at every step."""

    def __init__(self, epsilon: float):
        self.value = epsilon

    def __call__(self, step: int) -> float:
        return self.value


class LinearAnnealingSchedule(ExplorationSchedule):
    """Linearly decays from `start` to `end` over `num_steps`, then holds."""

    def __init__(self, start: float, end: float, num_steps: int):
        self.start = start
        self.end = end
        self.num_steps = num_steps
        self._decay = (start - end) / num_steps

    def __call__(self, step: int) -> float:
        return max(self.end, self.start - self._decay * step)


class DecayingAnnealingSchedule(ExplorationSchedule):
    """Exponentially decays from `start` toward `end` with time-constant `num_steps`."""

    def __init__(self, start: float, end: float, num_steps: int):
        self.start = start
        self.end = end
        self.num_steps = num_steps

    def __call__(self, step: int) -> float:
        return self.end + (self.start - self.end) * np.exp(-step / self.num_steps)


_SCHEDULE_REGISTRY: dict[str, type[ExplorationSchedule]] = {
    "Constant":         ConstantSchedule,
    "LinearAnnealing":  LinearAnnealingSchedule,
    "DecayingAnnealing": DecayingAnnealingSchedule,
}


def build_schedule(config: dict) -> ExplorationSchedule:
    config = dict(config)                          # don't mutate the original
    schedule_type = config.pop("type")
    cls = _SCHEDULE_REGISTRY.get(schedule_type)
    if cls is None:
        raise ValueError(
            f"Unknown schedule type '{schedule_type}'. "
            f"Valid options: {list(_SCHEDULE_REGISTRY)}"
        )
    return cls(**config)


class OUNoise:
    def __init__(
        self,
        action_dim: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
    ):
        self.action_dim = action_dim
        self.mu = mu * np.ones(action_dim)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self, sigma: float | None = None) -> np.ndarray:
        sigma = sigma if sigma is not None else self.sigma
        dx = self.theta * (self.mu - self.state) + sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state