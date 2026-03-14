from abc import abstractmethod
import torch
import numpy as np
from .sac import SAC
from src.utils.exploration import build_schedule, OUNoise


def _to_tensor(state, device):
    return torch.FloatTensor(state).to(device).unsqueeze(0)
 
 
def _mean_action(policy, state_tensor):
    """Return the deterministic mean action (tanh(μ)) from the policy."""
    with torch.no_grad():
        # rsample returns (sample, log_prob, mean) — index 2 is tanh(μ)
        return policy.rsample(state_tensor)[2].cpu().numpy()[0]
 

class EpsilonGreedySAC(SAC):
    """
    Base class for SAC with epsilon-greedy exploration.
    Subclasses only need to implement `_get_epsilon()`.
    """

    def __init__(self, *args, epsilon_schedule: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self._schedule = build_schedule(epsilon_schedule)
        self._step = 0

    def sample_action(self, state):
        epsilon = self._schedule(self._step)
        if self._is_training: self._step += 1

        if self._is_training and self._rng.random() < epsilon:
            return self._action_space.sample()

        state = torch.FloatTensor(state).to(self._device).unsqueeze(0)

        with torch.no_grad():
            if self._is_training:
                action = self._policy.rsample(state)[0]
            else:
                action = self._policy.rsample(state)[2]
        return action.cpu().numpy()[0]


class OUNoiseSAC(SAC):
    """
    SAC variant that always adds OU noise to the mean action during training,
    with noise magnitude (sigma) controlled by a schedule.
 
    This is the DDPG/TD3-style noise strategy:
    - No epsilon gating — every training step uses a noisy action
    - Sigma is annealed according to the schedule, shrinking exploration over time
    - At evaluation time the clean deterministic mean is used
 
    Config example
    --------------
    {
        "agent_name": "OUNoiseSAC",
        "parameters": {
            ...shared SAC params...
            "sigma_schedule": {
                "type": "LinearAnnealing",
                "start": 0.3,
                "end": 0.05,
                "num_steps": 100000
            },
            "ou_theta": 0.15
        }
    }
    """
 
    def __init__(
        self,
        *args,
        sigma_schedule: dict,
        ou_theta: float = 0.15,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._schedule = build_schedule(sigma_schedule)
        self._ou_noise = OUNoise(self._action_space.shape[0], theta=ou_theta, sigma=1.0)
        self._step = 0
 
    def on_episode_end(self):
        """Reset OU state so noise correlation doesn't bleed across episodes."""
        self._ou_noise.reset()
 
    def sample_action(self, state):
        sigma = self._schedule(self._step)
        if self._is_training:
            self._step += 1
 
        state_t = _to_tensor(state, self._device)
        mean = _mean_action(self._policy, state_t)
 
        if self._is_training:
            noise = self._ou_noise.sample(sigma=sigma)
            return np.clip(mean + noise, self._action_space.low, self._action_space.high)
 
        return mean