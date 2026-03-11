import torch
import numpy as np
from .sac import SAC


class ConstantEpsilonGreedySAC(SAC):
    def __init__(
        self, 
        gamma, 
        tau, 
        target_update_interval, 
        critic_lr, 
        actor_lr, 
        actor_hidden_dim, 
        critic_hidden_dim, 
        replay_capacity, 
        seed, 
        batch_size,
        epsilon: float, 
        betas, 
        env, 
        alpha_lr: float = None, 
        baseline_actions=-1, 
        cuda=False, 
        clip_stddev=1000, 
        init=None, 
        auto_entropy_tuning: bool = False, 
        alpha: float = 0.2, 
        activation="relu"
    ):
        super().__init__(gamma, tau, target_update_interval, critic_lr, actor_lr, actor_hidden_dim, critic_hidden_dim, replay_capacity, seed, batch_size, betas, env, alpha_lr, baseline_actions, cuda, clip_stddev, init, auto_entropy_tuning, alpha, activation)
        self._rng = np.random.default_rng(seed)
        self._epsilon = epsilon
    
    def sample_action(self, state):
        if self._is_training and self._rng.random() < self._epsilon:
            return self._action_space.sample()
        state = torch.FloatTensor(state).to(self._device).unsqueeze(0)
        with torch.no_grad():
            if self._is_training:
                # Standard SAC stochastic action
                action = self._policy.rsample(state)[0]
            else:
                # Deterministic mean action for evaluation
                action = self._policy.rsample(state)[2]

        return action.cpu().numpy()[0]
        