import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from src.agent.base import BaseAgent
import src.utils.nn_utils as nn_utils
from src.networks import SquashedGaussian, DoubleQ
from src.utils.replay_buffer import TorchReplayBuffer as ExperienceReplayBuffer


class SAC(BaseAgent):
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
        betas,
        env,
        alpha_lr: float = None,
        baseline_actions=-1,
        cuda=False,
        clip_stddev=1000,
        init=None,
        auto_entropy_tuning: bool = True,
        alpha: float = None,
        activation="relu",
    ):
        super(SAC, self).__init__()
        self._env = env

        if batch_size > replay_capacity:
            raise ValueError("batch size cannot be greater than replay capacity")
        
        self._action_space = env.action_space
        self._obs_space = env.observation_space
        if len(self._obs_space.shape) != 1:
            raise ValueError("SAC only supports vector observations")
        
        if auto_entropy_tuning and alpha_lr is None:
            raise ValueError("For auto entropy setting, must provide the learning rate for alpha as alpha_lr")
        if not auto_entropy_tuning and alpha is None:
            raise ValueError("For fixed entropy setting, must provide a alpha value")
        self.auto_entropy_tuning = auto_entropy_tuning
        
        self.torch_rng = torch.manual_seed(seed)
        self._rng = np.random.default_rng(seed)

        self._is_training = True
        self._gamma = gamma
        self._tau = tau
        
        self._device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        self._batch_size = batch_size
        self._replay = ExperienceReplayBuffer(
            replay_capacity,
            seed,
            self._obs_space.shape,
            self._action_space.shape[0],
            device=self._device
        )

        self._target_update_interval = target_update_interval
        self._update_number = 0

        self._init_critic(
            self._obs_space,
            self._action_space,
            critic_hidden_dim,
            init,
            activation,
            critic_lr,
            betas,
        )

        self._init_policy(
            self._obs_space,
            self._action_space,
            actor_hidden_dim,
            init,
            activation,
            actor_lr,
            betas,
            clip_stddev,
        )
        
        if self.auto_entropy_tuning:
            self._alpha_lr = alpha_lr
            self.auto_entropy_tuning = auto_entropy_tuning
            self._target_entropy = -torch.prod(
                torch.Tensor(self._action_space.shape).to(self._device)
            ).item()
            self._log_alpha = torch.zeros(1, requires_grad=True, device=self._device)
            self._alpha = self._log_alpha.exp().detach()
            self._alpha_optim = Adam([self._log_alpha], lr=self._alpha_lr, betas=betas)
        else:
            print(f"Initialized SAC with fixed entropy with alpha={alpha}")
            self._alpha = alpha
    
    def reset(self):
        pass

    def eval(self):
        self._is_training = False
    
    def train(self):
        self._is_training = True
    
    def get_parameters(self):
        pass
    
    def save_model(self, env_name, actor_path=None, critic_path=None):
        critic_path = critic_path if critic_path is not None else f"{env_name}_sac_critic.pth"
        actor_path = actor_path if actor_path is not None else f"{env_name}_sac_actor.pth"

        self.save_model_state_dict(critic_path, actor_path)
    
    def save_model_state_dict(self, critic_path, actor_path):
        torch.save(self._critic.state_dict(), critic_path)
        torch.save(self._policy.state_dict(), actor_path)

    def load_model(self, actor_path, critic_path):
        pass
    
    def _init_critic(
        self,
        obs_space,
        action_space,
        hidden_dim,
        init,
        activation,
        lr,
        betas,
    ):
        num_inputs = obs_space.shape[0]
        self._critic = DoubleQ(
            num_inputs,
            action_space.shape[0],
            hidden_dim,
            init,
            activation
        ).to(self._device)

        self._critic_target = DoubleQ(
            num_inputs,
            action_space.shape[0],
            hidden_dim,
            init,
            activation
        ).to(self._device)

        nn_utils.hard_update(self._critic_target, self._critic)

        self._critic_optim = Adam(self._critic.parameters(), lr=lr, betas=betas)
    
    def _init_policy(
        self,
        obs_space,
        action_space,
        hidden_dim,
        init,
        activation,
        lr,
        betas,
        clip_stddev,
    ):
        num_inputs = obs_space.shape[0]
        self._policy = SquashedGaussian(
            num_inputs,
            action_space.shape[0],
            hidden_dim,
            init=init,
            activation=activation,
            action_space=action_space,
            clip_stddev=clip_stddev
        ).to(self._device)

        self._policy_optim = Adam(self._policy.parameters(), lr=lr, betas=betas)

    def _get_q(self, state_batch, action_batch):
        q1, q2 = self._critic(state_batch, action_batch)
        return torch.min(q1, q2)
    
    def sample_action(self, state):
        state = torch.FloatTensor(state).to(self._device).unsqueeze(0)
        if self._is_training:
            action = self._policy.rsample(state)[0]
        else:
            action = self._policy.rsample(state)[2]

        return action.detach().cpu().numpy()[0]
    
    def add_to_replay(self, state, action, reward, next_state, done_mask):
        return self._replay.push(state, action, reward, next_state, done_mask)
    
    def update(self, state, action, reward, next_state, done_mask):
        self.add_to_replay(state, action, reward, next_state, done_mask)

        state_batch, action_batch, reward_batch, next_state_batch, done_mask_batch = self._replay.sample(self._batch_size)
        if state_batch is None: return

        self._update_critic(state_batch, action_batch, reward_batch, next_state_batch, done_mask_batch)
        self._update_actor(state_batch)
    
    def _update_actor(self, state_batch):
        pi, log_pi = self._policy.rsample(state_batch)[:2]
        q = self._get_q(state_batch, pi)
    
        policy_loss = ((self._alpha * log_pi) - q).mean()
        self._policy_optim.zero_grad()
        policy_loss.backward()
        self._policy_optim.step()

        if self.auto_entropy_tuning:
            alpha_loss = -(self._log_alpha * (log_pi + self._target_entropy).detach()).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()

            self._alpha = self._log_alpha.exp().detach()

    def _update_critic(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
        done_mask_batch
    ):
        with torch.no_grad():
            next_state_action, next_state_log_pi = self._policy.rsample(next_state_batch)[:2]
            next_q1, next_q2 = self._critic_target(next_state_batch, next_state_action)
            min_next_q = torch.min(next_q1, next_q2) - self._alpha * next_state_log_pi
            q_target = reward_batch + done_mask_batch * self._gamma * min_next_q
        
        q1, q2 = self._critic(state_batch, action_batch)
        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)
        q_loss = q1_loss + q2_loss

        self._critic_optim.zero_grad()
        q_loss.backward()
        self._critic_optim.step()

        self._update_number += 1
        if self._update_number % self._target_update_interval == 0:
            self._update_number = 0
            nn_utils.soft_update(self._critic_target, self._critic, self._tau)