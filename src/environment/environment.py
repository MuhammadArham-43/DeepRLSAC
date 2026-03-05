import gymnasium as gym
import numpy as np
from src.environment.pendulum_env import PendulumEnv

class Environment:
    def __init__(
        self,
        config,
        monitor=False,
        monitor_after=0
    ):
        self.overwrite_rewards = config["overwrite_rewards"]
        self.rewards = config["rewards"]
        self.start_state = np.array(config["start_state"])

        self.steps = 0
        self.episodes = 0

        self.monitor = monitor
        self.steps_until_monitor = monitor_after
        self.env_name = config["env_name"]
        seed = config["seed"]
        config["monitor"] = monitor
        config["render_mode"] = "human" if monitor else None
        self.env = env_factory(config)
        print(f"Seeding environment: {seed}")
        self.env.reset(seed=seed)
        self.steps_per_episode = config["steps_per_episode"]

        self.env._max_episode_steps = self.steps_per_episode + 10

        if "info" in dir(self.env):
            self.info = self.env.info
        else:
            self.info = {}
    
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def observation_space(self):
        return self.env.observation_space
    
    def seed(self, seed):
        self.env.reset(seed=seed)
    
    def reset(self):
        self.steps = 0
        self.episodes += 1
        state, info = self.env.reset()

        if self.start_state.shape[0] != 0:
            state = self.start_state
            self.env.state = state
        
        return state, {"orig_state": state}
    
    def render(self):
        self.env.render()
    
    def step(self, action):
        if self.monitor and self.steps_until_monitor < 0:
            self.render()
        
        self.steps += 1
        self.steps_until_monitor -= (1 if self.monitor >= 0 else 0)

        state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        info["orig_state"] = state
        
        if done:
            info["steps_exceeded"] = False
            if self.overwrite_rewards:
                reward = self.rewards["goal"]
            return state, reward, done, info
        
        if self.overwrite_rewards:
            reward = self.rewards["timestep"]
        
        if self.steps >= self.steps_per_episode > 0:
            done = True
            info["steps_exceeded"] = True
        
        return state, reward, done, info

def env_factory(config):
    name = config["env_name"]
    seed = config["seed"]
    env = None
    monitor = config.get("monitor", False)
    render_mode = config.get("render_mode", None)
    
    if name == "PendulumContinuous-v0":
        max_steps = config["total_timesteps"]
        env = PendulumEnv()
        print("Created PendulumContinous-v0 environment")
    else:
        if monitor:
            env = gym.make(name, render_mode=render_mode).env
        else:
            env = gym.make(name).env
    
    env.reset(seed=seed)
    return env