import pygame
import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np



class PendulumEnv(gymnasium.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30
    }

    def __init__(
        self,
        continuous_action: bool = True,
        g: float = 10.0,
        trig_features: bool = False,
        seed: int = None,
        monitor: bool = False,
        render_mode: str = "human"
    ):
        super(PendulumEnv, self).__init__()
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.length = 1.0
        self.monitor = monitor
        self.render_mode = render_mode

        self.viewer = None
        self.window = None
        self.continous_actions = continuous_action

        if self.continous_actions:
            self.action_space = spaces.Box(
                low=-self.max_torque,
                high=self.max_torque,
                shape=(1,),
                dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(3)

        self.trig_features = trig_features
        if self.trig_features:
            high = np.array([1.,1., self.max_speed], dtype=np.float32)
            self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        else:
            low = np.array([-np.pi, -self.max_speed], dtype=np.float32)
            high = np.array([np.pi, self.max_speed], dtype=np.float32)
            self.observation_space = spaces.Box(
                low=low,
                high=high,
                dtype=np.float32
            )
        
        self.np_random, _ = seeding.np_random(seed)
    
    def step(self, u):
        th, thdot = self.state
        g = self.g
        m = self.m 
        length = self.length
        dt = self.dt

        if self.continous_actions:
            u = np.clip(u, -self.max_torque, self.max_torque)[0]
        else:
            assert self.action_space.contains(u), "%r (%s) invalid"%(u, type(u))
            u = (u - 1) * self.max_torque
        
        self.last_u = u
        newthdot = thdot + (-3 * g / (2 * length) * np.sin(th + np.pi)
                            + 3. / (m * length ** 2) * u) * dt
        newth = angle_normalize(th + newthdot * dt)
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        reward = np.cos(newth)

        terminated = False
        truncated = False

        if self.trig_features:
            return self._get_obs(), reward, terminated, truncated, {}

        return self.state, reward, terminated, truncated, {}
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        state = np.array([np.pi, 0.])
        self.state = angle_normalize(state)
        self.last_u = None

        obs = self._get_obs() if self.trig_features else self.state
        info = {}
        return obs, info

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self):
        if self.window is None and self.monitor:
            pygame.init()
            self.window = pygame.display.set_mode((500, 500))
            self.clock = pygame.time.Clock()

        surface = pygame.Surface((500, 500))
        surface.fill((255, 255, 255))
        center = np.array([250, 250])
        length = 200
        angle = self.state[0]
        end = center - length * np.array([np.sin(angle), np.cos(angle)])

        pygame.draw.line(surface, (200, 60, 60), center, end, 10)
        pygame.draw.circle(surface, (0, 0, 0), center.astype(int), 10)

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
            self.window.blit(surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None


def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)