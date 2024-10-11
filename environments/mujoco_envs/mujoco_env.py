import numpy as np
import gymnasium as gym
from .mujoco_render import MJRenderer

class MujocoEnv(gym.Env):
    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = float("inf"),
        step_limit: int = 200,
        is_render: bool = False,
    ):
        self.physics = self.load_models()
        self.physics.model.opt.timestep = physics_dt
        self.control_dt = control_dt
        self.n_substeps = int(control_dt // physics_dt)
        self.time_limit = time_limit
        self.step_limit = step_limit
        self.is_render = is_render
        self.random = np.random.RandomState(seed)

        self.step_num = 0 
        self.renderer = MJRenderer(self.physics)

    def load_models(self):
        return NotImplementedError

    def render(self):
        self.renderer.refresh_window()
        self.renderer.render_to_window()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def time_limit_exceeded(self) -> bool:
        return self.physics.data.time >= self.time_limit
    
    def step_limit_exceeded(self) -> bool:
        return self.step_num > self.step_limit