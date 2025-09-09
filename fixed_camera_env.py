import numpy as np
import torch
import gymnasium as gym

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SimConfig
from min_scene import SimpleSceneBuilder
import sapien.core as sapien

def apply_force_safe(actor, f, idxs=None, mode="force"):
    f = np.asarray(f, dtype=np.float32)
    if f.ndim == 2 and f.shape[0] == 1:   # (1,3) -> (3,)
        f = f[0]
    actor.apply_force(f)

@register_env("FixedCamera-v0", max_episode_steps=200)
class FixedCameraEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["none"]
    SUPPORTED_REWARD_MODES = ["none"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids="none", **kwargs)
        self.action_space = gym.spaces.Box(low=np.zeros(0, np.float32), high=np.zeros(0, np.float32), shape=(0,), dtype=np.float32)
        self.single_action_space = self.action_space
        self._orig_single_action_space = self.action_space
        self._t = 0.0  # for moving items

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[1.0, 0.0, 1.0], target=[0.0, 0.75, 0.0])
        return [CameraConfig("fixed_camera", pose=pose, width=256, height=256, fov=np.pi / 2)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[1.0, 0.0, 1.0], target=[0.0, 0.75, 0.0])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=np.pi / 2)

    def _load_agent(self, options: dict):
        super()._load_agent(options)

    def _load_scene(self, options: dict):
        # Build the scene via our builder
        self.scene_builder = SimpleSceneBuilder(self)
        # For now, a single “config id” for all envs; switch to per-env ids for curriculum
        self.scene_builder.build(build_config_idxs=0)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Delegate reset of props to the builder
        self.scene_builder.initialize(env_idx)

    def step(self, action):
        # Example: move the kinematic ball
        self._t += 0.1
        if hasattr(self.scene_builder, "moving_ball"):
            for i in range(self.num_envs):
                x = 0.25 * np.sin(1.5 * self._t + i * 0.2)
                y = 0.22 * np.cos(1.9 * self._t + i * 0.2)
                z = 0.25 + 0.05 * np.sin(2.7 * self._t + i * 0.2)
                self.scene_builder.moving_ball._objs[i].set_pose(sapien.Pose([x, y, z]))

                # Fixed-velocity update
                """
                v = torch.zeros((1, 3), dtype=torch.float32)
                v[:, 1] = 0.1
                self.scene_builder.gravity_ball.set_linear_velocity(v)
                """
            
        if hasattr(self.scene_builder, "gravity_ball"):
            for i in range(self.num_envs):
                # Force-add acceleration, gravity ball
                f = np.zeros((1, 3), dtype=np.float32)
                angle_rad = np.deg2rad((15*int(self._t))%360)   # Circle!
                print((int(self._t))%360)
                direction = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])
                f = 0.2 * direction
                print(f)
                apply_force_safe(self.scene_builder.gravity_ball, f)
                #impulse = np.array([0.5, 0.0, 0.0], np.float32)
                #self.scene_builder.gravity_ball.apply_impulse(impulse, mode="impulse")
                

        return super().step(None)

