import numpy as np
import torch
import gymnasium as gym

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SimConfig


@register_env("FixedCamera-v0", max_episode_steps=200)
class FixedCameraEnv(BaseEnv):
    """Simple environment with no robot and a single fixed camera."""

    SUPPORTED_ROBOTS = ["none"]
    SUPPORTED_REWARD_MODES = ["none"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids="none", **kwargs)
        # Provide a dummy action space so that gym APIs expecting an action
        # space can still function. The environment ignores any actions.
        self.action_space = gym.spaces.Box(
            low=np.zeros(0, dtype=np.float32),
            high=np.zeros(0, dtype=np.float32),
            shape=(0,),
            dtype=np.float32,
        )
        self.single_action_space = self.action_space
        self._orig_single_action_space = self.action_space

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[1.0, 0.0, 1.0], target=[0.0, 0.0, 0.0])
        return [
            CameraConfig(
                "fixed_camera", pose=pose, width=128, height=128, fov=np.pi / 2
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[1.0, 0.0, 1.0], target=[0.0, 0.0, 0.0])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=np.pi / 2)

    def _load_agent(self, options: dict):
        # No robot is loaded; robot_uids="none" causes BaseEnv to skip agent setup.
        super()._load_agent(options)

    def _load_scene(self, options: dict):
        # Add a ground plane to the otherwise empty scene.
        self.ground = build_ground(self.scene)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Nothing to randomize for now.
        pass

    def evaluate(self):
        return {}

    def _get_obs_agent(self):
        # No proprioceptive observation since there is no robot.
        return torch.zeros((self.num_envs, 0), device=self.device)

    def _get_obs_extra(self, info: dict):
        return {}

    def step(self, action):
        # Ignore external actions and step the simulation forward.
        return super().step(None)
