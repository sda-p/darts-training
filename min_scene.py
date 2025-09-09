import numpy as np
import sapien.core as sapien
import sapien.render
import torch
from typing import List, Tuple, Dict, Union
from collections import defaultdict

from mani_skill.utils.scene_builder import SceneBuilder  # base class
from mani_skill.utils.building.ground import build_ground

class SimpleSceneBuilder(SceneBuilder):
    """Small, self-contained builder for FixedCameraEnv."""

    # If you define lighting yourself, set this True so ManiSkill won't add defaults
    builds_lighting = True

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        # Keep handles to important actors/articulations to reset later
        self._default_object_poses: List[Tuple[Union[sapien.Actor, sapien.Articulation], sapien.Pose]] = []
        self.cubes: List[sapien.Actor] = []

    def build(self, build_config_idxs: Union[int, List[int]]):
        if isinstance(build_config_idxs, int):
            build_config_idxs = [build_config_idxs] * self.env.num_envs
        assert len(build_config_idxs) == self.env.num_envs

        scene = self.scene  # shorthand

        # ---- Lighting (use color scale as "intensity") ----
        scene.set_ambient_light([0.3, 0.3, 0.3])
        scene.add_directional_light(direction=[-1, -1, -2], color=[3.0, 2.9, 2.8], shadow=True)
        scene.add_point_light([2.0, 2.0, 3.0], color=[40.0, 38.0, 42.0])
        scene.add_point_light([-2.0, -1.5, 2.5], color=[25.0, 22.0, 22.0])

        # ---- Ground plane (static) ----
        self.ground = build_ground(scene)  # returns an actor spanning all sub-scenes

        # Optional: apply a render material (via renderer)
        """
        mat = sapien.render.RenderMaterial()
        mat.set_base_color([0.65, 0.7, 0.75, 1.0])
        mat.set_roughness(0.8)
        mat.set_metallic(0.0)
        for shape in self.ground.get_actor_visual_mesh():
            shape.set_material(mat)
        """

        # ---- A simple static wall (per-env) ----
        wall_builder = scene.create_actor_builder()
        wall_builder.add_box_visual(half_size=[0.01, 1.0, 0.5], material=sapien.render.RenderMaterial(base_color=[0.5, 0.5, 0.5, 1.0]))
        wall_builder.add_box_collision(half_size=[0.01, 1.0, 0.5], density=800)
        wall_builder.initial_pose = sapien.Pose([0.3, 0.0, 0.5])
        wall_builder.set_scene_idxs(list(range(self.env.num_envs)))
        self.wall = wall_builder.build_static(name="wall")

        # ---- Dynamic cubes (per-env, randomized positions) ----
        rng = np.random.default_rng(0)
        self.cubes.clear()
        for i in range(6):
            b = scene.create_actor_builder()
            hs = rng.uniform(0.03, 0.07, size=3)
            b.add_box_visual(half_size=hs, material=sapien.render.RenderMaterial(base_color=list(rng.random(3)) + [1.0]))
            b.add_box_collision(half_size=hs, density=600)
            # IMPORTANT: set a reasonable initial pose for GPU init
            b.initial_pose = sapien.Pose([0.0, 0.0, 0.12], [1, 0, 0, 0])
            b.set_scene_idxs(list(range(self.env.num_envs)))
            actor = b.build(name=f"cube_{i}")
            self.cubes.append(actor)

        # Stash for resets
        self._default_object_poses = [(actor, None) for actor in self.cubes]

        # Set per-env start poses using the ManiSkill Actor wrapper + reset mask
        # (sample once and reuse for all cubes, or resample per actor if you prefer)
        p = torch.zeros((self.env.num_envs, 3))
        p[..., :2] = torch.rand((self.env.num_envs, 2)) * 0.2 - 0.1  # [-0.1, 0.1] in x,y
        p[..., 2] = 0.12
        q = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(self.env.num_envs, -1)  # identity quat
        from mani_skill.utils.structs.pose import Pose
        pose_all = Pose.create_from_pq(p=p, q=q)

        _prev = self.scene._reset_mask.clone()
        self.scene._reset_mask[:] = True  # update all envs
        for actor in self.cubes:
            actor.set_pose(pose_all)
        self.scene._reset_mask = _prev

        # Kinematic ball: also give builder an initial_pose and then set per-env if needed
        kb = scene.create_actor_builder()
        kb.add_sphere_visual(radius=0.05, material=sapien.render.RenderMaterial(base_color=[0.2, 0.8, 1, 1]))
        kb.add_sphere_collision(radius=0.05, density=800)
        kb.initial_pose = sapien.Pose([0.0, -0.3, 0.25], [1, 0, 0, 0])  # avoids warning
        kb.set_scene_idxs(list(range(self.env.num_envs)))
        self.moving_ball = kb.build_kinematic(name="moving_ball")


        # Moving ball: like above, but with gravity
        mb = scene.create_actor_builder()
        mb.add_sphere_visual(radius=0.05, material=sapien.render.RenderMaterial(base_color=[0.7, 0.3, 0.5, 1]))
        mb.add_sphere_collision(radius=0.05, density=800)  # gives it mass
        mb.initial_pose = sapien.Pose([0.0, -0.3, 0.25], [1, 0, 0, 0])  # good spawn
        mb.set_scene_idxs(list(range(self.env.num_envs)))

        # DYNAMIC (affected by gravity & contacts)
        self.gravity_ball = mb.build(name="gravity_ball")

        # If you need to merge background actors or set collision groups, do it here
        # (see the ReplicaCAD example you pasted).

    def initialize(self, env_idx):
        # normalize indices
        if isinstance(env_idx, int):
            idxs = [env_idx]
        elif hasattr(env_idx, "tolist"):
            idxs = list(env_idx.tolist())
        else:
            idxs = list(env_idx)

        n = len(idxs)

        # Torch sampling keeps types/devices consistent
        p = torch.zeros((n, 3), dtype=torch.float32)
        p[:, :2] = torch.rand((n, 2)) * 1.2 - 0.6   # uniform [-0.6, 0.6] for x,y
        p[:, 2] = 0.12
        q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32).expand(n, -1)

        from mani_skill.utils.structs.pose import Pose
        pose_batch = Pose.create_from_pq(p=p, q=q)

        prev = self.scene._reset_mask.clone()
        self.scene._reset_mask[:] = False
        self.scene._reset_mask[idxs] = True
        for actor, _ in self._default_object_poses:
            actor.set_pose(pose_batch)

        # inside build(...) right after self.ball is created, or inside initialize(...)
        v0 = torch.zeros((n, 3), dtype=torch.float32)
        v0[:, 0] = 0.3  # kick along +X as an example

        prev = self.scene._reset_mask.clone()
        self.scene._reset_mask[:] = True  # or only True for selected env indices
        self.gravity_ball.set_linear_velocity(v0)

        #self.moving_ball.set_linear_velocity(v0)

        self.scene._reset_mask = prev

