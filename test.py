import gymnasium as gym
import mani_skill.envs
from gymnasium.wrappers import TimeLimit
import time

env = gym.make(
    "PickCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=1,
    obs_mode="state", # there is also "state_dict", "rgbd", ...
    control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
    render_mode="human",
    human_render_camera_configs=dict(shader_pack="rt"),
    max_episode_steps=2000
)

# override max steps (e.g. 2000 instead of default 200)
env = TimeLimit(env, max_episode_steps=2000)

print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset(seed=0)
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()           # draw a frame
    time.sleep(1/30.0)     # ~30 FPS; try 1/15 for slower
env.close()
