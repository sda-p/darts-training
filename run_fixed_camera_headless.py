# run_fixed_camera_viewer.py
import fixed_camera_env  # ensure registration
import gymnasium as gym
import time

env = gym.make(
    "FixedCamera-v0",
    num_envs=1,
    obs_mode="none",       # we’ll just view
    render_mode="human",   # open a window
    max_episode_steps=2000
)

env.reset(seed=0)

done = False
while not done:
    # viewer updates on env.step(); your env ignores actions
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
    env.render()           # draw a frame
    if terminated or truncated:
        break
    time.sleep(1/30.0)     # ~30 FPS; try 1/15 for slower

env.close()

