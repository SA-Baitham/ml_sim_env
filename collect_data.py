from environments.mujoco_envs.pick_cube_env import PickCubeEnv
import json

pick_cube_env = PickCubeEnv(is_render=True)
pick_cube_env.collect_data()

# while True:
#     # pick_cube_env.physics.step()
#     pick_cube_env.reset()
#     pick_cube_env.render()
