# from environments.mujoco_envs.pick_cube_env import PickCubeEnv
from environments.mujoco_envs.pick_cube_env_complex import PickCubeEnv
import json

# combinations
random_dynamics_list = [
    ["link mass"],
    ['clean'], 
    ["link mass"], 
    ["joint damping"], 
    ["joint friction"], 
    ["actuator gain"],
    ["link inertia"],
    ["joint stiffness"],
    ['gravity'],
    ]

for dynamic in random_dynamics_list:
    print(f"####################### Collecting data for {dynamic} #######################")
    pick_cube_env = PickCubeEnv(is_render=True, random_dynamics_to_apply=dynamic)
    pick_cube_env.collect_data(dynamic)

    pick_cube_env.close()

# while True:
#     # pick_cube_env.physics.step()
#     pick_cube_env.reset()
#     pick_cube_env.render()
