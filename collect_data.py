# from environments.mujoco_envs.pick_cube_env import PickCubeEnv
from environments.mujoco_envs.pick_cube_env_complex import PickCubeEnv
import yaml

# combinations of random dynamics to apply
random_dynamics_list = [
    ['clean'], 
    ["link_mass"], 
    ["joint_damping"], 
    ["joint_friction"], 
    ["actuator_gain"],
    ["link_inertia"],
    ["joint_stiffness"],
    ['gravity'],
    ]

def load_config(config_file_path):
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Example usage
config_file_path = 'random_dynamics_config.yaml'
random_dynamics_config = load_config(config_file_path)

for dynamic in random_dynamics_list:
    print(f"####################### Collecting data for {dynamic} #######################")
    pick_cube_env = PickCubeEnv(is_render=True, random_dynamics_to_apply=dynamic, random_dynamics_config=random_dynamics_config)
    pick_cube_env.collect_data(dynamic)

    pick_cube_env.close()