# from environments.mujoco_envs.pick_cube_env import PickCubeEnv
from environments.mujoco_envs.pick_cube_env import PickCubeEnv
import yaml

# combinations of random dynamics to apply
def load_config(config_file_path):
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Example usage
config_file_path = 'configs.yaml'
configs = load_config(config_file_path)

for randomization_to_apply in configs["randomization_list"]:
    print(f"####################### Collecting data for {' + '.join(randomization_to_apply)} #######################")
    pick_cube_env = PickCubeEnv(is_render=True, randomizations_to_apply=randomization_to_apply, configs=configs)
    pick_cube_env.collect_data()

    # pick_cube_env.close()