import os
from omegaconf import OmegaConf
import hydra
import h5py

from environments.mujoco_envs.pick_cube_env import PickCubeEnv
from utils import set_seed

@hydra.main(config_path="config", config_name="replay_default")
def replay_data(conf: OmegaConf):
    set_seed(conf.seed)
    episode_idx = 0
    pick_cube_env = PickCubeEnv() #TODO: Make gym style(ex: gym.make)

    with h5py.File(os.path.join(conf.dataset_folder_path, "episode_{}.hdf5".format(episode_idx)), 'r') as root:
        options = {}
        options["generated_cube_pose"] = root.attrs["info"]
        action = root['/action']

        pick_cube_env.reset(options)

        for act in action:
                       
            pick_cube_env.step(act)
            pick_cube_env.render()

if __name__ == "__main__":
    replay_data()