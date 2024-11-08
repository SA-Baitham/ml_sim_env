import h5py
import os
from glob import glob
import numpy as np

dataset_path = "/home/ahmed/Desktop/workspace/ml_sim_env/dataset_for_training_corrected_orientation2/pick_cube"

dataset_categories_names = os.listdir(dataset_path)
# append /pick_cube/ to each category
dataset_categories = [f"{dataset_path}/{category}" for category in dataset_categories_names]

# compare the same episode in each category and flag if there are two with the exact same observations
# find number of episodes by counting number of hdf5 files in the first category
num_episodes = len(glob(f"{dataset_categories[0]}/*.hdf5"))
print(f"first category: {dataset_categories[0]}")
print(f"Number of episodes: {num_episodes}")

for episode in range(num_episodes):
    observations = []
    for category in dataset_categories:
        with h5py.File(f"{category}/episode_{episode}.hdf5", "r") as root:
            observation = np.array(root['observations']['images']['hand_eye_cam'])
            observations.append(observation)

    
    for i in range(len(observations)):
        for j in range(i+1, len(observations)):
            if np.array_equal(observations[i], observations[j]):
                print(f"Episode {episode} in category {dataset_categories_names[i]} is the same as episode {episode} in category {dataset_categories_names[j]}")
                break
        else:
            continue
        break
    else:
        print(f"Episode {episode} is unique in all categories")