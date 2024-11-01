import os
import h5py
import cv2
from glob import glob
# from natsort import natsorted

## load hdf5

category = "salt_and_pepper"

# filename = "/home/plaif_train/syzy/motion/mo_plaif_act/dataset/pick_cube/episode_23.hdf5"
filename = f"/home/ahmed/Desktop/workspace/ml_sim_env/dataset_for_training/pick_cube/f{category}/"

files = glob(filename+"/*.hdf5")
for file in files:
    with h5py.File(file, "r") as root:

        print("Keys: %s" % root.keys()) # ['action', 'observations']

        # actions = root['action'] 
        observations = root['observations']
        
        # for action in actions:
        #     print(action)  # shape (205, 7), type "<f4">

        # print(observations.keys()) # ['images', 'qpos', 'qvel']
        # print(observations['images']['top_cam'].shape) 
        
        imgs = observations['images']['top_cam']
        len_imgs = len(imgs)
        img_idx = 0
        
        while(True):
            cv2.imshow(f"top_cam_{category}", imgs[img_idx][::-1])
            c = cv2.waitKey(0)
            if c==27:    # Esc key to stop
                break
            elif c==ord('n'):  # normally -1 returned,so don't print it
                if len_imgs-1 > img_idx:
                    img_idx += 1
            elif c==ord('b'):
                if img_idx > 0:
                    img_idx -= 1
            elif c == ord('q'):
                break
        
        
        print("qpos: ", observations['qpos'])
        print("qvel: ", observations['qvel'])
    
    
"""

# root
#     ├── action - (205, 7) 
#     └── observations
#           ├── images
#           │     └── top_cam (205, 7) - 480, 640, 3
#           ├── qpos (205, 7) # joint_traj
#           └── qvel (205, 7)


""" 


## save hdf5

# ===========================================================
# Start the recording
# ===========================================================
"""
episode_length = 100
for i in range(episode_length):
    print("Episode: ", i)
    joint_traj, actions, qvels, top_frames, target_pose = make_trajectory(env) 
    
    # random_start_env()
    ## save hdf5 file
    camera_names = ["top_cam"]
    data_dict = {
        "/observations/qpos": [],
        "/observations/qvel": [],
        "/action": [],
    }
    for cam_name in camera_names:
        data_dict[f"/observations/images/{cam_name}"] = []

    data_dict["/observations/qpos"] = joint_traj
    data_dict["/observations/qvel"] = qvels
    data_dict["/action"] = actions
    data_dict[f"/observations/images/top_cam"] = top_frames

    max_timesteps = len(joint_traj)
    dataset_path = str(folder_path)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    with h5py.File(
        dataset_path + f"/episode_{i}.hdf5", "w", rdcc_nbytes=1024**2 * 2
    ) as root:
        root.attrs["sim"] = True
        root.attrs["info"] = target_pose
        obs = root.create_group("observations")
        image = obs.create_group("images")
        for cam_name in camera_names:
            _ = image.create_dataset(
                cam_name,
                (max_timesteps, 480, 640, 3),
                dtype="uint8",
                chunks=(1, 480, 640, 3),
                compression='gzip', compression_opts=9
            )
        qpos = obs.create_dataset("qpos", (max_timesteps, 7), compression='gzip', compression_opts=9)
        qvel = obs.create_dataset("qvel", (max_timesteps, 7), compression='gzip', compression_opts=9)
        action = root.create_dataset("action", (max_timesteps, 7), compression='gzip', compression_opts=9)

        for name, array in data_dict.items():
            root[name][...] = array

"""