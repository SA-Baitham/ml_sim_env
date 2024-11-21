import os
import h5py
import cv2
from glob import glob
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import torch
import os
import copy
# from natsort import natsorted


## load hdf5

# category = "robot_color"
dataset_path = "/home/ahmed/Desktop/workspace/ml_sim_env/dataset_failure_gt/pick_cube"

categories = os.listdir(dataset_path)
num_categories = len(categories)

# grid height and width based on number of categories to make square grid
grid_height = int(np.ceil(np.sqrt(num_categories)))
grid_width = int(np.ceil(num_categories / grid_height))

def convert_array_to_pil(depth_map):
    # Input: depth_map -> HxW numpy array with depth values 
    # Output: colormapped_im -> HxW numpy array with colorcoded depth values
    mask = depth_map!=0
    disp_map = 1/depth_map
    vmax = np.percentile(disp_map[mask], 95)
    vmin = np.percentile(disp_map[mask], 5)
    normalizer = mpl.colors.PowerNorm(gamma=0.35, vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    mask = np.repeat(np.expand_dims(mask,-1), 3, -1)
    colormapped_im = (mapper.to_rgba(disp_map)[:, :, :3] * 255).astype(np.uint8)
    colormapped_im[~mask] = 255
    return colormapped_im

# files = glob(filename+"/*.hdf5")
num_episodes = len(glob(f"{dataset_path}/{categories[0]}/*.hdf5"))

files = sorted(glob(f"{dataset_path}/{categories[0]}/*.hdf5"))

break_all = False
for episode in range(num_episodes):
        print(f"EPISODE: {episode}")
        synced_images = []
        for i, category in enumerate(categories):
            file = f"{dataset_path}/{category}/episode_{episode}.hdf5"
            file = files[episode]
            print(f"{i}. {category}: {file}")
            if break_all:
                break

            with h5py.File(file, "r") as root:

                # print("Keys: %s" % root.keys()) # ['action', 'observations']

                # actions = root['action'] 
                observations = root['observations']
                
                # for action in actions:
                #     print(action)  # shape (205, 7), type "<f4">

                # print(observations.keys()) # ['images', 'qpos', 'qvel']
                # print(observations['images']['top_cam'].shape) 

                # cam = 'hand_eye_depth_cam'
                cam = 'hand_eye_cam'
                
                imgs = observations['images'][cam]

                if 'depth' in cam:
                    imgs = np.array([convert_array_to_pil(img) for img in imgs])

                len_imgs, img_height, img_width, _ = imgs.shape

                # add text on the image to show the category at the bottom and horizontally centered
                # get the text size
                labeled_imgs = []
                for step, img in enumerate(imgs):
                    text = category + " " + str(step)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.4
                    thickness = 1
                    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                    img = cv2.putText(img, text, (img_width//2-text_size[0]//2, img_height-text_size[1]), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                    labeled_imgs.append(img)
                    

                synced_images.append(np.array(labeled_imgs))

        # concatenate the synced images into a square grid
        # black grid_imgs
        grid_imgs = np.zeros((len_imgs, grid_height*img_height, grid_width*img_width, 3), dtype=np.uint8)

        for i in range(grid_height):
            for j in range(grid_width):
                if i*grid_width+j >= num_categories:
                    break
                grid_imgs[:, i*img_height:(i+1)*img_height, j*img_width:(j+1)*img_width] = synced_images[i*grid_width+j]

        img_idx = 0

        while(True):
            cv2.imshow(f"top_cam_{category}_{episode}", grid_imgs[img_idx][...,::-1])
            c = cv2.waitKey(0)
            if c==27:    # Esc key to stop
                break
            elif c==ord('n'):  # normally -1 returned, so don't print it
                if len_imgs-1 > img_idx:
                    img_idx += 1
            elif c==ord('b'):
                if img_idx > 0:
                    img_idx -= 1
            elif c == ord('q'):
                cv2.destroyAllWindows()
                break
            elif c == ord('f'):
                break_all = True
                break
        
        
        # print("qpos: ", observations['qpos'])
        # print("qvel: ", observations['qvel'])
    
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