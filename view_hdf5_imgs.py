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
from environments.utils import frames_to_gif
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

files = {category: sorted(glob(f"{dataset_path}/{category}/*.hdf5")) for category in categories}

break_all = False
for episode in range(num_episodes):
        print(f"EPISODE: {episode}")
        synced_images = []
        for i, category in enumerate(categories):
            file = f"{dataset_path}/{category}/episode_{episode}.hdf5"
            file = files[category][episode]
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
                cam = 'top_cam'
                
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

        # save episode as a gif
        frames_to_gif("episode_vid", grid_imgs, episode)


        while(True):
            cv2.imshow(f"{cam}_{category}_{episode}", grid_imgs[img_idx])
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