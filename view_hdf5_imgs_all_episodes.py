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
category = "clean"
dataset_path = "/home/ahmed/Desktop/workspace/ml_sim_env/dataset_failure_gt/pick_cube"

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
num_episodes = len(glob(f"{dataset_path}/{category}/*.hdf5"))
files = sorted(glob(f"{dataset_path}/{category}/*.hdf5"))

break_all = False
for file in files:
    print(file)
    
    if break_all:
        break

    with h5py.File(file, "r") as root:

        # print("Keys: %s" % root.keys()) # ['action', 'observations']

        # actions = root['action'] 
        observations = root['observations']

        # cam = 'hand_eye_depth_cam'
        cam = 'top_cam'
        
        img = np.array(observations['images'][cam])

        if 'depth' in cam:
            img = np.array([convert_array_to_pil(img) for img in img])

        # make image transparent with 50% opacity
        if "6_0" in file:
            output = img
        elif "_0." in file:
            # opacity = 1/(episode+1)
            # output = (img*opacity + output*(1-opacity)).astype("uint8")
            output[np.abs(output[..., 0]-img[..., 0])>2] = [255, 0, 0]

for file in files:
    if "_0.hdf5" in file:
        continue
    print(file)
    
    if break_all:
        break

    with h5py.File(file, "r") as root:

        # print("Keys: %s" % root.keys()) # ['action', 'observations']

        # actions = root['action'] 
        observations = root['observations']

        # cam = 'hand_eye_depth_cam'
        cam = 'top_cam'
        
        img = np.array(observations['images'][cam])

        if 'depth' in cam:
            img = np.array([convert_array_to_pil(img) for img in img])

        # make image transparent with 50% opacity
        if "6_1" in file:
            output_noise = img
        else:
            # opacity = 1/(episode+1)
            # output_noise = (img*opacity + output_noise*(1-opacity)).astype("uint8")
            output_noise[np.abs(output_noise[..., 0]-img[..., 0])>2] = [255, 0, 255]

opacity = 0.5
output[np.abs(output[..., 0]-output_noise[..., 0])>2] = (1-opacity)*output[np.abs(output[..., 0]-output_noise[..., 0])>2] + opacity*output_noise[np.abs(output[..., 0]-output_noise[..., 0])>2]

img_idx = 0

while(True):
    # resize image to fit screen
    output_img = cv2.resize(output[img_idx], (640*2, 480*2))
    cv2.imshow(f"top_cam_{category}", output_img[...,::-1])
    c = cv2.waitKey(0)
    if c==27:    # Esc key to stop
        break
    elif c==ord('n'):  # normally -1 returned, so don't print it
        if len(output)-1 > img_idx:
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