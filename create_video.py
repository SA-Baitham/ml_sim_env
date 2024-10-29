import h5py
import glob
import numpy as np
import os
import imageio
from PIL import Image


dataset_path = "dataset_random_dynamics_complex/pick_cube/"

# add folders inside dataset path into a list
dynamics_types = glob.glob(dataset_path + "/*/")

for dynamics_type in dynamics_types:
    # read hdf5 inside the folder
    hdf5_files = glob.glob(dynamics_type + "/*.hdf5")
    for hdf5_file in hdf5_files:
        with h5py.File(hdf5_file, "r") as f:
            # save oservations into a video.. images are stored in f["observations"]["images"]["top_cam"][i], where i is the index of the image
            gif_path = os.path.join(dynamics_type, os.path.basename(hdf5_file).replace(".hdf5", ".gif"))
            images = f["observations"]["images"]["top_cam"]
            # resize images to half their resolution
            pil_images = [Image.fromarray(image) for image in images]
            # pil_images = [image.resize((image.width // 2, image.height // 2)) for image in pil_images]
            print(len(pil_images))
            # fps = 30
            # imageio.mimsave(gif_path, images, duration=0.01)
            # Create the GIF with a specific duration (e.g., 0.1 seconds per frame)
            pil_images[0].save(
            gif_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=50,  # duration in milliseconds per frame
            loop=0  # loop indefinitely
            )
            print(f"Saved gif to {gif_path}")

