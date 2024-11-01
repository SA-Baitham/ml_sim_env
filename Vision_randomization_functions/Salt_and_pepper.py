#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:28:12 2024

@author: surjeet
"""

import h5py
import numpy as np
from skimage.util import random_noise
import os
import glob

def add_salt_and_pepper_noise(image, amount=0.02, salt_vs_pepper=0.5):
    """
    Adds reduced salt-and-pepper noise to an image.
    
    Parameters:
        image (numpy.ndarray): The original image.
        amount (float): Proportion of image pixels to alter with noise (reduced).
        salt_vs_pepper (float): Proportion of salt vs. pepper noise.
        
    Returns:
        numpy.ndarray: Image with reduced salt-and-pepper noise added.
    """
    noisy_image = random_noise(image, mode='s&p', amount=amount, salt_vs_pepper=salt_vs_pepper)
    noisy_image = (noisy_image * 255).astype(np.uint8)
    return noisy_image

def apply_transformations_in_place(input_folder, amount=0.02, salt_vs_pepper=0.5):
    """
    Applies reduced salt-and-pepper noise to 'hand_eye_cam' and 'top_cam' images in each HDF5 file.
    
    Parameters:
        input_folder (str): Directory containing the input HDF5 files.
        amount (float): Reduced amount of salt-and-pepper noise to add.
        salt_vs_pepper (float): Ratio of salt vs. pepper noise.
    """
    # Get list of HDF5 files in the input directory
    file_list = glob.glob(os.path.join(input_folder, '*.hdf5'))
    total_files = len(file_list)
    print(f"Found {total_files} HDF5 files to process.")
    
    # Process each .h5 file in place
    for i, file_path in enumerate(file_list, 1):
        with h5py.File(file_path, 'r+') as f:
            print(f"Processing file {i}/{total_files}: {file_path}")
            
            for cam in ['hand_eye_cam', 'top_cam']:
                if f'observations/images/{cam}' in f:
                    images = f[f'observations/images/{cam}']
                    
                    # Loop through all images in the dataset and apply noise
                    for j in range(images.shape[0]):
                        original_image = images[j][:]
                        noisy_image = add_salt_and_pepper_noise(original_image, amount, salt_vs_pepper)
                        
                        # Save noisy image back to file
                        images[j] = noisy_image
                
            print(f"File {file_path} processed and modified in place.")
    
    print("Processing complete.")

# Example usage:
input_folder = "/home/surjeet/Desktop/Random Dynamics/salt_and_pepper"  # Folder containing multiple HDF5 files

apply_transformations_in_place(input_folder)
