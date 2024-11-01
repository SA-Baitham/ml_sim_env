#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:11:39 2024

@author: surjeet
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
import cv2  # OpenCV for HSV conversion

def convert_to_hsv(image):
    """
    Converts an image from RGB to HSV.

    Parameters:
        image (numpy.ndarray): The original image in RGB format.

    Returns:
        numpy.ndarray: Image converted to HSV format.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

def convert_to_rgb(image):
    """
    Converts an image from HSV to RGB.

    Parameters:
        image (numpy.ndarray): The image in HSV format.

    Returns:
        numpy.ndarray: Image converted back to RGB format.
    """
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

def apply_hsv_transformation(input_folder):
    """
    Applies HSV transformation to 'hand_eye_cam' and 'top_cam' images in each HDF5 file.

    Parameters:
        input_folder (str): Directory containing the input HDF5 files.
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

                    # Loop through all images in the dataset
                    for j in range(images.shape[0]):
                        original_image = images[j][:]
                        hsv_image = convert_to_hsv(original_image)
                        # Optionally perform any manipulation on hsv_image here
                        # For example, to modify saturation:
                        # hsv_image[:, :, 1] = hsv_image[:, :, 1] * 1.5  # Increase saturation
                        rgb_image = convert_to_rgb(hsv_image)

                        # Save transformed image back to file
                        images[j] = rgb_image

            print(f"File {file_path} processed and modified in place.")

    print("Processing complete.")

# Example usage:
input_folder = "/home/surjeet/Desktop/Random Dynamics/HSV"  # Folder containing multiple HDF5 files

apply_hsv_transformation(input_folder)
