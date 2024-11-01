import h5py
import numpy as np
import os
import glob

def normalize_image(image):
    """
    Normalizes an image to the range [0, 1].

    Parameters:
        image (numpy.ndarray): The original image with values in [0, 255].

    Returns:
        numpy.ndarray: Normalized image with values in [0, 1].
    """
    normalized_image = image / 255.0
    return normalized_image

def apply_normalization_in_place(input_folder):
    """
    Applies normalization to 'hand_eye_cam' and 'top_cam' images in each HDF5 file.

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
                    images = f[f'observations/images/{cam}']  # Corrected line
                    
                    # Loop through all images in the dataset
                    for j in range(images.shape[0]):
                        original_image = images[j][:]
                        normalized_image = normalize_image(original_image)
                        
                        # Convert normalized image back to uint8 for saving
                        normalized_image_uint8 = (normalized_image * 255).astype(np.uint8)
                        
                        # Save normalized image back to file
                        images[j] = normalized_image_uint8
                
            print(f"File {file_path} processed and modified in place.")
    
    print("Processing complete.")

# Example usage:
input_folder = "/home/surjeet/Desktop/Random Dynamics/normalization"  # Folder containing multiple HDF5 files

apply_normalization_in_place(input_folder)
