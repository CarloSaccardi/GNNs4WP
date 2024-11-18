import os
import numpy as np

# Input and output folder paths
input_folder = "data/CERRA/static"
output_folder = "data/CERRA/static"
os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists

# Parameters for slicing
original_size = 1069
new_size = 300
start_idx = (original_size - new_size) // 2  # Starting index for the 300x300 grid
end_idx = start_idx + new_size  # Ending index

# Process each file
for filename in os.listdir(input_folder):
    if filename.endswith(".npy"):
        # Load the .npy file
        file_path = os.path.join(input_folder, filename)
        array = np.load(file_path)
        
        # Check the array shape to ensure compatibility
        if array.shape[1:3] != (original_size, original_size):
            print(f"Skipping {filename}: unexpected shape {array.shape}")
            continue
        
        # Slice the central 300x300 grid
        resized_array = array[:, start_idx:end_idx, start_idx:end_idx, :]
        
        # Save the resized array
        output_path = os.path.join(output_folder, filename)
        np.save(output_path, resized_array)
        print(f"Resized and saved {filename} to {output_path}")
