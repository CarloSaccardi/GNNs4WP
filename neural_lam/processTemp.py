import os
import numpy as np

# Directories
data_folder = '/aspire/CarloData/samples'
output_folder = '/aspire/CarloData/samplesStacked'

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get sorted list of files in the directory
files = sorted(f for f in os.listdir(data_folder) if f.endswith('.npy'))

# Process files in batches of 21
sequence_length = 21
for i in range(0, len(files), sequence_length):
    # Check if there are enough files left to form a full sequence
    if i + sequence_length > len(files):
        break

    # Extract the batch of files
    batch_files = files[i:i+sequence_length]
    
    # Load and stack the numpy tensors
    stacked_data = np.stack([np.load(os.path.join(data_folder, f)) for f in batch_files])
    
    # Save the stacked tensor
    output_filename = batch_files[0]  # Use the name of the first file in the sequence
    output_path = os.path.join(output_folder, output_filename)
    np.save(output_path, stacked_data)

    print(f"Saved: {output_path}")