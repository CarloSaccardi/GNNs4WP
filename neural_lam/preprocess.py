import os
import numpy as np
import xarray as xr
from glob import glob
from datetime import datetime

# Specify the path to the CERRA dataset's 2020 folder
data_folder = '/aspire/CarloData/CERRA/2019'

# Create a list of all _PRES.grb files in the 2020 folder
file_pattern = os.path.join(data_folder, '*_PRES.grb')
files = sorted(glob(file_pattern))

# Output folder for saving .npy files
output_folder = '/aspire/CarloData/samples'
os.makedirs(output_folder, exist_ok=True)

xy_coord = np.load('data/CERRA/static/nwp_xy.npy')
surface_geopotential = np.load('data/CERRA/static/surface_geopotential.npy')
boundary = np.load("data/CERRA/static/border_mask.npy")
boundary = boundary.astype(int)

# Loop through each file
for file_path in files:
    # Extract the date and time from the filename
    base_name = os.path.basename(file_path)
    datetime_str = base_name.split('_')  # CERRA_YYYYMMDDTT_PRES.grb
    dd_tt = datetime_str[-2].split("-")  # Split DD and TT
    datetime_str = datetime_str[1] + datetime_str[2] + dd_tt[0] + dd_tt[1]  # Reorder to YYYYMMDDHH
    date_time = datetime.strptime(datetime_str, "%Y%m%d%H")
    output_filename = f'nwp_{datetime_str}_mbr000.npy'
    output_path = os.path.join(output_folder, output_filename)

    # Load the dataset
    ds = xr.open_dataset(file_path, engine='cfgrib')
    
    # Check if required variables are present
    if all(var in ds for var in ["u", "v", "z", "t", "r"]):
        # Stack the main variables along the new axis
        main_vars = np.stack([ds[var].values for var in ["u", "v", "z", "t", "r"]], axis=-1)
        
        
        # Stack the time encoding arrays along the last axis
        time_encoded_vars = np.concatenate([main_vars], axis=-1)[0]

        # Save the stacked array as .npy
        np.save(output_path, time_encoded_vars)
        #delete the _PRES.grb file
        print(f"Saved {output_path}")
    else:
        print(f"Warning: File {file_path} is missing required variables.")
