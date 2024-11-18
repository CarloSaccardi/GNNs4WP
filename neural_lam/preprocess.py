import os
import numpy as np
import xarray as xr
from glob import glob
from datetime import datetime
import cfgrib
import matplotlib.pyplot as plt
import easydict
from pyproj import Transformer
from neural_lam import constants



def PRES_to_npy(data_folder="'/aspire/CarloData/CERRA/2019'", output_folder="'/aspire/CarloData/samples'"):
    """
    data_folder: str - The folder containing the _PRES.grb files
    output_folder: str - The folder to save the .npy files
    
    This function converts the _PRES.grb files to .npy files containing the main variables (u, v, z, t, r) stacked along the last axis.
    
    """
    # Create a list of all _PRES.grb files in the 2020 folder
    file_pattern = os.path.join(data_folder, '*_PRES.grb')
    files = sorted(glob(file_pattern))

    os.makedirs(output_folder, exist_ok=True)

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
            

def concatenate_npy_files(data_folder="'/aspire/CarloData/samples'", output_folder="'/aspire/CarloData/samplesStacked'"):
    """         
    data_folder: str - The folder containing the .npy files
    output_folder: str - The folder to save the stacked .npy files
    
    Each .npy file in datafolder has shape (pressure_levels, x, y, variables). For now we only consider the first pressure_level, making each variable a 2d tensor. 
    Moreover, this function stacks consecutinve 21 in time (yyyymmddtt : yyyymmdd(tt+21)) files in data_folder along the first axis and saves the stacked tensor in output_folder.
    Now the shape of the stacked tensor is (21, x, y, variables), where 21 is the number of time steps.
    """

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
        
        
def resize(input_folder = "data/CERRA/static", output_folder = "data/CERRA/static"):
    """
    input_folder: str - The folder containing the .npy files to resize
    output_folder: str - The folder to save the resized .npy files
    
    This function resizes the 1069x1069 arrays in the input_folder to 300x300 arrays and saves them in the output_folder.
    1069x1069 represent the entire Europe, which for now is too big and runs in memory issues.
    300x300 is a more manageable size but have to find an alternative on how we select the area of interest. For now we just take the central 300x300 grid.
    """

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
            
def LatLon_to_LambertProj(path, save_dir, plot=False):
    
    """
    Convert latitude and longitude coordinates to Lambert Conformal projection coordinates.
    """
    
    ds = xr.open_dataset(path, engine='cfgrib')

    lon = ds['longitude'].values
    lat = ds['latitude'].values
    lon[lon > 180] = lon[lon > 180] - 360

    pp = easydict.EasyDict(constants.LAMBERT_PROJ_PARAMS_CERRA)
    mycrs = f"+proj=lcc +lat_0={pp.lat_0} +lon_0={pp.lon_0} +lat_1={pp.lat_1} +lat_2={pp.lat_2} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

    crstrans = Transformer.from_crs("EPSG:4326", mycrs, always_xy=True)
    x, y = crstrans.transform(lon, lat)

    xy = np.array([x, y]) # shape = (2, n_lon, n_lat)
    
    if plot:
        plt.figure(figsize=(10, 8))
        plt.scatter(x, y, s=1, color='blue', alpha=0.5)
        plt.xlabel("X Coordinates")
        plt.ylabel("Y Coordinates")
        plt.grid(True)
        plt.show()
        plt.savefig('grid_representationWRONG.png')
        
    np.save(save_dir, xy)
    
    
def get_topography(path, save_dir):
    """
    Extract the surface geopotential from a GRIB file and
    """
    
    surface_data = cfgrib.open_datasets(path)
    surface_geopotential = surface_data[-1]["orog"].values
    np.save(save_dir, surface_geopotential)
    
    
def create_border_mask(shape=(1069, 1069), border_width=10, filename='border_mask.npy'):
    """
    Create a mask with True values along the border and False elsewhere.
    
    Parameters:
    - shape (tuple): The shape of the mask array, default is (1069, 1069).
    - border_width (int): The width of the border (in grid nodes) to set as True.
    - filename (str): The filename to save the mask as, default is 'border_mask.npy'.
    
    Returns:
    - None: Saves the mask array to a .npy file.
    """
    # Initialize the mask array with False
    border_mask = np.zeros(shape, dtype=bool)

    # Set the border to True
    border_mask[:border_width, :] = True         # Top border
    border_mask[-border_width:, :] = True        # Bottom border
    border_mask[:, :border_width] = True         # Left border
    border_mask[:, -border_width:] = True        # Right border

    # Save the mask to a .npy file
    np.save(filename, border_mask)
    print(f"Border mask saved to {filename}")
    
    
    
if __name__ == "__main__":
    create_border_mask(shape=(300, 300), border_width=10, filename='data/CERRA/static/border_mask.npy')