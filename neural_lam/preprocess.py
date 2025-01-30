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
import logging
import pathlib

import matplotlib
import matplotlib.pyplot as plt

from neural_lam import constants, utils




def plot_on_axis_cerra(ax, data, alpha=None, vmin=None, vmax=None, ax_title=None):
    """
    Plot weather state on given axis
    """
    #ax.coastlines()  # Add coastline outlines
    data_grid = data.reshape(300,300)#.cpu().numpy()
    im = ax.imshow(
        data_grid,
        origin="lower",
        #extent=constants.GRID_LIMITS_CERRA,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        cmap="plasma",
    )

    if ax_title:
        ax.set_title(ax_title, size=15)
    return im

def plot_on_axis_era5(ax, data, alpha=None, vmin=None, vmax=None, ax_title=None):
    """
    Plot weather state on given axis
    """
    #ax.coastlines()  # Add coastline outlines
    data_grid = data.reshape(81,81)#.cpu().numpy()
    im = ax.imshow(
        data_grid,
        origin="lower",
        #extent=constants.GRID_LIMITS_ERA5,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        cmap="plasma",
    )

    if ax_title:
        ax.set_title(ax_title, size=15)
    return im


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_ensemble_prediction(
    low_res, high_res, obs_mask, title=None, vrange=None
):
    """
    Plot example predictions, ground truth, mean and std.-dev.
    from ensemble forecast

    low_res: (S, N_grid,)
    high_res: (N_grid,)
    ens_mean: (N_grid,)
    ens_std: (N_grid,)
    obs_mask: (N_grid,)
    (optional) title: title of plot
    (optional) vrange: tuple of length with common min and max of values
        (not for std.)
    """
    low_res = low_res.flatten()
    high_res = high_res.flatten()
    # Get common scale for values
    # Set up masking of border region
    """
    mask_reshaped = obs_mask.reshape(*constants.GRID_SHAPE_CERRA)
    pixel_alpha = (
        mask_reshaped.clamp(0.7, 1).cpu().numpy()
    )  # Faded border region
    """

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 15),
        #subplot_kw={"projection": constants.LAMBERT_PROJ},
    )
    axes = axes.flatten()

    # Plot high_res, ensemble mean and std.
    gt_im = plot_on_axis_cerra(
        axes[0],
        high_res,
        alpha=None,
        vmin=high_res.min().item(),
        vmax=high_res.max().item(),
        ax_title="High Resolution",
    )
    plot_on_axis_era5(
        axes[1],
        low_res,
        alpha=None,
        vmin=low_res.min().item(),
        vmax=low_res.max().item(),
        ax_title="Low Resolution",
    )
    # Turn off unused axes
    for ax in axes[(3 + low_res.shape[0]) :]:
        ax.axis("off")

    # Add colorbars
    values_cbar = fig.colorbar(
        gt_im, ax=axes[:2], aspect=60, location="bottom", shrink=0.9
    )
    values_cbar.ax.tick_params(labelsize=10)
    
    if title:
        fig.suptitle(title, size=20)
        
    #save the figure
    plt.savefig('grid_representation.png')
        
    return fig

def interpolate_xr(cerra_path, era5_path, output_folder):
    """
    Interpolate the CERRA data to the ERA5 grid.
    
    Parameters:
    - cerra_path (xr.Dataset): CERRA dataset to interpolate.
    - era5_path (xr.Dataset): ERA5 dataset to interpolate to.
    
    Returns:
    - xr.Dataset: Interpolated CERRA dataset.
    """
    # Create a list of all _PRES.grb files in the 2020 folder
    file_pattern = os.path.join(cerra_path, '*_PRES.grb')
    files = sorted(glob(file_pattern))

    os.makedirs(output_folder, exist_ok=True)
    
    f = pathlib.Path(era5_path)
    ds = xr.open_dataset(f)

    # Loop through each file
    for file_path in files:
        #load cerra as xr array
        cerra_sample = xr.open_dataset(file_path, engine='cfgrib')
        
        #load era5 as xr array
        f = pathlib.Path(era5_path)
        ds = xr.open_dataset(f)
        
        cerrasample_interp = cerra_sample.interp_like(ds)
    



def PRES_to_npy(data_folder="/aspire/CarloData/CERRA_interpolated/2018/PRES", output_folder="/aspire/CarloData/CERRA_interpolated/2018/samples"):
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
            time_encoded_vars = np.concatenate([main_vars], axis=-1)[0] # Assuming the first pressure level

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

class CERRA():
    
    
    def __init__(self, grid_size: int, npy_samples_path_in: str, npy_samples_path_out: str, original_grb_path: str, default_grid_size: int):

        
        self.grid_size = grid_size # The size of the grid to resize the .npy files to
        self.npy_samples_path_in = npy_samples_path_in # The folder containing the original 1069x1069 .npy files
        self.npy_samples_path_out = npy_samples_path_out # The folder to save the resized .npy files
        self.original_grb_path = original_grb_path # The path to the original .grb file needed to compute static features
        self.default_grid_size = default_grid_size # The default size of the input .npy arrays
        
    def ensure_dir(self, path: str) -> str:
        """
        Ensure the specified directory exists, creating it if necessary.
        
        Parameters:
        - path (str): Path to the directory.
        
        Returns:
        - str: The path to the ensured directory.
        """
        os.makedirs(path, exist_ok=True)
        return path
    
    def extract_subgrid(self, center_lat: float, center_lon: float, grid_size: int, folder: str = "samples", plot: bool = True):
        """
        Extract a sub-grid from the CERRA dataset centered at a specified location and of a given size.

        Parameters:
        - center_lat (float): Latitude of the center point of the desired sub-grid.
        - center_lon (float): Longitude of the center point of the desired sub-grid.
        - grid_size (int): Size of the sub-grid (grid_size x grid_size).
        - folder (str): Folder to save the extracted sub-grid.
        - plot (bool): Whether to plot the extracted sub-grid for visualization.
        
        Returns:
        - None: Saves the extracted sub-grid to the specified folder.
        """
        save_dir = self.ensure_dir(os.path.join(self.npy_samples_path_out, folder))

        # Load latitude and longitude arrays
        ds = xr.open_dataset(self.original_grb_path, engine='cfgrib')
        lat = ds['latitude'].values
        lon = ds['longitude'].values
        lon[lon > 180] -= 360  # Adjust longitude range

        # Find the nearest indices for the center point
        lat_idx = np.abs(lat - center_lat).argmin()
        lon_idx = np.abs(lon - center_lon).argmin()
        
        # Calculate start and end indices for slicing
        half_size = grid_size // 2
        start_lat_idx = max(lat_idx - half_size, 0)
        end_lat_idx = min(lat_idx + half_size, lat.shape[0])
        start_lon_idx = max(lon_idx - half_size, 0)
        end_lon_idx = min(lon_idx + half_size, lon.shape[1])

        # Log extracted indices for debugging
        logging.info(f"Extracting grid with size {grid_size}x{grid_size}")
        logging.info(f"Latitude indices: {start_lat_idx} to {end_lat_idx}")
        logging.info(f"Longitude indices: {start_lon_idx} to {end_lon_idx}")

        # Process each .npy file
        for filename in sorted(os.listdir(self.npy_samples_path_in)):
            if not filename.endswith(".npy"):
                logging.warning(f"Skipping non-npy file: {filename}")
                continue

            file_path = os.path.join(self.npy_samples_path_in, filename)
            try:
                array = np.load(file_path)
            except Exception as e:
                logging.error(f"Error loading {filename}: {e}")
                continue

        # Extract the sub-grid
        subgrid = array[start_lat_idx:end_lat_idx, start_lon_idx:end_lon_idx, :]

        # Save the sub-grid
        output_path = os.path.join(save_dir, filename)
        np.save(output_path, subgrid)
        logging.info(f"Sub-grid saved to {output_path}")

        # Plot the sub-grid for visual inspection
        if plot:
            plt.figure(figsize=(8, 6))
            plt.imshow(subgrid[:, :, 0], cmap="viridis")  # Plot the first variable layer
            plt.title(f"Sub-grid centered at ({center_lat}, {center_lon})")
            plt.xlabel("Longitude Index")
            plt.ylabel("Latitude Index")
            plt.colorbar(label="Variable Value")
            plt.show()
        
            
    def resize(self, folder: str = "samples"):
        """
        Resize .npy files from the input path to a central cropped grid and save them.

        Parameters:
        - folder (str): Name of the folder where resized files will be saved.
        """
        save_dir = self.ensure_dir(os.path.join(self.npy_samples_path_out, folder))
        original_size = self.DEFAULT_GRID_SIZE

        # Calculate slicing indices
        start_idx = (original_size - self.grid_size) // 2
        end_idx = start_idx + self.grid_size

        for filename in sorted(os.listdir(self.npy_samples_path_in)):
            
            if filename.endswith(".npy"):
                # Load the .npy file
                file_path = os.path.join(self.npy_samples_path_in, filename)
                array = np.load(file_path)
                
                # Check the array shape to ensure compatibility
                """
                if array.shape[:2] != (original_size, original_size):
                    logging.warning(f"Skipping {filename}: unexpected shape {array.shape}")
                    continue
                """
                
                # Slice the central 300x300 grid
                resized_array = array[start_idx:end_idx, start_idx:end_idx, :]
                
                # Save the resized array
                output_path = os.path.join(save_dir, filename)
                np.save(output_path, resized_array)
                logging.info(f"Resized and saved {filename} to {output_path}")
                
                                        
                
    def LatLon_to_LambertProj(self, filename: str = "nwp_xy_interpolated.npy", folder: str = "static", interpolate_to_ERA5: bool=True, plot: bool = False):
        """
        Convert latitude and longitude coordinates to Lambert Conformal projection coordinates.

        Parameters:
        - filename (str): Name of the file to save the projected coordinates.
        - folder (str): Name of the folder where the file will be saved.
        - plot (bool): Whether to visualize the coordinates in a scatter plot.
        """
        save_dir = self.ensure_dir(os.path.join(self.npy_samples_path_out, folder))
        
        #ds = xr.open_dataset(self.original_grb_path, engine='cfgrib')
        f = pathlib.Path(self.original_grb_path)
        ds = xr.open_dataset(f)

        lon = ds['longitude'].values
        lat = ds['latitude'].values
        lon[lon > 180] = lon[lon > 180] - 360
        
        if len(lon.shape) == 1:
            lon, lat = np.meshgrid(lon, lat)

        #pp = easydict.EasyDict(constants.LAMBERT_PROJ_PARAMS_CERRA)
        #mycrs = f"+proj=lcc +lat_0={pp.lat_0} +lon_0={pp.lon_0} +lat_1={pp.lat_1} +lat_2={pp.lat_2} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

        #crstrans = Transformer.from_crs("EPSG:4326", mycrs, always_xy=True)
        #lon, lat = crstrans.transform(lon, lat)


        xy = np.array([lon, lat]) # shape = (2, n_lon, n_lat)
        if self.grid_size != self.default_grid_size:
            original_size = self.default_grid_size
            start_idx = (original_size - self.grid_size) // 2
            end_idx = start_idx + self.grid_size
            
            
            xy = xy[:, start_idx:end_idx, start_idx:end_idx]
        
        np.save(os.path.join(save_dir, filename), xy)
        
        if plot:
            plt.figure(figsize=(10, 8))
            plt.scatter(xy[0], xy[1], s=1, color='blue', alpha=0.5)
            plt.xlabel("X Coordinates")
            plt.ylabel("Y Coordinates")
            plt.grid(True)
            plt.show()
            plt.savefig('grid_representation.png')
            
        
        
    def get_topography(self, folder: str, filename: str):
        """
        Extract the surface geopotential from a GRIB file and save it.
        
        Parameters:
        - path (str): Path to the GRIB file.
        - save_dir (str): Path to save the extracted data.
        """
        save_dir = self.ensure_dir(os.path.join(self.npy_samples_path_out, folder, filename))
        
        try:
            #surface_data = cfgrib.open_datasets(path)
            surface_data = xr.open_dataset(self.original_grb_path, engine='cfgrib')
            surface_geopotential = surface_data[-1]["orog"].values
            np.save(save_dir, surface_geopotential)
            logging.info(f"Topography data saved to {save_dir}")
        except Exception as e:
            logging.error(f"Error extracting topography: {e}")
        
        
    def create_border_mask(self, border_width: int = 10, folder: str = "static", filename: str = 'border_mask.npy'):
        """
        Create a mask with True values along the border and False elsewhere.

        Parameters:
        - border_width (int): Width of the border to set as True.
        - folder (str): Folder to save the mask.
        - filename (str): Name of the file to save the mask.
        """
        border_mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        border_mask[:border_width, :] = True
        border_mask[-border_width:, :] = True
        border_mask[:, :border_width] = True
        border_mask[:, -border_width:] = True

        save_dir = self.ensure_dir(os.path.join(self.npy_samples_path_out, folder))
        np.save(os.path.join(save_dir, filename), border_mask)
        logging.info(f"Border mask saved to {os.path.join(save_dir, filename)}")
        
        
    def create_dataset(self, 
                       samples_folder: str = "samples", 
                       static_folder: str = "static", 
                       border_width: int = 10, 
                       plot: bool = False,
                       interpolate_to_ERA5: bool = False):
        """
        Create the pre-processed dataset by orchestrating the individual steps.
        Parameters:
        - samples_folder (str): Folder name to save resized .npy files.
        - static_folder (str): Folder name to save static features.
        - border_width (int): Border width for the mask.
        - plot (bool): Whether to plot the Lambert projection.
        """
        # Step 1: Resize samples
        #logging.info("Starting resizing of samples...")
        #self.resize(folder=samples_folder)

        # Step 2: Generate Lambert projection static file
        #self.interpolate(folder=static_folder, interpolate_to_ERA5=interpolate_to_ERA5, plot=plot)
        #logging.info("Generating Lambert projection static file...")
        #self.LatLon_to_LambertProj(folder=static_folder, interpolate_to_ERA5=interpolate_to_ERA5, plot=plot)

        # Step 3: Get topography data
        logging.info("Extracting topography...")

        self.get_topography(folder="static", filename="surface_geopotential.npy")

        # Step 4: Create border mask
        logging.info("Creating border mask...")
        #self.create_border_mask(border_width=border_width, folder=static_folder)

        logging.info("Dataset creation complete.")

        
        
    
if __name__ == "__main__":
    """
    cerra = CERRA(grid_size=300, 
                  npy_samples_path_in="data/CERRA_interpolated/samples", 
                  npy_samples_path_out="data/CERRA_interpolated", 
                  original_grb_path="/aspire/CarloData/CERRA_interpolated/2017/PRES/CERRA_2017_01_01-00_PRES.grb",
                  default_grid_size=300)
    
    
    cerra = CERRA(grid_size=81, 
                  npy_samples_path_in="data/ERA5/60_n2_40_18/2017/samples", 
                  npy_samples_path_out="data/ERA5/60_n2_40_18/2017", 
                  original_grb_path="/aspire/CarloData/ERA5/60_n2_40_18/2017/data_stream-oper_stepType-instant.nc",
                  default_grid_size=81)
    
    
    cerra.create_dataset(samples_folder="samples", 
                         static_folder="static", 
                         border_width=10, 
                         plot=True,
                         interpolate_to_ERA5=True)
    
    """
    
    high_res = np.load("/aspire/CarloData/MASK_GNN_DATA/CERRA_interpolated_300x300/samples/val/nwp_2019082100_mbr000.npy")
    low_res = np.load("/aspire/CarloData/MASK_GNN_DATA/ERA5_60_n2_40_18/samples/val/nwp_2019082100_mbr000.npy")
    plot_ensemble_prediction(low_res[:,:,0], high_res[:,:,0], obs_mask=None, title=None, vrange=None)
    
    #plot_ensemble_prediction(low_res[:,:,0], high_res[:,:,0], obs_mask=None, title=None, vrange=None)
    

                         
    
    
    
    
    
    