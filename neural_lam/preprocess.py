import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# Open the GRIB file
ds = xr.open_dataset('data/CERRA/rawdata/CERRA_2019_01_01-00_PRES.grb', engine='cfgrib')
#plot pressure
temp = ds['t'].values
y_coords = ds['longitude'].values
x_coords = ds['latitude'].values
y_coords[y_coords > 180] = y_coords[y_coords > 180] - 360

###########################################################################################################
temp_slice = temp[0, :, :]  # This gives a (1069, 1069) 2D array

# Plot the selected time slice
plt.figure(figsize=(10, 8))
plt.imshow(temp_slice, origin='lower', cmap='coolwarm')
plt.colorbar(label='Temperature')
plt.title('Temperature at First Time Slice')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
plt.savefig('temperature_first_time_slice.png')
###########################################################################################################

# Plotting the x-y coordinate grid
plt.figure(figsize=(10, 8))
plt.scatter(y_coords, x_coords, s=1, color='blue', alpha=0.5)
plt.xlabel("X Coordinates")
plt.ylabel("Y Coordinates")
plt.grid(True)

# Display the plot
plt.show()
plt.savefig('grid_representationWRONG.png')


"""
extent = [lons.min(), lons.max(), lats.min(), lats.max()]

# Set up a plot with a geographic projection (e.g., PlateCarree for latitude/longitude data)
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

# Plot the temperature data using imshow, with the defined extent to match the lat/lon bounds
# Setting origin='upper' if latitude values go from max to min
temp_plot = ax.imshow(temp[0, :, :], extent=extent, origin='upper', transform=ccrs.PlateCarree(), cmap='coolwarm')

# Add a color bar
cbar = plt.colorbar(temp_plot, ax=ax, orientation='horizontal', pad=0.05)
cbar.set_label('Temperature')

# Add coastlines and country borders for geographic context
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Set title
plt.title("Temperature Overlay with Geographic Area (Europe)")

# Display and save the plot
plt.show()
plt.savefig('temperature_with_geographic_context.png')

"""








# Extract latitude and longitude as 2D arrays
latitudes = ds['latitude'].values  # Shape (x, y)
longitudes = ds['longitude'].values  # Shape (x, y)

# Stack latitude and longitude arrays along a new dimension to get shape (2, x, y)
coordinates_tensor = np.stack([latitudes, longitudes], axis=0)

# Save to 'nwp_xy.npy'
np.save('data/CERRA/static/nwp_xy.npy', coordinates_tensor)


# Load grid positions
static_dir_path = os.path.join("data", "meps_example", "static")
graph_dir_path = os.path.join("graphs", "multiscale")
os.makedirs(graph_dir_path, exist_ok=True)

xy = np.load(os.path.join(static_dir_path, "nwp_xy.npy"))

# Extract x and y coordinates from the grid
x_coords = xy[0, :, :]
y_coords = xy[1, :, :]

# Plotting the x-y coordinate grid
plt.figure(figsize=(10, 8))
plt.scatter(x_coords, y_coords, s=1, color='blue', alpha=0.5)
plt.xlabel("X Coordinates")
plt.ylabel("Y Coordinates")
plt.grid(True)

# Display the plot
plt.show()
plt.savefig('grid_representation.png')