# Standard library
import os
from argparse import ArgumentParser

# Third-party
import numpy as np
import torch


def main():
    """
    Pre-compute all static features related to the grid nodes
    """
    parser = ArgumentParser(description="Training arguments")
    parser.add_argument(
        "--dataset_low",
        type=str,
        default="/aspire/CarloData/MASK_GNN_DATA/ERA5_60_n2_40_18",
        help="Dataset to load grid point coordinates from "
        "(default: meps_example)",
    )
    parser.add_argument(
        "--dataset_high",
        type=str,
        default="/aspire/CarloData/MASK_GNN_DATA/CERRA_interpolated_300x300",
        help="Dataset to load grid point coordinates from "
        "(default: meps_example)",
    )
    args = parser.parse_args()

    static_dir_path_highRes = os.path.join(args.dataset_high, "static")
    static_dir_path_lowRes = os.path.join(args.dataset_low, "static")
    
    for static_dir_path in [static_dir_path_highRes, static_dir_path_lowRes]:
        # -- Static grid node features --
        grid_xy = torch.tensor(
            np.load(os.path.join(static_dir_path, "nwp_xy.npy"))
        )  # (2, N_x, N_y)
        grid_xy = grid_xy.flatten(1, 2).T  # (N_grid, 2)
        pos_max = torch.max(torch.abs(grid_xy))
        grid_xy = grid_xy / pos_max  # Divide by maximum coordinate

        geopotential = torch.tensor(
            np.load(os.path.join(static_dir_path, "surface_geopotential.npy"))
        )  # (N_x, N_y)
        geopotential = geopotential.flatten(0, 1).unsqueeze(1)  # (N_grid,1)
        gp_min = torch.min(geopotential)
        gp_max = torch.max(geopotential)
        # Rescale geopotential to [0,1]
        geopotential = (geopotential - gp_min) / (gp_max - gp_min)  # (N_grid, 1)

        # Concatenate grid features
        grid_features = torch.cat(
            (grid_xy, geopotential), dim=1
        )  # (N_grid, 4)

        torch.save(grid_features.to(torch.float32), os.path.join(static_dir_path, "grid_features.pt"))


if __name__ == "__main__":
    main()
