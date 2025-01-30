# Standard library
import os
from argparse import ArgumentParser

# Third-party
import numpy as np
import torch
from tqdm import tqdm

# First-party
from neural_lam import constants
from neural_lam.weather_dataset import ERA5toCERRA


def main():
    """
    Pre-compute parameter weights to be used in loss function
    """
    parser = ArgumentParser(description="Training arguments")
    parser.add_argument(
        "--dataset_cerra",
        type=str,
        default="/aspire/CarloData/MASK_GNN_DATA/CERRA_interpolated_300x300",
        help="Dataset, corresponding to name in data directory "
        "(default: meps_example)",
    )
    parser.add_argument(
        "--dataset_era5",
        type=str,
        default="/aspire/CarloData/MASK_GNN_DATA/ERA5_60_n2_40_18",
        help="Dataset, corresponding to name in data directory "
        "(default: meps_example)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size when iterating over the dataset",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of workers in data loader (default: 4)",
    )
    args = parser.parse_args()

    static_dir_path_highRes = os.path.join("data", args.dataset_cerra, "static")
    static_dir_path_lowRes = os.path.join("data", args.dataset_era5, "static")

    # Load dataset without any subsampling
    ds = ERA5toCERRA(
            args.dataset_cerra,
            args.dataset_era5,
            split="train",
            standardize=False,
        )  # Without standardization
    loader = torch.utils.data.DataLoader(
        ds, args.batch_size, shuffle=False, num_workers=args.n_workers
    )
    # Compute mean and std.-dev. of each parameter (+ flux forcing)
    # across full dataset
    print("Computing mean and std.-dev. for parameters...")
    means_highRes = []
    squares_highRes = []
    means_lowRes = []
    squares_lowRes = []
    for highRes, lowRes in tqdm(loader):

        means_highRes.append(torch.mean(highRes, dim=1))  # (N_batch, d_features,)
        means_lowRes.append(torch.mean(lowRes, dim=1))  # (N_batch, d_features,)
        
        squares_highRes.append(
            torch.mean(highRes**2, dim=1)
        )
        squares_lowRes.append(
            torch.mean(lowRes**2, dim=1)
        )

    mean_highRes = torch.mean(torch.cat(means_highRes, dim=0), dim=0)  # (d_features)
    second_moment = torch.mean(torch.cat(squares_highRes, dim=0), dim=0)
    std_highRes = torch.sqrt(second_moment - mean_highRes**2)  # (d_features)
    
    mean_lowRes = torch.mean(torch.cat(means_lowRes, dim=0), dim=0)  # (d_features)
    second_moment = torch.mean(torch.cat(squares_lowRes, dim=0), dim=0)
    std_lowRes = torch.sqrt(second_moment - mean_lowRes**2)  # (d_features)


    print("Saving mean, std.-dev...")
    torch.save(mean_highRes, os.path.join(static_dir_path_highRes, "parameter_mean.pt"))
    torch.save(std_highRes, os.path.join(static_dir_path_highRes, "parameter_std.pt"))
    torch.save(mean_lowRes, os.path.join(static_dir_path_lowRes, "parameter_mean.pt"))
    torch.save(std_lowRes, os.path.join(static_dir_path_lowRes, "parameter_std.pt"))


if __name__ == "__main__":
    main()
