# Standard library
import datetime as dt
import glob
import os

# Third-party
import numpy as np
import torch

# First-party
from neural_lam import constants, utils

import torch.nn.functional as F


class ERA5toCERRA(torch.utils.data.Dataset):
    """
    For our dataset:
    N_t' = 65
    N_t = 65//subsample_step (= 21 for 3h steps)
    dim_x = 268
    dim_y = 238
    N_grid = 268x238 = 63784
    d_features = 17 (d_features' = 18)
    d_forcing = 5
    """

    def __init__(
        self,
        dataset_name_CERRA,
        dataset_name_ERA5,
        split,
        standardize=True,
        subset=False,
    ):
        super().__init__()

        assert split in ("train", "val", "test"), "Unknown dataset split"
        self.sample_dir_path_CERRA = os.path.join(
            "data", dataset_name_CERRA, "samples", split
        )
        
        self.sample_dir_path_era5 = os.path.join(
            "data", dataset_name_ERA5, "samples", split
        )

        member_file_regexp = (
             "nwp*mbr*.npy"
        )
        
        #get file names for CERRA
        sample_paths_CERRA = glob.glob(
            os.path.join(self.sample_dir_path_CERRA, member_file_regexp)
        )
        self.sample_names_CERRA = [path.split("/")[-1][4:-4] for path in sample_paths_CERRA]
        self.sample_names_CERRA = sorted(self.sample_names_CERRA)
        
        #get file names for ERA5
        sample_paths_era5 = glob.glob(
            os.path.join(self.sample_dir_path_era5, member_file_regexp)
        )
        self.sample_names_era5 = [path.split("/")[-1][4:-4] for path in sample_paths_era5]
        self.sample_names_era5 = sorted(self.sample_names_era5)

        if subset:
            self.sample_names_CERRA = self.sample_names_CERRA[:50]  # Limit to 50 samples
            self.sample_names_era5 = self.sample_names_era5[:50]  # Limit to 50 samples


        # Set up for standardization
        self.standardize = standardize
        if standardize:
            
            #standardize metrics for CERRA
            ds_stats_CERRA = utils.load_dataset_stats(dataset_name_CERRA, "cpu")
            self.data_mean_CERRA, self.data_std_CERRA = (
                ds_stats_CERRA["data_mean"],
                ds_stats_CERRA["data_std"],
            )
            
            #standardize metrics for ERA5
            ds_stats_era5 = utils.load_dataset_stats(dataset_name_ERA5, "cpu")
            self.data_mean_era5, self.data_std_era5 = (
                ds_stats_era5["data_mean"],
                ds_stats_era5["data_std"],
            )

        # If subsample index should be sampled (only duing training)
        self.random_subsample = split == "train"


    def __len__(self):
        assert len(self.sample_names_CERRA) == len(self.sample_names_era5), "Different number of samples in CERRA and ERA5"
        return len(self.sample_names_CERRA)

    def __getitem__(self, idx):
        # === Sample ===

        sample_names_CERRA = self.sample_names_CERRA[idx]
        sample_names_era5 = self.sample_names_era5[idx]
        
        sample_path_CERRA = os.path.join(
            self.sample_dir_path_CERRA, f"nwp_{sample_names_CERRA}.npy"
        )
        sample_path_era5 = os.path.join(
            self.sample_dir_path_era5, f"nwp_{sample_names_era5}.npy"
        )
        try:
            sample_CERRA = torch.tensor(
                np.load(sample_path_CERRA), dtype=torch.float32
            )  # (N_t', dim_x, dim_y, d_features')
            sample_era5 = torch.tensor(
                np.load(sample_path_era5), dtype=torch.float32
            )  # (N_t', dim_x, dim_y, d_features')
        except ValueError:
            print(f"Failed to load {sample_path_CERRA}")
            print(f"Failed to load {sample_path_era5}")

        # Only use every ss_step:th time step, sample which of ss_step
        # possible such time series
            
        # Flatten spatial dim
        sample_CERRA = sample_CERRA.flatten(0, 1)  # (N_grid, d_features)
        sample_era5 = sample_era5.flatten(0, 1)  # (N_grid, d_features)
        

        if self.standardize:
            # Standardize sample
            sample_CERRA = (sample_CERRA - self.data_mean_CERRA) / self.data_std_CERRA
            sample_era5 = (sample_era5 - self.data_mean_era5) / self.data_std_era5
            
        return sample_CERRA, sample_era5
    
    
    
class ERA5toCERRA2(torch.utils.data.Dataset):
    """
    For our dataset:
    N_t' = 65
    N_t = 65//subsample_step (= 21 for 3h steps)
    dim_x = 268
    dim_y = 238
    N_grid = 268x238 = 63784
    d_features = 17 (d_features' = 18)
    d_forcing = 5
    """
    
    def __init__(
        self,
        dataset_name_CERRA,
        dataset_name_ERA5,
        split,
        standardize=True,
        subset=False,
    ):
        super().__init__()
        
        assert split in ("train", "val", "test"), "Unknown dataset split"
        
        # Determine mode based on which dataset names are provided.
        if dataset_name_CERRA is None and dataset_name_ERA5 is None:
            raise ValueError("At least one dataset must be provided.")
        elif dataset_name_CERRA is not None and dataset_name_ERA5 is not None:
            self.mode = "both"
        elif dataset_name_CERRA is not None:
            self.mode = "CERRA_only"
        else:
            self.mode = "ERA5_only"
        
        member_file_regexp = "*.npy"
        self.split = split
        
        # Load CERRA dataset if available.
        if self.mode in ("both", "CERRA_only"):
            self.sample_dir_path_CERRA = os.path.join("data", dataset_name_CERRA, "samples", split)
            sample_paths_CERRA = glob.glob(os.path.join(self.sample_dir_path_CERRA, member_file_regexp))
            self.sample_names_CERRA = sorted([os.path.basename(path)[4:-4] for path in sample_paths_CERRA])
        
        # Load ERA5 dataset if available.
        if self.mode in ("both", "ERA5_only"):
            self.sample_dir_path_era5 = os.path.join("data", dataset_name_ERA5, "samples", split)
            sample_paths_era5 = glob.glob(os.path.join(self.sample_dir_path_era5, member_file_regexp))
            self.sample_names_era5 = sorted([os.path.basename(path)[4:-4] for path in sample_paths_era5])
        
        # Optionally restrict to a subset of samples.
        if subset:
            if self.mode in ("both", "CERRA_only"):
                self.sample_names_CERRA = self.sample_names_CERRA[:50]
            if self.mode in ("both", "ERA5_only"):
                self.sample_names_era5 = self.sample_names_era5[:50]
        
        # Set up standardization if requested.
        self.standardize = standardize
        if standardize:
            if self.mode in ("both", "CERRA_only"):
                ds_stats_CERRA = utils.load_dataset_stats(dataset_name_CERRA, "cpu")
                self.data_mean_CERRA, self.data_std_CERRA = ds_stats_CERRA["data_mean"], ds_stats_CERRA["data_std"]
            if self.mode in ("both", "ERA5_only"):
                ds_stats_era5 = utils.load_dataset_stats(dataset_name_ERA5, "cpu")
                self.data_mean_era5, self.data_std_era5 = ds_stats_era5["data_mean"], ds_stats_era5["data_std"]
        
        # If subsampling should occur (only during training)
        self.random_subsample = (split == "train")
    
    def __len__(self):
        if self.mode == "both":
            assert len(self.sample_names_CERRA) == len(self.sample_names_era5), "Different number of samples in CERRA and ERA5"
            return len(self.sample_names_CERRA)
        elif self.mode == "CERRA_only":
            return len(self.sample_names_CERRA)
        else:  # ERA5_only
            return len(self.sample_names_era5)
    
    def __getitem__(self, idx):
        if self.mode == "both":
            sample_name_CERRA = self.sample_names_CERRA[idx]
            sample_name_era5 = self.sample_names_era5[idx]
            sample_path_CERRA = os.path.join(self.sample_dir_path_CERRA, f"nwp_{sample_name_CERRA}.npy")
            sample_path_era5 = os.path.join(self.sample_dir_path_era5, f"nwp_{sample_name_era5}.npy")
            try:
                sample_CERRA = torch.tensor(np.load(sample_path_CERRA), dtype=torch.float32)
                sample_era5 = torch.tensor(np.load(sample_path_era5), dtype=torch.float32)
            except ValueError:
                print(f"Failed to load {sample_path_CERRA}")
                print(f"Failed to load {sample_path_era5}")
            # Flatten spatial dimensions.
            
            sample_era5 = self.upsample(sample_era5, sample_CERRA)
            
            sample_CERRA = sample_CERRA.flatten(0, 1)
            sample_era5 = sample_era5.flatten(0, 1)
            
            if self.standardize:
                sample_CERRA = (sample_CERRA - self.data_mean_CERRA) / self.data_std_CERRA
                sample_era5 = (sample_era5 - self.data_mean_era5) / self.data_std_era5
                
                
            if self.split == "test":
                mean_CERRA = self.data_mean_CERRA[:, None, None]
                std_CERRA = self.data_std_CERRA[:, None, None]
                mean_era5 = self.data_mean_era5[:, None, None]
                std_era5 = self.data_std_era5[:, None, None]
                diz_stats = {
                    "mean_CERRA": mean_CERRA,
                    "std_CERRA": std_CERRA,
                    "mean_era5": mean_era5,
                    "std_era5": std_era5
                }
                return sample_CERRA, sample_era5, diz_stats, sample_name_CERRA
            
            else:
                return sample_CERRA, sample_era5
        
        elif self.mode == "CERRA_only":
            sample_name_CERRA = self.sample_names_CERRA[idx]
            sample_path_CERRA = os.path.join(self.sample_dir_path_CERRA, f"nwp_{sample_name_CERRA}.npy")
            try:
                sample_CERRA = torch.tensor(np.load(sample_path_CERRA), dtype=torch.float32)
            except ValueError:
                print(f"Failed to load {sample_path_CERRA}")
            sample_CERRA = sample_CERRA.flatten(0, 1)
            if self.standardize:
                sample_CERRA = (sample_CERRA - self.data_mean_CERRA) / self.data_std_CERRA
            return sample_CERRA
        
        else:  # ERA5_only
            sample_name_era5 = self.sample_names_era5[idx]
            sample_path_era5 = os.path.join(self.sample_dir_path_era5, f"nwp_{sample_name_era5}.npy")
            try:
                sample_era5 = torch.tensor(np.load(sample_path_era5), dtype=torch.float32)
            except ValueError:
                print(f"Failed to load {sample_path_era5}")
            sample_era5 = sample_era5.flatten(0, 1)
            if self.standardize:
                sample_era5 = (sample_era5 - self.data_mean_era5) / self.data_std_era5
            return sample_era5


    def upsample(self, lr_tensor, hr_tensor):
        """
        Upsample the input tensor to match the target tensor's spatial dimensions.
        """
    
        # Flatten spatial dimensions.
        hr_tensor = hr_tensor.permute(2, 0, 1)
        lr_tensor = lr_tensor.permute(2, 0, 1)
        # 1) add batch dim
        era5_batched = lr_tensor.unsqueeze(0)                # [1, C, H_old, W_old]

        # 2) pick the target spatial size from sample_CERRA
        target_size = hr_tensor.shape[-2:]                  # (H_new, W_new)

        # 3) interpolate
        upsampled = F.interpolate(
            era5_batched,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )                                                      # [1, C, H_new, W_new]

        # 5) permute back to original shape
        upsampled = upsampled.permute(0, 2, 3, 1)            # [H_new, W_new, C]
        # 4) drop the batch dim
        return upsampled.squeeze(0)                      # [C, H_new, W_new]