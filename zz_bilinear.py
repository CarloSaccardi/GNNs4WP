import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from neural_lam.weather_dataset import ERA5toCERRA
from argparse import ArgumentParser
import os
import matplotlib.pyplot as plt
from neural_lam import constants, vis
import pytorch_lightning as pl
import numpy as np



os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def interpolate_and_evaluate(high_res, low_res, low_res_size=(81, 81), high_res_size=(300, 300)):
    batch_size, _, num_vars = low_res.shape

    # Reshape to (B, C, H, W)
    low_res = low_res.view(batch_size, low_res_size[0], low_res_size[1], num_vars).permute(0, 3, 1, 2)
    high_res = high_res.view(batch_size, high_res_size[0], high_res_size[1], num_vars).permute(0, 3, 1, 2)

    # Bilinear interpolation
    upsampled = F.interpolate(low_res, size=high_res_size, mode='bilinear', align_corners=False)

    # Compute MSE and SSIM per variable
    mse_values = torch.mean((upsampled - high_res) ** 2, dim=[2, 3])  # Mean over spatial dims
    mse_per_var = mse_values.mean(dim=0).tolist()  # Average over batch
    avg_mse = sum(mse_per_var) / num_vars  # Average over variables

    ssim_values = []
    for j in range(num_vars):
        per_var_ssim = [
            ssim(upsampled[i, j].cpu().numpy(), high_res[i, j].cpu().numpy(),
                 data_range=high_res[i, j].max().item() - high_res[i, j].min().item())
            for i in range(batch_size)
        ]
        ssim_values.append(sum(per_var_ssim) / batch_size)  # Average over batch
    avg_ssim = sum(ssim_values) / num_vars  # Average over variables

    return upsampled, mse_per_var, avg_mse, ssim_values, avg_ssim


def plot_first_variable(upsampled_grid):
    """Plots the first variable of the first sample in the batch."""
    first_variable = upsampled_grid[0, 0].cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(first_variable, cmap='viridis')
    plt.colorbar(label="Value")
    plt.title("First Variable of Upsampled Grid")
    plt.show()
    plt.savefig("upsampled_grid.png")
    


def main():
    """
    Main function for training and evaluating models
    """
    parser = ArgumentParser(
        description="Train or evaluate NeurWP models for LAM"
    )

    # General options
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
        "--subset_ds",
        type=int,
        default=0,
        help="Use only a small subset of the dataset, for debugging"
        "(default: 0=false)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=3,
        help="Number of workers in data loader (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="upper epoch limit (default: 200)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="batch size (default: 4)"
    )

    args = parser.parse_args()
    
    # Instantiate model + trainer
    if torch.cuda.is_available():
        device_name = "cuda"
    else:
        device_name = "cpu"

    train_loader = torch.utils.data.DataLoader(
        ERA5toCERRA(
            args.dataset_cerra,
            args.dataset_era5,
            split="train",
            subset=bool(args.subset_ds),
        ),
        args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        ERA5toCERRA(
            args.dataset_cerra,
            args.dataset_era5,
            split="val",
            subset=bool(args.subset_ds),
        ),
        args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
    )
    
    print("Evaluating on validation set:")
    mse_per_var_list = []
    avg_mse_list = []
    ssim_values_list = []
    avg_ssim_list = []
    
    mask = np.ones((300, 300))
    
    for idx, (high_res, low_res) in enumerate(val_loader):
        high_res, low_res = high_res.to(device_name), low_res.to(device_name)
        upsampled_grid, mse_per_var, avg_mse, ssim_values, avg_ssim = interpolate_and_evaluate(high_res, low_res)
        
        if idx == 0:
            
            fig = vis.plot_ensemble_prediction(
                    upsampled_grid[0,0,:,:],
                    high_res[0,:,0],
                    obs_mask = torch.tensor(mask).unsqueeze(-1).to(device_name),
                    title=f"Bilinear-interpolation",
                )
            fig.savefig("bilinear_interpolation.png")
            
        mse_per_var_list.append(mse_per_var)
        avg_mse_list.append(avg_mse)
        ssim_values_list.append(ssim_values)
        avg_ssim_list.append(avg_ssim)
        
    mse_per_var = torch.tensor(mse_per_var_list).mean(dim=0).tolist()
    avg_mse = torch.tensor(avg_mse_list).mean().item()
    ssim_values = torch.tensor(ssim_values_list).mean(dim=0).tolist()
    avg_ssim = torch.tensor(avg_ssim_list).mean().item()
    
    
    
    
    print(f"Validation MSE per variable: {mse_per_var}")
    print(f"Validation Average MSE: {avg_mse:.6f}, Validation SSIM per variable: {ssim_values}")
    print(f"Validation Average SSIM: {avg_ssim:.6f}")


if __name__ == "__main__":
    main()
