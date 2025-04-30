import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from argparse import ArgumentParser
from neural_lam.weather_dataset import ERA5toCERRA  # Ensure this module is available
from neural_lam import vis  # This imports your visualization functions

# --------------------------- Masking Function ---------------------------
def block_random_masking(x, mask_ratio, grid_size, block_size):
    """
    Perform random masking on a flattened grid of nodes, grouped into blocks, 
    based on the original 2D grid layout.

    Args:
        x: Tensor of shape (batch, num_nodes, latent_dim), e.g., (N, 90000, D).
        mask_ratio: Fraction of nodes to mask.
        grid_size: Side length of the grid (e.g., 300 for a 300x300 grid).
        block_size: Size of each block (e.g., 50 for a 50x50 block).

    Returns:
        x_masked: Tensor of shape (batch, num_kept_nodes, latent_dim).
        mask: Binary mask of shape (batch, num_nodes), where 0 indicates an observed pixel and 1 a masked one.
        ids_restore: The indices required to restore the original order.
        ids_keep: Indices of the kept nodes.
    """
    N, num_nodes, D = x.shape
    assert num_nodes == grid_size * grid_size, "num_nodes must match grid_size squared"
    num_blocks = (grid_size // block_size) ** 2  # e.g., if grid_size=300 and block_size=50, then 6*6=36 blocks

    # Compute 2D grid indices from flattened indices
    row_indices = torch.arange(grid_size, device=x.device).repeat_interleave(grid_size)
    col_indices = torch.arange(grid_size, device=x.device).repeat(grid_size)

    # Determine block indices for each node
    block_row_indices = row_indices // block_size
    block_col_indices = col_indices // block_size
    block_indices = block_row_indices * (grid_size // block_size) + block_col_indices  # shape: (num_nodes,)

    # Randomly mask blocks: determine how many nodes are kept
    len_keep = int(num_nodes * (1 - mask_ratio))
    # Generate random noise per block and expand to node level:
    noise = torch.rand(num_blocks, device=x.device).repeat(N, 1)  # shape: (N, num_blocks)
    noise = noise[:, block_indices]  # Each node gets the noise value of its block
    ids_shuffle = torch.argsort(noise, dim=1)  # Lower noise means "keep"
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]  # Indices of nodes to keep

    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    mask = torch.ones([N, num_nodes], device=x.device)
    mask[:, :len_keep] = 0  # 0 = observed, 1 = masked
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore, ids_keep

# ------------------------ Interpolation & Evaluation ------------------------
def interpolate_and_evaluate(high_res, low_res, mask, 
                             low_res_size=(81, 81), high_res_size=(300, 300),
                             eps=1e-8):
    """
    Upsample the low_res grid to high_res size via bilinear interpolation and compute 
    reconstruction metrics only on the reconstructed (masked) portion.

    Args:
        high_res: Tensor of shape (B, num_nodes, num_vars) for the full high-resolution grid.
        low_res: Tensor of shape (B, num_low_nodes, num_vars) for the low-resolution grid.
        mask: Binary mask of shape (B, high_res_size[0]*high_res_size[1]) with 0 for observed and 1 for missing.
        low_res_size: Tuple (H, W) for the low-resolution grid.
        high_res_size: Tuple (H, W) for the high-resolution grid.
        eps: Small constant to avoid division by zero.

    Returns:
        upsampled:      Tensor of shape (B, num_vars, H, W) from interpolation.
        mse_per_var:    List of MSE values computed on the masked (reconstructed) pixels per variable.
        avg_mse:        Average MSE over missing pixels and variables.
        mae_per_var:    List of MAE values computed on the masked (reconstructed) pixels per variable.
        avg_mae:        Average MAE over missing pixels and variables.
        ssim_per_var:   List of SSIM values computed on the masked region for each variable.
        avg_ssim:       Average SSIM over missing regions and variables.
    """
    batch_size = low_res.shape[0]
    num_vars = low_res.shape[-1]

    # Reshape inputs to (B, C, H, W)
    low_res = low_res.view(batch_size, low_res_size[0], low_res_size[1], num_vars).permute(0, 3, 1, 2)
    high_res = high_res.view(batch_size, high_res_size[0], high_res_size[1], num_vars).permute(0, 3, 1, 2)

    # Bilinear interpolation from low_res to high_res dimensions
    upsampled = F.interpolate(low_res, size=high_res_size, mode='bilinear', align_corners=False)

    # Prepare the mask: if provided as (B, H*W), reshape to (B, H, W), then unsqueeze to (B, 1, H, W)
    if mask.dim() == 2:
        mask = mask.view(batch_size, high_res_size[0], high_res_size[1])
    mask = mask.float().unsqueeze(1)  # now shape is (B, 1, H, W)

    # Create composite output: use high_res values in observed regions (mask==0) 
    # and upsampled values in the missing regions (mask==1)
    composite = torch.where(mask == 0, high_res, upsampled)

    # -- Compute Masked MSE only on missing (masked) pixels --
    squared_error = (upsampled - high_res) ** 2
    masked_squared_error = squared_error * mask  # errors only at masked pixels
    num_missing_pixels = mask.sum(dim=[2, 3])  # (B, 1)
    mse_per_sample_var = masked_squared_error.sum(dim=[2, 3]) / (num_missing_pixels + eps)  # (B, num_vars)
    mse_per_var = mse_per_sample_var.mean(dim=0).tolist()  # averaged over batch
    avg_mse = mse_per_sample_var.mean().item()

    # -- Compute Masked MAE only on missing (masked) pixels --
    abs_error = torch.abs(upsampled - high_res)
    masked_abs_error = abs_error * mask
    mae_per_sample_var = masked_abs_error.sum(dim=[2, 3]) / (num_missing_pixels + eps)  # (B, num_vars)
    mae_per_var = mae_per_sample_var.mean(dim=0).tolist()  # averaged over batch
    avg_mae = mae_per_sample_var.mean().item()

    # -- Compute Masked SSIM: for each sample and each variable, compute over the bounding box of missing pixels --
    ssim_values = np.zeros((batch_size, num_vars))
    for i in range(batch_size):
        m_np = mask[i, 0].cpu().numpy()  # shape: (H, W)
        missing_indices = np.argwhere(m_np == 1)
        for j in range(num_vars):
            if missing_indices.size == 0:
                ssim_values[i, j] = 1.0
            else:
                rmin, cmin = missing_indices.min(axis=0)
                rmax, cmax = missing_indices.max(axis=0)
                cropped_high = high_res[i, j, rmin:rmax+1, cmin:cmax+1].cpu().numpy()
                cropped_up   = upsampled[i, j, rmin:rmax+1, cmin:cmax+1].cpu().numpy()
                data_range = cropped_high.max() - cropped_high.min()
                if data_range < eps:
                    ssim_values[i, j] = 1.0
                else:
                    ssim_values[i, j] = ssim(cropped_up, cropped_high, data_range=data_range)
    ssim_per_var = ssim_values.mean(axis=0).tolist()
    avg_ssim = float(ssim_values.mean())

    return upsampled, mse_per_var, avg_mse, mae_per_var, avg_mae, ssim_per_var, avg_ssim

# ----------------------------- Main Routine -----------------------------
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    parser = ArgumentParser(description="Evaluate Bilinear Interpolation with Block Masking on High-Resolution Grid")
    parser.add_argument("--dataset_cerra", type=str,
                        default="/aspire/CarloData/MASK_GNN_DATA/CERRA_interpolated_300x300",
                        help="Path to CERRA dataset (default provided)")
    parser.add_argument("--dataset_era5", type=str,
                        default="/aspire/CarloData/MASK_GNN_DATA/ERA5_60_n2_40_18",
                        help="Path to ERA5 dataset (default provided)")
    parser.add_argument("--subset_ds", type=int, default=0,
                        help="Use only a small subset of the dataset, for debugging (0=false)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--n_workers", type=int, default=3, help="Number of workers in data loader (default: 3)")
    parser.add_argument("--epochs", type=int, default=200, help="Upper epoch limit (default: 200)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    # Masking-related arguments
    parser.add_argument("--mask_ratio", type=float, default=0.75,
                        help="Fraction of nodes (via blocks) to mask (default: 0.2)")
    parser.add_argument("--block_size", type=int, default=50,
                        help="Size of each masking block (e.g., 50 for 50x50 block)")
    parser.add_argument("--grid_size", type=int, default=300,
                        help="Side length of the grid (e.g., 300 for a 300x300 grid)")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create validation DataLoader from the ERA5toCERRA dataset.
    val_loader = torch.utils.data.DataLoader(
        ERA5toCERRA(args.dataset_cerra, args.dataset_era5, split="val", subset=bool(args.subset_ds)),
        args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
    )

    print("Evaluating on validation set:")
    mse_per_var_list = []
    avg_mse_list = []
    mae_per_var_list = []
    avg_mae_list = []
    ssim_values_list = []
    avg_ssim_list = []

    for idx, (high_res, low_res) in enumerate(val_loader):
        # Move tensors to device.
        high_res = high_res.to(device)  # Expected shape: (B, num_nodes, num_vars) with num_nodes = grid_size^2
        low_res = low_res.to(device)    # Expected shape: (B, num_low_nodes, num_vars)

        # Apply block random masking on the high-res grid.
        # The high-res grid serves as the ground truth.
        _, mask, _, _ = block_random_masking(high_res, args.mask_ratio, args.grid_size, args.block_size)
        # 'mask' is of shape (B, num_nodes) with 0 for observed and 1 for masked.

        # Interpolate from low_res to high_res and evaluate metrics on masked regions.
        (upsampled_grid, mse_per_var, avg_mse, 
         mae_per_var, avg_mae, ssim_per_var, avg_ssim) = interpolate_and_evaluate(
            high_res, low_res, mask,
            low_res_size=(81, 81),
            high_res_size=(args.grid_size, args.grid_size)
        )

        # For plotting, form the full composite grid that matches the original high_res grid.
        # Reshape the ground truth to (B, num_vars, H, W)
        batch_size = high_res.shape[0]
        num_vars = high_res.shape[-1]
        high_res_grid = high_res.view(batch_size, args.grid_size, args.grid_size, num_vars).permute(0, 3, 1, 2)
        # Reshape the mask from (B, num_nodes) to (B, H, W, 1)
        obs_mask = mask.view(batch_size, args.grid_size, args.grid_size).unsqueeze(-1).float()
        # Compute the composite grid: observed regions use ground truth, masked regions use the interpolated values.
        composite_grid = torch.where(obs_mask.permute(0, 3, 1, 2) == 0, high_res_grid, upsampled_grid)

        # Plot using your provided plotting function.
        if idx == 0:
            fig = vis.plot_ensemble_prediction(
                composite_grid[0, 0, :, :],         # Predicted grid (composite; full resolution)
                high_res_grid[0, 0, :, :].detach(),    # Ground truth grid (full resolution) for the first variable
                obs_mask=obs_mask[0],                  # Observed mask (full resolution) for the first sample
                title="Bilinear-interpolation"
            )
            fig.savefig("bilinear_interpolation.png")

        mse_per_var_list.append(mse_per_var)
        avg_mse_list.append(avg_mse)
        mae_per_var_list.append(mae_per_var)
        avg_mae_list.append(avg_mae)
        ssim_values_list.append(ssim_per_var)
        avg_ssim_list.append(avg_ssim)

    # Aggregate metrics over all batches.
    mse_per_var_avg = np.mean(mse_per_var_list, axis=0)
    avg_mse_avg = np.mean(avg_mse_list)
    mae_per_var_avg = np.mean(mae_per_var_list, axis=0)
    avg_mae_avg = np.mean(avg_mae_list)
    ssim_per_var_avg = np.mean(ssim_values_list, axis=0)
    avg_ssim_avg = np.mean(avg_ssim_list)

    print(f"Validation MSE per variable: {mse_per_var_avg}")
    print(f"Validation Average MSE: {avg_mse_avg:.6f}")
    print(f"Validation MAE per variable: {mae_per_var_avg}")
    print(f"Validation Average MAE: {avg_mae_avg:.6f}")
    print(f"Validation SSIM per variable: {ssim_per_var_avg}")
    print(f"Validation Average SSIM: {avg_ssim_avg:.6f}")

if __name__ == "__main__":
    main()
