import torch
import pytorch_lightning as pl
import importlib
import wandb
import matplotlib.pyplot as plt
from neural_lam import constants
from neural_lam import vis
from typing import Callable, Optional, Tuple
import random

network_module = importlib.import_module("physicsnemo.models.diffusion")




class UNetWrapper(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        
        # for compatibility with older versions that took only 1 dimension
        if isinstance(args.img_resolution, int):
            self.img_shape_x = self.img_shape_y = args.img_resolution
        else:
            self.img_shape_y = args.img_resolution[0]
            self.img_shape_x = args.img_resolution[1]

        self.img_in_channels = args.img_in_channels
        self.img_out_channels = args.img_out_channels
        self.lr = args.lr
        self.wandb_project = args.wandb_project
        
        self.model_kwargs = {
            'checkpoint_level': args.checkpoint_level,
            'N_grid_channels': args.N_grid_channels,
            'embedding_type': args.embedding_type,
            'model_channels': args.model_channels,
            'channel_mult': args.channel_mult,
            'attn_resolutions': args.attn_resolutions,
        }

        model_class = getattr(network_module, args.model_type)
        self.model = model_class(
            img_resolution=args.img_resolution,
            in_channels=args.img_in_channels + args.N_grid_channels + args.img_out_channels,
            out_channels=args.img_out_channels,
            **self.model_kwargs,
        )

        self.loss_fn = RegressionLoss()

    def forward(
        self,
        x: torch.Tensor,
        img_lr: torch.Tensor,
        force_fp32: bool = False,
        **model_kwargs: dict,
    ) -> torch.Tensor:
        """
        Forward pass of the UNet wrapper model.

        This method concatenates the input tensor with the low-resolution conditioning tensor
        and passes the result through the underlying model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor, typically zero-filled, of shape (B, C_hr, H, W).
        img_lr : torch.Tensor
            Low-resolution conditioning image of shape (B, C_lr, H, W).
        force_fp32 : bool, optional
            Whether to force FP32 precision regardless of the `use_fp16` attribute,
            by default False.
        **model_kwargs : dict
            Additional keyword arguments to pass to the underlying model
            `self.model` forward method.

        Returns
        -------
        torch.Tensor
            Output tensor (prediction) of shape (B, C_hr, H, W).

        Raises
        ------
        ValueError
            If the model output dtype doesn't match the expected dtype.
        """
        # SR: concatenate input channels
        if img_lr is not None:
            x = torch.cat((x, img_lr), dim=1)


        F_x = self.model(
            x,  # (c_in * x).to(dtype),
            torch.zeros(x.shape[0], device=x.device),  # c_noise.flatten()
            class_labels=None,
            **model_kwargs,
        )

        # skip connection
        D_x = F_x.to(torch.float32)
        return D_x

    def training_step(self, batch, *args):
        batch_size = batch[0].shape[0]
        img_clean, img_lr, *rest = batch
        img_clean = img_clean.float()
        img_lr = img_lr.float()
        loss_map, _, _ = self.loss_fn(
                            net=self,
                            img_clean=img_clean,
                            img_lr=img_lr
                        )
        loss = loss_map.sum() / batch_size
        
        log_dict = {
            "train_loss": loss,
        }
        self.log_dict(
            log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )
        return loss

    def validation_step(self, batch, *args):
        batch_size = batch[0].shape[0]
        img_clean, img_lr = batch
        img_clean = img_clean.float()
        img_lr = img_lr.float()
        val_map, ground_truth, predictions = self.loss_fn(
                                                net=self,
                                                img_clean=img_clean,
                                                img_lr=img_lr
                                            )
        val_loss = val_map.sum() / batch_size
        
        # Log loss per time step forward and mean
        val_log_dict = {
            "val_loss": val_loss
        }
        self.log_dict(
            val_log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )
        
        batch_idx = args[0]
        
        # Plot some example predictions using prior and encoder
        if (
            self.trainer.is_global_zero
            and batch_idx == 0
            and self.current_epoch % 10 == 0
            and self.wandb_project is not None
        ):
            self.load_metrics_and_plots(predictions, ground_truth, batch_idx, mask=None)
    
    
    def test_step(self, batch, batch_idx: int) -> dict:
        """
        Evaluate model on a test batch and store metrics.
        Computes overall MSE, MAE, RMSE, and SSIM as well as per-variable metrics.
        """
        batch_size = batch[0].shape[0]
        img_clean, img_lr, diz_stats = batch
        img_clean = img_clean.float()
        img_lr = img_lr.float()
        
        # Get loss, ground truth, and predictions from the loss function.
        # Note: ground_truth and predictions are assumed to be in (B, C, H, W)
        _, ground_truth, predictions = self.loss_fn(
            net=self,
            img_clean=img_clean,
            img_lr=img_lr
        )
        
        # Overall metrics across all variables.
        mse_all = torch.mean((predictions - ground_truth) ** 2)
        mae_all = torch.mean(torch.abs(predictions - ground_truth))
        rmse_all = torch.sqrt(mse_all)
        
        from torchmetrics.functional import structural_similarity_index_measure as ssim_func
        overall_data_range = (ground_truth.max() - ground_truth.min()).item()
        ssim_all = ssim_func(predictions, ground_truth, data_range=overall_data_range)
        
        # Define variable names in order.
        var_names = ['u10', 'v10', 't2m', 'sshf', 'zust']
        
        mse_vars = {}
        mae_vars = {}
        rmse_vars = {}
        ssim_vars = {}
        for i, var_name in enumerate(var_names):
            # Extract the i-th variable (shape: (B, H, W)).
            pred_i = predictions[:, i, :, :]
            gt_i = ground_truth[:, i, :, :]
            
            mse_val = torch.mean((pred_i - gt_i) ** 2)
            mse_vars[f"test_mse_{var_name}"] = mse_val
            mae_vars[f"test_mae_{var_name}"] = torch.mean(torch.abs(pred_i - gt_i))
            rmse_vars[f"test_rmse_{var_name}"] = torch.sqrt(mse_val)
            
            # SSIM expects the input to be (B, C, H, W); for a single channel, unsqueeze.
            data_range_i = (gt_i.max() - gt_i.min()).item()
            ssim_vars[f"test_ssim_{var_name}"] = ssim_func(pred_i.unsqueeze(1),
                                                            gt_i.unsqueeze(1),
                                                            data_range=data_range_i)
        
        # Combine all metrics.
        log_metrics = {
            "test_mse": mse_all,
            "test_mae": mae_all,
            "test_rmse": rmse_all,
            "test_ssim": ssim_all,
        }
        log_metrics.update(mse_vars)
        log_metrics.update(mae_vars)
        log_metrics.update(rmse_vars)
        log_metrics.update(ssim_vars)
        
        self.log_dict(log_metrics, prog_bar=False, on_epoch=True, sync_dist=True)
        
        self.test_metrics_and_plots(predictions, ground_truth, img_lr)
        
        return log_metrics
    
    
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return opt
            
            
    def load_metrics_and_plots(self, prediction, high_res, batch_idx, mask=None):
        
        #reshap from (B, C, H, W) to (B, num_grid_nodes, C)
        prediction = prediction.permute(0, 2, 3, 1).flatten(1, 2)
        high_res = high_res.permute(0, 2, 3, 1).flatten(1, 2)
        
        if mask is None:
            mask = torch.ones_like(high_res[:, :, 0])
        
        # Plot samples
        log_plot_dict = {}

        var_i = random.randint(0, len(constants.PARAM_NAMES_SHORT_CERRA) - 1)
        var_name = constants.PARAM_NAMES_SHORT_CERRA[var_i]
        var_unit = constants.PARAM_UNITS_CERRA[var_i]
        
        sample = random.randint(0, prediction.shape[0] - 1)

        pred_states = prediction[
            sample, :, var_i
        ]  # (S, num_grid_nodes)
        
        target_state = high_res[
            sample, :, var_i
        ]  # (num_grid_nodes,)

        plot_title = (
            f"{var_name} ({var_unit})"
        )

        # Make plots
        log_plot_dict[
            f"pred_{var_name}"
        ] = vis.plot_ensemble_prediction(
            pred_states,
            target_state,
            obs_mask = mask[sample],
            title=f"{plot_title} (prior)",
        )

        if not self.trainer.sanity_checking:
            # Log all plots to wandb
            wandb.log(log_plot_dict)

        plt.close("all")   
        
        
        
    def test_metrics_and_plots(self, prediction, high_res, img_lr, diz_stats):
        """
        Plot a random sample for a random variable from the batch. The figure includes 
        the low resolution input, target (high_res), prediction, and residual.
        The overall figure title indicates the variable name, and the plot is saved.
        """
        import os
        import matplotlib.pyplot as plt
        
        high_res_mean = diz_stats["mean_CERRA"]
        high_res_std = diz_stats["std_CERRA"]
        low_res_mean = diz_stats["mean_era5"]
        low_res_std = diz_stats["std_era5"]
        
        prediction = prediction * high_res_std + high_res_mean
        high_res = high_res * high_res_std + high_res_mean
        img_lr = img_lr * low_res_std + low_res_mean

        # Select a random sample from the batch and a random variable index.
        sample = random.randint(0, prediction.shape[0] - 1)
        var_i = random.randint(0, prediction.shape[1] - 1)
        
        # Retrieve variable name and unit from constants.
        var_name = constants.PARAM_NAMES_SHORT_CERRA[var_i]
        var_unit = constants.PARAM_UNITS_CERRA[var_i]
        
        # Extract the images for the selected variable and sample.
        input_img = img_lr[sample, var_i, :, :].detach().cpu().numpy()
        target_img = high_res[sample, var_i, :, :].detach().cpu().numpy()
        pred_img   = prediction[sample, var_i, :, :].detach().cpu().numpy()
        residual_img = target_img - pred_img
        
        # Create a plot with four subplots.
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(input_img, cmap='plasma', origin='lower')
        axes[0].set_title("Input")
        axes[0].axis("off")
        
        axes[1].imshow(target_img, cmap='plasma', origin='lower')
        axes[1].set_title("Target")
        axes[1].axis("off")
        
        axes[2].imshow(pred_img, cmap='plasma', origin='lower')
        axes[2].set_title("Prediction")
        axes[2].axis("off")
        
        axes[3].imshow(residual_img, cmap='plasma', origin='lower')
        axes[3].set_title("Residual")
        axes[3].axis("off")
        
        # Set overall title with variable name and unit.
        fig.suptitle(f"{var_name} ({var_unit})", fontsize=16)
        
        # Save plot to a given folder.
        save_dir = "plots"
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{var_name}_sample_{sample}.png")
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    
    
class RegressionLoss:
    """
    Regression loss function for the deterministic predictions.
    Note: this loss does not apply any reduction.

    Attributes
    ----------
    sigma_data: float
        Standard deviation for data. Deprecated and ignored.

    Note
    ----
    Reference: Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C.,Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric Downscaling.
    arXiv preprint arXiv:2309.15214.
    """

    def __init__(self):
        """
        Arguments
        ----------
        """
        return

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: torch.Tensor,
        img_lr: torch.Tensor,
        augment_pipe: Optional[
            Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]
        ] = None,
    ) -> torch.Tensor:
        """
        Calculate and return the regression loss for
        deterministic predictions.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network model that will make predictions.
            Expected signature: `net(x, img_lr,
            augment_labels=augment_labels, force_fp32=False)`, where:
                x (torch.Tensor): Tensor of shape (B, C_hr, H, W). Is zero-filled.
                img_lr (torch.Tensor): Low-resolution input of shape (B, C_lr, H, W)
                augment_labels (torch.Tensor, optional): Optional augmentation
                labels, returned by `augment_pipe`.
                force_fp32 (bool, optional): Whether to force the model to use
                fp32, by default False.
            Returns:
                torch.Tensor: Predictions of shape (B, C_hr, H, W)

        img_clean : torch.Tensor
            High-resolution input images of shape (B, C_hr, H, W).
            Used as ground truth and for data augmentation if 'augment_pipe' is provided.

        img_lr : torch.Tensor
            Low-resolution input images of shape (B, C_lr, H, W).
            Used as input to the neural network.

        augment_pipe : callable, optional
            An optional data augmentation function.
            Expected signature:
                img_tot (torch.Tensor): Concatenated high and low resolution
                    images of shape (B, C_hr+C_lr, H, W)
            Returns:
                Tuple[torch.Tensor, Optional[torch.Tensor]]:
                    - Augmented images of shape (B, C_hr+C_lr, H, W)
                    - Optional augmentation labels

        Returns
        -------
        torch.Tensor
            A tensor representing the per-sample element-wise squared
            difference between the network's predictions and the high
            resolution images `img_clean` (possibly data-augmented by
            `augment_pipe`).
            Shape: (B, C_hr, H, W), same as `img_clean`.
        """
        weight = (
            1.0  # (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        )

        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]

        zero_input = torch.zeros_like(y, device=img_clean.device)
        D_yn = net(zero_input, y_lr, force_fp32=False, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)

        return loss, y, D_yn
    