import torch
import pytorch_lightning as pl
import importlib
import wandb
import matplotlib.pyplot as plt
from neural_lam import constants
from neural_lam import vis
from typing import Callable, Optional, Tuple, Union

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
            
            
    def load_metrics_and_plots(self, prediction, high_res, batch_idx, mask=None):
        
        #reshap from (B, C, H, W) to (B, num_grid_nodes, C)
        prediction = prediction.permute(0, 2, 3, 1).flatten(1, 2)
        high_res = high_res.permute(0, 2, 3, 1).flatten(1, 2)
        
        if mask is None:
            mask = torch.ones_like(high_res[:, :, 0])
        
        # Plot samples
        log_plot_dict = {}

        for var_i in constants.VAL_PLOT_VARS_CERRA:
            var_name = constants.PARAM_NAMES_SHORT_CERRA[var_i]
            var_unit = constants.PARAM_UNITS_CERRA[var_i]

            pred_states = prediction[
                batch_idx, :, var_i
            ]  # (S, num_grid_nodes)
            
            target_state = high_res[
                batch_idx, :, var_i
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
                obs_mask = mask[batch_idx],
                title=f"{plot_title} (prior)",
            )

        if not self.trainer.sanity_checking:
            # Log all plots to wandb
            wandb.log(log_plot_dict)

        plt.close("all")   

    

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return opt
    
    
    
    
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
    