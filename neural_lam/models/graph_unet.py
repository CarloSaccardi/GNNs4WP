import logging

import torch
from torch import nn
import torch.distributions as tdists
from typing import Union
import wandb
import matplotlib.pyplot as plt

import torch.nn.functional as F

from neural_lam import constants, mask_utils, utils, vis
from neural_lam.models.base_gnn_module import BaseGraphModule
from neural_lam.models.hi_graph_latent_decoder import HiGraphLatentUnet

import random

logger = logging.getLogger(__name__)


class GraphUNet(BaseGraphModule):
    """
    Graph-based Ensemble Forecasting Model with optional masking support.
    Extends BaseGraphModule for graph/static loading.
    """

    def __init__(self, args) -> None:
        super().__init__(args)
        self.run_name = args.run_name

        # Metric storage
        self.test_MSEs = []
        self.test_MAEs = []
        self.test_MSEs_masked = []
        self.test_MAEs_masked = []

        # Blueprint for MLP embedders
        blueprint = [args.hidden_dim] * (args.hidden_layers + 1)

        # Embedders: grid and graph↔mesh
        self.high_res_embedder = utils.make_mlp([self.grid_dim] + blueprint)
        self.g2m_embedder = utils.make_mlp([self.g2m_dim] + blueprint)
        self.m2g_embedder = utils.make_mlp([self.m2g_dim] + blueprint)

        # Mesh structure embedding layers
        mesh_dims = [feat.shape[1] for feat in self.mesh_static_features]
        m2m_dim = self.m2m_features[0].shape[1]
        up_dim = self.mesh_up_features[0].shape[1]
        down_dim = self.mesh_down_features[0].shape[1]

        self.mesh_embedders = nn.ModuleList(
            utils.make_mlp([dim] + blueprint) for dim in mesh_dims
        )
        self.m2m_embedders = nn.ModuleList(
            utils.make_mlp([m2m_dim] + blueprint) for _ in mesh_dims
        )
        self.mesh_up_embedders = nn.ModuleList(
            utils.make_mlp([up_dim] + blueprint) for _ in mesh_dims[:-1]
        )
        self.mesh_down_embedders = nn.ModuleList(
            utils.make_mlp([down_dim] + blueprint) for _ in mesh_dims[:-1]
        )

        # Latent UNet processor
        latent_dim = args.latent_dim or args.hidden_dim
        self.gnn_unet = HiGraphLatentUnet(
            self.g2m_edge_index,
            self.m2m_edge_index,
            self.m2g_edge_index,
            self.mesh_up_edge_index,
            self.mesh_down_edge_index,
            self.variational,
            args.hidden_dim,
            latent_dim,
            args.processor_layers,
            hidden_layers=args.hidden_layers,
            output_std=self.output_std,
            output_dist="diagonal",
        )

    def _log_mesh_structure(self) -> None:
        """
        Log hierarchy of mesh nodes and edges.
        """
        levels = len(self.mesh_static_features)
        print(f"Mesh hierarchy: {levels} levels")
        for i, feat in enumerate(self.mesh_static_features):
            nodes = feat.shape[0]
            same_edges = self.m2m_features[i].shape[0]
            print(f"Level {i}: {nodes} nodes, {same_edges} same-level edges")
            if i + 1 < levels:
                up = self.mesh_up_features[i].shape[0]
                down = self.mesh_down_features[i].shape[0]
                print(f"Between {i}<->{i+1}: {up} up edges, {down} down edges")

    def embed_all(self, high_res: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Embed grid, nodes, and edges into hidden representations.
        Returns:
            high_res_emb: (B, N, H)
            graph_emb: dict of embeddings per relation
        """
        B = high_res.shape[0]
        # Grid embedding
        high_res_emb = self.high_res_embedder(high_res)

        # Basic graph embeddings
        graph_emb = {
            "g2m_edge_index": mask_utils.adjust_g2m_edge_index(self.g2m_edge_index),
            "g2m": self.expand_to_batch(self.g2m_embedder(self.g2m_features), B),
            "m2g": self.expand_to_batch(self.m2g_embedder(self.m2g_features), B),
        }

        # Mesh nodes
        graph_emb["mesh"] = [
            self.expand_to_batch(embed(feat), B)
            for embed, feat in zip(self.mesh_embedders, self.mesh_static_features)
        ]

        # Same-level edges
        graph_emb["m2m"] = [
            self.expand_to_batch(embed(feat), B)
            for embed, feat in zip(self.m2m_embedders, self.m2m_features)
        ]

        # Up/down edges
        graph_emb["mesh_up"] = [
            self.expand_to_batch(embed(feat), B)
            for embed, feat in zip(self.mesh_up_embedders, self.mesh_up_features)
        ]
        graph_emb["mesh_down"] = [
            self.expand_to_batch(embed(feat), B)
            for embed, feat in zip(self.mesh_down_embedders, self.mesh_down_features)
        ]

        return high_res_emb, graph_emb

    def encode_decode(
        self, high_res_emb: torch.Tensor, graph_emb: dict
    ) -> tuple[tdists.Distribution, torch.Tensor, torch.Tensor]:
        """
        Pass embeddings through the latent UNet.
        Returns:
            latent_dist, pred_mean, pred_std
        """
        return self.gnn_unet(high_res_emb, graph_emb)

    def compute_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        latent_dist: Union[tdists.Distribution, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute reconstruction MSE and KL divergence loss.
        """
        batch_size = prediction.shape[0]
        diff_squared = (prediction - target)**2
        total_loss = diff_squared.sum() / batch_size
            
        return total_loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Single training iteration.
        """
        # Unpack: batch can be tuple or single tensor
        ground_truth, high_res = (
            (batch[0], batch[1]) if isinstance(batch, (list, tuple)) and len(batch) == 2
            else (batch, None)
        )

        high_res_emb, graph_emb = self.embed_all(high_res)
        latent_dist, pred_mean, _ = self.encode_decode(high_res_emb, graph_emb)
        loss = self.compute_loss(pred_mean, ground_truth, latent_dist)

        # Log metrics
        log_data = {
            "train_loss": loss,
        }
        self.log_dict(log_data, prog_bar=True, on_epoch=True, sync_dist=True)

        return loss
        

    def validation_step(self, batch, *args):
        """
        Run validation on single batch
        """
        
        # Unpack: batch can be tuple or single tensor
        ground_truth, high_res = (
            (batch[0], batch[1]) if isinstance(batch, (list, tuple)) and len(batch) == 2
            else (batch, None)
        )
        
        high_res_grid_emb, graph_emb = self.embed_all(high_res)
        var_dist, prediction, _ = self.encode_decode(high_res_grid_emb, graph_emb)
        val_loss = self.compute_loss(prediction, ground_truth, var_dist)
        
        
        # Log loss per time step forward and mean
        val_log_dict = {
            "val_loss": val_loss,
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
            self.load_metrics_and_plots(prediction, ground_truth, batch_idx, mask=None)
            
    def test_step(self, batch, batch_idx: int) -> dict:
        """
        Evaluate model on a test batch and store metrics.
        Computes overall MSE, MAE, RMSE, and SSIM as well as per-variable metrics.
        """
        batch_size = batch[0].shape[0]
        ground_truth, low_res, diz_stats = batch
        ground_truth = ground_truth.float()
        low_res = low_res.float()
        
        # Get loss, ground truth, and predictions from the loss function.
        # Note: ground_truth and predictions are assumed to be in (B, C, H, W)
        low_res_grid_emb, graph_emb = self.embed_all(low_res)
        var_dist, predictions, _ = self.encode_decode(low_res_grid_emb, graph_emb)
        
        #reshape from (B, n_nodes, C) to (B, C, H, W) where H and W are 368x368
        B, n_nodes, C = predictions.shape
        C_low_res = low_res.shape[-1]
        H, W = 368, 368
        assert H * W == n_nodes

        # swap and reshape in one go:
        predictions = (
            predictions
            .permute(0, 2, 1)          # (B, C, n_nodes)
            .view(B, C, H, W)          # (B, C, H, W)
        )
        # And do the same for ground_truth and low_res:
        ground_truth = ground_truth.permute(0, 2, 1).view(B, C, H, W)
        low_res     = low_res    .permute(0, 2, 1).view(B, C_low_res, H, W)
         
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
        
        self.test_metrics_and_plots(predictions, ground_truth, low_res, diz_stats)
        
        return log_metrics
            
            
    def load_metrics_and_plots(self, prediction, high_res, batch_idx, mask=None):
        
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
        
        high_res_mean = diz_stats["mean_CERRA"].unsqueeze(-1).unsqueeze(-1)
        high_res_std = diz_stats["std_CERRA"].unsqueeze(-1).unsqueeze(-1)
        low_res_mean = diz_stats["mean_era5"].unsqueeze(-1).unsqueeze(-1)
        low_res_std = diz_stats["std_era5"].unsqueeze(-1).unsqueeze(-1)
        
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
        save_dir = "plot_tests"
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{var_name}_sample_{sample}.png")
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        
    # def test_step(self, batch, batch_idx: int) -> dict:
    #     """
    #     Evaluate model on a test batch and store metrics.
    #     """
    #     ground_truth, high_res = (
    #         (batch[0], batch[1]) if isinstance(batch, (list, tuple)) and len(batch) == 2
    #         else (batch, None)
    #     )
    #     high_res_emb, graph_emb = self.embed_all(high_res)
    #     latent_dist, pred_mean, _ = self.encode_decode(high_res_emb, graph_emb)
    #     mse, _ = utils.compute_MSE_entiregrid(pred_mean, ground_truth)
    #     mae, _ = utils.compute_MAE_entiregrid(pred_mean, ground_truth)
    #     # If mask available, compute masked metrics
    #     mask = graph_emb.get("mask")
    #     if mask is not None:
    #         mse_masked, _ = utils.compute_MSE_masked(pred_mean, ground_truth, mask)
    #         mae_masked, _ = utils.compute_MAE_masked(pred_mean, ground_truth, mask)
    #     else:
    #         mse_masked = mae_masked = torch.tensor(float('nan'), device=pred_mean.device)

    #     self.test_MSEs.append(mse)
    #     self.test_MAEs.append(mae)
    #     self.test_MSEs_masked.append(mse_masked)
    #     self.test_MAEs_masked.append(mae_masked)

    #     self.log_dict({
    #         "test_mse": mse,
    #         "test_mae": mae,
    #         "test_mse_masked": mse_masked,
    #         "test_mae_masked": mae_masked,
    #     }, prog_bar=False, on_epoch=True, sync_dist=True)

    #     return {"test_mse": mse, "test_mae": mae,
    #             "test_mse_masked": mse_masked, "test_mae_masked": mae_masked}

    # def on_test_epoch_end(self) -> None:
    #     """
    #     Aggregate and log test metrics at epoch end.
    #     """
    #     def mean(tensor_list):
    #         return torch.stack([t for t in tensor_list if not torch.isnan(t)]).mean()

    #     metrics = {
    #         "test_mse_mean": mean(self.test_MSEs),
    #         "test_mae_mean": mean(self.test_MAEs),
    #         "test_mse_masked_mean": mean(self.test_MSEs_masked),
    #         "test_mae_masked_mean": mean(self.test_MAEs_masked),
    #     }
    #     self.log_dict(metrics, prog_bar=True, sync_dist=True)
    #     print(
    #         f"Test MSE: {metrics['test_mse_mean']:.4f}, "
    #         f"MAE: {metrics['test_mae_mean']:.4f}\n"
    #         f"Masked MSE: {metrics['test_mse_masked_mean']:.4f}, "
    #         f"Masked MAE: {metrics['test_mae_masked_mean']:.4f}"
    #     )