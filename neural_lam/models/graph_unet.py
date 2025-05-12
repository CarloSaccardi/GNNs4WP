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
        grid_feats = torch.cat(
            [high_res, self.expand_to_batch(self.grid_static_features, B)], dim=-1
        )
        high_res_emb = self.high_res_embedder(grid_feats)

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
        mse = F.mse_loss(prediction, target, reduction="mean")
        # MSE per feature: mean over batch and nodes
        diff = prediction - target
        mse_per_var = diff.pow(2).mean(dim=1).mean(dim=0)

        if self.variational:
            # KL divergence to standard normal
            std_normal = tdists.Normal(
                torch.zeros_like(latent_dist.loc), torch.ones_like(latent_dist.scale)
            )
            kl_term = tdists.kl.kl_divergence(latent_dist, std_normal).sum(dim=-1).mean()
            total_loss = mse + self.kl_beta * kl_term
            
        else:
            # No KL divergence
            kl_term = torch.tensor(0.0, device=prediction.device)
            total_loss = mse
            
        return total_loss, mse_per_var, kl_term

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
        loss, mse_per_var, kl_term = self.compute_loss(pred_mean, ground_truth, latent_dist)

        # Log metrics
        log_data = {
            "train_loss": loss,
            "train_kl": kl_term,
            **{
                f"train_mse_{constants.PARAM_NAMES_SHORT_CERRA[i]}_{constants.PARAM_UNITS_CERRA[i]}": mse_per_var[i]
                for i in range(mse_per_var.numel())
            },
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
        val_loss, val_mse_per_var, kl_term = self.compute_loss(prediction, ground_truth, var_dist)
        
        
        # Log loss per time step forward and mean
        val_log_dict = {
            "val_loss": val_loss,
            "val_kl_div": kl_term,
            **{
                f"val_MSE_{constants.PARAM_NAMES_SHORT_CERRA[var_i]} {constants.PARAM_UNITS_CERRA[var_i]}": val_mse_per_var[var_i]
                for var_i in range(len(val_mse_per_var))
            },
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
            
            
    def load_metrics_and_plots(self, prediction, ground_truth, batch_idx, mask=None):
        
        if mask is None:
            mask = torch.ones_like(ground_truth[:, :, 0])
        
        # Plot samples
        log_plot_dict = {}

        for var_i in constants.VAL_PLOT_VARS_CERRA:
            var_name = constants.PARAM_NAMES_SHORT_CERRA[var_i]
            var_unit = constants.PARAM_UNITS_CERRA[var_i]

            pred_states = prediction[
                batch_idx, :, var_i
            ]  # (S, num_grid_nodes)
            
            target_state = ground_truth[
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
        
    def test_step(self, batch, batch_idx: int) -> dict:
        """
        Evaluate model on a test batch and store metrics.
        """
        ground_truth, high_res = (
            (batch[0], batch[1]) if isinstance(batch, (list, tuple)) and len(batch) == 2
            else (batch, None)
        )
        high_res_emb, graph_emb = self.embed_all(high_res)
        latent_dist, pred_mean, _ = self.encode_decode(high_res_emb, graph_emb)
        mse, _ = utils.compute_MSE_entiregrid(pred_mean, ground_truth)
        mae, _ = utils.compute_MAE_entiregrid(pred_mean, ground_truth)
        # If mask available, compute masked metrics
        mask = graph_emb.get("mask")
        if mask is not None:
            mse_masked, _ = utils.compute_MSE_masked(pred_mean, ground_truth, mask)
            mae_masked, _ = utils.compute_MAE_masked(pred_mean, ground_truth, mask)
        else:
            mse_masked = mae_masked = torch.tensor(float('nan'), device=pred_mean.device)

        self.test_MSEs.append(mse)
        self.test_MAEs.append(mae)
        self.test_MSEs_masked.append(mse_masked)
        self.test_MAEs_masked.append(mae_masked)

        self.log_dict({
            "test_mse": mse,
            "test_mae": mae,
            "test_mse_masked": mse_masked,
            "test_mae_masked": mae_masked,
        }, prog_bar=False, on_epoch=True, sync_dist=True)

        return {"test_mse": mse, "test_mae": mae,
                "test_mse_masked": mse_masked, "test_mae_masked": mae_masked}

    def on_test_epoch_end(self) -> None:
        """
        Aggregate and log test metrics at epoch end.
        """
        def mean(tensor_list):
            return torch.stack([t for t in tensor_list if not torch.isnan(t)]).mean()

        metrics = {
            "test_mse_mean": mean(self.test_MSEs),
            "test_mae_mean": mean(self.test_MAEs),
            "test_mse_masked_mean": mean(self.test_MSEs_masked),
            "test_mae_masked_mean": mean(self.test_MAEs_masked),
        }
        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        print(
            f"Test MSE: {metrics['test_mse_mean']:.4f}, "
            f"MAE: {metrics['test_mae_mean']:.4f}\n"
            f"Masked MSE: {metrics['test_mse_masked_mean']:.4f}, "
            f"Masked MAE: {metrics['test_mae_masked_mean']:.4f}"
        )