import math
import os
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.distributions as tdists
import pytorch_lightning as pl
import wandb
from pytorch_lightning.utilities import grad_norm

from neural_lam import constants, mask_utils, utils, vis
from neural_lam.models.ar_model import ARModel
from neural_lam.models.hi_graph_latent_encoder import HiGraphLatentEncoder, HiGraphLatentEncoderCond
from neural_lam.models.hi_graph_latent_decoder import HiGraphLatentDecoder


class GraphEFM(ARModel):
    """
    Graph-based Ensemble Forecasting Model with optional spatial masking.
    Extends ARModel to encode/decode hierarchical graph latents.
    """

    def __init__(self, args):
        super().__init__(args)
        self.save_hyperparameters(args)

        # Runtime metadata
        self.run_name = args.run_name

        # Storage for test metrics
        self.test_MSEs = []
        self.test_MAEs = []
        self.test_MSEs_masked = []
        self.test_MAEs_masked = []

        # Build feature embedders
        self._init_embedders(args)

        # Latent dimension (default to hidden_dim)
        self.latent_dim = args.latent_dim or args.hidden_dim

        #Cond. + Enc. + Dec
        #Conditioner
        self.conditioner = HiGraphLatentEncoder(
            latent_dim=self.latent_dim,
            g2m_edge_index=self.g2m_edge_index_lr,
            m2m_edge_index=self.m2m_edge_index_lr,
            mesh_up_edge_index=self.mesh_up_edge_index_lr,
            hidden_dim=args.hidden_dim,
            intra_level_layers=args.encoder_processor_layers,
            hidden_layers=args.hidden_layers,
            output_dist="diagonal",
        )

        #Encoder
        self.encoder = HiGraphLatentEncoderCond(
            latent_dim=self.latent_dim,
            g2m_edge_index=self.g2m_edge_index_hr,
            m2m_edge_index=self.m2m_edge_index_hr,
            mesh_up_edge_index=self.mesh_up_edge_index_hr,
            hidden_dim=args.hidden_dim,
            intra_level_layers=args.encoder_processor_layers,
            hidden_layers=args.hidden_layers,
            output_dist="diagonal",
            conditioner=self.conditioner,
        )

        # Decoder
        self.decoder = HiGraphLatentDecoder(
            g2m_edge_index=self.g2m_edge_index_hr,
            m2g_edge_index=self.m2g_edge_index_hr,
            m2m_edge_index=self.m2m_edge_index_hr,
            mesh_down_edge_index=self.mesh_down_edge_index_hr,
            mesh_up_edge_index=self.mesh_up_edge_index_hr,
            hidden_dim=args.hidden_dim,
            latent_dim=self.latent_dim,
            intra_level_layers=args.processor_layers,
            hidden_layers=args.hidden_layers,
            output_std=bool(args.output_std),
        )

    def _init_embedders(self, args):
        """
        Create MLP embedders for grid, graph, and mesh features at both resolutions.
        """
        blueprint = [args.hidden_dim] * (args.hidden_layers + 1)

        # Grid embedders
        self.high_res_embedder = utils.make_mlp([self.grid_dim_hr, *blueprint])
        self.low_res_embedder = utils.make_mlp([self.grid_dim_lr, *blueprint])
        self.high_res_static_embedder = utils.make_mlp([self.grid_dim_hr_static, *blueprint])

        # Graph embedders
        self.g2m_embedder_hr = utils.make_mlp([self.g2m_dim_hr, *blueprint])
        self.m2g_embedder_hr = utils.make_mlp([self.m2g_dim_hr, *blueprint])
        self.g2m_embedder_lr = utils.make_mlp([self.g2m_dim_lr, *blueprint])

        # Mesh-level embedders
        self.mesh_embedders_hr = self._make_level_embedders(
            self.mesh_features_hr, blueprint
        )
        self.mesh_embedders_lr = self._make_level_embedders(
            self.mesh_features_lr, blueprint
        )

        # Mesh up/down embedders
        self.mesh_up_embedders_hr = self._make_level_embedders(
            self.mesh_up_features_hr, blueprint
        )
        self.mesh_up_embedders_lr = self._make_level_embedders(
            self.mesh_up_features_lr, blueprint
        )
        self.mesh_down_embedders_hr = self._make_level_embedders(
            self.mesh_down_features_hr, blueprint
        )

        # m2m embedders
        self.m2m_embedders_hr = self._make_level_embedders(
            self.m2m_features_hr, blueprint
        )
        self.m2m_embedders_lr = self._make_level_embedders(
            self.m2m_features_lr, blueprint
        )

    def _make_level_embedders(
        self,
        feature_list: list[torch.Tensor],
        blueprint: list[int],
    ) -> torch.nn.ModuleList:
        """
        Generate an MLP embedder for each level of features.

        Args:
            feature_list: list of [N_i x D_i] tensors.
            blueprint: list of hidden dims.
            skip_first: if True, omit feature_list[0].
            skip_last: if True, omit last element.
        """
        modules = []
        for feat in feature_list:
            modules.append(utils.make_mlp([feat.size(1), *blueprint]))
        return torch.nn.ModuleList(modules)

    def embed_all(self, data: torch.Tensor, resolution: str = 'hr'):
        """
        Embed grid and hierarchical graph features for a given resolution.

        Args:
            data: Tensor of shape (B, N, C) for resolution 'hr' or 'lr'
            resolution: 'hr' or 'lr'

        Returns:
            grid_emb: Tensor (B, N, H)
            graph_emb: dict[str, Tensor or list[Tensor]]
        """
        
        if resolution not in ('hr', 'lr'):
            raise ValueError("resolution must be 'hr' or 'lr'")
        B = data.size(0)

        # Select resolution-specific modules
        grid_embedder = getattr(self, f'{resolution}_res_embedder',
                                self.high_res_embedder if resolution == 'hr' else self.low_res_embedder)
        static_buf = getattr(self, f'grid_static_features_{resolution}')
        g2m_embedder = getattr(self, f'g2m_embedder_{resolution}')
        g2m_feats = getattr(self, f'g2m_features_{resolution}')
        g2m_edge_idx = getattr(self, f'g2m_edge_index_{resolution}')

        # Grid embedding
        grid_features = torch.cat((data, self.expand_to_batch(static_buf, B)), dim=-1)
        grid_emb = grid_embedder(grid_features)

        # Base graph embedding dict
        graph_emb = {
            'g2m_edge_index': mask_utils.adjust_g2m_edge_index(g2m_edge_idx),
            'g2m': self.expand_to_batch(g2m_embedder(g2m_feats), B),
        }
        # HR-only m2g embedding
        if resolution == 'hr':
            graph_emb['m2g'] = self.expand_to_batch(
                self.m2g_embedder_hr(self.m2g_features_hr), B
            )
            graph_emb["grid_static_features_hr"] = self.expand_to_batch(
                self.high_res_static_embedder(static_buf), B
            )

        # Determine embed keys per resolution
        keys = ['mesh', 'm2m', 'mesh_up']
        if resolution == 'hr':
            keys.append('mesh_down')

        # Loop and embed each key
        for key in keys:
            emb_list = []
            emb_mods = getattr(self, f'{key}_embedders_{resolution}', None)
            feat_list = getattr(self, f'{key}_features_{resolution}', None)
            if emb_mods is None or feat_list is None:
                continue
            for mod, feat in zip(emb_mods, feat_list):
                emb_list.append(self.expand_to_batch(mod(feat), B))
            graph_emb[key] = emb_list

        return grid_emb, graph_emb



    def encode_sample_decode(
        self,
        high_res_emb: torch.Tensor,
        low_res_emb: torch.Tensor,
        graph_emb_hr: dict,
        graph_emb_lr: dict,
    ):
        """
        Produces a conditioned Normal distribution q(z | x_hi, x_lo).
        """
        # ----------------------------
        # 1) compute the joint posterior
        # ----------------------------
        latent_dist, skip_in, skip_up = self.encoder(
            high_emb=high_res_emb, 
            low_emb=low_res_emb,
            graph_hr=graph_emb_hr,
            graph_lr=graph_emb_lr,
        )
        # latent_dist.mean: (B, N_mesh, d_latent)
        # latent_dist.std:  (B, N_mesh, d_latent)

        # ----------------------------
        # 2) sample z ~ q(z|…)
        # ----------------------------
        z = latent_dist.rsample()  
        
        pred_mean, model_pred_std = self.decoder(
            z, skip_in, skip_up, graph_emb_hr
        )  # both (B, num_grid_nodes, d_state)

        return latent_dist, pred_mean, model_pred_std
        

    def compute_loss(self, prediction, target, latent_dist):
        """
        if mask is not None:
            mask = mask.unsqueeze(-1)
            mask = mask.repeat(1, 1, 5)
            squared_error = ((prediction - target) ** 2) * mask 
            mse_batches = squared_error.sum(dim=1) / mask.sum(dim=1)
            mse_per_var = mse_batches.mean(dim=0)
            mse = mse_batches.mean()
        else:
        """

        squared_error = (prediction - target) ** 2
        mse_batches = squared_error.mean(dim=1)
        mse_per_var = mse_batches.mean(dim=0)
        mse = mse_batches.mean()
        
        # Compute KL divergence
        standard_normal = tdists.Normal(torch.zeros_like(latent_dist.loc), torch.ones_like(latent_dist.scale))
        kl_div = tdists.kl.kl_divergence(latent_dist, standard_normal).sum(dim=-1).mean()
        
        train_loss = mse + kl_div * self.kl_beta
        
        return train_loss, mse_per_var, kl_div


    def training_step(self, batch):
        """
        Train on single batch

        batch, containing:
        init_states: (B, 2, num_grid_nodes, d_state)
        target_states: (B, pred_steps, num_grid_nodes, d_state)
        forcing_features: (B, pred_steps, num_grid_nodes, d_forcing), where
            index 0 corresponds to index 1 of init_states
        """
        high_res, low_res = batch if len(batch) == 2 else (batch, None)
        high_res_grid_emb, graph_emb_hr = self.embed_all(high_res, "hr")
        low_res_grid_emb, graph_emb_lr = self.embed_all(low_res, "lr")
        var_dist, pred_mean, _ = self.encode_sample_decode(high_res_grid_emb, low_res_grid_emb, graph_emb_hr, graph_emb_lr)
        loss, mse_per_var, kl_term = self.compute_loss(pred_mean, high_res, var_dist)
        
        
        log_dict = {
            "train_loss": loss,
            "train_kl_div": kl_term,
            **{
                f"train_mse_{constants.PARAM_NAMES_SHORT_CERRA[var_i]} {constants.PARAM_UNITS_CERRA[var_i]}": mse_per_var[var_i]
                for var_i in range(len(mse_per_var))
            },
        }
        self.log_dict(
            log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )
        
        return loss
        

    def validation_step(self, batch, *args):
        """
        Run validation on single batch
        """
        
        high_res, low_res = batch if len(batch) == 2 else (batch, None)
        high_res_grid_emb, graph_emb_hr = self.embed_all(high_res, "hr")
        low_res_grid_emb, graph_emb_lr = self.embed_all(low_res, "lr")
        var_dist, prediction, _ = self.encode_sample_decode(high_res_grid_emb, low_res_grid_emb, graph_emb_hr, graph_emb_lr)
        val_loss, val_mse_per_var, kl_term = self.compute_loss(prediction, high_res, var_dist)
        
        
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
            self.load_metrics_and_plots(prediction, high_res, batch_idx)
            
            
    def load_metrics_and_plots(self, prediction, high_res, mask, batch_idx):
        
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
        
        
    def test_step(self, batch):
        """
        Run testing on a single batch.
        """
        high_res, low_res = batch if len(batch) == 2 else (batch, None)
        
        high_res_grid_emb, graph_emb, mask, ids_restore = self.embed_all(high_res, low_res)
        var_dist, prediction, _ = self.encode_sample_decode(high_res_grid_emb, graph_emb, ids_restore)
        
        test_MSE, _ = utils.compute_MSE_entiregrid(prediction, high_res)
        test_MAE, _ = utils.compute_MAE_entiregrid(prediction, high_res)
        test_MSE_masked, _ = utils.compute_MSE_masked(prediction, high_res, mask)
        test_MAE_masked, _ = utils.compute_MAE_masked(prediction, high_res, mask)
        
        
        self.test_MSEs.append(test_MSE)
        self.test_MAEs.append(test_MAE)
        self.test_MSEs_masked.append(test_MSE_masked)
        self.test_MAEs_masked.append(test_MAE_masked)
        
        return {"test_MSE": test_MSE, "test_MAE": test_MAE, "test_MSE_masked": test_MSE_masked, "test_MAE_masked": test_MAE_masked}



    def on_test_epoch_end(self):
        """
        Called at the end of the testing epoch to aggregate results.
        """
        test_MSE_mean = sum(self.test_MSEs) / len(self.test_MSEs)
        test_MAE_mean = sum(self.test_MAEs) / len(self.test_MAEs)
        test_MSE_masked_mean = sum(self.test_MSEs_masked) / len(self.test_MSEs_masked)
        test_MAE_masked_mean = sum(self.test_MAEs_masked) / len(self.test_MAEs_masked)
        
        self.log("test_MSE_mean", test_MSE_mean)
        self.log("test_MAE_mean", test_MAE_mean)
        self.log("test_MSE_masked_mean", test_MSE_masked_mean)
        self.log("test_MAE_masked_mean", test_MAE_masked_mean)
        print(f"Mean Test MSE: {test_MSE_mean}, Mean Test MAE: {test_MAE_mean}")
        print(f"Mean Test MSE masked: {test_MSE_masked_mean}, Mean Test MAE masked: {test_MAE_masked_mean}")
        
        
    def on_after_backward(self) -> None:
        for name, p in self.named_parameters():
            # only look at things you intended to learn
            if p.requires_grad and p.grad is None:
                print(f"[UNUSED] {name}") 
        
        
        
            
            
    

        
        
        
        
        
        
        
        
        

