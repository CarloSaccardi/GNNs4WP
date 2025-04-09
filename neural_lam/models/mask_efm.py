# Third-party
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import math

# First-party
from neural_lam import constants, mask_utils, utils, vis
from neural_lam.models.ar_model import ARModel
from neural_lam.models.hi_graph_latent_decoder import HiGraphLatentDecoder
from neural_lam.models.hi_graph_latent_encoder import HiGraphLatentEncoder
import torch.distributions as tdists

from pytorch_lightning.utilities import grad_norm


class GraphEFM_mask(ARModel):
    """
    Graph-based Ensemble Forecasting Model
    """

    def __init__(self, args):
        super().__init__(args)
        
        self.test_MSEs = []
        self.test_MAEs = []
        self.test_MSEs_masked = []
        self.test_MAEs_masked = []
        
        self.run_name = args.run_name

        # Define sub-models
        # Feature embedders for grid
        self.mlp_blueprint_end = [args.hidden_dim] * (args.hidden_layers + 1)
        self.high_res_embedder = utils.make_mlp([self.grid_dim] + self.mlp_blueprint_end)  # For states up to t-1
        # Embedders for mesh
        self.g2m_embedder = utils.make_mlp([self.g2m_dim] + self.mlp_blueprint_end)
        self.m2g_embedder = utils.make_mlp([self.m2g_dim] + self.mlp_blueprint_end)

        ################### Print some useful info ###################
        print("Loaded hierarchical graph with structure:")
        level_mesh_sizes = [mesh_feat.shape[0] for mesh_feat in self.mesh_static_features]
        num_levels = len(self.mesh_static_features)
        for level_index, level_mesh_size in enumerate(level_mesh_sizes):
            same_level_edges = self.m2m_features[level_index].shape[0]
            print(
                f"level {level_index} - {level_mesh_size} nodes, "
                f"{same_level_edges} same-level edges"
            )

            if level_index < (num_levels - 1):
                up_edges = self.mesh_up_features[level_index].shape[0]
                down_edges = self.mesh_down_features[level_index].shape[0]
                print(f"  {level_index}<->{level_index+1}")
                print(f" - {up_edges} up edges, {down_edges} down edges")
        ##############################################################
                
        # Embedders
        # Assume all levels have same static feature dimensionality
        mesh_dim = self.mesh_static_features[1].shape[1] #first mesh features has shape (6561, 4) since it represent the era5 grid. The other meshes have shape (x, 2). There is a dedicated ambedder for the first mesh. This code should be modified to take into account the different shapes of the mesh features.
        m2m_dim = self.m2m_features[0].shape[1]
        mesh_up_dim = self.mesh_up_features[0].shape[1]
        mesh_down_dim = self.mesh_down_features[0].shape[1]
        
        if args.dataset_era5 is None or args.dataset_cerra is None:
            #first embedder has to project 2 features, as the first mesh is just a projection of the original grid
            self.mesh_embedders = torch.nn.ModuleList(
                [
                    utils.make_mlp([mesh_dim] + self.mlp_blueprint_end)
                    for _ in range(num_levels)
                ]
            )
        else:
            #first embedder has to project 8 features, as the first mesh a lower resolution version of the original grid
            self.low_res_embedder = utils.make_mlp([self.grid_dim] + self.mlp_blueprint_end)
            self.mesh_embedders = torch.nn.ModuleList( [self.low_res_embedder] +
                [
                    utils.make_mlp([mesh_dim] + self.mlp_blueprint_end)
                    for _ in range(num_levels-1)
                ]
            )
        self.mesh_up_embedders = torch.nn.ModuleList(
            [
                utils.make_mlp([mesh_up_dim] + self.mlp_blueprint_end)
                for _ in range(num_levels - 1)
            ]
        )
        self.mesh_down_embedders = torch.nn.ModuleList(
            [
                utils.make_mlp([mesh_down_dim] + self.mlp_blueprint_end)
                for _ in range(num_levels - 1)
            ]
        )
        self.m2m_embedders = torch.nn.ModuleList(
            [
                utils.make_mlp([m2m_dim] + self.mlp_blueprint_end)
                for _ in range(num_levels)
            ]
        )
        latent_dim = (
            args.latent_dim if args.latent_dim is not None else args.hidden_dim
        )
        # Prior
        """
        if args.learn_prior:
            if self.hierarchical_graph:
                self.prior_model = HiGraphLatentEncoder(
                    latent_dim,
                    self.g2m_edge_index,
                    self.m2m_edge_index,
                    self.mesh_up_edge_index,
                    args.hidden_dim,
                    args.prior_processor_layers,
                    hidden_layers=args.hidden_layers,
                    output_dist=args.prior_dist,
                )
            else:
                self.prior_model = GraphLatentEncoder(
                    latent_dim,
                    self.g2m_edge_index,
                    self.m2m_edge_index,
                    args.hidden_dim,
                    args.prior_processor_layers,
                    hidden_layers=args.hidden_layers,
                    output_dist=args.prior_dist,
                )
        else:
            self.prior_model = ConstantLatentEncoder(
                latent_dim,
                self.num_mesh_nodes,
                output_dist=args.prior_dist,
            )
        """
        # Enc. + Dec.
        # Encoder
        self.encoder = HiGraphLatentEncoder(
            latent_dim,
            self.g2m_edge_index,
            self.m2m_edge_index,
            self.mesh_up_edge_index,
            args.hidden_dim,
            args.encoder_processor_layers,
            hidden_layers=args.hidden_layers,
            output_dist="diagonal",
        )
        # Decoder
        self.decoder = HiGraphLatentDecoder(
            self.g2m_edge_index,
            self.m2m_edge_index,
            self.m2g_edge_index,
            self.mesh_up_edge_index,
            self.mesh_down_edge_index,
            args.hidden_dim,
            latent_dim,
            args.processor_layers,
            hidden_layers=args.hidden_layers,
            output_std=bool(args.output_std),
        )


    def embedd_all(self, high_res, low_res):
        """
        embed all node and edge representations

        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)

        Returns:
        grid_emb: (B, num_grid_nodes, d_h)
        graph_embedding: dict with entries of shape (B, *, d_h)
        """
        batch_size = high_res.shape[0]
        # Embed high-res grid nodes
        high_res_grid_features = torch.cat(
            (
                high_res,
                self.expand_to_batch(self.grid_static_features, batch_size),
            ),
            dim=-1,
        )  # (B, num_grid_nodes, grid_dim)
        high_res_grid_emb = self.high_res_embedder(high_res_grid_features)
        high_res_grid_emb = high_res_grid_emb + self.pos_embed
        # Masking
        if self.mask_ratio is not None:
            high_res_grid_emb, mask, ids_restore, g2m_features, g2m_edge_index = mask_utils.masking(high_res_grid_emb, 
                                                                                 #self.pos_embed, 
                                                                                 self.g2m_edge_index, 
                                                                                 self.g2m_features,
                                                                                 self.mask_ratio,
                                                                                 int(self.num_grid_nodes**.5),
                                                                                 self.masking_block_size)
            graph_emb = {"g2m_edge_index": g2m_edge_index}  # Ensure it is part of the masking block
        else:
            mask = ids_restore = None
            g2m_features = self.g2m_features
            graph_emb = {"g2m_edge_index": mask_utils.adjust_g2m_edge_index(self.g2m_edge_index)}
        # Graph embedding
        graph_emb.update({
            "g2m": self.expand_to_batch(self.g2m_embedder(g2m_features), batch_size),
            "m2g": self.expand_to_batch(self.m2g_embedder(self.m2g_features), batch_size),
        })
        # Embed mesh nodes
        graph_emb["mesh"] = [
            emb(torch.cat((low_res, self.expand_to_batch(node_static_features, batch_size)), dim=-1))
            if (indx ==0 and low_res is not None) else self.expand_to_batch(emb(node_static_features), batch_size)
            for indx, (emb, node_static_features) in enumerate(zip(self.mesh_embedders, self.mesh_static_features))
        ]
        # Embed mesh edges, in-between levels
        graph_emb["m2m"] = [
            self.expand_to_batch(emb(edge_feat), batch_size)
            for emb, edge_feat in zip(
                self.m2m_embedders, self.m2m_features
            )
        ]
        #Embed mesh edges, up
        graph_emb["mesh_up"] = [
            self.expand_to_batch(emb(edge_feat), batch_size)
            for emb, edge_feat in zip(
                self.mesh_up_embedders, self.mesh_up_features
            )
        ]
        #Embed mesh edges, down
        graph_emb["mesh_down"] = [
            self.expand_to_batch(emb(edge_feat), batch_size)
            for emb, edge_feat in zip(
                self.mesh_down_embedders, self.mesh_down_features
            )
        ]

        return high_res_grid_emb, graph_emb, mask, ids_restore

    
    def encode_sample_decode(
        self, high_res_emb, graph_emb, ids_restore
    ):
        """
        Estimate (masked) likelihood using given distribution over
        latent variables

        latent_dist: distribution, (B, num_mesh_nodes, d_latent)
        current_state: (B, num_grid_nodes, d_state)
        last_state: (B, num_grid_nodes, d_state)
        high_res_emb: (B, num_grid_nodes, d_state)
        g2m_emb: (B, M_g2m, d_h)
        m2m_emb: (B, M_m2m, d_h)
        m2g_emb: (B, M_m2g, d_h)

        Returns:
        likelihood_term: (B,)
        pred_mean: (B, num_grid_nodes, d_state)
        pred_std: (B, num_grid_nodes, d_state) or (d_state,)
        """
        
        # Compute variational approximation (encoder)
        latent_dist = self.encoder(
            high_res_emb, graph_emb=graph_emb
        )  # Gaussian, (B, num_mesh_nodes, d_latent)
        
        # Sample from variational distribution
        latent_samples = latent_dist.rsample()  # (B, num_mesh_nodes, d_latent)

        if ids_restore is not None:
            # Compute reconstruction (decoder)        
            mask_tokens = self.mask_token.repeat(high_res_emb.shape[0], ids_restore.shape[1] - high_res_emb.shape[1], 1)
            x_ = torch.cat([high_res_emb, mask_tokens], dim=1)
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, high_res_emb.shape[2]))  # unshuffle
            full_grid_rep = x_ + self.decoder_pos_embed
        else:
            full_grid_rep = None
        
        pred_mean, model_pred_std = self.decoder(
            high_res_emb, latent_samples, graph_emb, full_grid_rep
        )  # both (B, num_grid_nodes, d_state)

        return latent_dist, pred_mean, model_pred_std
        

    def compute_loss(self, prediction, target, latent_dist, mask):
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
        if self.masked_loss:
            n_features = prediction.shape[-1]
            mask = mask.unsqueeze(-1)
            mask = mask.repeat(1, 1, n_features)
            squared_error = ((prediction - target) ** 2) * mask 
            mse_batches = squared_error.sum(dim=1) / mask.sum(dim=1)
            mse_per_var = mse_batches.mean(dim=0)
            mse = mse_batches.mean()
            
        else:
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
        
        high_res_grid_emb, graph_emb, mask, ids_restore = self.embedd_all(high_res,low_res)
        var_dist, pred_mean, _ = self.encode_sample_decode(high_res_grid_emb, graph_emb, ids_restore)
        loss, mse_per_var, kl_term = self.compute_loss(pred_mean, high_res, var_dist, mask)
        
        
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
        
        high_res_grid_emb, graph_emb, mask, ids_restore = self.embedd_all(high_res,low_res)
        var_dist, prediction, _ = self.encode_sample_decode(high_res_grid_emb, graph_emb, ids_restore)
        val_loss, val_mse_per_var, kl_term = self.compute_loss(prediction, high_res, var_dist, mask)
        
        
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
            self.load_metrics_and_plots(prediction, high_res, mask, batch_idx)
            
            
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
        
        high_res_grid_emb, graph_emb, mask, ids_restore = self.embedd_all(high_res, low_res)
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
        
        
        
        
            
            
    

        
        
        
        
        
        
        
        
        

