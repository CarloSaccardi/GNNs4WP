# Standard library
import os

# Third-party
import matplotlib.pyplot as plt
import numpy as np
from neural_lam.models.hi_graph_latent_encoder import HiGraphLatentEncoder
import torch
from torch import nn

# First-party
from neural_lam import constants, utils


class CondEncoder(nn.Module):
    """
    Generic auto-regressive weather model.
    Abstract class that can be extended.
    """

    # pylint: disable=arguments-differ
    # Disable to override args/kwargs from superclass

    def __init__(self, args):
        super().__init__()
        
        ########################################
        # Load conditioning graph specifics
        ########################################
        self.mlp_blueprint_end = [args.hidden_dim] * (args.hidden_layers + 1)
        self.hierarchical_graph_cond, graph_ldict_cond = utils.load_graph(args.graph_conditioner)
        for name, attr_value in graph_ldict_cond.items():
            if "down" not in name and "m2g" not in name:
                # Make BufferLists module members and register tensors as buffers
                name = "cond_" + name
                if isinstance(attr_value, torch.Tensor):
                    self.register_buffer(name, attr_value, persistent=False)
                else:
                    setattr(self, name, attr_value)
        # Load static features for grid/data
        static_data_dict_cond = utils.load_static_data(args.dataset_era5)
        for static_data_name, static_data_tensor in static_data_dict_cond.items():
            self.register_buffer(
                "cond_" + static_data_name, static_data_tensor, persistent=False
            )
        
        ########################################
        #Determine weather to predict latent sd or not
        ########################################
        # Double grid output dim. to also output std.-dev.
        """
        self.output_std = bool(args.output_std)
        if self.output_std:
            self.cond_grid_output_dim = 2 * constants.GRID_STATE_DIM_CERRA  # Pred. dim. in grid cell
        else:
            self.cond_grid_output_dim = constants.GRID_STATE_DIM_CERRA  # Pred. dim. in grid cell
        """
        # grid_dim from data + static
        self.cond_num_grid_nodes, grid_static_dim = self.cond_grid_static_features.shape 
        self.cond_grid_dim = constants.GRID_STATE_DIM_CERRA + grid_static_dim
        
        ########################################
        # Define embedders
        ########################################
        num_levels = len(self.cond_mesh_static_features)
        cond_mesh_dim = self.cond_mesh_static_features[0].shape[1]
        cond_m2m_dim = self.cond_m2m_features[0].shape[1]
        cond_mesh_up_dim = self.cond_mesh_up_features[0].shape[1]
        
        self.cond_g2m_dim = self.cond_g2m_features.shape[1]
        self.cond_g2m_embedder = utils.make_mlp([self.cond_g2m_dim] + self.mlp_blueprint_end)
        
        self.cond_grid_embedder = utils.make_mlp([self.cond_grid_dim] + self.mlp_blueprint_end)
        
        self.cond_mesh_embedders = torch.nn.ModuleList([
            utils.make_mlp([cond_mesh_dim] + self.mlp_blueprint_end)
            for _ in range(num_levels)
        ])
        self.cond_mesh_up_embedders = torch.nn.ModuleList([
            utils.make_mlp([cond_mesh_up_dim] + self.mlp_blueprint_end)
            for _ in range(num_levels - 1)
        ])
        self.cond_m2m_embedders = torch.nn.ModuleList([
            utils.make_mlp([cond_m2m_dim] + self.mlp_blueprint_end)
            for _ in range(num_levels)
        ])
        latent_dim = (
            args.latent_dim if args.latent_dim is not None else args.hidden_dim
        )
        ########################################
        # Define cond encoder
        ########################################
        self.conditioning_encoder = HiGraphLatentEncoder(
            latent_dim,
            self.cond_g2m_edge_index,
            self.cond_m2m_edge_index,
            self.cond_mesh_up_edge_index,
            args.hidden_dim,
            args.prior_processor_layers,
            hidden_layers=args.hidden_layers,
            output_dist=args.prior_dist,
        )
        
        
    def embed_all_cond(self, conditioning_graph):
        """
        embed all node and edge representations

        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)

        Returns:
        grid_emb: (B, num_grid_nodes, d_h)
        graph_embedding: dict with entries of shape (B, *, d_h)
        """
        batch_size = conditioning_graph.shape[0]
        # Embed high-res grid nodes
        cond_grid_features = torch.cat(
            (
                conditioning_graph,
                self.expand_to_batch(self.cond_grid_static_features, batch_size),
            ),
            dim=-1,
        )  # (B, num_grid_nodes, grid_dim)
        cond_grid_emb = self.cond_grid_embedder(cond_grid_features)

        cond_graph_emb = {
            "g2m_edge_index": self.adjust_g2m_edge_index(),
             "g2m": self.expand_to_batch(self.cond_g2m_embedder(self.cond_g2m_features), batch_size),
            }  

        # Embed mesh nodes
        cond_graph_emb["mesh"] = [
            self.expand_to_batch(emb(node_static_features), batch_size)
            for (emb, node_static_features) in zip(
                self.cond_mesh_embedders, self.cond_mesh_static_features
            )
        ]
        # Embed mesh edges, in-between levels
        cond_graph_emb["m2m"] = [
            self.expand_to_batch(emb(edge_feat), batch_size)
            for emb, edge_feat in zip(
                self.cond_m2m_embedders, self.cond_m2m_features
            )
        ]
        #Embed mesh edges, up
        cond_graph_emb["mesh_up"] = [
            self.expand_to_batch(emb(edge_feat), batch_size)
            for emb, edge_feat in zip(
                self.cond_mesh_up_embedders, self.cond_mesh_up_features
            )
        ]
        return cond_grid_emb, cond_graph_emb
    
    def adjust_g2m_edge_index(self):
        # Normalize and offset sender indices by the number of receivers.
        cond_g2m_edge_index = self.cond_g2m_edge_index - self.cond_g2m_edge_index.min(dim=1, keepdim=True)[0]
        self.num_rec = int(cond_g2m_edge_index[1].max().item() + 1)
        cond_g2m_edge_index[0] += self.num_rec
        # Reindex senders to a continuous range starting at self.num_rec.
        _, cond_g2m_edge_index[0] = torch.unique(cond_g2m_edge_index[0], sorted=True, return_inverse=True)
        cond_g2m_edge_index[0] += self.num_rec
        return cond_g2m_edge_index
    
    @staticmethod
    def expand_to_batch(x, batch_size):
        """
        Expand tensor with initial batch dimension
        """
        return x.unsqueeze(0).expand(batch_size, -1, -1)
    
    def forward(self, conditioning_graph):
        """
        Forward pass of the model.

        Args:
        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)

        Returns:
        pred: (B, num_grid_nodes, pred_dim)
        """
        high_res_grid_emb, cond_graph_emb = self.embed_all_cond(conditioning_graph)
        latent = self.conditioning_encoder(high_res_grid_emb, graph_emb=cond_graph_emb)
        return latent
    