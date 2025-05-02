# Third-party
from torch import nn

# First-party
from neural_lam import utils
from neural_lam.models.interaction_net import InteractionNet, PropagationNet
#from neural_lam.models.base_graph_latent_decoder import BaseGraphLatentDecoder
from neural_lam import constants, utils


class HiGraphLatentDecoder(nn.Module):
    """
    Decoder that maps grid input + latent variable on mesh to prediction on grid
    Uses hierarchical graph
    """

    def __init__(
        self,
        g2m_edge_index,
        m2m_edge_index,
        m2g_edge_index,
        mesh_up_edge_index,
        mesh_down_edge_index,
        hidden_dim,
        latent_dim,
        intra_level_layers,
        hidden_layers=1,
        output_std=True,
    ):
        super().__init__()

        """# MLP for residual mapping of grid rep.
        self.grid_update_mlp = utils.make_mlp(
            [hidden_dim] * (hidden_layers + 2)
        )"""

        # Embedder for latent variable
        self.latent_embedder = utils.make_mlp(
            [latent_dim] + [hidden_dim] * (hidden_layers + 1)
        )

        # Either output input-dependent per-grid-node std or
        # use common per-variable std
        self.output_std = output_std
        if self.output_std:
            output_dim = 2 * constants.GRID_STATE_DIM_CERRA
        else:
            output_dim = constants.GRID_STATE_DIM_CERRA

        # Mapping to parameters of state distribution
        self.param_map = utils.make_mlp(
            [hidden_dim] * (hidden_layers + 1) + [output_dim], layer_norm=False
        )

        # GNN from grid to mesh
        """self.g2m_gnn = InteractionNet(
            g2m_edge_index,
            hidden_dim,
            hidden_layers=hidden_layers,
            update_edges=False,
        )"""
        # GNN from mesh to grid
        self.m2g_gnn = PropagationNet(
            m2g_edge_index,
            hidden_dim,
            hidden_layers=hidden_layers,
            update_edges=False,
        )

        # GNNs going up through mesh levels
        """self.mesh_up_gnns = nn.ModuleList(
            [
                # Note: We keep these as InteractionNets
                InteractionNet(
                    edge_index,
                    hidden_dim,
                    hidden_layers=hidden_layers,
                    update_edges=False,
                )
                for edge_index in mesh_up_edge_index
            ]
        )"""
        # GNNs going down through mesh levels
        self.mesh_down_gnns = nn.ModuleList(
            [
                PropagationNet(
                    edge_index,
                    hidden_dim,
                    hidden_layers=hidden_layers,
                    update_edges=False,
                )
                for edge_index in mesh_down_edge_index
            ]
        )
        # GNNs applied on intra-level in-between up and down propagation
        # Identity mappings if intra_level_layers = 0
        """self.intra_up_gnns = nn.ModuleList(
            [
                utils.make_gnn_seq(
                    edge_index, intra_level_layers, hidden_layers, hidden_dim
                )
                for edge_index in m2m_edge_index
            ]
        )"""
        self.intra_down_gnns = nn.ModuleList(
            [
                utils.make_gnn_seq(
                    edge_index, intra_level_layers, hidden_layers, hidden_dim
                )
                for edge_index in list(m2m_edge_index)[:-1]
                # Not needed for level L
            ]
        )

    def combine_with_latent(
        self, latent_emb, skip_in, skip_up, graph_emb_hr
    ):
        """
        Combine the grid representation with representation of latent variable.
        The output should be on the grid again.

        original_grid_rep: (B, num_grid_nodes, d_h)
        latent_rep: (B, num_mesh_nodes, d_h)
        residual_grid_rep: (B, num_grid_nodes, d_h)

        Returns:
        grid_rep: (B, num_grid_nodes, d_h)
        """
        # Map to bottom mesh level

        # Run intra-level processing for highest mesh level
        rep = latent_emb

        # Down hierarchy
        # Propagate down before running intra-level processing
        for (
            down_gnn,
            intra_gnn_seq,
            mesh_down_level_rep,
            m2m_level_rep,
            mesh_level_rep,
            skip_in_,
            skip_down_,
        ) in zip(
            reversed(self.mesh_down_gnns[1:]),
            reversed(self.intra_down_gnns[1:]),
            reversed(graph_emb_hr["mesh_down"][1:]),
            reversed(graph_emb_hr["m2m"][1:-1]),  # Residual connections to up pass
            reversed(graph_emb_hr["mesh"][1:-1]),  # ^
            reversed(skip_in[1:-1]),
            reversed(skip_up[:-1]),
        ):  # Loop goes L-1 times, from intra level processing at l=L-1 to l=1
            # Apply down GNN, don't need to store these reps.
            new_mesh_rep = down_gnn(
                rep, mesh_level_rep, mesh_down_level_rep
            )  # (B, num_mesh_nodes[l], d_h)

            rep = new_mesh_rep + skip_down_  
            # Run same level processing on level l
            current_mesh_rep, _ = intra_gnn_seq(
                rep, m2m_level_rep
            )  # (B, num_mesh_nodes[l], d_h)
            
            rep = current_mesh_rep + skip_in_

        
        
        # Map to bottom mesh level. 
        new_mesh_rep = self.mesh_down_gnns[0](
            rep, graph_emb_hr["mesh"][0], graph_emb_hr["mesh_down"][0]
        )  # (B, num_mesh_nodes[l], d_h)

        # Run same level processing on level l
        current_mesh_rep, _ = self.intra_down_gnns[0](
            new_mesh_rep, graph_emb_hr["m2m"][0]
        )  # (B, num_mesh_nodes[l], d_h)
        
        rep = current_mesh_rep + skip_in[0]
        
        # Map back to grid
        grid_rep = self.m2g_gnn(
            rep, graph_emb_hr["grid_static_features_hr"], graph_emb_hr["m2g"]
        )  # (B, num_mesh_nodes[0], d_h)

        return grid_rep
    
    
    def forward(self, latent_samples, skip_in, skip_up, graph_emb_hr):
        """
        Compute prediction (mean and std.-dev.) of next weather state

        grid_rep: (B, num_grid_nodes, d_h)
        latent_samples: (B, N_mesh, d_latent)
        last_state: (B, num_grid_nodes, d_state)
        graph_emb: dict with graph embedding vectors, entries at least
            g2m: (B, M_g2m, d_h)
            m2m: (B, M_g2m, d_h)
            m2g: (B, M_m2g, d_h)

        Returns:
        mean: (B, N_mesh, d_latent), predicted mean
        std: (B, N_mesh, d_latent), predicted std.-dev.
        """
        # To mesh
        latent_emb = self.latent_embedder(latent_samples)  # (B, N_mesh, d_h)
        

        combined_grid_rep = self.combine_with_latent(
            latent_emb, skip_in, skip_up, graph_emb_hr
        )

        state_params = self.param_map(
            combined_grid_rep
        )  # (B, N_mesh, d_state_params)

        if self.output_std:
            mean_delta, std_raw = state_params.chunk(
                2, dim=-1
            )  # (B, num_grid_nodes, d_state),(B, num_grid_nodes, d_state)
            # pylint: disable-next=not-callable
            pred_std = nn.functional.softplus(std_raw)  # positive std.
        else:
            mean_delta = state_params  # (B, num_grid_nodes, d_state)
            pred_std = None

        #pred_mean = last_state + mean_delta
        pred_mean = mean_delta

        return pred_mean, pred_std
