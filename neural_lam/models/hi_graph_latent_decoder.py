# Third-party imports
from torch import nn
import torch
from torch import distributions as tdists

# First-party imports
from neural_lam import constants, utils
from neural_lam.models.interaction_net import InteractionNet, PropagationNet


class HiGraphLatentUnet(nn.Module):
    """
    Decoder that maps grid input + latent variable on mesh to prediction on grid
    using a hierarchical graph.
    """

    def __init__(
        self,
        g2m_edge_index,
        m2m_edge_index,
        m2g_edge_index,
        mesh_up_edge_index,
        mesh_down_edge_index,
        variational: bool,
        hidden_dim: int,
        latent_dim: int,
        intra_level_layers: int,
        hidden_layers: int = 1,
        output_std: bool = True,
        output_dist: str = "isotropic",
    ):
        super().__init__()
        # Save config
        self.variational = variational
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.intra_level_layers = intra_level_layers

        # Determine output distribution settings
        self.output_dist = output_dist
        if output_dist == "isotropic":
            self.output_dim = latent_dim
        elif output_dist == "diagonal":
            self.output_dim = 2 * latent_dim
            self.latent_std_eps = 1e-4
        else:
            raise ValueError(f"Unknown output distribution: {output_dist}")

        # Residual MLP for grid updates
        self.grid_update_mlp = utils.make_mlp(
            [hidden_dim] * (hidden_layers + 2)
        )

        # Output parameter mapping
        self.output_std = output_std
        param_output_dim = (
            2 * constants.GRID_STATE_DIM_CERRA
            if output_std
            else constants.GRID_STATE_DIM_CERRA
        )
        self.param_map = utils.make_mlp(
            [hidden_dim] * (hidden_layers + 1) + [param_output_dim],
            layer_norm=False,
        )

        # Build GNN components
        self.g2m_gnn = PropagationNet(
            g2m_edge_index, hidden_dim, hidden_layers=hidden_layers, update_edges=False
        )
        self.m2g_gnn = PropagationNet(
            m2g_edge_index, hidden_dim, hidden_layers=hidden_layers, update_edges=False
        )

        # Hierarchical propagation nets
        self.mesh_up_gnns = self._make_propagation_list(mesh_up_edge_index)
        self.mesh_down_gnns = self._make_propagation_list(mesh_down_edge_index)

        # Intra-level GNN sequences
        self.intra_up_gnns = self._make_intra_level_seq(
            m2m_edge_index, intra_level_layers
        )
        # Skip last level for downward
        self.intra_down_gnns = self._make_intra_level_seq(
            list(m2m_edge_index)[:-1], intra_level_layers
        )

        # Latent parameter map for variational case
        if variational:
            self.latent_param_map = utils.make_mlp(
                [hidden_dim] * (hidden_layers + 1) + [self.output_dim],
                layer_norm=False,
            )

    def _make_propagation_list(self, edge_indices):  # helper to build PropagationNet lists
        return nn.ModuleList(
            PropagationNet(
                idx, self.hidden_dim, hidden_layers=self.hidden_layers, update_edges=False
            )
            for idx in edge_indices
        )

    def _make_intra_level_seq(self, edge_indices, layers):  # helper for intra-level GNNs
        return nn.ModuleList(
            utils.make_gnn_seq(idx, layers, self.hidden_layers, self.hidden_dim)
            for idx in edge_indices
        )

    def encode(self, grid_rep, graph_emb):
        """
        Encodes grid representation and graph embeddings into latent and intermediate mesh reps.

        Returns:
            latent_dist or final mesh rep,
            list of m2m level reps,
            list of mesh level reps
        """
        # Initial propagation to bottom mesh
        current = self.g2m_gnn(
            grid_rep,
            graph_emb["mesh"][0],
            graph_emb["g2m"],
            graph_emb.get("g2m_edge_index"),
        )

        mesh_level_reps, m2m_level_reps = [], []
        # Upward hierarchy
        for up_gnn, intra_seq, up_emb, m2m_emb, mesh_emb in zip(
            self.mesh_up_gnns,
            self.intra_up_gnns[:-1],
            graph_emb["mesh_up"],
            graph_emb["m2m"][:-1],
            graph_emb["mesh"][1:],
        ):
            new_mesh, new_m2m = intra_seq(current, m2m_emb)
            mesh_level_reps.append(new_mesh)
            m2m_level_reps.append(new_m2m)
            current = up_gnn(new_mesh, mesh_emb, up_emb)

        # Top-level intra processing
        current, _ = self.intra_up_gnns[-1](current, graph_emb["m2m"][-1])

        if self.variational:
            params = self.latent_param_map(current)
            if self.output_dist == "diagonal":
                mean, std_raw = params.chunk(2, dim=-1)
                std = self.latent_std_eps + nn.functional.softplus(std_raw)
            else:
                mean, std = params, torch.ones_like(params)
            return tdists.Normal(mean, std), m2m_level_reps, mesh_level_reps

        return current, m2m_level_reps, mesh_level_reps

    def decode(
        self,
        mesh_rep,
        residual_grid_rep,
        graph_emb,
        m2m_level_reps,
        mesh_level_reps,
    ):
        """
        Decodes mesh representation and residual grid rep back to grid rep.
        """
        current = mesh_rep
        # Downward hierarchy
        for down_gnn, intra_seq, down_emb, m2m_emb, mesh_emb in zip(
            reversed(self.mesh_down_gnns),
            reversed(self.intra_down_gnns),
            reversed(graph_emb["mesh_down"]),
            reversed(m2m_level_reps),
            reversed(mesh_level_reps),
        ):
            new_mesh = down_gnn(current, mesh_emb, down_emb)
            current, _ = intra_seq(new_mesh, m2m_emb)

        # Back to grid
        return self.m2g_gnn(current, residual_grid_rep, graph_emb.get("m2g"))

    def forward(self, grid_rep, graph_emb):
        """
        Forward pass: encodes, samples latent, decodes, and maps to state params.
        """
        # Residual update
        residual = grid_rep + self.grid_update_mlp(grid_rep)
        latent_dist, m2m_reps, mesh_reps = self.encode(grid_rep, graph_emb)
        samples = latent_dist.rsample() if self.variational else latent_dist
        decoded = self.decode(samples, residual, graph_emb, m2m_reps, mesh_reps)

        state_params = self.param_map(decoded)
        if self.output_std:
            mean, raw = state_params.chunk(2, dim=-1)
            std = nn.functional.softplus(raw)
        else:
            mean, std = state_params, None

        return latent_dist, mean, std
