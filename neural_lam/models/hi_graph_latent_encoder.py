# Third-party
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

# First-party
from neural_lam import utils
from neural_lam.models.interaction_net import PropagationNet
from torch import distributions as tdists
import torch


class HiGraphLatentEncoder(nn.Module):
    """
    Encoder that maps from grid to mesh and defines a latent distribution
    on mesh.
    Uses a hierarchical mesh graph.
    """

    def __init__(
        self,
        latent_dim,
        g2m_edge_index,
        m2m_edge_index,
        mesh_up_edge_index,
        hidden_dim,
        intra_level_layers,
        hidden_layers=1,
        output_dist="isotropic",
    ):
        super().__init__()

        # Mapping to parameters of latent distribution
        self.output_dist = output_dist
        if output_dist == "isotropic":
            # Isotopic Gaussian, output only mean (\Sigma = I)
            self.output_dim = latent_dim
        elif output_dist == "diagonal":
            # Isotopic Gaussian, output mean and std
            self.output_dim = 2 * latent_dim

            # Small epsilon to prevent enccoding to dist. with std.-dev. 0
            self.latent_std_eps = 1e-4
        else:
            assert False, f"Unknown encoder output distribution: {output_dist}"

        # GNN from grid to mesh
        self.g2m_gnn = PropagationNet(
            g2m_edge_index,
            hidden_dim,
            hidden_layers=hidden_layers,
            update_edges=False,
        )

        # GNNs going up through mesh levels
        self.mesh_up_gnns = nn.ModuleList(
            [
                PropagationNet(
                    edge_index,
                    hidden_dim,
                    hidden_layers=hidden_layers,
                    update_edges=False,
                )
                for edge_index in mesh_up_edge_index
            ]
        )

        # GNNs applied on intra-level in-between upwards propagation
        # Identity mappings if intra_level_layers = 0
        self.intra_level_gnns = nn.ModuleList(
            [
                utils.make_gnn_seq(
                    edge_index, intra_level_layers, hidden_layers, hidden_dim
                )
                for edge_index in m2m_edge_index
            ]
        )


class HiGraphLatentEncoderCond(HiGraphLatentEncoder):
    """
    Hierarchical latent encoder with multi-scale conditioning.
    At each graph level, sums the high-res and low-res streams before proceeding.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        g2m_edge_index,
        m2m_edge_index,
        mesh_up_edge_index,
        hidden_dim: int,
        intra_level_layers: int,
        hidden_layers: int = 1,
        output_dist: str = "diagonal",
        conditioner: HiGraphLatentEncoder,    # <— pass in your LR encoder here
    ):
        # Initialize the “main” (HR) encoder
        super().__init__(
            latent_dim=latent_dim,
            g2m_edge_index=g2m_edge_index,
            m2m_edge_index=m2m_edge_index,
            mesh_up_edge_index=mesh_up_edge_index,
            hidden_dim=hidden_dim,
            intra_level_layers=intra_level_layers,
            hidden_layers=hidden_layers,
            output_dist=output_dist,
        )
        # Store the LR encoder for conditioning
        self.conditioner: HiGraphLatentEncoder = conditioner
        
        # Final map to parameters
        self.latent_param_map = utils.make_mlp(
            [hidden_dim] * (hidden_layers + 1) + [self.output_dim],
            layer_norm=False,
        )
        
        
    def compute_dist_params_cond(
        self,
        high_emb: torch.Tensor,
        low_emb: torch.Tensor,
        graph_hr: Dict[str, List],
        graph_lr: Dict[str, List],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Run both HR and LR streams in lockstep, summing at each scale.

        Args:
            high_emb: (B, N_grid_hr, d_h) high-res grid embeddings
            low_emb:  (B, N_grid_lr, d_h) low-res grid embeddings
            graph_hr: dict of HR graph features (mesh, g2m, m2m, mesh_up, mesh_down, g2m_edge_index)
            graph_lr: dict of LR graph features (mesh, g2m, m2m, mesh_up, g2m_edge_index)

        Returns:
            params: (B, N_mesh_L, D_out) combined latent parameters
            skip_ins: list of LR skip-in tensors at each level
            skip_ups: list of LR skip-up tensors at each level
        """
        # Unpack HR graph features
        hr_meshes = graph_hr['mesh']
        hr_m2m_feats = graph_hr['m2m']
        hr_mesh_up_feats = graph_hr['mesh_up']
        hr_g2m = graph_hr['g2m']
        hr_g2m_edge = graph_hr['g2m_edge_index']

        # Unpack LR graph features
        lr_meshes = graph_lr['mesh']
        lr_m2m_feats = graph_lr['m2m']
        lr_mesh_up_feats = graph_lr['mesh_up']
        lr_g2m = graph_lr['g2m']
        lr_g2m_edge = graph_lr['g2m_edge_index']

        # Level 0: grid -> mesh
        hr_rep = self.g2m_gnn(high_emb, hr_meshes[0], hr_g2m, hr_g2m_edge)
        hr_in, _ = self.intra_level_gnns[0](hr_rep, hr_m2m_feats[0])
        rep = hr_in + low_emb

        skip_ins: List[torch.Tensor] = [low_emb]
        skip_ups: List[torch.Tensor] = []

        # Hierarchical up-propagation through levels 1..L
        for level, (up_gnn, intra_gnn) in enumerate(
            zip(self.mesh_up_gnns, self.intra_level_gnns[1:])
        ):
            # HR features for this level
            hr_mesh = hr_meshes[level + 1]
            hr_m2m = hr_m2m_feats[level + 1]
            hr_up_feat = hr_mesh_up_feats[level]

            # LR features for this level
            lr_mesh = lr_meshes[level]
            lr_m2m = lr_m2m_feats[level]
            lr_up_feat = lr_mesh_up_feats[level - 1]

            # Up-propagation
            hr_up = up_gnn(rep, hr_mesh, hr_up_feat)
            if level == 0:
                lr_up = self.conditioner.g2m_gnn(
                    low_emb, lr_meshes[0], lr_g2m, lr_g2m_edge
                )
            else:
                lr_up = self.conditioner.mesh_up_gnns[level - 1](
                    lr_in, lr_mesh, lr_up_feat
                )
            skip_ups.append(lr_up)
            rep = hr_up + lr_up

            # Intra-level refinement
            hr_in, _ = intra_gnn(rep, hr_m2m)
            lr_idx = 0 if level == 0 else level
            lr_in, _ = self.conditioner.intra_level_gnns[lr_idx](
                lr_up, lr_m2m
            )
            skip_ins.append(lr_in)
            rep = hr_in + lr_in

        # Final mapping to distribution parameters
        params = self.latent_param_map(rep)
        return params, skip_ins, skip_ups


    def forward(
        self,
        high_emb: torch.Tensor,
        low_emb: torch.Tensor,
        graph_hr: dict,
        graph_lr: dict
    ):
        # Compute combined params and split into (μ, σ)
        latent_params, skip_in, skip_up = self.compute_dist_params_cond(high_emb, low_emb, graph_hr, graph_lr)

        if self.output_dist == "diagonal":
            mu, std_raw = latent_params.chunk(2, dim=-1)
            std = self.latent_std_eps + F.softplus(std_raw)
        else:
            mu = latent_params
            std = torch.ones_like(mu)

        return tdists.Normal(mu, std), skip_in, skip_up