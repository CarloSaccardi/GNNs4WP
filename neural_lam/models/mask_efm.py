# Third-party
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

# First-party
from neural_lam import constants, metrics, utils, vis
from neural_lam.models.ar_model import ARModel
from neural_lam.models.constant_latent_encoder import ConstantLatentEncoder
from neural_lam.models.graph_latent_decoder import GraphLatentDecoder
from neural_lam.models.graph_latent_encoder import GraphLatentEncoder
from neural_lam.models.hi_graph_latent_decoder import HiGraphLatentDecoder
from neural_lam.models.hi_graph_latent_encoder import HiGraphLatentEncoder


class GraphEFM_mask(ARModel):
    """
    Graph-based Ensemble Forecasting Model
    """

    def __init__(self, args):
        super().__init__(args)


        # Load graph with static features
        self.hierarchical_graph, graph_ldict = utils.load_graph(args.graph)
        for name, attr_value in graph_ldict.items():
            # Make BufferLists module members and register tensors as buffers
            if isinstance(attr_value, torch.Tensor):
                self.register_buffer(name, attr_value, persistent=False)
            else:
                setattr(self, name, attr_value)

        # Specify dimensions of data
        # grid_dim from data + static
        grid_current_dim = self.grid_dim + constants.GRID_STATE_DIM_CERRA
        g2m_dim = self.g2m_features.shape[1]
        m2g_dim = self.m2g_features.shape[1]

        # Define sub-models
        # Feature embedders for grid
        self.mlp_blueprint_end = [args.hidden_dim] * (args.hidden_layers + 1)
        self.grid_prev_embedder = utils.make_mlp(
            [self.grid_dim] + self.mlp_blueprint_end
        )  # For states up to t-1
        self.grid_current_embedder = utils.make_mlp(
            [grid_current_dim] + self.mlp_blueprint_end
        )  # For states including t
        # Embedders for mesh
        self.g2m_embedder = utils.make_mlp([g2m_dim] + self.mlp_blueprint_end)
        self.m2g_embedder = utils.make_mlp([m2g_dim] + self.mlp_blueprint_end)
        if self.hierarchical_graph:
            # Print some useful info
            print("Loaded hierarchical graph with structure:")
            level_mesh_sizes = [
                mesh_feat.shape[0] for mesh_feat in self.mesh_static_features
            ]
            self.num_mesh_nodes = level_mesh_sizes[-1]
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
            # Embedders
            # Assume all levels have same static feature dimensionality
            mesh_dim = self.mesh_static_features[0].shape[1]
            m2m_dim = self.m2m_features[0].shape[1]
            mesh_up_dim = self.mesh_up_features[0].shape[1]
            mesh_down_dim = self.mesh_down_features[0].shape[1]

            # Separate mesh node embedders for each level
            self.mesh_embedders = torch.nn.ModuleList(
                [
                    utils.make_mlp([mesh_dim] + self.mlp_blueprint_end)
                    for _ in range(num_levels)
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
            # If not using any processor layers, no need to embed m2m
            self.embedd_m2m = (
                max(
                    args.prior_processor_layers,
                    args.encoder_processor_layers,
                    args.processor_layers,
                )
                > 0
            )
            if self.embedd_m2m:
                self.m2m_embedders = torch.nn.ModuleList(
                    [
                        utils.make_mlp([m2m_dim] + self.mlp_blueprint_end)
                        for _ in range(num_levels)
                    ]
                )
        else:
            self.num_mesh_nodes, mesh_static_dim = (
                self.mesh_static_features.shape
            )
            print(
                f"Loaded graph with {self.num_grid_nodes + self.num_mesh_nodes}"
                f"nodes ({self.num_grid_nodes} grid, "
                f"{self.num_mesh_nodes} mesh)"
            )
            mesh_static_dim = self.mesh_static_features.shape[1]
            self.mesh_embedder = utils.make_mlp(
                [mesh_static_dim] + self.mlp_blueprint_end
            )
            m2m_dim = self.m2m_features.shape[1]
            self.m2m_embedder = utils.make_mlp(
                [m2m_dim] + self.mlp_blueprint_end
            )

        latent_dim = (
            args.latent_dim if args.latent_dim is not None else args.hidden_dim
        )
        # Prior
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

        # Enc. + Dec.
        if self.hierarchical_graph:
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
        else:
            # Encoder
            self.encoder = GraphLatentEncoder(
                latent_dim,
                self.g2m_edge_index,
                self.m2m_edge_index,
                args.hidden_dim,
                args.encoder_processor_layers,
                hidden_layers=args.hidden_layers,
                output_dist="diagonal",
            )
            # Decoder
            self.decoder = GraphLatentDecoder(
                self.g2m_edge_index,
                self.m2m_edge_index,
                self.m2g_edge_index,
                args.hidden_dim,
                latent_dim,
                args.processor_layers,
                hidden_layers=args.hidden_layers,
                output_std=bool(args.output_std),
            )

        # Add lists for val and test errors of ensemble prediction
        self.val_metrics.update(
            {
                "spread_squared": [],
                "ens_mse": [],
            }
        )
        self.test_metrics.update(
            {
                "ens_mae": [],
                "ens_mse": [],
                "crps_ens": [],
                "spread_squared": [],
            }
        )





    def embedd_current(
        self,
        prev_state,
        prev_prev_state,
        forcing,
        current_state,
    ):
        """
        embed grid representation including current (target) state. Used as
        input to the encoder, which is conditioned also on the target.

        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)
        current_state: (B, num_grid_nodes, feature_dim), X_{t+1}

        Returns:
        current_emb: (B, num_grid_nodes, d_h)
        """
        batch_size = prev_state.shape[0]

        grid_current_features = torch.cat(
            (
                prev_prev_state,
                prev_state,
                forcing,
                self.expand_to_batch(self.grid_static_features, batch_size),
                current_state,
            ),
            dim=-1,
        )  # (B, num_grid_nodes, grid_current_dim)

        return self.grid_current_embedder(
            grid_current_features
        )  # (B, num_grid_nodes, d_h)

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
        batch_size = low_res.shape[0]

        high_res_grid_features = torch.cat(
            (
                high_res,
                self.expand_to_batch(self.grid_static_features, batch_size),
            ),
            dim=-1,
        )  # (B, num_grid_nodes, grid_dim)

        grid_emb = self.grid_prev_embedder(high_res_grid_features)
        # (B, num_grid_nodes, d_h)

        # Graph embedding
        graph_emb = {
            "g2m": self.expand_to_batch(
                self.g2m_embedder(self.g2m_features), batch_size
            ),  # (B, M_g2m, d_h)
            "m2g": self.expand_to_batch(
                self.m2g_embedder(self.m2g_features), batch_size
            ),  # (B, M_m2g, d_h)
        }

        if self.hierarchical_graph:
            graph_emb["mesh"] = [
                self.expand_to_batch(emb(node_static_features), batch_size)
                for emb, node_static_features in zip(
                    self.mesh_embedders,
                    self.mesh_static_features,
                )
            ]  # each (B, num_mesh_nodes[l], d_h)

            if self.embedd_m2m:
                graph_emb["m2m"] = [
                    self.expand_to_batch(emb(edge_feat), batch_size)
                    for emb, edge_feat in zip(
                        self.m2m_embedders, self.m2m_features
                    )
                ]
            else:
                # Need a placeholder otherwise, just use raw features
                graph_emb["m2m"] = list(self.m2m_features)

            graph_emb["mesh_up"] = [
                self.expand_to_batch(emb(edge_feat), batch_size)
                for emb, edge_feat in zip(
                    self.mesh_up_embedders, self.mesh_up_features
                )
            ]
            graph_emb["mesh_down"] = [
                self.expand_to_batch(emb(edge_feat), batch_size)
                for emb, edge_feat in zip(
                    self.mesh_down_embedders, self.mesh_down_features
                )
            ]
        else:
            graph_emb["mesh"] = self.expand_to_batch(
                self.mesh_embedder(self.mesh_static_features), batch_size
            )  # (B, num_mesh_nodes, d_h)
            graph_emb["m2m"] = self.expand_to_batch(
                self.m2m_embedder(self.m2m_features), batch_size
            )  # (B, M_m2m, d_h)

        return grid_emb, graph_emb

    def compute_step_loss(
        self,
        high_res,
        low_res,
    ):
        """
        Perform forward pass and compute loss for one time step

        prev_states: (B, 2, num_grid_nodes, d_features), X^{t-p}, ..., X^{t-1}
        current_state: (B, num_grid_nodes, d_features) X^t
        forcing_features: (B, num_grid_nodes, d_forcing) corresponding to
            index 1 of prev_states
        """
        # embed all features
        high_res_grid_emb, graph_emb = self.embedd_all(
            high_res,
            low_res,
        )

        # Compute variational approximation (encoder)
        var_dist = self.encoder(
            high_res_grid_emb, graph_emb=graph_emb
        )  # Gaussian, (B, num_mesh_nodes, d_latent)

        # Compute likelihood
        likelihood_term, pred_mean, pred_std = self.estimate_likelihood(
            var_dist, high_res_grid_emb, graph_emb
        )
        if self.kl_beta > 0:
            # Compute prior
            prior_dist = self.prior_model(
                high_res_grid_emb, graph_emb=graph_emb
            )  # Gaussian, (B, num_mesh_nodes, d_latent)

            # Compute KL
            kl_term = torch.sum(
                torch.distributions.kl_divergence(var_dist, prior_dist),
                dim=(1, 2),
            )  # (B,)
        else:
            # If beta=0, do not need to even compute prior nor KL
            kl_term = None  # Set to None to crash if erroneously used

        return likelihood_term, kl_term, pred_mean, pred_std
    
    
    def estimate_likelihood(
        self, latent_dist, grid_prev_emb, graph_emb
    ):
        """
        Estimate (masked) likelihood using given distribution over
        latent variables

        latent_dist: distribution, (B, num_mesh_nodes, d_latent)
        current_state: (B, num_grid_nodes, d_state)
        last_state: (B, num_grid_nodes, d_state)
        grid_prev_emb: (B, num_grid_nodes, d_state)
        g2m_emb: (B, M_g2m, d_h)
        m2m_emb: (B, M_m2m, d_h)
        m2g_emb: (B, M_m2g, d_h)

        Returns:
        likelihood_term: (B,)
        pred_mean: (B, num_grid_nodes, d_state)
        pred_std: (B, num_grid_nodes, d_state) or (d_state,)
        """
        # Sample from variational distribution
        latent_samples = latent_dist.rsample()  # (B, num_mesh_nodes, d_latent)

        # Compute reconstruction (decoder)
        pred_mean, model_pred_std = self.decoder(
            grid_prev_emb, latent_samples, graph_emb
        )  # both (B, num_grid_nodes, d_state)

        if self.output_std:
            pred_std = model_pred_std  # (B, num_grid_nodes, d_state)
        else:
            # Use constant set std.-devs.
            pred_std = self.per_var_std  # (d_f,)

        # Compute likelihood (negative loss, exactly likelihood for nll loss)
        # Note: There are some round-off errors here due to float32
        # and large values
        entry_likelihoods = -self.loss(
            pred_mean,
            pred_std,
            mask=self.interior_mask_bool,
            average_grid=False,
            sum_vars=False,
        )  # (B, num_grid_nodes', d_state)
        likelihood_term = torch.sum(entry_likelihoods, dim=(1, 2))  # (B,)
        return likelihood_term, pred_mean, pred_std


    def training_step(self, batch, mask_ration):
        """
        Train on single batch

        batch, containing:
        init_states: (B, 2, num_grid_nodes, d_state)
        target_states: (B, pred_steps, num_grid_nodes, d_state)
        forcing_features: (B, pred_steps, num_grid_nodes, d_forcing), where
            index 0 corresponds to index 1 of init_states
        """
        cerra, era5 = batch
        
        binary_mask_tensor, mask_idx = self.random_masking(cerra, mask_ratio=0.5)
        #self.plot_binary_mask(binary_mask_tensor)
        binary_mask_tensor = binary_mask_tensor.flatten()
        
        #mask 
        self.grid_static_features = self.grid_static_features * binary_mask_tensor.unsqueeze(1)
        #re-index mask_idx tensor: the indexes of the grid nodes reported in self.g2m_edge_index and self.g2m_features
        #are integers that start from 7371 (in the specifics og the current experiment settings).  This is beacsue we are 
        #dealing with a multi-layer mesh. Hence, the 0 index refers to the first node of the first mesh. Then there are all 
        # the nodes of the upper meshes, and lastly the nodes of the original grid. Consider that the first mesh has 6561 nodes 
        # (hence nodes indexes range from 0 to 6560), the second and third mesh combined have 810 nodes (hence nodes indexes range
        # from 6561 to 7370), and the original grid has 90000 nodes (hence nodes indexes range from 7371 to 97370).
        
        mask_idx_edges = mask_idx + self.g2m_edge_index[0,0]#self.g2m_edge_index[0,0] -1 = 7371
        edge_index_to_mask = torch.nonzero(torch.isin(self.g2m_edge_index, mask_idx_edges)[0], as_tuple=False).unique()
        self.g2m_features[edge_index_to_mask] = 0
        
        loss_like_term, loss_kl_term, pred_mean, pred_std = (
                self.compute_step_loss(
                    cerra,
                    era5,
                )
            )
        
        

    def random_masking(self, cerra, mask_ratio):
        """
        Randomly mask some variables at random lead times.

        Parameters:
        cerra (torch.Tensor): Input tensor of shape (batch, 9000, 5).
        mask_ratio (float): Ratio of subgrids to mask (e.g., 0.5 for 50%).

        Returns:
        torch.Tensor: Binary mask tensor of shape (grid_size, grid_size).
        """
        # Reshape cerra from (batch, 9000, 5) to (batch, 300, 300, 5)
        grid_size = int(cerra.shape[1] ** 0.5)

        # Define subgrid size
        subgrid_size = 10
        num_subgrids_per_row = grid_size // subgrid_size
        num_subgrids = num_subgrids_per_row ** 2

        # Mask a percentage of the subgrids
        num_masks = int(mask_ratio * num_subgrids)
        subgrid_indices = torch.arange(num_subgrids)
        masked_indices = subgrid_indices[torch.randperm(num_subgrids)[:num_masks]]

        # Create a binary mask of shape (grid_size, grid_size)
        binary_mask = torch.ones((grid_size, grid_size), dtype=torch.bool)

        for idx in masked_indices:
            row = idx // num_subgrids_per_row
            col = idx % num_subgrids_per_row
            binary_mask[row * subgrid_size:(row + 1) * subgrid_size, 
                        col * subgrid_size:(col + 1) * subgrid_size] = 0

        # Convert binary_mask to a tensor of 0s and 1s for output
        binary_mask_tensor = binary_mask.to(torch.float)

        indices_where_one = torch.nonzero(binary_mask_tensor.view(-1) == 1, as_tuple=False).squeeze()

        return binary_mask_tensor.to(self.device), indices_where_one.to(self.device)
    
    
    def plot_binary_mask(self, binary_mask_tensor, title="Binary Mask Visualization"):
        """
        Plot the binary mask tensor to visualize the masked areas.

        Parameters:
        binary_mask_tensor (torch.Tensor): Tensor of shape (x, y) with masking information.
        title (str): Title for the plot.
        """
        # Convert the binary mask tensor to a NumPy array for visualization
        mask_slice = binary_mask_tensor.cpu().numpy()

        # Plot the binary mask
        plt.figure(figsize=(8, 8))
        plt.imshow(mask_slice, cmap='gray', origin='upper', interpolation='nearest')
        plt.colorbar(label="Mask Value (1: Not Masked, 0: Masked)")
        plt.title(title)
        plt.xlabel("Grid X")
        plt.ylabel("Grid Y")
        plt.show()
        plt.savefig("mask_visualization.png")


        
        
        
        
        
        
        
        
        
        
        

