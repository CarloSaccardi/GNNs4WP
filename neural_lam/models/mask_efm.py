# Third-party
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import math

# First-party
from neural_lam import constants, metrics, utils, vis
from neural_lam.utils import get_2d_sincos_pos_embed
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
        assert (
            args.n_example_pred <= args.batch_size
        ), "Can not plot more examples than batch size in GraphEFM"
        #self.sample_obs_noise = bool(args.sample_obs_noise)
        #self.ensemble_size = args.ensemble_size
        #self.kl_beta = args.kl_beta
        #self.crps_weight = args.crps_weight


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
        #grid_current_dim = self.grid_dim + constants.GRID_STATE_DIM_CERRA
        
        g2m_dim = self.g2m_features.shape[1]
        m2g_dim = self.m2g_features.shape[1]

        # Define sub-models
        # Feature embedders for grid
        self.mlp_blueprint_end = [args.hidden_dim] * (args.hidden_layers + 1)
        self.high_res_embedder = utils.make_mlp(
            [self.grid_dim] + self.mlp_blueprint_end
        )  # For states up to t-1

        self.low_res_embedder = utils.make_mlp(
            [self.grid_dim] + self.mlp_blueprint_end
        )

        #self.grid_current_embedder = utils.make_mlp(
        #    [grid_current_dim] + self.mlp_blueprint_end
        #)  # For states including t
        # Embedders for mesh
        self.g2m_embedder = utils.make_mlp([g2m_dim] + self.mlp_blueprint_end)
        self.m2g_embedder = utils.make_mlp([m2g_dim] + self.mlp_blueprint_end)
        if self.hierarchical_graph:
            # Print some useful info
            print("Loaded hierarchical graph with structure:")
            level_mesh_sizes = [
                mesh_feat.shape[0] for mesh_feat in self.mesh_static_features
            ]
            #self.num_mesh_nodes = level_mesh_sizes[-1]#is this right? shouldn't it be the sum of all the mesh nodes?
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
            mesh_dim = self.mesh_static_features[1].shape[1] #first mesh features has shape (6561, 4) since it represent the era5 grid. The other meshes have shape (x, 2). There is a dedicated ambedder for the first mesh. This code should be modified to take into account the different shapes of the mesh features.
            m2m_dim = self.m2m_features[0].shape[1]
            mesh_up_dim = self.mesh_up_features[0].shape[1]
            mesh_down_dim = self.mesh_down_features[0].shape[1]

            # Separate mesh node embedders for each level
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
        
        
        ##### MAsking parameters #####
        self.num_nodes = 300**2
        
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, self.num_nodes, args.hidden_dim), requires_grad=False)  # fixed sin-cos embedding
        
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, args.hidden_dim))
        
        self.decoder_pos_embed = torch.nn.Parameter(torch.zeros(1, self.num_nodes, args.hidden_dim), requires_grad=False)  # fixed sin-cos embedding
        
        self.initialize_weights()
        
        
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_nodes**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_nodes**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        #w = self.patch_embed.proj.weight.data
        #torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        #torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        #self.apply(self._init_weights)
        
        

    def embedd_all(self, high_res, low_res, mask_ratio=None):
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

        high_res_grid_emb = self.high_res_embedder(high_res_grid_features)
        
        if mask_ratio is not None:
        
            #### Masking ####
            high_res_grid_emb = high_res_grid_emb + self.pos_embed
            high_res_grid_emb, mask, ids_restore, ids_keep = self.block_random_masking(high_res_grid_emb, mask_ratio)
            keep_uniques = torch.unique(ids_keep[0]) + self.g2m_edge_index[0,0]
            senders = self.g2m_edge_index[0]
            mask_edges = torch.isin(senders, keep_uniques)
            kept_indexes = torch.nonzero(mask_edges, as_tuple=True)[0]
            g2m_features = self.g2m_features[kept_indexes]
            #### Mask g2m edge index ####
            g2m_edge_index_mins = self.g2m_edge_index.min(dim=1, keepdim=True)[0]
            g2m_edge_index_max_1 = self.g2m_edge_index[1].max()
            g2m_edge_index = self.g2m_edge_index[:, mask_edges]
            g2m_edge_index = g2m_edge_index  - g2m_edge_index_mins
            self.num_rec = g2m_edge_index_max_1 + 1
            g2m_edge_index[0] = (
                g2m_edge_index[0] + self.num_rec
            )
            unique_senders = torch.unique(g2m_edge_index[0])
            sorted_senders = torch.sort(unique_senders).values
            new_senders = torch.arange(self.num_rec, self.num_rec + len(sorted_senders))
            sender_mapping = dict(zip(sorted_senders.tolist(), new_senders.tolist()))
            reindexed_senders = torch.tensor([sender_mapping[sender.item()] for sender in g2m_edge_index[0]])
            g2m_edge_index[0] = reindexed_senders
            
            ##############################
        
        else:
            g2m_features = self.g2m_features
            mask = None
            ids_restore = None


        # Graph embedding
        graph_emb = {
            "g2m": self.expand_to_batch(
                self.g2m_embedder(g2m_features), batch_size
            ),  # (B, M_g2m, d_h)
            "m2g": self.expand_to_batch(
                self.m2g_embedder(self.m2g_features), batch_size
            ),  # (B, M_m2g, d_h)
        }
        
        if mask_ratio is not None:
             graph_emb["g2m_edge_index"] = g2m_edge_index #TODO this should go in the masking block. graph_emb is defined after the masking block. 

        if self.hierarchical_graph:
            
            graph_emb["mesh"] = []
            for indx, (emb, node_static_features) in enumerate(zip(self.mesh_embedders, self.mesh_static_features)):
                
                if indx == 0:
                    #embed low_res grid with era5 data, just as the high_res grid
                    low_res_grid_features = torch.cat(
                        (
                            low_res,
                            self.expand_to_batch(node_static_features, batch_size),
                        ),
                        dim=-1,
                    )  # (B, num_mesh_nodes, mesh_dim)
                    
                    graph_emb["mesh"].append(emb(low_res_grid_features)) 
                    
                else:

                    graph_emb["mesh"].append(self.expand_to_batch(emb(node_static_features), batch_size))
                
                
                
                #self.expand_to_batch(emb(node_static_features), batch_size)
            

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

        return high_res_grid_emb, graph_emb, mask, ids_restore

    def compute_step_loss(
        self,
        high_res,
        low_res,
        mask_ratio=0.5,
    ):
        """
        Perform forward pass and compute loss for one time step

        prev_states: (B, 2, num_grid_nodes, d_features), X^{t-p}, ..., X^{t-1}
        current_state: (B, num_grid_nodes, d_features) X^t
        forcing_features: (B, num_grid_nodes, d_forcing) corresponding to
            index 1 of prev_states
        """
        # embed all features
        high_res_grid_emb, graph_emb, mask, ids_restore = self.embedd_all(high_res,low_res, mask_ratio)
        
        # Compute variational approximation (encoder)
        var_dist = self.encoder(
            high_res_grid_emb, graph_emb=graph_emb
        )  # Gaussian, (B, num_mesh_nodes, d_latent)

        # Compute likelihood
        likelihood_term, pred_mean, pred_std = self.estimate_likelihood(
            var_dist, high_res, high_res_grid_emb, graph_emb, ids_restore, mask
        )
        """
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
        """

        return likelihood_term, pred_mean, pred_std
    
    
    def estimate_likelihood(
        self, latent_dist, high_res_grid, high_res_emb, graph_emb, ids_restore, mask
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
        # Sample from variational distribution
        latent_samples = latent_dist.rsample()  # (B, num_mesh_nodes, d_latent)

        #graph_emb["g2m_edge_index"] = None
        # Compute reconstruction (decoder)
        
        mask_tokens = self.mask_token.repeat(high_res_emb.shape[0], ids_restore.shape[1] + 1 - high_res_emb.shape[1], 1)
        x_ = torch.cat([high_res_emb, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, high_res_emb.shape[2]))  # unshuffle
        full_grid_rep = x_ + self.decoder_pos_embed
        
        pred_mean, model_pred_std = self.decoder(
            high_res_emb, latent_samples, graph_emb, full_grid_rep
        )  # both (B, num_grid_nodes, d_state)

        if self.output_std:
            pred_std = model_pred_std  # (B, num_grid_nodes, d_state)
        else:
            # Use constant set std.-devs.
            pred_std = self.per_var_std  # (d_f,)

        # Compute MSE loss for masked areas        
        mask = mask.unsqueeze(-1)
        squared_error = ((pred_mean - high_res_grid) ** 2) * mask 
        mse_masked = squared_error.sum() / mask.sum()
        
        return mse_masked, pred_mean, pred_std


    def training_step(self, batch):
        """
        Train on single batch

        batch, containing:
        init_states: (B, 2, num_grid_nodes, d_state)
        target_states: (B, pred_steps, num_grid_nodes, d_state)
        forcing_features: (B, pred_steps, num_grid_nodes, d_forcing), where
            index 0 corresponds to index 1 of init_states
        """
        cerra, era5 = batch
        
        """
        binary_mask_tensor, mask_idx = self.random_masking(cerra, mask_ratio=0.5, subgrid_size=50)
        #self.plot_binary_mask(binary_mask_tensor)
        binary_mask_tensor = binary_mask_tensor.flatten()#shape [90000]
        
        #mask 
        cerra = cerra * binary_mask_tensor.unsqueeze(1)
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
    
        """
        
        loss, _, _ = (
                self.compute_step_loss(
                    cerra,
                    era5,
                    mask_ratio=0.5,
                )
            )
        
        log_dict = {
            "train_loss": loss,
        }

        self.log_dict(
            log_dict, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
        return loss
             
    
    def block_random_masking(self, x, mask_ratio, grid_size=300, block_size=50):
        """
        Perform random masking on a flattened grid of nodes, grouped into blocks, based on the original 2D grid layout.

        Args:
            x: Tensor of shape (batch, num_nodes, latent_dim), e.g., (N, 90000, D).
            mask_ratio: Fraction of blocks to mask (e.g., 0.4 for 40% masked).
            grid_size: Size of the grid (e.g., 300x300 for num_nodes=90000).
            block_size: Size of each block (e.g., 50x50).

        Returns:
            x_masked: Tensor of shape (batch, num_kept_nodes, latent_dim).
            mask: Binary mask of shape (batch, num_nodes), 0 for kept, 1 for masked.
        """
        N, num_nodes, D = x.shape
        assert num_nodes == grid_size * grid_size, "num_nodes must match grid size"
        num_blocks = (grid_size // block_size) ** 2  # Total number of blocks (e.g., 36x36=1296)

        # Step 1: Compute block indices for each node
        # Reshape flat indices (0 to 89999) back to 2D grid coordinates (row, col)
        row_indices = torch.arange(grid_size, device=x.device).repeat_interleave(grid_size)
        col_indices = torch.arange(grid_size, device=x.device).repeat(grid_size)

        # Determine block row and column indices for each node
        block_row_indices = row_indices // block_size
        block_col_indices = col_indices // block_size

        # Assign a unique block index to each block
        block_indices = block_row_indices * (grid_size // block_size) + block_col_indices
        # `block_indices` has shape (90000,) and assigns each node to a block (0 to 1295)

        # Step 2: Randomly mask a subset of blocks
        len_keep = int(num_nodes * (1 - mask_ratio))
        noise = torch.rand(num_blocks, device=x.device).repeat(N,1)  # Random noise for each block
        noise = noise[:, block_indices]  # Expand noise to all nodes and shuffle blocks
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]  # Blocks to keep
        
        #ids_restore = ids_restore[:, block_indices] # Restore the original block order
        #ids_keep = ids_keep[:, block_indices]  # Keep the first subset of blocks
        
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, num_nodes], device=x.device)
        
        #new_len_keep = len_keep * block_size**2
        mask[:, :len_keep] = 0  # 0 is keep, 1 is remove
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore, ids_keep

    
    
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
        

    def validation_step(self, batch, *args):
        """
        Run validation on single batch
        """
        
        high_res, low_res = batch
        prediction, pred_std, mask = self.predict_step(high_res, low_res)
        
        target = high_res 

        mask = mask.unsqueeze(-1)
        squared_error = ((prediction - target) ** 2) * mask 
        val_loss = squared_error.sum() / mask.sum()

        # Log loss per time step forward and mean
        val_log_dict = {}
        val_log_dict["val_loss"] = val_loss
        self.log_dict(
            val_log_dict, on_step=False, on_epoch=True, sync_dist=True
        )
        
        batch_idx = args[0]
        
        
        
        # Plot some example predictions using prior and encoder
        if (
            self.trainer.is_global_zero
            and batch_idx == 0
        ):

            # Plot samples
            log_plot_dict = {}

            for var_i in constants.VAL_PLOT_VARS_CERRA:
                var_name = constants.PARAM_NAMES_SHORT_CERRA[var_i]
                var_unit = constants.PARAM_UNITS_CERRA[var_i]

                pred_states = prediction[
                    batch_idx, :, var_i
                ]  # (S, num_grid_nodes)
                
                target_state = target[
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
                    obs_mask = 1 - mask[batch_idx].flatten(),
                    title=f"{plot_title} (prior)",
                )

            if not self.trainer.sanity_checking:
                # Log all plots to wandb
                wandb.log(log_plot_dict)

            plt.close("all")   
    
            
            
            
    def predict_step(self, prev_state, prev_prev_state):
        """
        Sample one time step prediction

        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)

        Returns:
        new_state: (B, num_grid_nodes, feature_dim)
        """
        # embed all features
        grid_prev_emb, graph_emb, mask, ids_restore= self.embedd_all(
            prev_state, prev_prev_state, mask_ratio=0.5
        )

        #graph_emb["g2m_edge_index"] = None
        # Compute prior
        prior_dist = self.encoder(
            grid_prev_emb, graph_emb=graph_emb
        )  # (B, num_mesh_nodes, d_latent)

        # Sample from prior
        latent_samples = prior_dist.rsample()
        # (B, num_mesh_nodes, d_latent)
        
        mask_tokens = self.mask_token.repeat(grid_prev_emb.shape[0], ids_restore.shape[1] + 1 - grid_prev_emb.shape[1], 1)
        x_ = torch.cat([grid_prev_emb, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, grid_prev_emb.shape[2]))  # unshuffle
        full_grid_rep = x_ + self.decoder_pos_embed

        # Compute reconstruction (decoder)
        pred_mean, pred_std = self.decoder(
            grid_prev_emb, latent_samples, graph_emb, full_grid_rep
        )  # (B, num_grid_nodes, d_state)

        return pred_mean, pred_std, mask
            
            
    

        
        
        
        
        
        
        
        
        

