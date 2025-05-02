import os

import torch
import pytorch_lightning as pl

from neural_lam import constants, metrics, utils


class ARModel(pl.LightningModule):
    """
    Generic auto-regressive weather model. Extend for specific architectures.
    """

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        # Load high- and low-resolution graphs
        self._load_graph(args.graph_hr, suffix="_hr")
        self._load_graph(args.graph_lr, suffix="_lr")

        # Load static data features
        self._load_static_data(args.dataset_cerra, suffix="_hr")
        self._load_static_data(args.dataset_era5, suffix="_lr")

        # Dimensions
        self._init_dimensions()

        # Training settings
        self.kl_beta = args.kl_beta
        self.lr = args.lr
        self.output_std = args.output_std
        self.step_length = args.step_length
        self.wandb_project = args.wandb_project

        # Output dimension (double if predicting std)
        state_dim = constants.GRID_STATE_DIM_CERRA
        self.grid_output_dim = state_dim * (2 if self.output_std else 1)

        # Model loss and metrics
        self.loss_fn = metrics.get_metric(args.loss)
        self.val_metrics = {"mse": []}
        self.test_metrics = {"mse": [], "mae": []}
        if self.output_std:
            self.test_metrics["output_std"] = []

        # Optional optimizer state restore
        self.opt_state = None

        # For storing spatial loss maps
        self.spatial_loss_maps = []
        

    def _load_graph(self, graph_path: str, suffix: str):
        graph, buffers = utils.load_graph(graph_path)
        setattr(self, f"hierarchical_graph{suffix}", graph)
        # which substrings to skip for the low-res case
        skip_if_lr = ("m2g", "down")
        for name, tensor in buffers.items():
            # guard-clause: if we're in _lr mode and name contains any of the skip substrings
            if suffix == "_lr" and any(sub in name for sub in skip_if_lr):
                continue
            buffer_name = f"{name}{suffix}"
            if isinstance(tensor, torch.Tensor):
                self.register_buffer(buffer_name, tensor, persistent=False)
            else:
                setattr(self, buffer_name, tensor)
                

    def _load_static_data(self, data_path: str, suffix: str):
        data_dict = utils.load_static_data(data_path)
        for name, tensor in data_dict.items():
            buffer_name = f"{name}{suffix}"
            self.register_buffer(buffer_name, tensor, persistent=False)
            

    def _init_dimensions(self):
        # Graph feature dimensions
        self.g2m_dim_hr = self.g2m_features_hr.size(1)
        self.m2g_dim_hr = self.m2g_features_hr.size(1)
        self.g2m_dim_lr = self.g2m_features_lr.size(1)

        # Grid static features (high resolution)
        num_nodes_hr, static_dim_hr = self.grid_static_features_hr.size()
        self.grid_dim_hr = constants.GRID_STATE_DIM_CERRA + static_dim_hr
        self.num_grid_nodes_hr = num_nodes_hr
        self.grid_dim_hr_static = static_dim_hr

        # Grid static features (low resolution)
        num_nodes_lr, static_dim_lr = self.grid_static_features_lr.size()
        self.grid_dim_lr = constants.GRID_STATE_DIM_CERRA + static_dim_lr
        self.num_grid_nodes_lr = num_nodes_lr



    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, betas=(0.9, 0.95)
        )
        if self.opt_state:
            opt.load_state_dict(self.opt_state)

        return opt
    
    @staticmethod
    def expand_to_batch(x, batch_size):
        """
        Expand tensor with initial batch dimension
        """
        return x.unsqueeze(0).expand(batch_size, -1, -1)

