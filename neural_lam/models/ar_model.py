# Standard library
import os

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from neural_lam.utils import get_2d_sincos_pos_embed

# First-party
from neural_lam import constants, metrics, utils, vis


class ARModel(pl.LightningModule):
    """
    Generic auto-regressive weather model.
    Abstract class that can be extended.
    """

    # pylint: disable=arguments-differ
    # Disable to override args/kwargs from superclass

    def __init__(self, args):
        super().__init__()
        
        # Load graph with static features
        self.hierarchical_graph, graph_ldict = utils.load_graph(args.graph)
        for name, attr_value in graph_ldict.items():
            # Make BufferLists module members and register tensors as buffers
            if isinstance(attr_value, torch.Tensor):
                self.register_buffer(name, attr_value, persistent=False)
            else:
                setattr(self, name, attr_value)
        # Specify dimensions of data
        self.g2m_dim = self.g2m_features.shape[1]
        self.m2g_dim = self.m2g_features.shape[1]
        self.kl_beta = args.kl_beta
        self.wandb_project = args.wandb_project
        
        self.save_hyperparameters()
        self.lr = args.lr
        # Load static features for grid/data
        static_data_dict = utils.load_static_data(args.dataset_cerra) if args.dataset_cerra else utils.load_static_data(args.dataset_era5)
        for static_data_name, static_data_tensor in static_data_dict.items():
            self.register_buffer(
                static_data_name, static_data_tensor, persistent=False
            )

        # Double grid output dim. to also output std.-dev.
        self.output_std = bool(args.output_std)
        
        if self.output_std:
            self.grid_output_dim = 2 * constants.GRID_STATE_DIM_CERRA  # Pred. dim. in grid cell
        else:
            self.grid_output_dim = constants.GRID_STATE_DIM_CERRA  # Pred. dim. in grid cell

        # grid_dim from data + static
        self.num_grid_nodes, grid_static_dim = self.grid_static_features.shape  # 63784 = 268x238
        self.grid_dim = constants.GRID_STATE_DIM_CERRA + grid_static_dim

        # Instantiate loss function
        self.loss = metrics.get_metric(args.loss)
        self.step_length = args.step_length  # Number of hours per pred. step

        # For making restoring of optimizer state optional (slight hack)
        self.opt_state = None

        # For storing spatial loss maps during evaluation
        self.spatial_loss_maps = []
        
        # Add lists for val and test errors of ensemble prediction
        self.val_metrics = {
            "mse": [],
        }
        self.test_metrics = {
            "mse": [],
            "mae": [],
        }
        if self.output_std:
            self.test_metrics["output_std"] = []  # Treat as metric

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

