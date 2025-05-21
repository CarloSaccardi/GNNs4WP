import torch
import pytorch_lightning as pl

from neural_lam import constants, metrics, utils


class BaseGraphModule(pl.LightningModule):
    """
    Base LightningModule for graph-based models:
      - Loads hierarchical graph and static data
      - Sets up optimizer and common utilities
    """

    def __init__(self, args) -> None:
        super().__init__()
        # Core hyperparameters
        self.variational = args.variational
        self.lr = args.lr
        self.kl_beta = args.kl_beta
        self.wandb_project = args.wandb_project
        self.output_std = bool(args.output_std)
        self.loss = args.loss
        self.step_length = args.step_length
        self.save_hyperparameters()

        # Load and register graph buffers
        self._init_graph(args.graph)
        # Load and register static grid data
        self._init_static(args.dataset_cerra, args.dataset_era5)

        # Determine output dimensionality for grid predictions
        base_dim = constants.GRID_STATE_DIM_IN
        self.grid_output_dim = base_dim * (2 if self.output_std else 1)
        # Full node feature dimension: dynamic + static
        # _, static_dim = self.grid_static_features.shape
        self.grid_dim = base_dim #+ static_dim

        # Loss function
        self.loss_fn = metrics.get_metric(self.loss)

        # Metric containers
        self.val_metrics = {'mse': []}
        self.test_metrics = {'mse': [], 'mae': []}
        if self.output_std:
            self.test_metrics['output_std'] = []

        # Placeholder for optimizer state restoration
        self.opt_state = None
        # For storing spatial loss maps during evaluation
        self.spatial_loss_maps = []

    def _init_graph(self, graph_path: str) -> None:
        """
        Load hierarchical graph and register its features as buffers.
        """
        graph, graph_dict = utils.load_graph(graph_path)
        for name, val in graph_dict.items():
            if isinstance(val, torch.Tensor):
                self.register_buffer(name, val, persistent=False)
            else:
                setattr(self, name, val)
        # Derive graph-related dimensions
        self.g2m_dim = self.g2m_features.shape[1]
        self.m2g_dim = self.m2g_features.shape[1]

    def _init_static(self, cerra_path: str, era5_path: str) -> None:
        """
        Load and register static grid data from CERRA or ERA5.
        """
        dataset = cerra_path or era5_path
        static_dict = utils.load_static_data(dataset)
        for name, tensor in static_dict.items():
            self.register_buffer(name, tensor, persistent=False)

    def configure_optimizers(self):
        """
        Sets up AdamW optimizer with saved learning rate.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, betas=(0.9, 0.95)
        )
        if self.opt_state:
            optimizer.load_state_dict(self.opt_state)
        return optimizer

    @staticmethod
    def expand_to_batch(x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Expand tensor to shape [batch_size, *x.shape].
        """
        return x.unsqueeze(0).expand(batch_size, -1, -1)
