# Standard library
import os

# Third-party
import numpy as np
import torch
import torch_geometric as pyg
from torch import nn
from tueplots import bundles, figsizes

# First-party
from neural_lam import constants
from neural_lam.models.interaction_net import InteractionNet
import matplotlib.pyplot as plt
import easydict
from pyproj import Transformer
from neural_lam import constants
import xarray as xr
import cfgrib


def load_dataset_stats(dataset_name, device="cpu"):
    """
    Load arrays with stored dataset statistics from pre-processing
    """
    static_dir_path = os.path.join(dataset_name, "static")

    def loads_file(fn):
        return torch.load(
            os.path.join(static_dir_path, fn), map_location=device
        )

    data_mean = loads_file("parameter_mean.pt")  # (d_features,)
    data_std = loads_file("parameter_std.pt")  # (d_features,)

    return {
        "data_mean": data_mean,
        "data_std": data_std,
    }


def load_static_data(dataset_name, device="cpu"):
    """
    Load static files related to dataset
    """
    static_dir_path = os.path.join(dataset_name, "static")

    def loads_file(fn):
        return torch.load(
            os.path.join(static_dir_path, fn), map_location=device
        )

    grid_static_features = loads_file(
        "grid_features.pt"
    )  # (N_grid, d_grid_static)

    # Load parameter std for computing validation errors in original data scale
    data_mean = loads_file("parameter_mean.pt")  # (d_features,)
    data_std = loads_file("parameter_std.pt")  # (d_features,)

    return {
        "grid_static_features": grid_static_features,
        "data_mean": data_mean,
        "data_std": data_std,
    }


class BufferList(nn.Module):
    """
    A list of torch buffer tensors that sit together as a Module with no
    parameters and only buffers.

    This should be replaced by a native torch BufferList once implemented.
    See: https://github.com/pytorch/pytorch/issues/37386
    """

    def __init__(self, buffer_tensors, persistent=True):
        super().__init__()
        self.n_buffers = len(buffer_tensors)
        for buffer_i, tensor in enumerate(buffer_tensors):
            self.register_buffer(f"b{buffer_i}", tensor, persistent=persistent)

    def __getitem__(self, key):
        return getattr(self, f"b{key}")

    def __len__(self):
        return self.n_buffers

    def __iter__(self):
        return (self[i] for i in range(len(self)))


def load_graph(graph_name, device="cpu"):
    """
    Load all tensors representing the graph
    """
    # Define helper lambda function
    graph_dir_path = os.path.join("graphs", graph_name)

    def loads_file(fn):
        return torch.load(os.path.join(graph_dir_path, fn), map_location=device)

    # Load edges (edge_index)
    m2m_edge_index = BufferList(
        loads_file("m2m_edge_index.pt"), persistent=False
    )  # List of (2, M_m2m[l])
    g2m_edge_index = loads_file("g2m_edge_index.pt")  # (2, M_g2m)
    m2g_edge_index = loads_file("m2g_edge_index.pt")  # (2, M_m2g)

    n_levels = len(m2m_edge_index)
    hierarchical = n_levels > 1  # Nor just single level mesh graph

    # Load static edge features
    m2m_features = loads_file("m2m_features.pt")  # List of (M_m2m[l], d_edge_f)
    g2m_features = loads_file("g2m_features.pt")  # (M_g2m, d_edge_f)
    m2g_features = loads_file("m2g_features.pt")  # (M_m2g, d_edge_f)

    # Normalize by dividing with longest edge (found in m2m)
    longest_edge = max(
        torch.max(level_features[:, 0]) for level_features in m2m_features
    )  # Col. 0 is length
    m2m_features = BufferList(
        [level_features / longest_edge for level_features in m2m_features],
        persistent=False,
    )
    g2m_features = g2m_features / longest_edge
    m2g_features = m2g_features / longest_edge

    # Load static node features
    mesh_static_features = loads_file(
        "mesh_features.pt"
    )  # List of (N_mesh[l], d_mesh_static)

    # Some checks for consistency
    assert (
        len(m2m_features) == n_levels
    ), "Inconsistent number of levels in mesh"
    assert (
        len(mesh_static_features) == n_levels
    ), "Inconsistent number of levels in mesh"

    if hierarchical:
        # Load up and down edges and features
        mesh_up_edge_index = BufferList(
            loads_file("mesh_up_edge_index.pt"), persistent=False
        )  # List of (2, M_up[l])
        mesh_down_edge_index = BufferList(
            loads_file("mesh_down_edge_index.pt"), persistent=False
        )  # List of (2, M_down[l])

        mesh_up_features = loads_file(
            "mesh_up_features.pt"
        )  # List of (M_up[l], d_edge_f)
        mesh_down_features = loads_file(
            "mesh_down_features.pt"
        )  # List of (M_down[l], d_edge_f)

        # Rescale
        mesh_up_features = BufferList(
            [
                edge_features / longest_edge
                for edge_features in mesh_up_features
            ],
            persistent=False,
        )
        mesh_down_features = BufferList(
            [
                edge_features / longest_edge
                for edge_features in mesh_down_features
            ],
            persistent=False,
        )

        mesh_static_features = BufferList(
            mesh_static_features, persistent=False
        )
    else:
        # Extract single mesh level
        m2m_edge_index = m2m_edge_index[0]
        m2m_features = m2m_features[0]
        mesh_static_features = mesh_static_features[0]

        (
            mesh_up_edge_index,
            mesh_down_edge_index,
            mesh_up_features,
            mesh_down_features,
        ) = ([], [], [], [])

    return hierarchical, {
        "g2m_edge_index": g2m_edge_index,
        "m2g_edge_index": m2g_edge_index,
        "m2m_edge_index": m2m_edge_index,
        "mesh_up_edge_index": mesh_up_edge_index,
        "mesh_down_edge_index": mesh_down_edge_index,
        "g2m_features": g2m_features,
        "m2g_features": m2g_features,
        "m2m_features": m2m_features,
        "mesh_up_features": mesh_up_features,
        "mesh_down_features": mesh_down_features,
        "mesh_features": mesh_static_features,
    }


def make_mlp(blueprint, layer_norm=True):
    """
    Create MLP from list blueprint, with
    input dimensionality: blueprint[0]
    output dimensionality: blueprint[-1] and
    hidden layers of dimensions: blueprint[1], ..., blueprint[-2]

    if layer_norm is True, includes a LayerNorm layer at
    the output (as used in GraphCast)
    """
    hidden_layers = len(blueprint) - 2
    assert hidden_layers >= 0, "Invalid MLP blueprint"

    layers = []
    for layer_i, (dim1, dim2) in enumerate(zip(blueprint[:-1], blueprint[1:])):
        layers.append(nn.Linear(dim1, dim2))
        if layer_i != hidden_layers:
            layers.append(nn.SiLU())  # Swish activation

    # Optionally add layer norm to output
    if layer_norm:
        layers.append(nn.LayerNorm(blueprint[-1]))

    return nn.Sequential(*layers)


def fractional_plot_bundle(fraction):
    """
    Get the tueplots bundle, but with figure width as a fraction of
    the page width.
    """
    bundle = bundles.neurips2023(usetex=False, family="serif")
    bundle.update(figsizes.neurips2023())
    original_figsize = bundle["figure.figsize"]
    bundle["figure.figsize"] = (
        original_figsize[0] / fraction,
        original_figsize[1],
    )
    return bundle


def init_wandb_metrics(wandb_logger):
    """
    Set up wandb metrics to track
    """
    experiment = wandb_logger.experiment
    experiment.define_metric("val_mean_loss", summary="min")
    for step in constants.VAL_STEP_LOG_ERRORS:
        experiment.define_metric(f"val_loss_unroll{step}", summary="min")


class IdentityModule(nn.Module):
    """
    A identity operator that can return multiple inputs
    """

    def forward(self, *args):
        """Return input args"""
        return args


def make_gnn_seq(edge_index, num_gnn_layers, hidden_layers, hidden_dim):
    """
    Make a sequential GNN module propagating both node and edge representations
    """
    if num_gnn_layers == 0:
        # If no layers, return identity
        return IdentityModule()
    return pyg.nn.Sequential(
        "mesh_rep, edge_rep",
        [
            (
                InteractionNet(
                    edge_index,
                    hidden_dim,
                    hidden_layers=hidden_layers,
                ),
                "mesh_rep, mesh_rep, edge_rep -> mesh_rep, edge_rep",
            )
            for _ in range(num_gnn_layers)
        ],
    )


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


    
def plot_binary_mask(binary_mask_tensor, title="Binary Mask Visualization"):
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
    
    
def compute_MSE_entiregrid(prediction, targets):
    """
    Compute the Mean Squared Error (MSE) for the entire grid.

    Parameters:
    preds (torch.Tensor): Tensor of shape (batch_size, grid_size, grid_size) with the predicted values.
    targets (torch.Tensor): Tensor of shape (batch_size, grid_size, grid_size) with the target values.
    mask (torch.Tensor): Tensor of shape (grid_size, grid_size) with the binary mask.

    Returns:
    mse (float): Mean Squared Error (MSE) for the entire grid.
    """
    squared_error = (prediction - targets) ** 2
    mse_batches = squared_error.mean(dim=1)
    #mse_per_var = mse_batches.mean(dim=0)
    
    return mse_batches.mean(), mse_batches.mean(dim=0)

def compute_MSE_masked(prediction, targets, mask):

    mask = mask.unsqueeze(-1)
    mask = mask.repeat(1, 1, 5)
    squared_error = ((prediction - targets) ** 2) * mask 
    mse_batches = squared_error.sum(dim=1) / mask.sum(dim=1)
    #mse_per_var = mse_batches.mean(dim=0)
    return mse_batches.mean(), mse_batches.mean(dim=0)

def compute_MAE_entiregrid(prediction, targets):
    """
    Compute the Mean Absolute Error (MAE) for the entire grid.

    Parameters:
    preds (torch.Tensor): Tensor of shape (batch_size, grid_size, grid_size) with the predicted values.
    targets (torch.Tensor): Tensor of shape (batch_size, grid_size, grid_size) with the target values.
    mask (torch.Tensor): Tensor of shape (grid_size, grid_size) with the binary mask.

    Returns:
    mae (float): Mean Absolute Error (MAE) for the entire grid.
    """
    absolute_error = torch.abs(prediction - targets)
    mae_batches = absolute_error.mean(dim=1)
    #mae_per_var = mae_batches.mean(dim=0)
    
    return mae_batches.mean(), mae_batches.mean(dim=0)


def compute_MAE_masked(prediction, targets, mask):
    mask = mask.unsqueeze(-1)
    mask = mask.repeat(1, 1, 5)
    absolute_error = torch.abs(prediction - targets) * mask
    mae_batches = absolute_error.sum(dim=1) / mask.sum(dim=1)
    #mae_per_var = mae_batches.mean(dim=0)
    return mae_batches.mean(), mae_batches.mean(dim=0)