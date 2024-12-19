# Standard library
import os
from argparse import ArgumentParser

# Third-party
import random
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
import networkx
import numpy as np
import scipy.spatial
import torch
import torch_geometric as pyg
from torch_geometric.utils.convert import from_networkx

#set cuda visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


def plot_graph_nodes_only(graph, title=None):
    fig, axis = plt.subplots(figsize=(8, 8), dpi=200)  # W,H
    pos = graph.pos

    # Move all to cpu and numpy, compute (in)-degrees
    degrees = (
        pyg.utils.degree(graph.edge_index[1], num_nodes=pos.shape[0]).cpu().numpy()
    )
    pos = pos.cpu().numpy()

    # Plot nodes
    node_scatter = axis.scatter(
        pos[:, 0],
        pos[:, 1],
        c=degrees,
        s=3,
        marker="o",
        zorder=2,
        cmap="viridis",
        clim=None,
    )

    plt.colorbar(node_scatter, aspect=50)

    if title is not None:
        axis.set_title(title)

    return fig, axis

def plot_graph_nodes_only_2graph(graph, graph2=None, title=None):
    fig, axis = plt.subplots(figsize=(8, 8), dpi=200)  # W,H
    
    # Plot nodes of the first graph
    pos1 = graph.pos
    degrees1 = (
        pyg.utils.degree(graph.edge_index[1], num_nodes=pos1.shape[0]).cpu().numpy()
    )
    pos1 = pos1.cpu().numpy()

    node_scatter1 = axis.scatter(
        pos1[:, 0],
        pos1[:, 1],
        c=degrees1,
        s=3,
        marker="o",
        zorder=2,
        cmap="viridis",
        label="Graph 1"
    )

    # Plot nodes of the second graph if provided
    if graph2 is not None:
        pos2 = graph2.pos
        degrees2 = (
            pyg.utils.degree(graph2.edge_index[1], num_nodes=pos2.shape[0]).cpu().numpy()
        )
        pos2 = pos2.cpu().numpy()

        node_scatter2 = axis.scatter(
            pos2[:, 0],
            pos2[:, 1],
            c="red",  # Single color for graph 2 nodes
            s=3,
            marker="o",
            zorder=3,
            label="Graph 2"
        )

    # Add a legend and colorbar for the first graph
    plt.colorbar(node_scatter1, aspect=50, label="Degrees (Graph 1)")
    axis.legend()

    if title is not None:
        axis.set_title(title)

    return fig, axis




def plot_graph(graph, title=None):
    fig, axis = plt.subplots(figsize=(8, 8), dpi=200)  # W,H
    edge_index = graph.edge_index
    pos = graph.pos

    # Fix for re-indexed edge indices only containing mesh nodes at
    # higher levels in hierarchy
    edge_index = edge_index - edge_index.min()

    if pyg.utils.is_undirected(edge_index):
        # Keep only 1 direction of edge_index
        edge_index = edge_index[:, edge_index[0] < edge_index[1]]  # (2, M/2)
    # TODO: indicate direction of directed edges

    # Move all to cpu and numpy, compute (in)-degrees
    degrees = (
        pyg.utils.degree(edge_index[1], num_nodes=pos.shape[0]).cpu().numpy()
    )
    edge_index = edge_index.cpu().numpy()
    pos = pos.cpu().numpy()

    # Plot edges
    from_pos = pos[edge_index[0]]  # (M/2, 2)
    to_pos = pos[edge_index[1]]  # (M/2, 2)
    edge_lines = np.stack((from_pos, to_pos), axis=1)
    axis.add_collection(
        matplotlib.collections.LineCollection(
            edge_lines, lw=0.4, colors="black", zorder=1
        )
    )

    # Plot nodes
    node_scatter = axis.scatter(
        pos[:, 0],
        pos[:, 1],
        c=degrees,
        s=3,
        marker="o",
        zorder=2,
        cmap="viridis",
        clim=None,
    )

    plt.colorbar(node_scatter, aspect=50)

    if title is not None:
        axis.set_title(title)

    return fig, axis


def sort_nodes_internally(nx_graph):
    # For some reason the networkx .nodes() return list can not be sorted,
    # but this is the ordering used by pyg when converting.
    # This function fixes this.
    H = networkx.DiGraph()
    H.add_nodes_from(sorted(nx_graph.nodes(data=True)))
    H.add_edges_from(nx_graph.edges(data=True))
    return H


def save_edges(graph, name, base_path):
    torch.save(
        graph.edge_index, os.path.join(base_path, f"{name}_edge_index.pt")
    )
    edge_features = torch.cat((graph.len.unsqueeze(1), graph.vdiff), dim=1).to(
        torch.float32
    )  # Save as float32
    torch.save(edge_features, os.path.join(base_path, f"{name}_features.pt"))


def save_edges_list(graphs, name, base_path):
    torch.save(
        [graph.edge_index for graph in graphs],
        os.path.join(base_path, f"{name}_edge_index.pt"),
    )
    edge_features = [
        torch.cat((graph.len.unsqueeze(1), graph.vdiff), dim=1).to(
            torch.float32
        )
        for graph in graphs
    ]  # Save as float32
    torch.save(edge_features, os.path.join(base_path, f"{name}_features.pt"))


def from_networkx_with_start_index(nx_graph, start_index):
    pyg_graph = from_networkx(nx_graph)
    pyg_graph.edge_index += start_index
    return pyg_graph


def mk_2d_graph_with_masking(xy, nx, ny, block_size, mask_percentage, seed=None):
    """
    Create a 2D grid graph with masked blocks of nodes and edges.

    Parameters:
        nx, ny (int): Number of nodes in the x and y directions (grid size).
        block_size (int): Size of each block (block_size x block_size).
        mask_percentage (float): Percentage of blocks to mask (0 to 100).
        seed (int, optional): Seed for reproducibility.

    Returns:
        networkx.DiGraph: Directed graph with masked nodes and edges.
    """
    # Set random seed for reproducibility
    g = networkx.grid_2d_graph(ny, nx)
    g.clear_edges()

    for node in g.nodes:
        g.nodes[node]["pos"] = np.array([xy[0][node], xy[1][node]])
        #g.nodes[node]["mask"] = False


    # Divide into blocks and select blocks to mask
    g_masked = g.copy()
    num_blocks_x = nx // block_size
    num_blocks_y = ny // block_size
    total_blocks = num_blocks_x * num_blocks_y

    blocks = [(bx, by) for bx in range(num_blocks_x) for by in range(num_blocks_y)]
    num_blocks_to_mask = int(mask_percentage / 100 * total_blocks)
    masked_blocks = random.sample(blocks, num_blocks_to_mask)

    # Mask nodes and edges
    masked_nodes = set()
    edges_to_remove = set() 
    for bx, by in masked_blocks:
        for i in range(bx * block_size, (bx + 1) * block_size):
            for j in range(by * block_size, (by + 1) * block_size):
                if (j, i) in g.nodes:
                    #g.nodes[(j, i)]["mask"] = True
                    masked_nodes.add((j, i))

    # Mask edges connected to masked nodes
    for u, v in list(g.edges):
        if u in masked_nodes or v in masked_nodes:
            #g.edges[u, v]["mask"] = True
            #g.edges[v, u]["mask"] = True
            edges_to_remove.add((u, v))
            
    g_masked.remove_nodes_from(masked_nodes)
    g_masked.remove_edges_from(edges_to_remove)
            

    return g, g_masked



def mk_2d_graph(xy, nx, ny):
    xm, xM = np.amin(xy[0][0, :]), np.amax(xy[0][0, :])
    ym, yM = np.amin(xy[1][:, 0]), np.amax(xy[1][:, 0])

    # avoid nodes on border
    dx = (xM - xm) / nx
    dy = (yM - ym) / ny
    lx = np.linspace(xm + dx / 2, xM - dx / 2, nx)
    ly = np.linspace(ym + dy / 2, yM - dy / 2, ny)

    mg = np.meshgrid(lx, ly)
    g = networkx.grid_2d_graph(len(ly), len(lx))

    for node in g.nodes:
        g.nodes[node]["pos"] = np.array([mg[0][node], mg[1][node]])

    # add diagonal edges
    g.add_edges_from(
        [((x, y), (x + 1, y + 1)) for x in range(nx - 1) for y in range(ny - 1)]
        + [
            ((x + 1, y), (x, y + 1))
            for x in range(nx - 1)
            for y in range(ny - 1)
        ]
    )

    # turn into directed graph
    dg = networkx.DiGraph(g)
    for u, v in g.edges():
        d = np.sqrt(np.sum((g.nodes[u]["pos"] - g.nodes[v]["pos"]) ** 2))
        dg.edges[u, v]["len"] = d
        dg.edges[u, v]["vdiff"] = g.nodes[u]["pos"] - g.nodes[v]["pos"]
        dg.add_edge(v, u)
        dg.edges[v, u]["len"] = d
        dg.edges[v, u]["vdiff"] = g.nodes[v]["pos"] - g.nodes[u]["pos"]

    return dg


def prepend_node_index(graph, new_index):
    # Relabel node indices in graph, insert (graph_level, i, j)
    ijk = [tuple((new_index,) + x) for x in graph.nodes]
    to_mapping = dict(zip(graph.nodes, ijk))
    return networkx.relabel_nodes(graph, to_mapping, copy=True)


def main():
    parser = ArgumentParser(description="Graph generation arguments")
    parser.add_argument(
        "--dataset_low",
        type=str,
        default="ERA5/2017/samples_60x60",
        help="Dataset to load grid point coordinates from "
        "(default: meps_example)",
    )
    parser.add_argument(
        "--dataset_high",
        type=str,
        default="CERRA",
        help="Dataset to load grid point coordinates from "
        "(default: meps_example)",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default="hierarchical_masked",
        help="Name to save graph as (default: multiscale)",
    )
    parser.add_argument(
        "--plot",
        type=int,
        default=1,
        help="If graphs should be plotted during generation "
        "(default: 0 (false))",
    )
    parser.add_argument(
        "--levels",
        type=int,
        help="Limit multi-scale mesh to given number of levels, "
        "from bottom up (default: None (no limit))",
    )
    parser.add_argument(
        "--hierarchical",
        type=int,
        default=1,
        help="Generate hierarchical mesh graph (default: 0, no)",
    )
    args = parser.parse_args()
    
    graph_dir_path = os.path.join("graphs", args.graph)
    os.makedirs(graph_dir_path, exist_ok=True)
    
    # Load grid positions low resolution
    static_dir_path_low = os.path.join("data", args.dataset_low, "static")
    xy_low = np.load(os.path.join(static_dir_path_low, "nwp_xy_base.npy"))
    
    # Load grid positions high resolution
    static_dir_path_high = os.path.join("data", args.dataset_high, "static")
    xy_high = np.load(os.path.join(static_dir_path_high, "nwp_xy_base.npy"))

    grid_xy_low = torch.tensor(xy_low)
    pos_max_low = torch.max(torch.abs(grid_xy_low))
    
    grid_xy_high = torch.tensor(xy_high)
    pos_max_high = torch.max(torch.abs(grid_xy_high))
    
    n = xy_low.shape[1]
    g_low = mk_2d_graph(xy_low, n, n)
    
    Ny, Nx = xy_high.shape[1:]
    _, G_grid_masked = mk_2d_graph_with_masking(xy_high, Ny, Nx, 10, 50, seed=None)
    
    plot_graph_nodes_only_2graph(from_networkx(G_grid_masked), from_networkx(g_low), title=f"era5_CERRA")
    plt.show()
    plt.savefig(f"era5_cerra.png")
    
    
if __name__ == "__main__":
    main()
