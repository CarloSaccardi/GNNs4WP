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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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


def load_grid(dataset):
    """
    Load the grid positions from a given dataset directory.
    Returns:
    - xy: the numpy array from nwp_xy.npy,
    - grid_xy: the torch tensor version of xy,
    - pos_max: the maximum absolute value in grid_xy.
    """
    static_dir = os.path.join("data", dataset, "static")
    xy = np.load(os.path.join(static_dir, "nwp_xy.npy"))
    grid_xy = torch.tensor(xy)
    pos_max = torch.max(torch.abs(grid_xy))
    return xy, pos_max


def main():
    parser = ArgumentParser(description="Graph generation arguments")
    parser.add_argument(
        "--dataset_low",
        type=str,
        default=None, #"/aspire/CarloData/MASK_GNN_DATA/ERA5_60_n2_40_18",
        help="Dataset to load grid point coordinates from "
        "(default: meps_example)",
    )
    parser.add_argument(
        "--dataset_high",
        type=str,
        default="/aspire/CarloData/MASK_GNN_DATA/CERRA_interpolated_300x300",
        help="Dataset to load grid point coordinates from "
        "(default: meps_example)",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default="hierarchical_highRes_only_fewLayers",
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
    parser.add_argument(
        "--lowRes_grid_features",
        type=str,
        default="/aspire/CarloData/MASK_GNN_DATA/ERA5_60_n2_40_18/static/grid_features.pt",
        help="path to .pt file containing grid features for low resolution grid",
    )
    args = parser.parse_args()
    
    graph_dir_path = os.path.join("graphs", args.graph)
    os.makedirs(graph_dir_path, exist_ok=True)
    

    # Load datasets if provided.
    xy_low, pos_max_low = (None, None)
    if args.dataset_low is not None:
        assert args.lowRes_grid_features, "Path to low resolution grid features must be provided"
        xy_low, pos_max_low = load_grid(args.dataset_low)

    xy_high, pos_max_high = (None, None)
    if args.dataset_high is not None:
        xy_high, pos_max_high = load_grid(args.dataset_high)

    # Ensure that at least one dataset is provided.
    if xy_low is None and xy_high is None:
        raise ValueError("At least one of dataset_low or dataset_high must be provided.")

    # Determine the reference dataset based on availability.
    # - If both are provided, we use high-res as reference and combine pos_max.
    # - If only one is provided, that one becomes the reference.
    if xy_low is not None and xy_high is not None:
        xy_ref = xy_high
        pos_max_ref = max(pos_max_low, pos_max_high)
    elif xy_high is not None:
        xy_ref = xy_high
        pos_max_ref = pos_max_high
    else:
        xy_ref = xy_low
        pos_max_ref = pos_max_low

    #
    # Mesh geometry computation
    #
    nx = 3  # number of children per node (resulting in nx**2 leaves)
    nlev = int(np.log(max(xy_ref.shape)) / np.log(nx))
    nleaf = nx ** nlev  # total number of leaves

    # Limit the number of mesh levels if args.levels is provided.
    mesh_levels = nlev if not args.levels else min(nlev, args.levels)

    print(f"nlev: {nlev}, nleaf: {nleaf}, mesh_levels: {mesh_levels}")
    print(f"pos_max: {pos_max_ref}")

    #
    # Build multi-resolution mesh graphs using the reference grid.
    #
    G = []

    for lev in range(1, mesh_levels-1): #for lev in range(1, mesh_levels + 1):
        # Update n for the next level based on the total leaves and current level
        n = int(nleaf / (nx ** lev))
        # Create mesh graph
        g = mk_2d_graph(xy_ref, n, n)
        if args.plot:
            plot_graph(from_networkx(g), title=f"Mesh graph, level {lev}")
            plt.show()
            plt.savefig(f"mesh_level_{lev}.png")
        G.append(g)

    if args.hierarchical:
        # Relabel nodes of each level with level index first
        G = [
            prepend_node_index(graph, level_i)
            for level_i, graph in enumerate(G)
        ]

        num_nodes_level = np.array([len(g_level.nodes) for g_level in G])
        # First node index in each level in the hierarchical graph
        first_index_level = np.concatenate(
            (np.zeros(1, dtype=int), np.cumsum(num_nodes_level[:-1]))
        )

        # Create inter-level mesh edges
        up_graphs = []
        down_graphs = []
        for from_level, to_level, G_from, G_to, start_index in zip(
            range(1, mesh_levels),
            range(0, mesh_levels - 1),
            G[1:],
            G[:-1],
            first_index_level[: mesh_levels - 1],
        ):
            # start out from graph at from level
            G_down = G_from.copy()
            G_down.clear_edges()
            G_down = networkx.DiGraph(G_down)

            # Add nodes of to level
            G_down.add_nodes_from(G_to.nodes(data=True))

            # build kd tree for mesh point pos
            # order in vm should be same as in vm_xy
            v_to_list = list(G_to.nodes)
            v_from_list = list(G_from.nodes)
            v_from_xy = np.array([xy for _, xy in G_from.nodes.data("pos")])
            kdt_m = scipy.spatial.KDTree(v_from_xy)

            # add edges from mesh to grid
            for v in v_to_list:
                # find 1(?) nearest neighbours (index to vm_xy)
                neigh_idx = kdt_m.query(G_down.nodes[v]["pos"], 1)[1]
                u = v_from_list[neigh_idx]

                # add edge from mesh to grid
                G_down.add_edge(u, v)
                d = np.sqrt(
                    np.sum(
                        (G_down.nodes[u]["pos"] - G_down.nodes[v]["pos"]) ** 2
                    )
                )
                G_down.edges[u, v]["len"] = d
                G_down.edges[u, v]["vdiff"] = (
                    G_down.nodes[u]["pos"] - G_down.nodes[v]["pos"]
                )

            # relabel nodes to integers (sorted)
            G_down_int = networkx.convert_node_labels_to_integers(
                G_down, first_label=start_index, ordering="sorted"
            )  # Issue with sorting here
            G_down_int = sort_nodes_internally(G_down_int)
            pyg_down = from_networkx_with_start_index(G_down_int, start_index)

            # Create up graph, invert downwards edges
            up_edges = torch.stack(
                (pyg_down.edge_index[1], pyg_down.edge_index[0]), dim=0
            )
            pyg_up = pyg_down.clone()
            pyg_up.edge_index = up_edges

            up_graphs.append(pyg_up)
            down_graphs.append(pyg_down)

            if args.plot:
                plot_graph(
                    pyg_down, title=f"Down graph, {from_level} -> {to_level}"
                )
                plt.show()
                plt.savefig(f"down_level_{from_level}_to_{to_level}.png")

                plot_graph(
                    pyg_down, title=f"Up graph, {to_level} -> {from_level}"
                )
                plt.show()
                plt.savefig(f"up_level_{to_level}_to_{from_level}.png")

        # Save up and down edges
        save_edges_list(up_graphs, "mesh_up", graph_dir_path)
        save_edges_list(down_graphs, "mesh_down", graph_dir_path)

        # Extract intra-level edges for m2m
        m2m_graphs = [
            from_networkx_with_start_index(
                networkx.convert_node_labels_to_integers(
                    level_graph, first_label=start_index, ordering="sorted"
                ),
                start_index,
            )
            for level_graph, start_index in zip(G, first_index_level)
        ]

        mesh_pos = [graph.pos.to(torch.float32) for graph in m2m_graphs]

        # For use in g2m and m2g
        G_bottom_mesh = G[0]

        joint_mesh_graph = networkx.union_all([graph for graph in G])
        all_mesh_nodes = joint_mesh_graph.nodes(data=True)

    else:
        # combine all levels to one graph
        G_tot = G[0]
        for lev in range(1, len(G)):
            nodes = list(G[lev - 1].nodes)
            n = int(np.sqrt(len(nodes)))
            ij = (
                np.array(nodes)
                .reshape((n, n, 2))[1::nx, 1::nx, :]
                .reshape(int(n / nx) ** 2, 2)
            )
            ij = [tuple(x) for x in ij]
            G[lev] = networkx.relabel_nodes(G[lev], dict(zip(G[lev].nodes, ij)))
            G_tot = networkx.compose(G_tot, G[lev])

        # Relabel mesh nodes to start with 0
        G_tot = prepend_node_index(G_tot, 0)

        # relabel nodes to integers (sorted)
        G_int = networkx.convert_node_labels_to_integers(
            G_tot, first_label=0, ordering="sorted"
        )

        # Graph to use in g2m and m2g
        G_bottom_mesh = G_tot
        all_mesh_nodes = G_tot.nodes(data=True)

        # export the nx graph to PyTorch geometric
        pyg_m2m = from_networkx(G_int)
        m2m_graphs = [pyg_m2m]
        mesh_pos = [pyg_m2m.pos.to(torch.float32)]

        if args.plot:
            plot_graph(pyg_m2m, title="Mesh-to-mesh")
            plt.show()
            plt.savefig("mesh_to_mesh.png")

    # Save m2m edges
    save_edges_list(m2m_graphs, "m2m", graph_dir_path)

    # Divide mesh node pos by max coordinate of grid cell
    mesh_pos = [pos / pos_max_ref for pos in mesh_pos]
    #replace first mesh features with low resolution grid features. This tensor has surface geopotential and position
    if args.dataset_low is not None and args.dataset_high is not None:
        ValueError("double check the dimensions of the meshes. The first mesh has to be dimension of era5, secon mesh has to be smaller than era5")
        firstMesh_features = torch.load(args.lowRes_grid_features)
        mesh_pos = [firstMesh_features] + mesh_pos[1:]

    # Save mesh positions
    torch.save(
        mesh_pos, os.path.join(graph_dir_path, "mesh_features.pt")
    )  # mesh pos, in float32

    #
    # Grid2Mesh
    #

    # radius within which grid nodes are associated with a mesh node
    # (in terms of mesh distance)
    DM_SCALE = 0.67

    # mesh nodes on lowest level
    vm = G_bottom_mesh.nodes
    vm_xy = np.array([xy for _, xy in vm.data("pos")])
    # distance between mesh nodes
    dm = np.sqrt(
        np.sum((vm.data("pos")[(0, 1, 0)] - vm.data("pos")[(0, 0, 0)]) ** 2)
    )

    # grid nodes
    Ny, Nx = xy_ref.shape[1:]
    #G_grid, G_grid_masked = mk_2d_graph_with_masking(xy_ref, Ny, Nx, 10, 50, seed=None)

    G_grid = networkx.grid_2d_graph(Ny, Nx)
    G_grid.clear_edges()
        # vg features (only pos introduced here)
    for node in G_grid.nodes:
        # pos is in feature but here explicit for convenience
        G_grid.nodes[node]["pos"] = np.array([xy_ref[0][node], xy_ref[1][node]])

    
    if args.plot:
        if args.dataset_low is not None and args.dataset_high is not None:
            plot_graph_nodes_only_2graph(from_networkx(G_grid), from_networkx(G[0]), title=f"grid to mesh")
            plt.show()
            plt.savefig(f"gird2mesh.png")
        
        plot_graph_nodes_only(from_networkx(G_grid), title=f"Mesh graph, complete")
        plt.show()
        plt.savefig(f"complete_mesh.png")
    
    
    # add 1000 to node key to separate grid nodes (1000,i,j) from mesh nodes
    # (i,j) and impose sorting order such that vm are the first nodes
    G_grid = prepend_node_index(G_grid, 1000)

    # build kd tree for grid point pos
    # order in vg_list should be same as in vg_xy
    vg_list = list(G_grid.nodes)
    vg_xy = np.array([[xy_ref[0][node[1:]], xy_ref[1][node[1:]]] for node in vg_list])
    kdt_g = scipy.spatial.KDTree(vg_xy)

    # now add (all) mesh nodes, include features (pos)
    G_grid.add_nodes_from(all_mesh_nodes)

    # Re-create graph with sorted node indices
    # Need to do sorting of nodes this way for indices to map correctly to pyg
    G_g2m = networkx.Graph()
    G_g2m.add_nodes_from(sorted(G_grid.nodes(data=True)))

    # turn into directed graph
    G_g2m = networkx.DiGraph(G_g2m)

    # add edges
    for v in vm:
        # find neighbours (index to vg_xy)
        neigh_idxs = kdt_g.query_ball_point(vm[v]["pos"], dm * DM_SCALE)
        for i in neigh_idxs:
            u = vg_list[i]
            # add edge from grid to mesh
            G_g2m.add_edge(u, v)
            d = np.sqrt(
                np.sum((G_g2m.nodes[u]["pos"] - G_g2m.nodes[v]["pos"]) ** 2)
            )
            G_g2m.edges[u, v]["len"] = d
            G_g2m.edges[u, v]["vdiff"] = (
                G_g2m.nodes[u]["pos"] - G_g2m.nodes[v]["pos"]
            )

    pyg_g2m = from_networkx(G_g2m)

    if args.plot:
        plot_graph(pyg_g2m, title="Grid-to-mesh")
        plt.show()

    #
    # Mesh2Grid
    #

    # start out from Grid2Mesh and then replace edges
    G_m2g = G_g2m.copy()
    G_m2g.clear_edges()

    # build kd tree for mesh point pos
    # order in vm should be same as in vm_xy
    vm_list = list(vm)
    kdt_m = scipy.spatial.KDTree(vm_xy)

    # add edges from mesh to grid
    for v in vg_list:
        # find 4 nearest neighbours (index to vm_xy)
        neigh_idxs = kdt_m.query(G_m2g.nodes[v]["pos"], 4)[1]
        for i in neigh_idxs:
            u = vm_list[i]
            # add edge from mesh to grid
            G_m2g.add_edge(u, v)
            d = np.sqrt(
                np.sum((G_m2g.nodes[u]["pos"] - G_m2g.nodes[v]["pos"]) ** 2)
            )
            G_m2g.edges[u, v]["len"] = d
            G_m2g.edges[u, v]["vdiff"] = (
                G_m2g.nodes[u]["pos"] - G_m2g.nodes[v]["pos"]
            )

    # relabel nodes to integers (sorted)
    G_m2g_int = networkx.convert_node_labels_to_integers(
        G_m2g, first_label=0, ordering="sorted"
    )
    pyg_m2g = from_networkx(G_m2g_int)

    if args.plot:
        plot_graph(pyg_m2g, title="Mesh-to-grid")
        plt.show()

    # Save g2m and m2g everything
    # g2m
    save_edges(pyg_g2m, "g2m", graph_dir_path)
    # m2g
    save_edges(pyg_m2g, "m2g", graph_dir_path)

if __name__ == "__main__":
    main()
