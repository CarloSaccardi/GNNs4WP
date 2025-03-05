import torch

def masking(grid_emb,pos_embed, g2m_edge_index, g2m_features, mask_ratio, grid_size, block_size):
    #### Masking ####
    grid_emb = grid_emb + pos_embed
    grid_emb, mask, ids_restore, ids_keep = block_random_masking(grid_emb, mask_ratio, grid_size, block_size)
    keep_uniques = torch.unique(ids_keep[0]) + g2m_edge_index[0,0]
    senders = g2m_edge_index[0]
    mask_edges = torch.isin(senders, keep_uniques)
    kept_indexes = torch.nonzero(mask_edges, as_tuple=True)[0]
    g2m_features = g2m_features[kept_indexes]
    #### Mask g2m edge index ####
    g2m_edge_index_mins = g2m_edge_index.min(dim=1, keepdim=True)[0]
    g2m_edge_index_max_1 = g2m_edge_index[1].max()
    g2m_edge_index = g2m_edge_index[:, mask_edges]
    g2m_edge_index = g2m_edge_index  - g2m_edge_index_mins
    num_rec = g2m_edge_index_max_1 + 1
    g2m_edge_index[0] = (
        g2m_edge_index[0] + num_rec
    )
    unique_senders = torch.unique(g2m_edge_index[0])
    sorted_senders = torch.sort(unique_senders).values
    new_senders = torch.arange(num_rec, num_rec + len(sorted_senders))
    sender_mapping = dict(zip(sorted_senders.tolist(), new_senders.tolist()))
    reindexed_senders = torch.tensor([sender_mapping[sender.item()] for sender in g2m_edge_index[0]])
    g2m_edge_index[0] = reindexed_senders
    
    return mask, ids_restore, g2m_features, g2m_edge_index


def block_random_masking(x, mask_ratio, grid_size, block_size):
    """
    Perform random masking on a flattened grid of nodes, grouped into blocks, based on the original 2D grid layout.

    Args:
        x: Tensor of shape (batch, num_nodes, latent_dim), e.g., (N, 90000, D).
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



def adjust_g2m_edge_index(g2m_edge_index):
    # Normalize and offset sender indices by the number of receivers.
    g2m_edge_index = g2m_edge_index - g2m_edge_index.min(dim=1, keepdim=True)[0]
    num_rec = int(g2m_edge_index[1].max().item() + 1)
    g2m_edge_index[0] += num_rec
    # Reindex senders to a continuous range starting at self.num_rec.
    _, g2m_edge_index[0] = torch.unique(g2m_edge_index[0], sorted=True, return_inverse=True)
    g2m_edge_index[0] += num_rec
    return g2m_edge_index