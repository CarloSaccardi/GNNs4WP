import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import argparse
from pathlib import Path

# Percentiles to compute
PERCENTILES = [0.01, 0.1, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 99.9, 99.99]

def load_npy_file(file_path):
    """Load a single .npy file."""
    return np.load(file_path)

def collect_npy_files(base_dir, region_name, data_type='CERRA'):
    """
    Collect all .npy files from train/test/val directories for a given region.
    """
    region_path = os.path.join(base_dir, region_name, data_type, "samples")
    subsets = ["train", "test", "val"]

    all_files = []
    for subset in subsets:
        subset_path = os.path.join(region_path, subset)
        if os.path.exists(subset_path):
            all_files.extend([
                os.path.join(subset_path, f)
                for f in os.listdir(subset_path)
                if f.endswith('.npy')
            ])

    if len(all_files) == 0:
        raise RuntimeError(f"No files found for region {region_name} in {data_type}/samples.")

    all_files.sort()
    return all_files

def compute_climatology_for_region(base_dir, region_name, data_type='CERRA', output_root='climatology'):
    """
    Compute mean, std, and percentiles for a region by combining all train/test/val .npy files.
    """
    print(f"Processing region: {region_name}")
    all_files = collect_npy_files(base_dir, region_name, data_type)

    num_workers = max(1, min(cpu_count() - 1, 16))  # Use up to 16 CPU cores
    with Pool(num_workers) as pool:
        data_list = list(tqdm(pool.imap(load_npy_file, all_files),
                              total=len(all_files), desc=f"Loading {region_name}"))

    # Stack along a new axis: shape (N, H, W, C)
    data_array = np.stack(data_list, axis=0)
    del data_list  

    print(f"[{region_name}] Computing mean, std, and percentiles...")
    mean_array = np.mean(data_array, axis=0)
    std_array = np.std(data_array, axis=0)

    percentiles = {}
    for p in PERCENTILES:
        percentiles[p] = np.percentile(data_array, p, axis=0)

    # Save results
    output_dir = os.path.join(output_root, region_name)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "mean.npy"), mean_array)
    np.save(os.path.join(output_dir, "std.npy"), std_array)
    for p, arr in percentiles.items():
        np.save(os.path.join(output_dir, f"percentile_{p}.npy"), arr)

    print(f"[{region_name}] Climatology saved in {output_dir}")
    return mean_array, std_array, percentiles

def process_all_regions(base_dir, regions, data_type='CERRA', output_root='climatology'):
    results = {}
    for region in regions:
        try:
            results[region] = compute_climatology_for_region(base_dir, region, data_type, output_root)
        except RuntimeError as e:
            print(f"Skipping {region}: {e}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=Path("data"),
        help="Base directory containing region data."
    )
    parser.add_argument(
        "--regions",
        nargs='+',
        default=["CentralEurope", "Iberia", "Scandinavia"],
        help="List of region names to process."
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="CERRA",
        help="Data type (e.g., CERRA, ERA5)."
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("climatology"),
        help="Directory to save climatology results."
    )
    args = parser.parse_args()

    process_all_regions(
        str(args.base_dir),
        args.regions,
        data_type=args.data_type,
        output_root=str(args.output_root)
    )