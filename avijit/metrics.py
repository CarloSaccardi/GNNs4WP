# Calculate metrics 

import os
import numpy as np
# Import pySTEPS verification modules
from pysteps.verification import detcontscores, detcatscores, spatialscores, probscores

# Base directories
base_dir = "/path/to/Avijit"  # replace with actual base path
results_dir = os.path.join(base_dir, "RESULTS")
metrics_dir = os.path.join(base_dir, "Metrics_Results")
climatology_dir = os.path.join(base_dir, "climatology")  # assumed climatology results directory

# Define variable names in order of channels (if known)
var_names = ["u10", "v10", "t2m", "sshf", "zust"]  # adjust if needed

# Regions to process (subdirectories in RESULTS)
regions = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]

# Percentile levels to use for thresholds
perc_levels = [25, 50, 75]

for region in regions:
    region_path = os.path.join(results_dir, region)
    # Ensure output directory for region exists
    os.makedirs(os.path.join(metrics_dir, region), exist_ok=True)
    print(f"Processing region: {region}")
    # Identify model subfolders (exclude "Target")
    subdirs = [d for d in os.listdir(region_path) if os.path.isdir(os.path.join(region_path, d))]
    model_names = [d for d in subdirs if d.lower() != "target"]
    # Load list of target files (assuming .npy files) and sort them
    target_dir = os.path.join(region_path, "Target")
    target_files = sorted([f for f in os.listdir(target_dir) if f.endswith(".npy")])
    # Pre-compute thresholds for each variable using climatology or target data
    thresholds = {}  # thresholds[var_name] = {25: thr25_value, 50: thr50_value, 75: thr75_value}
    for idx, var in enumerate(var_names):
        thr_values = {}
        # Try to load climatology percentiles for this region & variable (if available)
        climatology_file = os.path.join(climatology_dir, f"{region}_{var}_climatology.npy")
        if os.path.exists(climatology_file):
            clim_data = np.load(climatology_file)  # assume this is a 2D array or similar
            # Compute spatial mean of percentile maps if climatology data contains percentiles
            # If climatology file is a single array, interpret it as the mean or median field (which may not directly give percentiles)
            # Here we assume we might have stored percentile fields separately; adjust as needed.
            # For demonstration, we'll compute percentiles from the climatology field's distribution:
            all_values = clim_data.flatten()
            thr_values[25] = np.nanmean(np.percentile(all_values, 25))
            thr_values[50] = np.nanmean(np.percentile(all_values, 50))
            thr_values[75] = np.nanmean(np.percentile(all_values, 75))
        else:
            # If no climatology file, compute percentiles from all target data for this variable
            all_values = []
            for fname in target_files:
                data = np.load(os.path.join(target_dir, fname))
                # Extract the variable channel
                var_field = data[..., idx]  # shape (H, W)
                all_values.append(var_field.flatten())
            if all_values:
                all_values = np.concatenate(all_values)
                thr_values[25] = np.nanmean(np.percentile(all_values, 25))
                thr_values[50] = np.nanmean(np.percentile(all_values, 50))
                thr_values[75] = np.nanmean(np.percentile(all_values, 75))
            else:
                # No data, set thresholds to None or 0 as default
                thr_values[25] = None
                thr_values[50] = None
                thr_values[75] = None
        thresholds[var] = thr_values
        # Create variable directory in output
        var_out_dir = os.path.join(metrics_dir, region, var)
        os.makedirs(var_out_dir, exist_ok=True)
    # Initialize storage for metrics for each model, variable, and metric type
    metrics_data = {}  # metrics_data[model][var][category] = dict of metrics lists
    for model in model_names:
        metrics_data[model] = {}
        for var in var_names:
            metrics_data[model][var] = {
                "continuous": {},             # to store continuous metrics lists
                "categorical": {p: {} for p in perc_levels},  # store categorical metrics for each threshold
                "spatial": {p: {} for p in perc_levels},      # store spatial metrics for each threshold
                "probabilistic": {}           # store probabilistic metrics (e.g. CRPS)
            }
    # Determine metrics names (for continuous and categorical) by computing on the first file as reference
    if target_files:
        sample_tgt = np.load(os.path.join(target_dir, target_files[0]))
        for model in model_names:
            sample_pred = np.load(os.path.join(region_path, model, target_files[0]))
            # If ensemble (4D array), reduce to mean for sample continuous metrics determination
            sample_pred_det = sample_pred
            if sample_pred.ndim == 4:
                # average across ensemble members for deterministic comparison
                sample_pred_det = np.nanmean(sample_pred, axis=0)
            # Compute sample metrics for each variable to get metric keys
            for idx, var in enumerate(var_names):
                # Prepare sample arrays
                obs_field = sample_tgt[..., idx]
                pred_field = sample_pred_det[..., idx]
                # Continuous metrics keys
                cont_res = detcontscores.det_cont_fct(pred_field, obs_field, scores="")  # all scores
                if not metrics_data[model][var]["continuous"]:
                    # initialize continuous metrics lists with keys
                    for mname in cont_res.keys():
                        metrics_data[model][var]["continuous"][mname] = []
                # Categorical metrics keys for each threshold
                for p in perc_levels:
                    thr_val = thresholds[var][p]
                    if thr_val is not None:
                        cat_res = detcatscores.det_cat_fct(pred_field, obs_field, thr=thr_val, scores="")
                    else:
                        # If threshold is None, skip
                        cat_res = {}
                    if not metrics_data[model][var]["categorical"][p]:
                        for mname in cat_res.keys():
                            metrics_data[model][var]["categorical"][p][mname] = []
                # Spatial metrics keys for each threshold (FSS for each window size)
                for p in perc_levels:
                    if thresholds[var][p] is None:
                        continue
                    # Define window sizes
                    for win in [2, 5, 10, 25]:
                        key = f"FSS_scale{win}"
                        if key not in metrics_data[model][var]["spatial"][p]:
                            metrics_data[model][var]["spatial"][p][key] = []
                # Probabilistic metrics keys (if ensemble)
                if sample_pred.ndim == 4:
                    if "CRPS" not in metrics_data[model][var]["probabilistic"]:
                        metrics_data[model][var]["probabilistic"]["CRPS"] = []
            # Only need one model's metric keys initialization (assuming all models have same keys for cont/cat)
            break

    # Loop through each datetime file and compute metrics
    for fname in target_files:
        target_data = np.load(os.path.join(target_dir, fname))  # shape (H, W, C)
        for model in model_names:
            model_file = os.path.join(region_path, model, fname)
            if not os.path.exists(model_file):
                # If a model is missing this file, skip it for this date
                continue
            pred_data = np.load(model_file)
            # If ensemble forecast (4D array: M x H x W x C)
            is_ensemble = (pred_data.ndim == 4)
            # For deterministic metrics, if ensemble present, use ensemble mean as deterministic forecast
            if is_ensemble:
                pred_data_det = np.nanmean(pred_data, axis=0)  # shape (H, W, C)
            else:
                pred_data_det = pred_data  # shape (H, W, C)
            # Loop over each variable channel
            for idx, var in enumerate(var_names):
                obs_field = target_data[..., idx]
                pred_field = pred_data_det[..., idx]
                # Compute continuous metrics (if deterministic)
                if not is_ensemble:
                    cont_scores = detcontscores.det_cont_fct(pred_field, obs_field, scores="")
                    for mname, val in cont_scores.items():
                        metrics_data[model][var]["continuous"][mname].append(val)
                else:
                    # Optionally, we could also evaluate ensemble mean like above if needed.
                    # Here we focus on probabilistic metrics for ensemble.
                    pass
                # Compute categorical and spatial metrics for each threshold
                for p in perc_levels:
                    thr_val = thresholds[var][p]
                    if thr_val is None:
                        continue  # skip if threshold not defined
                    # Categorical metrics (dichotomous yes/no above threshold)
                    if not is_ensemble:
                        cat_scores = detcatscores.det_cat_fct(pred_field, obs_field, thr=thr_val, scores="")
                        for mname, val in cat_scores.items():
                            metrics_data[model][var]["categorical"][p][mname].append(val)
                    else:
                        # Could compute categorical on ensemble mean or probabilistic metrics for event; skipping here.
                        pass
                    # Spatial metrics (FSS at various scales)
                    # For ensemble, we could also use ensemble mean or each member; use mean for spatial deterministic comparison
                    fss_pred_field = pred_field if not is_ensemble else pred_data_det[..., idx]
                    for win in [2, 5, 10, 25]:
                        fss_val = spatialscores.fss(fss_pred_field, obs_field, thr=thr_val, scale=win)
                        metrics_data[model][var]["spatial"][p][f"FSS_scale{win}"].append(fss_val)
                # Probabilistic metric (CRPS) for ensemble forecasts
                if is_ensemble:
                    # Compute CRPS for the ensemble vs observation for this variable
                    ens_members = pred_data[..., idx]  # shape (M, H, W)
                    crps_val = probscores.CRPS(ens_members, obs_field)
                    metrics_data[model][var]["probabilistic"]["CRPS"].append(crps_val)
    # End of file loop

    # Save metrics to files for each model & variable
    for model in model_names:
        for var in var_names:
            var_out_dir = os.path.join(metrics_dir, region, var)
            # Continuous metrics file (only if we collected any)
            if metrics_data[model][var]["continuous"]:
                cont_file = os.path.join(var_out_dir, f"continuous_{model}.npy")
                np.save(cont_file, metrics_data[model][var]["continuous"], allow_pickle=True)
            # Categorical metrics files for each threshold
            for p in perc_levels:
                if metrics_data[model][var]["categorical"][p]:
                    cat_file = os.path.join(var_out_dir, f"categorical_{model}_thr{p}p.npy")
                    np.save(cat_file, metrics_data[model][var]["categorical"][p], allow_pickle=True)
            # Spatial metrics files for each threshold
            for p in perc_levels:
                if metrics_data[model][var]["spatial"][p]:
                    spatial_file = os.path.join(var_out_dir, f"spatial_{model}_thr{p}p.npy")
                    np.save(spatial_file, metrics_data[model][var]["spatial"][p], allow_pickle=True)
            # Probabilistic metrics file (if any data collected)
            if metrics_data[model][var]["probabilistic"]:
                prob_file = os.path.join(var_out_dir, f"probabilistic_{model}.npy")
                np.save(prob_file, metrics_data[model][var]["probabilistic"], allow_pickle=True)
    print(f"Metrics saved for region {region}.")
print("All regions processed.")
