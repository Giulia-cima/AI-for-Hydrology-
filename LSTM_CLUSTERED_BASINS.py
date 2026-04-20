# ============================================================
# Rainfall–Runoff LSTM Model (IMPROVED)
# ============================================================

import pandas as pd
import numpy as np
import tensorflow as tf
from helper_functions import is_match, train_cluster

# ============================================================
# Load data
# ============================================================

tf.config.threading.set_intra_op_parallelism_threads(20)
time_start= "2004-09-01 00:00:00"
time_end = "2015-08-31 23:00:00"

precipitation = pd.read_pickle( "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/mean_h_precipitation_20040901_20150831_cumulative_by_basin.pkl" ).loc[time_start:time_end]
area_data = pd.read_csv( "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/basin_stats.csv" )
runoff = pd.read_pickle( "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/list_q.pkl" )

area_bins = [0, 10, 100, 1000, 10000, np.inf]
basin_matrices = {}
basin_clusters = {i: [] for i in range(len(area_bins) - 1)}
# ============================================================
# Build basin matrices
# ============================================================

for basin in precipitation.columns:
    basin_norm = basin.lower().replace(" ", "_")
    # remove any special characters
    basin_norm = basin_norm.replace("à", "a")
    basin_norm = basin_norm.replace("è", "e")
    basin_norm = basin_norm.replace("é", "e")

    if basin_norm not in [r["basin"].replace(" ", "_").lower() for r in runoff]:
        basin_norm = is_match(basin_norm, runoff)
        if not basin_norm:
            print(f"Runoff data missing for basin: {basin}")
            continue
    print (f"Processing basin: {basin}")
    try:
        idx = area_data[area_data["basin"] == basin.lower()].index[0]
    except:
        print(f"{basin}")
    area_val = area_data['area_km2'].loc[idx]
    mean_H = area_data['mean_H'].loc[idx]
    # --- get runoff series ---
    r_values = next(
        r["discharge"]
        for r in runoff
        if r["basin"].replace(" ", "_").lower() == basin_norm
    )
    # IMPORTANT: assume runoff_list also stores time
    r_time = next(
        r["date"]
        for r in runoff
        if r["basin"].replace(" ", "_").lower() == basin_norm
    )
    runoff_series = pd.Series(
        r_values,
        index=pd.to_datetime(r_time)
    )
    # --- align runoff to precipitation index ---
    runoff_aligned = runoff_series.reindex(precipitation.index)
    # --- build dataframe safely ---
    df = pd.DataFrame({
        "precipitation": precipitation[basin].values,
        "Corr_time": 1.085*np.sqrt(area_val),
        "mean_H": mean_H,
        "runoff": runoff_aligned.values
    }, index=precipitation.index)
    df.loc[df["precipitation"] < 0, "precipitation"] = 0
    df.loc[df["runoff"] < 0, "runoff"] = 0
    df.dropna(inplace=True)
    basin_matrices[basin] = df
    # -------------------------
    # ASSIGN CLUSTER
    # -------------------------
    for i in range(len(area_bins) - 1):
        if area_bins[i] < area_val <= area_bins[i + 1]:
            basin_clusters[i].append(basin)
            break

print("Basins used:", len(basin_matrices))


for i in basin_clusters:
    print(f"Cluster {i} ({area_bins[i]} - {area_bins[i+1]} km²): {len(basin_clusters[i])} basins")


cluster_models = {}
cluster_histories = {}
cluster_scalers = {}

for cluster_id, basins in basin_clusters.items():

    if len(basins) == 0:
        print(f"Cluster {cluster_id} is empty → skipped")
        continue

    model, history, scaler = train_cluster(cluster_id, basins,basin_matrices)

    cluster_models[cluster_id] = model
    cluster_histories[cluster_id] = history
    cluster_scalers[cluster_id] = scaler

print("\n==============================")
print("CLUSTER SUMMARY")
print("==============================")

for cid in cluster_models:
    print(f"Cluster {cid}: trained")