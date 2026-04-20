# ============================================================
# Rainfall–Runoff LSTM Model (IMPROVED)
# ===========================================================
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt
from helper_functions import create_sequences, create_model, imbalance_score, compute_stats, is_match

# ============================================================
# TensorFlow threads
# ============================================================
tf.config.threading.set_intra_op_parallelism_threads(20)
# ============================================================
# Load data
# ============================================================
time_start= "2004-09-01 00:00:00"
time_end = "2015-08-31 23:00:00"

precipitation = pd.read_pickle( "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/mean_h_precipitation_20040901_20150831_cumulative_by_basin.pkl" ).loc[time_start:time_end]
area_data = pd.read_csv( "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/basin_stats.csv" )
runoff = pd.read_pickle( "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/list_q.pkl" )

# ============================================================
# Build basin matrices
# ============================================================
basin_matrices = {}
for basin in precipitation.columns:
    if basin == "PoPontelagoscuro":
        print(f"Skipping PoPontelagoscuro")
        continue
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

print("Basins used:", len(basin_matrices))
# ============================================================
# Split basins into train/val/test
# ============================================================

basin_list = list(basin_matrices.keys())

# Precompute basin runoff arrays
basin_data = {
    b: basin_matrices[b]["runoff"].values for b in basin_list
}

# Remove empty basins (IMPORTANT)
basin_data = {b: v for b, v in basin_data.items() if len(v) > 0}

# Compute means safely
basin_means = {b: np.mean(v) for b, v in basin_data.items() if len(v) > 0}

# Sort basins (largest signal first)
sorted_basins = sorted(basin_means.keys(), key=lambda b: basin_means[b], reverse=True)

# Initialize splits
splits = {
    "train": [],
    "val": [],
    "test": []
}

# Greedy assignment
for basin in sorted_basins:

    best_split = None
    best_score = np.inf

    for split_name in splits.keys():

        # Try assigning basin temporarily
        temp_splits = {k: v.copy() for k, v in splits.items()}
        temp_splits[split_name].append(basin)

        score = imbalance_score(temp_splits)

        if np.isnan(score):
            continue

        if score < best_score:
            best_score = score
            best_split = split_name

    # Safety check
    if best_split is None:
        raise ValueError(f"No valid split found for basin {basin}")

    splits[best_split].append(basin)


# Final splits
train_basins = splits["train"]
val_basins = splits["val"]
test_basins = splits["test"]

print("Train mean/std:", compute_stats(train_basins))
print("Val mean/std:",  compute_stats(val_basins))
print("Test mean/std:", compute_stats(test_basins))


# Build DataFrames
train_data = pd.concat([basin_matrices[b] for b in train_basins])
val_data = pd.concat([basin_matrices[b] for b in val_basins])
test_data = pd.concat([basin_matrices[b] for b in test_basins])

print(f"Basins used: {len(sorted_basins)}")

# ============================================================
# Scaling
# ============================================================
scaler = StandardScaler()
train_scaled = train_data.copy()
train_scaled[["precipitation","Corr_time", "mean_H", "runoff"]] = scaler.fit_transform(train_data[["precipitation","Corr_time","mean_H", "runoff"]])

val_scaled = val_data.copy()
val_scaled[["precipitation","Corr_time","mean_H", "runoff"]] = scaler.transform(val_data[["precipitation","Corr_time","mean_H", "runoff"]])

test_scaled = test_data.copy()
test_scaled[["precipitation","Corr_time","mean_H", "runoff"]] = scaler.transform(test_data[["precipitation","Corr_time","mean_H", "runoff"]])

# ============================================================
# Create sequences
# ============================================================
X_train, y_train = create_sequences(train_scaled.reset_index(drop=True))
X_val, y_val = create_sequences(val_scaled.reset_index(drop=True))
X_test, y_test   = create_sequences(test_scaled.reset_index(drop=True))

print("X_train shape:", X_train.shape)

# ============================================================
# Train model
# ============================================================
early_stopping = EarlyStopping(
    monitor='val_loss',   # usually what you want
    patience=50,          # epochs to wait before stopping
    restore_best_weights=True
)
model = create_model((X_train.shape[1], X_train.shape[2]))
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=30,
    callbacks=[early_stopping]
)

# store the model
model.save("/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/lstm_model.h5")

# plot training history
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('MAE over epochs')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.savefig("/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/training_history_new.png", dpi=300)
plt.close()
# ============================================================
# Evaluate
# ============================================================
y_pred = model.predict(X_test)

# Inverse scaling
runoff_mean = scaler.mean_[2]
runoff_std  = scaler.scale_[2]
y_pred_inv = y_pred* runoff_std + runoff_mean
y_test_inv = y_test * runoff_std + runoff_mean


# Define bins (last one open-ended)
bins = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 3000), (3000, np.inf)]

n_bins = len(bins)
ncols = 3
nrows = math.ceil(n_bins / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
axes = axes.flatten()

# Global limits for comparability
global_min = min(y_test_inv.min(), y_pred_inv.min())
global_max = max(y_test_inv.max(), y_pred_inv.max())

for i, (bmin, bmax) in enumerate(bins):
    if np.isinf(bmax):
        mask = (y_test_inv >= bmin)
        title = f">{bmin}"
    else:
        mask = (y_test_inv >= bmin) & (y_test_inv < bmax)
        title = f"{bmin}-{bmax}"

    ax = axes[i]

    if np.sum(mask) == 0:
        ax.set_title(f"{title}\n(no data)")
        ax.axis("off")
        continue

    # Scatter
    ax.scatter(y_test_inv[mask], y_pred_inv[mask], alpha=0.5)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test_inv[mask], y_pred_inv[mask]))
    bias = np.mean(y_pred_inv[mask] - y_test_inv[mask])
    n_points = np.sum(mask)

    # 1:1 line
    ax.plot([global_min, global_max], [global_min, global_max], 'r--')
    ax.set_xlim(global_min, global_max)
    ax.set_ylim(global_min, global_max)

    # Title + metrics
    ax.set_title(f"{title}\nN={n_points}, RMSE={rmse:.1f}, Bias={bias:.1f}")

    ax.set_xlabel("True runoff")
    ax.set_ylabel("Predicted runoff")

# Remove unused axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/pred_vs_true_binned_new.png", dpi=300)
plt.close()

