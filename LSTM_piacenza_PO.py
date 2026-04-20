import numpy as np
import pandas as pd
import os
import re
import math
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from helper_functions import create_sequences, custom_loss, stats

# ============================================================
# ============================================================
# SETTINGS
# ============================================================
shp_path = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/MASKS_SHAPE_PO/Piacenza_Po_catchment.shp"
dem_path = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/DEM_ITALIA/italy_dem_merged_COMPLETE.tif"
shp_name = "Piacenza_Po_catchment.shp"
dem_path = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/DEM_ITALIA/italy_dem_merged_COMPLETE.tif"
time_start = "2004-09-01 00:00:00"
time_end = "2015-08-31 23:00:00"

SEQ = 72
basin_name = os.path.splitext(shp_name)[0].replace("_catchment", "")
basin_name = re.sub(r'(?<=[a-z])(?=[A-Z])', '_', basin_name)
parts = basin_name.split('_')
if len(parts) > 1:
   basin_name = f"{parts[-1]}_{parts[0]}"
basin_name = basin_name.replace(" ", "_").replace("_", "")
basin_name = basin_name.lower()

# ============================================================
# LOAD INPUT DATA
# ============================================================
precipitation = pd.read_pickle(
    "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/mean_h_precipitation_20040901_20150831_cumulative_by_basin.pkl").loc[time_start:time_end]

runoff_list = pd.read_pickle("/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/list_q.pkl")

gdf = gpd.read_file(shp_path)

# FIX: reproject BEFORE buffer
gdf = gdf.to_crs("EPSG:32632")
gdf["geometry"] = gdf.buffer(0)

# ensure validity
gdf = gdf[gdf.is_valid]

area_m2 = gdf.geometry.area.sum()
area_km2 = area_m2 / 1e6


with rasterio.open(dem_path) as src:

    # align CRS to DEM
    gdf_dem = gdf.to_crs(src.crs)

    # clip DEM
    clipped, transform = mask(
        src,
        gdf_dem.geometry,
        crop=True,
        filled=True,
        nodata=np.nan
    )

    dem_clip = clipped[0]

    # valid values (robust)
    valid = dem_clip[np.isfinite(dem_clip)]

    if valid.size == 0:
        raise ValueError("Clipped DEM is empty — check CRS or geometry alignment")

    mean_H = float(np.mean(valid))
    max_H = float(np.max(valid))

print("Area km2:", area_km2)
print("Mean elevation:", mean_H)
print("Max elevation:", max_H)


# ============================================================
# MATCH BASIN
# ============================================================
for basin in precipitation.columns:
    basin_norm = basin.lower().replace(" ", "_")
    # remove any special characters
    basin_norm = basin_norm.replace("à", "a")
    basin_norm = basin_norm.replace("è", "e")
    basin_norm = basin_norm.replace("é", "e")

    if basin_norm == basin_name:
        # --- get runoff series ---
        r_values = next(
            r["discharge"]
            for r in runoff_list
            if r["basin"].replace(" ", "_").lower() == basin_norm
        )
        # IMPORTANT: assume runoff_list also stores time
        r_time = next(
            r["date"]
            for r in runoff_list
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
            "Corr_time": 1.085 * np.sqrt(area_km2),
            "mean_H": mean_H,
            "runoff": runoff_aligned.values
        }, index=precipitation.index)
        # --- cleaning ---
        df["precipitation"] = df["precipitation"].clip(lower=0)
        df["runoff"] = df["runoff"].clip(lower=0)
        df.dropna(inplace=True)
        df["month"] = df.index.month
    else:
        continue


# ============================================================
# 1. CREATE HYDROLOGICAL REGIMES (IMPORTANT FIX)
# ============================================================
# bins based on runoff distribution (not months!)
df["runoff_bin"] = pd.qcut(df["runoff"], q=10, duplicates="drop")

# ============================================================
# 2. STRATIFIED SPLIT (TRAIN / TEMP)
# ============================================================
sss1 = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.3,
    random_state=42
)

X = df.index
y = df["runoff_bin"]

for train_idx, temp_idx in sss1.split(X, y):
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

# ============================================================
# 3. SPLIT TEMP INTO VAL / TEST
# ============================================================
sss2 = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.5,
    random_state=42
)

X_temp = temp_df.index
y_temp = temp_df["runoff_bin"]

for val_idx, test_idx in sss2.split(X_temp, y_temp):
    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]

# ============================================================
# 4. OPTIONAL: DROP AUX COLUMN
# ============================================================
train_df = train_df.drop(columns=["runoff_bin"])
val_df = val_df.drop(columns=["runoff_bin"])
test_df = test_df.drop(columns=["runoff_bin"])

# ============================================================
# 5. STATS CHECK
# ============================================================
stats("TRAIN", train_df)
stats("VAL", val_df)
stats("TEST", test_df)

# ============================================================
# 6. SAMPLE COUNT CONTROL (YOU ASKED THIS)
# ============================================================
print("\n===== SAMPLE COUNTS =====")
print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))

# optional: exact percentages
total = len(df)
print("\n===== PERCENTAGES =====")
print("Train %:", len(train_df)/total*100)
print("Val %:", len(val_df)/total*100)
print("Test %:", len(test_df)/total*100)
# ============================================================
# SCALING (TRAIN ONLY FIT)
# ============================================================
scaler = StandardScaler()

cols = ["precipitation", "Corr_time", "mean_H", "runoff"]

train_scaled = train_df.copy()
val_scaled = val_df.copy()
test_scaled = test_df.copy()

train_scaled[cols] = scaler.fit_transform(train_df[cols])
val_scaled[cols] = scaler.transform(val_df[cols])
test_scaled[cols]= scaler.transform(test_df[cols])


# ============================================================
# SEQUENCE CREATION
# ============================================================
X_train, y_train = create_sequences(train_scaled, seq_length=SEQ)
X_val, y_val = create_sequences(val_scaled, seq_length=SEQ)
X_test, y_test = create_sequences(test_scaled, seq_length=SEQ)

# ============================================================
# LSTM MODEL
# ============================================================
model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(100, return_sequences=True),
        LSTM(50),
        Dense(1)])

# insert early stopping callback

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=custom_loss,
    metrics=['mae']
)

es = EarlyStopping(patience=50, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=30,
    callbacks=[es]
)


# ============================================================
# PREDICTION
# ============================================================
y_pred = model.predict(X_test)

runoff_mean = scaler.mean_[-1]
runoff_std = np.sqrt(scaler.var_[-1])

y_pred_inv = y_pred * runoff_std + runoff_mean
y_test_inv = y_test * runoff_std + runoff_mean

# ===========================================================
# METRICS
# ============================================================
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print("TEST RMSE:", rmse)
# SAVE THE PREDICTIONS
pred_df = pd.DataFrame({
    "y_true": y_test_inv.flatten(),
    "y_pred": y_pred_inv.flatten()
}, index=test_df.index[SEQ:])
pred_df.to_csv(
    "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/lstm_predictions_PIACENZA.csv"
)
plt.style.use("seaborn-v0_8-whitegrid")

# ============================================================
# BIN DEFINITION
# ============================================================
plt.style.use("seaborn-v0_8-whitegrid")

# ============================================================
# BIN DEFINITION
# ============================================================
bins = [
    (0, 500),
    (500, 1000),
    (1000, 1500),
    (1500, 2000),
    (2000, 3000),
    (3000, np.inf)
]

n_bins = len(bins)
ncols = 3
nrows = math.ceil(n_bins / ncols)

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(18, 10),
    constrained_layout=True
)
axes = np.array(axes).flatten()

# ============================================================
# GLOBAL LIMITS (square & consistent)
# ============================================================
global_min = np.nanmin([y_test_inv.min(), y_pred_inv.min()])
global_max = np.nanmax([y_test_inv.max(), y_pred_inv.max()])

pad = 0.05 * (global_max - global_min)
lims = (global_min - pad, global_max + pad)

# ============================================================
# PLOTTING
# ============================================================
for i, (bmin, bmax) in enumerate(bins):

    ax = axes[i]

    # --------------------------------------------------------
    # MASK
    # --------------------------------------------------------
    if np.isinf(bmax):
        mask = y_test_inv >= bmin
        title = f"Runoff > {bmin} mm"
    else:
        mask = (y_test_inv >= bmin) & (y_test_inv < bmax)
        title = f"{bmin}–{bmax} mm"

    mask = np.array(mask)

    if np.sum(mask) == 0:
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")
        continue

    y_true_bin = y_test_inv[mask]
    y_pred_bin = y_pred_inv[mask]

    # ========================================================
    # SCATTER (cleaner + more readable)
    # ========================================================
    ax.scatter(
        y_true_bin,
        y_pred_bin,
        s=10,
        alpha=0.2,
        color="steelblue",
        edgecolor="none"
    )

    # ========================================================
    # 1:1 LINE (strong visual anchor)
    # ========================================================
    ax.plot(
        lims, lims,
        linestyle="--",
        color="black",
        linewidth=2
    )

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")

    # ========================================================
    # METRICS (clean box, better typography)
    # ========================================================
    rmse = np.sqrt(mean_squared_error(y_true_bin, y_pred_bin))
    bias = np.mean(y_pred_bin - y_true_bin)
    n_points = len(y_true_bin)

    ax.text(
        0.03, 0.97,
        f"N = {n_points}\nRMSE = {rmse:.1f}\nBias = {bias:.1f}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="gray",
            alpha=0.9
        )
    )

    # ========================================================
    # TITLES
    # ========================================================
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Only outer labels (cleaner poster look)
    if i % ncols == 0:
        ax.set_ylabel("Predicted runoff (mm)", fontsize=12)
    else:
        ax.set_ylabel("")

    if i >= n_bins - ncols:
        ax.set_xlabel("Observed runoff (mm)", fontsize=12)
    else:
        ax.set_xlabel("")

    ax.tick_params(labelsize=10)

# ============================================================
# REMOVE EMPTY AXES
# ============================================================
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# ============================================================
# GLOBAL TITLE (optional but strong for posters)
# ============================================================
fig.suptitle(
    "LSTM Performance Across Runoff Bins",
    fontsize=18,
    fontweight="bold"
)

# ============================================================
# SAVE (poster quality)
# ============================================================
plt.savefig(
    "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/lstm_binned_poster_v2.png",
    dpi=400,
    bbox_inches="tight"
)

plt.close()