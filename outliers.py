# ============================================================
# Rainfall–Runoff LSTM Model
# ============================================================

import os
import re
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import seaborn as sns

def remove_outliers_iqr(df, column, factor=3):

    series = df[column]

    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR

    df_clean = df.copy()

    mask = (series >= lower) & (series <= upper)

    df_clean.loc[~mask, column] = np.nan

    return df_clean


def z_score_outliers(df, column, threshold=3):
    series = df[column]

    mean = series.mean()
    std = series.std()

    z_scores = (series - mean) / std

    df_clean = df.copy()

    mask = z_scores.abs() <= threshold

    df_clean.loc[~mask, column] = np.nan

    return df_clean


def plot_cleaning(df_before, df_after, column, basin_name, output_folder):

    # align indices
    after_aligned = df_after.reindex(df_before.index)

    # detect removed values
    removed_mask = ~(df_before[column].fillna(-9999).eq(
        after_aligned[column].fillna(-9999)
    ))

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(14,8),
        sharex=True
    )

    # --------------------------------------------------
    # ORIGINAL SERIES
    # --------------------------------------------------

    ax1.plot(
        df_before.index,
        df_before[column],
        color="tab:blue",
        alpha=0.7,
        label="Original"
    )

    ax1.scatter(
        df_before.index[removed_mask],
        df_before[column][removed_mask],
        color="red",
        s=10,
        label="Removed values"
    )

    ax1.set_ylabel(column)
    ax1.set_title(f"{column} - Original ({basin_name})")
    ax1.legend()

    # --------------------------------------------------
    # CLEANED SERIES
    # --------------------------------------------------

    ax2.plot(
        after_aligned.index,
        after_aligned[column],
        color="tab:orange",
        label="Cleaned"
    )

    ax2.set_ylabel(column)
    ax2.set_xlabel("Time")
    ax2.set_title(f"{column} - Cleaned")
    ax2.legend()

    plt.tight_layout()

    os.makedirs(output_folder, exist_ok=True)

    plt.savefig(
        os.path.join(output_folder, f"{basin_name}_{column}_cleaning.png"),
        dpi=300
    )

    plt.close()
# ============================================================
# Load precipitation data
# ============================================================

precipitation = pd.read_pickle(
"/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/mean_h_precipitation_20040901_20230831_cumulative_by_basin.pkl"
)

time_start = "2010-01-01 00:00:00"
time_end = "2020-12-31 23:00:00"

precipitation = precipitation.loc[time_start:time_end]


# ============================================================
# Load runoff data
# ============================================================

runoff = pd.read_pickle(
"/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/list_q.pkl"
)


# ============================================================
# Align time index
# ============================================================

full_time_index = pd.date_range(
    start="2004-01-01 00:00:00",
    end="2023-09-30 01:00:00",
    freq='h'
)

for item in runoff:

    discharge = item["discharge"]

    n_discharge = discharge.shape[0]
    n_time = len(full_time_index)

    if n_discharge < n_time:

        padded = xr.DataArray(
            np.full(n_time, np.nan),
            coords=[full_time_index],
            dims=["time"]
        )

        padded[:n_discharge] = discharge.values
        item["discharge"] = padded

    elif n_discharge > n_time:

        item["discharge"] = discharge.isel(time=slice(0, n_time))

    else:

        item["discharge"] = discharge.assign_coords(time=full_time_index)


for item in runoff:
    item["discharge"] = item["discharge"].sel(time=slice(time_start, time_end))


# ============================================================
# Build runoff dictionary
# ============================================================

runoff_dict = {}

for item in runoff:

    name = item["basin"]
    runoff_dict[name.replace(" ", "_")] = item["discharge"].values


print("Number of basins in runoff:", len(runoff_dict))


# ============================================================
# Combine precipitation + runoff
# ============================================================

list_basins = []

for basin in precipitation.columns:

    if basin in runoff_dict:

        df_combined = pd.DataFrame({
            "precipitation": precipitation[basin].values,
            "runoff": np.pad(
                runoff_dict[basin],
                (0, max(0, len(precipitation[basin]) - len(runoff_dict[basin]))),
                constant_values=np.nan
            )
        })

        df_combined.index = pd.to_datetime(precipitation[basin].index)

        df_before = df_combined.copy()

        # precipitation cleaning
        df_combined.loc[df_combined["precipitation"] >= 1e6, "precipitation"] = np.nan
        df_combined.loc[df_combined["precipitation"] < 0, "precipitation"] = 0
        # runoff cleaning
        df_combined.loc[df_combined["runoff"] >= 1e6, "runoff"] = np.nan
        df_combined.loc[df_combined["runoff"] < 0, "runoff"] = 0

        # drop nan by index
        df_combined.dropna(subset=["precipitation", "runoff"], how="all", inplace=True)

        # apply z-score outlier removal
        df_combined = z_score_outliers(df_combined, "precipitation", threshold=20)
        df_combined = z_score_outliers(df_combined, "runoff", threshold=15)


        plot_cleaning(
            df_before,
            df_combined,
            "precipitation",
            basin,
            "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/cleaning"
        )
        plot_cleaning(
            df_before,
            df_combined,
            "runoff",
            basin,
            "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/cleaning"
        )


