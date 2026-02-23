import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask

# -------------------------------------------------------------------------------------
# Paths and time
# -------------------------------------------------------------------------------------
path_MCM = "/home/idrologia/share/PhD_GiuliaBlandini_dati/MCM/geotiff_storiaBUFR/"
folder_masks = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/MASKS_SHAPE/"
folder_WL_Q = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/WL_Q_nc/"
start_date = "2004-09-01 00:00:00"
end_date = "2023-09-30 23:00:00"
period = pd.date_range(start=start_date, end=end_date, freq="h")

output_pkl = (
    "/home/idrologia/share/PhD_GiuliaBlandini_dati/"
    "AI_RIVER_LEVELS/output/mean_h_precipitation_20040901_20230831_cumulative_by_basin.pkl")

# -------------------------------------------------------------------------------------
# Load basin shapefiles
# -------------------------------------------------------------------------------------
basins = {}
for shp_file in os.listdir(folder_masks):
    if shp_file.endswith(".shp"):
        basin_name = os.path.splitext(shp_file)[0].replace("_catchment", "")
        basin_name = re.sub(r'(?<=[a-z])(?=[A-Z])', '_', basin_name)  # CamelCase → snake_case
        basin_name = basin_name.replace(" ", "_")
        gdf = gpd.read_file(os.path.join(folder_masks, shp_file))
        basins[basin_name] = gdf

# sort basins by name
basins = dict(sorted(basins.items()))
print("Basins loaded")

# -------------------------------------------------------------------------------------
# Load WL_Q files and map to basins
# -------------------------------------------------------------------------------------
wlq_files = pd.read_pickle("/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/plots_WLQ/list_file.pkl")
wlq_files_names = [
    re.sub(r"_\d{8}_\d{8}\.nc$", "", fname) for fname in wlq_files]

# create a normalized comparison: remove underscores and lowercase
wlq_files_norm = [re.sub("_", "", fname).lower() for fname in wlq_files_names]

# Filter basins to those with WL_Q files
valid_basins = {}
for basin_name, gdf in basins.items():
    norm_name = basin_name.replace("_", "").lower()
    if norm_name in wlq_files_norm:
        valid_basins[basin_name] = gdf

basins = valid_basins

# -------------------------------------------------------------------------------------
# Loop over time and compute mean precipitation
# -------------------------------------------------------------------------------------
results = []

# -------------------------------------------------------------------------------------
# Loop over time
# -------------------------------------------------------------------------------------
for date in period:

    path_to_file = os.path.join(
        path_MCM,
        date.strftime("%Y"),
        date.strftime("%m"),
        date.strftime("%d")
    )

    file_MCM = f"MCM_BUFR_{date.strftime('%Y%m%d%H')}0000.tif"
    full_path_MCM = os.path.join(path_to_file, file_MCM)

    # ------------------------------------------------------------------
    # Missing file → zero precipitation for all basins
    # ------------------------------------------------------------------
    if not os.path.exists(full_path_MCM):
        for basin_name in basins:
            results.append({
                "date": date,
                "basin": basin_name,
                "mean_precipitation": 0.0
            })
        print (f"File not found: {full_path_MCM}. Assigned zero precipitation for all basins.")
        continue

    # ------------------------------------------------------------------
    # Read raster
    # ------------------------------------------------------------------
    with rasterio.open(full_path_MCM) as src:

        nodata = src.nodata
        raster_crs = src.crs

        # Reproject basins once per raster
        basins_proj = {
            name: gdf.to_crs(raster_crs)
            for name, gdf in basins.items()
        }

        for basin_name, gdf in basins_proj.items():

            geoms = [geom for geom in gdf.geometry if geom is not None]

            try:
                out_image, _ = mask(src, geoms, crop=True, all_touched=True)
                data = out_image[0]

                if nodata is not None:
                    data = data[data != nodata]

                mean_precip = float(np.nanmean(data)) if data.size > 0 else 0.0

            except ValueError:
                # Basin outside raster
                mean_precip = -9999

            results.append({
                "date": date,
                "basin": basin_name,
                "mean_precipitation": mean_precip
            })

        print (f"Processed {date.strftime('%Y-%m-%d %H:%M:%S')}")


# -------------------------------------------------------------------------------------
# Build DataFrame with datetime index and basins as columns
# -------------------------------------------------------------------------------------
df = pd.DataFrame(results)
df_pivot = df.pivot(index="date", columns="basin", values="mean_precipitation")

# Sort columns alphabetically
df_pivot = df_pivot.reindex(sorted(df_pivot.columns), axis=1)
import matplotlib.pyplot as plt
import pandas as pd

# Assume df_pivot is already prepared
basins = df_pivot.columns.unique()

for basin in basins:
    df_basin = df_pivot[basin].copy()
    df_basin.index = pd.to_datetime(df_basin.index)

    # Resample to daily accumulated precipitation
    daily = df_basin.resample("D").sum()

    # Create figure with 2 subplots (vertical)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Subplot 1: Time series line plot
    axes[0].bar(df_basin.index, df_basin.values, width=1, color='blue')
    axes[0].set_title(f"{basin} - Mean Precipitation Time Series", fontsize=12)
    axes[0].grid(True)

    # Subplot 2: Daily accumulated precipitation
    axes[1].bar(daily.index, daily.values, width=1, color='skyblue')
    axes[1].set_title(f"{basin} - Daily Accumulated Precipitation", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    axes[1].set_xlabel("Date")

    plt.tight_layout()
    plt.savefig(f"/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/plots_WLQ/precipitation/{basin}_precip.png")
    plt.close()
    print (f"Saved precipitation figure for {basin}.")

print("Saved precipitation figures for all basins.")
# Save as pickle
df_pivot.to_pickle(output_pkl)
print(f"Saved mean precipitation time series to {output_pkl}")