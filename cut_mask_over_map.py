import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt
import logging

# -------------------------------------------------------------------------------------
# Paths and time
# -------------------------------------------------------------------------------------
path_MCM = "/home/idrologia/share/PhD_GiuliaBlandini_dati/MCM/geotiff_storiaBUFR/"
folder_masks = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/MASKS_SHAPE_PO/"
folder_WL_Q = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/WL_Q_nc/"
start_date = "2015-09-01 00:00:00"
end_date = "2023-08-31 23:00:00"
period = pd.date_range(start=start_date, end=end_date, freq="h")
output_pkl = (
    "/home/idrologia/share/PhD_GiuliaBlandini_dati/"
    "AI_RIVER_LEVELS/output/mean_h_precipitation_20150901_20230831_cumulative_by_basin.pkl"
)
intermediate_pkl = output_pkl.replace(".pkl", "_partial.pkl")
log_file = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/run_precipitation.log"

# -------------------------------------------------------------------------------------
# Set up logging
# -------------------------------------------------------------------------------------
logging.basicConfig(
    filename=log_file,
    filemode='a',  # append mode
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Script started")

# -------------------------------------------------------------------------------------
# Load basin shapefiles
# -------------------------------------------------------------------------------------
basins = {}
for shp_file in os.listdir(folder_masks):
    if shp_file.endswith(".shp"):
        basin_name = os.path.splitext(shp_file)[0].replace("_catchment", "")
        basin_name = re.sub(r'(?<=[a-z])(?=[A-Z])', '_', basin_name)
        parts = basin_name.split('_')
        if len(parts) > 1:
            basin_name = f"{parts[-1]}_{parts[0]}"
        basin_name = basin_name.replace(" ", "_").replace("_", "")
        gdf = gpd.read_file(os.path.join(folder_masks, shp_file))
        basins[basin_name] = gdf
basins = dict(sorted(basins.items()))
logging.info(f"Loaded {len(basins)} basins")

# -------------------------------------------------------------------------------------
# Load WL_Q files and map to basins
# -------------------------------------------------------------------------------------
wlq_files_path = [f for f in os.listdir(folder_WL_Q) if f.endswith(".nc")]
wlq_files_names = [re.sub(r"_\d{8}_\d{8}\.nc$", "", fname) for fname in wlq_files_path]
wlq_files_names = [name.replace("_", "").lower() for name in wlq_files_names]
wlq_files_names.sort()
logging.info(f"Found {len(wlq_files_names)} WL_Q files")

basin_to_wlq = {}
for basin_name in basins.keys():
    basin_name_clean = basin_name.replace("_", "").lower()
    basin_name_clean = re.sub(r'[àáâäæãåā]', 'a', basin_name_clean)
    matched_file = next((fname for fname in wlq_files_names if basin_name_clean in fname), None)
    if matched_file:
        basin_to_wlq[basin_name] = matched_file
    else:
        logging.warning(f"No match found for {basin_name} in WL_Q files.")

# -------------------------------------------------------------------------------------
# Loop over time and compute mean precipitation
# -------------------------------------------------------------------------------------
results = []
save_every_n_hours = 100  # store partial results every N hours

for i, date in enumerate(period, start=1):
    path_to_file = os.path.join(
        path_MCM,
        date.strftime("%Y"),
        date.strftime("%m"),
        date.strftime("%d")
    )
    file_MCM = f"MCM_BUFR_{date.strftime('%Y%m%d%H')}0000.tif"
    full_path_MCM = os.path.join(path_to_file, file_MCM)

    if not os.path.exists(full_path_MCM):
        for basin_name in basins:
            results.append({"date": date, "basin": basin_name, "mean_precipitation": 0.0})
        logging.warning(f"File not found: {full_path_MCM}. Assigned zero precipitation for all basins.")
        continue

    try:
        with rasterio.open(full_path_MCM) as src:
            nodata = src.nodata
            raster_crs = src.crs

            basins_proj = {name: gdf.to_crs(raster_crs) for name, gdf in basins.items()}

            for basin_name, gdf in basins_proj.items():
                geoms = [geom for geom in gdf.geometry if geom is not None]

                try:
                    out_image, _ = mask(src, geoms, crop=True, all_touched=True)
                    data = out_image[0]
                    if nodata is not None:
                        data = data[data != nodata]
                    mean_precip = float(np.nanmean(data)) if data.size > 0 else 0.0
                except ValueError:
                    mean_precip = -9999

                results.append({"date": date, "basin": basin_name, "mean_precipitation": mean_precip})

        logging.info(f"Processed {date.strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        logging.error(f"Error reading {full_path_MCM}: {e}")
        for basin_name in basins:
            results.append({"date": date, "basin": basin_name, "mean_precipitation": 0.0})

    # Save intermediate results every N hours
    if i % save_every_n_hours == 0:
        df_partial = pd.DataFrame(results)
        df_partial.to_pickle(intermediate_pkl)
        logging.info(f"Saved intermediate results after {i} hours to {intermediate_pkl}")

# -------------------------------------------------------------------------------------
# Build final DataFrame and save
# -------------------------------------------------------------------------------------
df = pd.DataFrame(results)
df_pivot = df.pivot(index="date", columns="basin", values="mean_precipitation")
df_pivot = df_pivot.reindex(sorted(df_pivot.columns), axis=1)
df_pivot.to_pickle(output_pkl)
logging.info(f"Saved final mean precipitation time series to {output_pkl}")

print("Script finished successfully.")