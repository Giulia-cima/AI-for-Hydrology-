import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
import os
import glob
import pandas as pd
import re



# -----------------------------
# FILE INPUT
# -----------------------------
dem_path = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/DEM_ITALIA/italy_dem_merged_COMPLETE.tif"
shp_folder = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/MASKS_SHAPE_PO/"

shp_files = glob.glob(os.path.join(shp_folder, "*.shp"))
results = []

# -----------------------------
# LOOP OVER SHAPEFILES
# -----------------------------
for shp_path in shp_files:
    basin_name = os.path.basename(shp_path).replace(".shp", "")
    print(f"\nProcessing: {basin_name}")
    basin_name = basin_name.replace("_catchment", "")
    basin_name = re.sub(r'(?<=[a-z])(?=[A-Z])', '_', basin_name)
    parts = basin_name.split('_')
    if len(parts) > 1:
        basin_name = f"{parts[-1]}_{parts[0]}"
    basin_name = basin_name.replace(" ", "")
    basin_name = basin_name.replace("_", "")
    basin_name = basin_name.lower()
    print(f"Normalized basin name: {basin_name}")

    # ============================================================
    # BASIN AREA + DEM STATS
    # ============================================================
    gdf = gpd.read_file(shp_path)
    gdf = gdf.buffer(0)
    gdf_proj = gdf.to_crs("EPSG:32632")

    area_m2 = gdf_proj.geometry.area.sum()
    area_km2 = area_m2 / 1e6

    with rasterio.open(dem_path) as src:
        gdf_proj = gdf.to_crs(src.crs)
        clipped, _ = mask(src, gdf_proj.geometry, crop=True)
        dem_clip = clipped[0]
        valid = dem_clip[dem_clip != src.nodata]
        mean_H = np.mean(valid)
        max_H = np.max(valid)

    # -----------------------------
    # STORE RESULTS
    # -----------------------------
    results.append({
        "basin": basin_name,
        "area_km2": area_km2,
        "mean_H": mean_H,
        "max_H": max_H,
    })

# -----------------------------
# SAVE TO CSV
# -----------------------------
df = pd.DataFrame(results)
df.to_csv("/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/basin_stats.csv", index=False)

print("\nDone! Results saved to basin_stats.csv")