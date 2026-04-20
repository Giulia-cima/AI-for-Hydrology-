import os
import glob
import rasterio
from rasterio.merge import merge

# Your folder
folder = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/DEM_ITALIA/"

# Output file
output_file = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/DEM_ITALIA/italy_dem_merged.tif"

# Get list of all tif files
tiff_file_list = glob.glob(os.path.join(folder, "*.tif"))

print(f"Found {len(tiff_file_list)} tiles")

# Open all rasters
src_files = [rasterio.open(fp) for fp in tiff_file_list]

# Merge
mosaic, out_transform = merge(src_files)

# Copy metadata from first tile
out_meta = src_files[0].meta.copy()

# Update metadata
out_meta.update({
    "driver": "GTiff",
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": out_transform,
    "compress": "lzw"
})

# Write merged DEM
with rasterio.open(output_file, "w", **out_meta) as dest:
    dest.write(mosaic)

# Close files
for src in src_files:
    src.close()

print("✅ Merge completed!")