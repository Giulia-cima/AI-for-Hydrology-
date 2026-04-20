import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds

# --- Read shapefile
acc_path = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/network_ita_v2.shp"
gdf = gpd.read_file(acc_path)

# --- Define raster grid (IMPORTANT 👇)
xmin, ymin, xmax, ymax = gdf.total_bounds

resolution = 0.01  # choose this carefully!!
width = int((xmax - xmin) / resolution)
height = int((ymax - ymin) / resolution)

transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

# --- Rasterize
shapes = [(geom, 1) for geom in gdf.geometry]

raster = rasterize(
    shapes,
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype=np.int8
)

# --- Save raster
output_path = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/network_ita_v2.tif"

with rasterio.open(
    output_path,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    dtype=raster.dtype,
    crs=gdf.crs,
    transform=transform,
    nodata=-99
) as dst:
    dst.write(raster, 1)