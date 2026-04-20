import os

import matplotlib.pyplot as plt
import numpy as np
import pandas
import geopandas as gpd
from pysheds.grid import Grid
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union


acc_path = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/network_1_km_v2.tif"
fdir_path = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/drainage_direction_v2.tif" # flow direction
out_dir = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/MASKS_Niscemi/"
out_shp = os.path.join(out_dir, "basin_niscemi.shp")
os.makedirs(out_dir, exist_ok=True)
niscemi_csv = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/niscemi.csv"

# Get the coordinates of the point
df = pandas.read_csv(niscemi_csv)
lon = df['Longitude'].values[0]
# FROM STR TO FLOAT
lon = float(lon.replace(',', '.'))
lat = df['Latitude'].values[0]
lat = float(lat.replace(',', '.'))

dirmap = (2, 1, 8, 7, 6, 5, 4, 3)

# Initialise grid with int8 to minimise space consumption
grid = Grid.from_raster(acc_path)
fdir = grid.read_raster(fdir_path, nodata=-9999)
fdir[fdir == -9999] = 0
fdir = np.abs(fdir).astype(np.uint8)
grid.dir = fdir
print("Files read successfully")

# Calculate the flow accumulation
acc = grid.accumulation(fdir, dirmap=dirmap)

# Snap the point to the nearest cell with flow accumulation > 1000
x_snap, y_snap = grid.snap_to_mask(acc > 1000, (lon, lat))
print(' Original point:', (lon, lat))
print(' Snapped point:', (x_snap, y_snap))

print("point snapped to nearest cell with flow accumulation > 1000")

# retrieve crs from x_snap and y_snap
raster_csr = grid.crs

catch = grid.catchment(
    fdir=grid.dir,
    x=x_snap,
    y=y_snap,
    dirmap=dirmap,
    recursionlimit=20000,
    nodata_out=0,
    xytype="coordinate",
    routing="d8",
    snap="center"
)

# Clip the bounding box to the catchment
grid.clip_to(catch)
clipped_catch = grid.view(catch)

print("catchment delineated and clipped to bounding box")

# Plot the catchment
fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)

plt.grid('on', zorder=0)
im = ax.imshow(np.where(clipped_catch, clipped_catch, np.nan), extent=grid.extent,
               zorder=1, cmap='Greys_r')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Delineated Catchment', size=14)
plt.savefig(os.path.join(out_dir, "basin_niscemi.png"), dpi=300)

print(" catchment plotted and saved as PNG")

mask = (clipped_catch > 0).astype(np.uint8)
geom_list = []

for shape_info in shapes(mask, mask=mask, transform=grid.affine):
    geom = shape(shape_info[0])
    geom_list.append(geom)

gdf = gpd.GeoDataFrame(
    geometry=gpd.GeoSeries(geom_list),
    crs=raster_csr
)

gdf.to_file(out_shp)
print("Shapefile salvato con le coordinate corrette.")