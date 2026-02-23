# CREATION OF MASKS FOR EACH BASIN TO TRIM THE MCM PRECIPITATION VALUES
# USING PYSHEDS
import os
import numpy as np
import geopandas as gpd
from pysheds.grid import Grid
from rasterio.features import shapes
from shapely.geometry import shape as shapely_shape

os.environ["OMP_NUM_THREADS"] = "30"

acc_path = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/network_1_km_v2.tif"
fdir_path = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/drainage_direction_v2.tif"
points_path = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/snapped.shp"
out_dir = "/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/MASKS_SHAPE/"
os.makedirs(out_dir, exist_ok=True)

dirmap = (2, 1, 8, 7, 6, 5, 4, 3)

# Initialise grid with int8 to minimise space consumption
grid = Grid.from_raster(acc_path)
fdir = grid.read_raster(fdir_path, nodata=-9999)
fdir[fdir == -9999] = 0
fdir = np.abs(fdir).astype(np.uint8)
grid.dir = fdir

pts = gpd.read_file(points_path)
print("Files read successfully")

# Loop points
for idx, r in pts.iterrows():
    name = str(r["Basin"]+r["Section"]) if "Basin" in r else f"basin_{idx}"
    x = r.geometry.x
    y = r.geometry.y

    print(f"Delineating basin: {name}")

    # Delineate catchment
    catch = grid.catchment(
        fdir=grid.dir,
        x=x,
        y=y,
        dirmap=dirmap,
        recursionlimit=20000,
        nodata_out=0,
        xytype="coordinate",
        routing="d8",
        snap="center"
    )

    mask = (catch > 0).astype(np.uint8)
    if mask.sum() == 0:
        print(f"{name}: KO (empty catchment)")
        continue

    # Polygonize and save shapefile
    geom_list = []
    for geom, val in shapes(mask, mask=mask.astype(bool), transform=grid.affine):
        if val == 1:
            geom_list.append(shapely_shape(geom))

    if not geom_list:
        print(f"{name}: KO (no polygons)")
        continue

    out_gdf = gpd.GeoDataFrame({"Basin": [name]}, geometry=[geom_list[0]], crs=pts.crs)
    out_path = os.path.join(out_dir, f"{name}_catchment.shp")
    out_gdf.to_file(out_path)

    print(f"{name}: OK")

print("Done.")
