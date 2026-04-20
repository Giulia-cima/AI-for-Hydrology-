import os
import pandas as pd
import numpy as np
import geopandas as gpd
from pysheds.grid import Grid
from rasterio.features import shapes


#=========================================================================================================
#== Configuration ==
#=========================================================================================================
os.environ["OMP_NUM_THREADS"] = "30"
acc_path_raster = '/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/network_1_km_v2.tif'
fdir_path = '/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/drainage_direction_v2.tif'
output_dir = '/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/output/MASKS_SHAPE_PO/'
output_shapefile = os.path.join(output_dir, 'catchments.shp')
points_csv = '/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/stations_coordinates_Q_PO.csv'
os.makedirs(output_dir, exist_ok=True)

#=========================================================================================================
# This script reads a drainage direction raster and a flow accumulation raster, then processes a
# list of points (with longitude and latitude) to extract the corresponding catchment areas.
# Each catchment area is saved as a shapefile in the specified output directory.
#=========================================================================================================

dirmap = (2, 1, 8, 7, 6, 5, 4, 3)
grid = Grid.from_raster(acc_path_raster)
fdir = grid.read_raster(fdir_path, nodata=-9999)
fdir[fdir == -9999] = 0
fdir = np.abs(fdir).astype(np.uint8)
grid.dir = fdir
acc = grid.accumulation(fdir, dirmap=dirmap)
points_df = pd.read_csv(points_csv)

for index, row in points_df.iterrows():
    x = row['Longitude']
    x = float(x.replace(',', '.'))
    y = row['Latitude']
    y = float(y.replace(',', '.'))

    section = row['Section']
    basin = row['Basin']
    name = f"{section}_{basin}"
    x_snap, y_snap = grid.snap_to_mask(acc>1000,(x, y))

    print (f"Processing point  {x} {y} snapped at {x_snap}, {y_snap} for {name}")

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

    clipped_catch = (catch > 0).astype(np.uint8)
    transform = grid.affine
    results = (
        {'properties': {'value': v}, 'geometry': s}
        for s, v in shapes(clipped_catch.astype(np.int16), mask=clipped_catch.astype(bool), transform=transform)
    )

    geoms = list(results)
    gdf_catch = gpd.GeoDataFrame.from_features(geoms)
    gdf_catch = gdf_catch[gdf_catch['value'] == 1]
    gdf_catch = gdf_catch.dissolve()
    gdf_catch['section'] = section
    gdf_catch['basin'] = basin
    name = f"{section}_{basin}"
    gdf_catch.set_crs(grid.crs, inplace=True)
    shp_path = os.path.join(output_dir, f"{name}_catchment.shp")
    gdf_catch.to_file(shp_path)
    print(f"Shapefile saved for {name}")

print("Processing complete.")