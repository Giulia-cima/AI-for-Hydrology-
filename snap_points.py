# Script to be used on large rasters to snap points to the nearest 1 (can be changed) cell.
# search radius in terms of number of pixel can be specified

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from shapely.geometry import Point

# Inputs
points_path = "snapped.shp"
raster_path = "network_1_km_v2.tif"
out_points_path = "points_snapped_to_1.shp"

target_value = 1
radius_px = 2   # search radius (pixels)

gdf = gpd.read_file(points_path)

with rasterio.open(raster_path) as src:
    transform = src.transform
    width = src.width
    height = src.height

    new_geoms = []
    snap_ok = []

    for geom in gdf.geometry:
        x, y = geom.x, geom.y
        row, col = rasterio.transform.rowcol(transform, x, y)

        # value under point
        val = src.read(1, window=Window(col, row, 1, 1))[0, 0]

        # already on 1
        if val == target_value:
            new_geoms.append(geom)
            snap_ok.append("OK")
            continue

        # fixed-radius window
        r0 = max(row - radius_px, 0)
        r1 = min(row + radius_px, height - 1)
        c0 = max(col - radius_px, 0)
        c1 = min(col + radius_px, width - 1)

        win = Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)
        block = src.read(1, window=win)

        rr, cc = np.where(block == target_value)

        # no 1 in radius
        if rr.size == 0:
            new_geoms.append(geom)
            snap_ok.append("KO")
            continue

        # nearest 1
        pr = row - r0
        pc = col - c0
        dr = rr - pr
        dc = cc - pc
        k = np.argmin(dr*dr + dc*dc)

        best_row = r0 + rr[k]
        best_col = c0 + cc[k]

        x2, y2 = rasterio.transform.xy(transform, best_row, best_col, offset="center")
        new_geoms.append(Point(x2, y2))
        snap_ok.append("OK")

gdf.geometry = new_geoms
gdf["snap_ok"] = snap_ok

gdf.to_file(out_points_path)

print(f"Saved: {out_points_path}")
print(f"OK: {snap_ok.count('OK')}  |  KO: {snap_ok.count('KO')}")

