import geopandas as gpd
import numpy as np
import rasterio
from pysheds.grid import Grid

# --- INPUT ---
fdir_path ='/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/drainage_direction_v2.tif'
acc_path_raster = '/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/network_1_km_v2.tif'


dirmap = (2, 1, 8, 7, 6, 5, 4, 3)

# Initialize the grid
grid = Grid.from_raster(acc_path_raster)
fdir = grid.read_raster(fdir_path, nodata=-9999)
fdir[fdir == -9999] = 0
fdir = np.abs(fdir).astype(np.uint8)
grid.dir = fdir

acc = grid.accumulation(fdir, dirmap=dirmap)

# --- CONVERSIONE fdir -> pnt ---
mapping = {
    2: 6,  # E
    1: 9,  # NE
    8: 8,  # N
    7: 7,  # NW
    6: 4,  # W
    5: 1,  # SW
    4: 2,  # S
    3: 3   # SE
}

pnt = np.full_like(fdir, np.nan, dtype=float)

for k, v in mapping.items():
    pnt[fdir == k] = v

# --- SAVE OUTPUT ---
with rasterio.open(acc_path_raster) as src:
    profile = src.profile

profile.update(dtype="float32", nodata=np.nan)

# salva raster_aree_monte
with rasterio.open("/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/aree_monte.tif", "w", **profile) as dst:
    dst.write(acc.astype("float32"), 1)

# salva raster_pnt

with rasterio.open("/home/idrologia/share/PhD_GiuliaBlandini_dati/AI_RIVER_LEVELS/pnt.tif", "w", **profile) as dst:
    dst.write(pnt.astype("float32"), 1)



# ============================================================
# UNION-FIND (for merging overlapping drainage basins)
# ============================================================
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


# ============================================================
# DRAINAGE AREAS (ROBUST VERSION)
# ============================================================
def aree_drenate(raster_pnt, raster_aree_monte, shape_sezioni, P=None):

    if P is None:
        P = np.array([[7, 8, 9],
                      [4, 5, 6],
                      [1, 2, 3]])

    # -----------------------------
    # READ RASTERS
    # -----------------------------
    with rasterio.open(raster_pnt) as src:
        pnt = src.read(1).astype(float)
        transform = src.transform
        crs = src.crs
        n_orig, m_orig = pnt.shape

    with rasterio.open(raster_aree_monte) as src:
        aree_monte = src.read(1).astype(float)

    # -----------------------------
    # LOAD SECTIONS
    # -----------------------------
    gdf = gpd.read_file(shape_sezioni).to_crs(crs)

    xy = np.array([
        [geom.representative_point().x, geom.representative_point().y]
        for geom in gdf.geometry
    ])

    rows, cols = rasterio.transform.rowcol(transform, xy[:, 0], xy[:, 1])

    mask = (rows >= 0) & (rows < n_orig) & (cols >= 0) & (cols < m_orig)
    rows, cols = rows[mask], cols[mask]

    sezioni = np.column_stack([rows, cols])

    if len(sezioni) == 0:
        raise ValueError("No valid section points inside raster")

    # -----------------------------
    # CLEAN DATA
    # -----------------------------
    pnt[pnt < 0] = np.nan
    aree_monte[aree_monte < 0] = np.nan

    # -----------------------------
    # PAD GRID (for flow handling)
    # -----------------------------
    pnt_pad = np.full((n_orig + 2, m_orig + 2), np.nan)
    pnt_pad[1:-1, 1:-1] = pnt

    pnt = pnt_pad
    n, m = pnt.shape
    pnt_flat = pnt.flatten()

    # -----------------------------
    # FLOW DIRECTION KERNEL
    # -----------------------------
    P = np.flipud(np.fliplr(P))
    dd = P.flatten()

    iD = np.array([-n-1, -n, -n+1,
                   -1,    0,   1,
                   n-1,  n,  n+1])

    # -----------------------------
    # SORT SECTIONS BY UPSLOPE AREA
    # -----------------------------
    I_sezioni = np.ravel_multi_index(
        (sezioni[:, 0], sezioni[:, 1]),
        (n_orig, m_orig)
    )

    aree_vals = aree_monte.flatten()[I_sezioni]
    ind_sort = np.argsort(aree_vals)

    sezioni = sezioni[ind_sort]
    ind_reverse = np.argsort(ind_sort)

    elenco_sezioni = [
        np.ravel_multi_index((s[0] + 1, s[1] + 1), (n, m))
        for s in sezioni
    ]

    # -----------------------------
    # MAIN LOOP
    # -----------------------------
    aree = []
    pixel_owner = {}   # pixel -> basin id
    uf = UnionFind(len(elenco_sezioni))

    for s_idx, seed in enumerate(elenco_sezioni):

        basin = set([seed])
        frontier = set([seed])

        while frontier:
            new_front = set()

            for pt in frontier:
                neigh = pt + iD

                valid = (neigh >= 0) & (neigh < pnt_flat.size)
                neigh = neigh[valid]

                # FLOW MATCH (SAFE: integer equality logic)
                mask_flow = (pnt_flat[neigh] == dd)
                valid_neigh = neigh[mask_flow]

                new_front.update(valid_neigh.tolist())

            basin.update(new_front)
            frontier = new_front - basin

        # -----------------------------
        # MERGE WITH PREVIOUS BASINS IF OVERLAP
        # -----------------------------
        for px in basin:
            if px in pixel_owner:
                uf.union(s_idx, pixel_owner[px])

        for px in basin:
            pixel_owner[px] = s_idx

        aree.append(basin)

    # -----------------------------
    # MERGE FINAL BASINS (DSU)
    # -----------------------------
    merged = {}
    for i in range(len(aree)):
        root = uf.find(i)
        merged.setdefault(root, set()).update(aree[i])

    final_aree = list(merged.values())

    # -----------------------------
    # BACK TO ORIGINAL GRID INDEX
    # -----------------------------
    aree_out = []

    for area in final_aree:
        i, j = np.unravel_index(list(area), (n, m))
        i = i - 1
        j = j - 1
        i = (n - 2) - i  # flip back

        idx = np.ravel_multi_index((i, j), (n_orig, m_orig))
        aree_out.append(idx)

    # restore original order of sections
    aree_out = [aree_out[i] for i in ind_reverse]

    return aree_out