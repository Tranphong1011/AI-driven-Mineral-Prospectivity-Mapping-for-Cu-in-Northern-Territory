from __future__ import annotations
from pathlib import Path
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd

def clip_raster(src_path: Path, aoi: gpd.GeoDataFrame, dst_path: Path) -> Path:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        geoms = [g.__geo_interface__ for g in aoi.geometry]
        out_img, out_transform = mask(src, geoms, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({"height": out_img.shape[1], "width": out_img.shape[2], "transform": out_transform})
    with rasterio.open(dst_path, "w", **out_meta) as dst:
        dst.write(out_img)
    return dst_path

def sample_raster_at_points(raster_path: Path, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    with rasterio.open(raster_path) as src:
        vals = list(src.sample(zip(xs, ys)))
    arr = np.array(vals)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr
