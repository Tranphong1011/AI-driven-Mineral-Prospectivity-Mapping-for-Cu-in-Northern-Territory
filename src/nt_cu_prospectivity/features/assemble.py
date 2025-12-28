from __future__ import annotations
from pathlib import Path
import pandas as pd
import geopandas as gpd
from .topography import build_dem_features
from .geology import build_geology_features
from .geophysics import build_geophysics_features
from .geochemistry import build_geochem_features

def assemble_feature_table(
    grid_polys: gpd.GeoDataFrame,
    grid_centroids: gpd.GeoDataFrame,
    crs_epsg: int,
    dem_path: Path,
    geology_file: Path,
    geology_layer: str,
    mag_raster: Path,
    aem_raster: Path | None = None,
    geochem_file: Path | None = None,
) -> pd.DataFrame:
    xs = grid_centroids.geometry.x.values
    ys = grid_centroids.geometry.y.values
    topo = build_dem_features(dem_path, xs, ys)
    geo = build_geology_features(geology_file, geology_layer, grid_polys, crs_epsg)
    geophys = build_geophysics_features(mag_raster, xs, ys, aem_raster=aem_raster)
    parts = [topo, geo, geophys]
    if geochem_file:
        parts.append(build_geochem_features(geochem_file, grid_centroids, crs_epsg))
    feat = pd.concat(parts, axis=1)
    feat.insert(0, "cell_id", grid_polys["cell_id"].values)
    feat.index = grid_polys.index
    return feat
