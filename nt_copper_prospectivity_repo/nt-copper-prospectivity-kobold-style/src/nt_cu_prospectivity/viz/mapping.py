from __future__ import annotations
from pathlib import Path
import geopandas as gpd
from ..utils.io import ensure_dir

def attach_predictions_to_grid(grid: gpd.GeoDataFrame, pred_mean, pred_std) -> gpd.GeoDataFrame:
    out = grid.copy()
    out["p_mean"] = pred_mean
    out["p_std"] = pred_std
    return out

def export_targeting_layers(gdf: gpd.GeoDataFrame, out_dir: Path) -> tuple[Path, Path]:
    out_dir = ensure_dir(out_dir)
    gpkg = out_dir / "targeting_map.gpkg"
    parquet = out_dir / "targeting_map.parquet"
    gdf.to_file(gpkg, layer="targeting", driver="GPKG")
    gdf.drop(columns=["geometry"]).to_parquet(parquet, index=False)
    return gpkg, parquet
