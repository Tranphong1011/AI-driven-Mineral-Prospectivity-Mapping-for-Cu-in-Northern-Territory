from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import typer
from rich import print

from .config import load_config
from .logging import setup_logging
from .utils.io import ensure_dir
from .utils.geoutils import load_aoi
from .features.grid import make_grid_from_aoi
from .features.assemble import assemble_feature_table
from .labels.ozmin import load_ozmin
from .labels.buffer import make_positive_buffers
from .labels.label_grid import label_grid_by_buffers
from .models.pu_ensemble import train_pu_ensemble
from .models.io import load_models, predict_ensemble
from .viz.mapping import attach_predictions_to_grid, export_targeting_layers
from .explain.shap_global import run_shap_global
from .explain.shap_local import run_shap_local

app = typer.Typer(add_completion=False)

@app.command()
def build_dataset(config: Path = typer.Option(..., exists=True)):
    setup_logging()
    cfg = load_config(config)
    ensure_dir(cfg.paths.data_processed)
    ensure_dir(cfg.outputs.model_dir)
    ensure_dir(cfg.outputs.pred_dir)
    ensure_dir(cfg.outputs.fig_dir)

    aoi = load_aoi(cfg.aoi.aoi_file, cfg.aoi.aoi_layer, cfg.aoi.bbox, cfg.project.crs_epsg)
    grid = make_grid_from_aoi(aoi.gdf, cfg.project.grid_size_m, cfg.project.crs_epsg)

    feat = assemble_feature_table(
        grid_polys=grid.gdf,
        grid_centroids=grid.centroids,
        crs_epsg=cfg.project.crs_epsg,
        dem_path=cfg.features.dem_raster,
        geology_file=cfg.features.geology_file,
        geology_layer=cfg.features.geology_layer,
        mag_raster=cfg.features.mag_raster,
        aem_raster=cfg.features.aem_raster,
        geochem_file=cfg.features.geochem_file,
    )

    oz = load_ozmin(cfg.labels.ozmin_file, cfg.labels.ozmin_layer, cfg.project.crs_epsg)
    buf = make_positive_buffers(oz, cfg.labels.buffer_m)
    y = label_grid_by_buffers(grid.gdf, buf)

    ml = feat.copy()
    ml["label"] = y.values
    ml["x"] = grid.centroids.geometry.x.values
    ml["y"] = grid.centroids.geometry.y.values
    ml.to_parquet(cfg.outputs.table_parquet, index=False)

    feature_cols = [c for c in ml.columns if c not in {"cell_id","label","x","y"}]
    Path(cfg.outputs.feature_cols_txt).write_text("\n".join(feature_cols), encoding="utf-8")

    grid_out = cfg.paths.data_processed / "grid.gpkg"
    grid.gdf.to_file(grid_out, layer="grid", driver="GPKG")

    print(f"[green]Saved:[/green] {cfg.outputs.table_parquet}")
    print(f"[green]Saved:[/green] {grid_out}")

@app.command()
def train(config: Path = typer.Option(..., exists=True)):
    setup_logging()
    cfg = load_config(config)
    df = pd.read_parquet(cfg.outputs.table_parquet)
    feature_cols = Path(cfg.outputs.feature_cols_txt).read_text(encoding="utf-8").splitlines()
    X = df[feature_cols].to_numpy()
    y = df["label"].to_numpy()
    settings = [s.model_dump() for s in cfg.model.pu_ensemble.settings]
    res = train_pu_ensemble(X, y, settings, cfg.model.estimator.params, cfg.outputs.model_dir, seed=cfg.model.random_seed)
    print(f"[green]Models saved to:[/green] {cfg.outputs.model_dir}")
    print(res.metrics.groupby('setting')[['pr_auc','recall@1%','recall@5%']].mean())

@app.command()
def predict(config: Path = typer.Option(..., exists=True)):
    setup_logging()
    cfg = load_config(config)
    df = pd.read_parquet(cfg.outputs.table_parquet)
    feature_cols = Path(cfg.outputs.feature_cols_txt).read_text(encoding="utf-8").splitlines()
    X = df[feature_cols].to_numpy()

    grid_gpkg = cfg.paths.data_processed / "grid.gpkg"
    grid = gpd.read_file(grid_gpkg, layer="grid").to_crs(epsg=cfg.project.crs_epsg)

    model_paths = load_models(cfg.outputs.model_dir)
    if not model_paths:
        raise RuntimeError(f"No models found in {cfg.outputs.model_dir}.")

    p_mean, p_std = predict_ensemble(model_paths, X)
    gdf_pred = attach_predictions_to_grid(grid, p_mean, p_std)
    gpkg, parquet = export_targeting_layers(gdf_pred, cfg.outputs.pred_dir)
    print(f"[green]Exported:[/green] {gpkg}")
    print(f"[green]Exported:[/green] {parquet}")

@app.command("shap-global")
def shap_global(config: Path = typer.Option(..., exists=True)):
    setup_logging()
    cfg = load_config(config)
    df = pd.read_parquet(cfg.outputs.table_parquet)
    feature_cols = Path(cfg.outputs.feature_cols_txt).read_text(encoding="utf-8").splitlines()
    X = df[feature_cols]
    model_paths = load_models(cfg.outputs.model_dir)
    if not model_paths:
        raise RuntimeError("No models found. Train first.")
    out = run_shap_global(model_paths[0], X, cfg.outputs.fig_dir)
    print(f"[green]Saved SHAP:[/green] {out}")

@app.command("shap-local")
def shap_local(config: Path = typer.Option(..., exists=True), top_n: int = 20):
    setup_logging()
    cfg = load_config(config)
    df = pd.read_parquet(cfg.outputs.table_parquet)
    feature_cols = Path(cfg.outputs.feature_cols_txt).read_text(encoding="utf-8").splitlines()
    X = df[feature_cols]
    model_paths = load_models(cfg.outputs.model_dir)
    if not model_paths:
        raise RuntimeError("No models found. Train first.")
    # choose top targets
    try:
        targeting = pd.read_parquet(cfg.outputs.pred_dir / "targeting_map.parquet")
        scores = targeting["p_mean"].to_numpy()
    except Exception:
        import joblib
        m = joblib.load(model_paths[0])
        scores = m.predict_proba(X)[:, 1]
    top_idx = list(np.argsort(-scores)[:top_n])
    outs = run_shap_local(model_paths[0], X, top_idx, Path(cfg.outputs.fig_dir) / "shap_local")
    print(f"[green]Saved {len(outs)} local SHAP plots.[/green]")

def main():
    app()

if __name__ == "__main__":
    main()
