from __future__ import annotations
from pathlib import Path
from typing import Any, Literal
import yaml
from pydantic import BaseModel, Field

class ProjectCfg(BaseModel):
    name: str = "nt_cu_prospectivity"
    crs_epsg: int = 28352
    grid_size_m: int = 500

class PathsCfg(BaseModel):
    data_raw: Path = Path("data/raw")
    data_interim: Path = Path("data/interim")
    data_processed: Path = Path("data/processed")
    outputs: Path = Path("outputs")

class AOICfg(BaseModel):
    aoi_file: Path | None = None
    aoi_layer: str | None = None
    bbox: list[float] | None = None

class LabelsCfg(BaseModel):
    ozmin_file: Path
    ozmin_layer: str
    buffer_m: int = 5000
    positive_status: list[str] = Field(default_factory=list)

class FeaturesCfg(BaseModel):
    dem_raster: Path
    geology_file: Path
    geology_layer: str
    faults_file: Path
    faults_layer: str
    mag_raster: Path
    aem_raster: Path | None = None
    geochem_file: Path | None = None

class EstimatorCfg(BaseModel):
    name: Literal["HistGradientBoostingClassifier"] = "HistGradientBoostingClassifier"
    params: dict[str, Any] = Field(default_factory=dict)

class PUSetting(BaseModel):
    name: str
    bg_ratio: int | Literal["all"]
    repeats: int

class PUEnsembleCfg(BaseModel):
    settings: list[PUSetting]

class ModelCfg(BaseModel):
    random_seed: int = 42
    estimator: EstimatorCfg = EstimatorCfg()
    pu_ensemble: PUEnsembleCfg

class OutputsCfg(BaseModel):
    table_parquet: Path
    feature_cols_txt: Path
    model_dir: Path
    pred_dir: Path
    fig_dir: Path

class AppCfg(BaseModel):
    project: ProjectCfg
    paths: PathsCfg
    aoi: AOICfg
    labels: LabelsCfg
    features: FeaturesCfg
    model: ModelCfg
    outputs: OutputsCfg

def load_config(path: str | Path) -> AppCfg:
    path = Path(path)
    cfg_dict = yaml.safe_load(path.read_text(encoding="utf-8"))
    return AppCfg.model_validate(cfg_dict)
