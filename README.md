# AI-driven Mineral Prospectivity Mapping for Cu in Northern Territory (NT), Australia

**Repository:** `AI-driven-Mineral-Prospectivity-Mapping-for-Cu-in-Northern-Territory`  
Industry-style **Positive–Unlabeled (PU)** mineral prospectivity mapping pipeline for **Copper (Cu)** in **Northern Territory, Australia**.

This project integrates **geology, geochemistry, geophysics (magnetic + gravity + TMI/RTP), and DEM-derived terrain** into a **500 m grid** and trains an **XGBoost (GPU)** model to produce:

- **Prospectivity probability surface** (ranking / target generation)
- **Uncertainty map** via ensemble standard deviation
- **Explainability** via **SHAP (global + local)**



## 1) Problem Formulation (PU Learning)

We formulate copper prospectivity mapping as a **Positive–Unlabeled (PU)** binary classification problem:

- **Positives (y=1):** known Cu occurrences (OZMIN / STRIKE mineral occurrences), buffered by **5 km**
- **Background/Unlabeled (y=0):** remaining grid cells treated as **background** (not true negatives)

The model outputs a **probabilistic prospectivity score** used for **ranking** and **target generation**, evaluated using **PR-AUC** and **Recall@K%** under **spatial block validation**.



## 2) Study Area & Spatial Reference

- **Region:** Northern Territory (NT), Australia  
- **Grid resolution:** **500 m**  
- **CRS:** all datasets are standardized to **EPSG:28352 (GDA94 / MGA Zone 52)**



## 3) Data Sources (Public)

All data used in this repository are publicly available.

### 3.1 Geological maps
- `GeologicUnitPolygons1M`
- `ShearDisplacementLines1M`
- `Contacts1M`

### 3.2 Geochemistry
- Stream sediment / soil / rock geochemistry (statistics computed per grid / radius)

### 3.3 Geophysics
- Magnetic anomaly map (NT subset)
- Magnetic line features
- Gravity (including line features / amplitudes)

### 3.4 DEM & terrain derivatives
A large DEM dataset (~24 GB) is processed into rasters such as:
- `aspect_30m.tif`
- `curv_profile_30m_MGA52.tif`
- `NT_DEM_30m_MGA52_bigtif.tif`
- `roughness_30m.tif`
- `slope_30m.tif`

### 3.5 Labels (Known Cu occurrences)
- OZMIN mineral deposits database + STRIKE mineral occurrences (Cu)

### Data portals (URLs)

STRIKE (Mineral Occurrences + Geochemistry): http://strike.nt.gov.au/wss.html
GA Portal (Magnetic anomaly / GADDS): https://portal.ga.gov.au/persona/gadds
GA eCat DEM metadata: https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/72759

## 4) Labeling Strategy

- **Buffer-based labeling:** Cu occurrences are buffered by **5 km**.
- The `label` column is used to construct:
  - **Binary target**: `y = 1` if `label > 0` else `0`
  - **Sample weighting** (positives only): `sample_weight` derived from **occurrence status/weight** (when provided)



## 5) Feature Engineering

### 5.1 Feature groups

- **Magnetic line features**
  - `dist_mag_line_m`, `mag_nearest_amp`, `mag_len_5km`, `mag_amp_max_5km`

- **Geochemistry statistics**
  - mean / count / median / rate (and related robust summaries)

- **Terrain derivatives (DEM-based)**
  - slope, curvature, roughness, elevation, aspect  
  - statistics computed as: **mean / std / p90**

- **Distance-based geology/structure**
  - faults / shear zones / contacts  
  - lithology/age-derived proximity & geological context features

### 5.2 Transformations

- `log` / `log1p`
- percentiles (e.g., `p90`)
- aggregation within radius (e.g., **5 km**)
- NaN-safe missing handling



## 6) Model & Training

### 6.1 Model

- **XGBoost (XGBClassifier)** with **GPU acceleration**
  - `tree_method="hist"`
  - `device="cuda"`

### 6.2 Spatial Validation

- **Spatial block split** to reduce spatial autocorrelation leakage:
  - `VAL_FRAC = 0.2`
  - `BLOCK_SIZE_M = 20,000` (**20 km** blocks)

### 6.3 PU / Background Sampling 

Three training settings are supported:

- `pos1_bg1`: positives : background = **1 : 1** (random BG sampling) × **10 repeats**
- `pos1_bg2`: positives : background = **1 : 2** (random BG sampling) × **10 repeats**
- `all_bg`: use **all background pool** × **1 run**

### 6.4 Ensemble Outputs

For each setting:

- Save per-repeat predictions over the full grid
- Compute:
  - `pred_mean.npy` = ensemble mean prospectivity score
  - `pred_std.npy`  = ensemble uncertainty (standard deviation)



## 7) Evaluation Metrics

The pipeline is optimized for **rare positives** and **target ranking**:

- **PR-AUC (Average Precision)**
- **Recall@1%**
- **Recall@5%**

Example results (current run):

| setting   | repeats | pr_auc (min) | pr_auc (max) | pr_auc (mean) |
|----------|---------|--------------|--------------|---------------|
| pos1_bg1 | 10      | 0.985736     | 0.987387     | ~0.9864       |
| pos1_bg2 | 10      | 0.989362     | 0.990261     | ~0.9898       |
| all_bg   | 1       | 0.991110     | 0.991110     | 0.991110      |

All metrics are exported to `metrics_all_repeats.csv`.


## 8) Explainability (XAI) — SHAP

We use **SHAP** for:

- **Global feature importance** (mean |SHAP|)
- **Local explanation** (per target cell / area)

### 8.1 Key drivers (global SHAP)

From SHAP summary plots, the most influential features include:

- `dist_mag_line_m` (dominant)
- `dem_mean`
- `mag_amp_max_5km`
- `grav_amp_max_5km`
- `tmi_p90`, `rtp_p90`, `tmi_mean`, `rtp_mean`, ...

**Interpretation (from beeswarm patterns):**
- Smaller `dist_mag_line_m` (closer to magnetic lineaments) tends to **increase** the prospectivity score.
- Magnetic / gravity amplitude features (e.g., `mag_amp_max_5km`, `grav_amp_max_5km`) often push the model toward **higher** prospectivity.
- Terrain and TMI/RTP statistics contribute **secondary but consistent** effects.

Notebooks:
- `notebooks/shap_global.ipynb`
- `notebooks/shap_local_targets.ipynb`

# Project Structure & Installation

## 9) Project Structure (Updated)


AI-driven-Mineral-Prospectivity-Mapping-for-Cu-in-Northern-Territory/
├── .gitignore
├── Makefile
├── README.md
├── configs/
│   └── base.yaml
├── environment.yml
├── gis_env/
├── notebooks/
│   ├── 01_topology_analyse.ipynb
│   ├── 02_geology_analyse.ipynb
│   ├── 03_geophysics_analyse.ipynb
│   ├── 04_mineral_deposit_analyse.ipynb
│   ├── 05_geochemistry_analyse.ipynb
│   ├── Analyse Result.ipynb
│   ├── Prediction.ipynb
│   ├── set_all_data_into_grid.ipynb
│   ├── shap_global.ipynb
│   └── shap_local_targets.ipynb
├── pyproject.toml
├── requirements.txt
├── scripts/
│   └── train_pu_ensemble_xgb.py
├── src/
│   └── nt_cu_prospectivity/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── explain/
│       │   ├── __init__.py
│       │   ├── shap_global.py
│       │   └── shap_local.py
│       ├── features/
│       │   ├── __init__.py
│       │   ├── assemble.py
│       │   ├── geochemistry.py
│       │   ├── geology.py
│       │   ├── geophysics.py
│       │   ├── grid.py
│       │   └── topography.py
│       ├── labels/
│       │   ├── __init__.py
│       │   ├── buffer.py
│       │   ├── label_grid.py
│       │   ├── ozmin.py
│       │   └── logging.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── io.py
│       │   ├── metrics.py
│       │   └── pu_ensemble.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── geoutils.py
│       │   ├── io.py
│       │   ├── raster.py
│       │   └── vector.py
│       └── viz/
│           ├── __init__.py
│           └── mapping.py
├── tests/
│   ├── test_config.py
│   ├── test_grid.py
│   └── test_metrics.py
└── train/
    ├── train_pu_ensemble.py
    └── train_pu_ensemble_xgb.py


