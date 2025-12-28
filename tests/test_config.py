from pathlib import Path
from nt_cu_prospectivity.config import load_config

def test_load_config():
    cfg = load_config(Path('configs/base.yaml'))
    assert cfg.project.grid_size_m > 0
