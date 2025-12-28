import numpy as np
from nt_cu_prospectivity.models.metrics import recall_at_k_percent

def test_recall():
    y=np.array([1,0,1,0]); s=np.array([0.9,0.8,0.2,0.1])
    r=recall_at_k_percent(y,s,50)
    assert 0<=r<=1
