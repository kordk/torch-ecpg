import torch
import pandas as pd
import numpy as np
from tecpg.processing import tecpg_mlr_lstsq

# dummy data
M = pd.DataFrame(np.random.rand(10, 50), index=[f"cg{i}" for i in range(10)])
G = pd.DataFrame(np.random.rand(5, 50), index=[f"gene{i}" for i in range(5)])
C = pd.DataFrame(np.random.rand(2, 50), index=["cov1", "cov2"]).T

# Note: threshold is arbitrary, just want some to pass for testing
res_deep = tecpg_mlr_lstsq(M, G, C, p_thresh=1.0, compute_ig_deep=True)
print(res_deep.head())
