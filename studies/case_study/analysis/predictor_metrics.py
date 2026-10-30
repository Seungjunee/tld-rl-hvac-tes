"""Per-building predictor quality with metrics appropriate for intermittent load.

MAPE explodes on near-zero actuals; WH cooling load is 0 ~58% of the time, so
MAPE misrepresents a predictor that R^2 says is good. Compute, on the held-out
val split, per horizon: MAPE (filtered), WAPE (sum|e|/sum|a|), nRMSE (RMSE/mean),
and R^2. WAPE/nRMSE are the standard choices for intermittent demand.
"""

import sys

sys.path.insert(0, "/workspaces/sinergym")
import numpy as np
import torch

from studies.case_study.core.load_predictor import (
    BUILDING_INPUT_VARIABLES,
    LoadDataset,
    LoadPredictor,
)

ROOT = "/workspaces/sinergym"
DS = {
    "office_5zone": "build_office_observation_dataset",
    "office_lbuilding_8zone": "build_lbuilding_observation_dataset",
    "warehouse_3zone": "build_warehouse_observation_dataset",
}

for reg, lab in [
    ("office_5zone", "OF"),
    ("office_lbuilding_8zone", "LB"),
    ("warehouse_3zone", "WH"),
]:
    ck = torch.load(
        f"{ROOT}/artifacts/case_study/predictors/{reg}_lstm.pt",
        map_location="cpu",
        weights_only=False,
    )
    H, S = int(ck["horizon"]), int(ck["seq_len"])
    ds = LoadDataset(
        f"{ROOT}/artifacts/case_study/exports/{DS[reg]}/observations_processed.csv",
        BUILDING_INPUT_VARIABLES[reg],
        S,
        H,
    )
    m = LoadPredictor(
        len(BUILDING_INPUT_VARIABLES[reg]), ck["hidden_dim"], ck["num_layers"], H
    )
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    n = len(ds)
    val = range(int(n * 0.8), n)
    P, A = [], []
    with torch.no_grad():
        for i in val:
            x, y = ds[i]
            P.append(m(x.unsqueeze(0)).squeeze(0).numpy())
            A.append(y.numpy())
    P = np.clip(np.array(P) * ds.target_std + ds.target_mean, 0, None)
    A = np.array(A) * ds.target_std + ds.target_mean
    amean = A[A > 0].mean()
    thr = 0.1 * amean

    def mape(h):
        msk = A[:, h] > thr
        return np.mean(np.abs(P[msk, h] - A[msk, h]) / A[msk, h]) * 100

    def wape(h):
        return np.sum(np.abs(P[:, h] - A[:, h])) / np.sum(np.abs(A[:, h])) * 100

    def nrmse(h):
        return np.sqrt(np.mean((P[:, h] - A[:, h]) ** 2)) / A[:, h].mean() * 100

    r2 = 1 - ((P - A) ** 2).sum() / ((A - A.mean()) ** 2).sum()
    zero = (A[:, 0] <= thr).mean() * 100
    print(
        f"{lab}: zero-load {zero:.0f}% | R2 {r2:.3f} | "
        f"MAPE h1/h12/mean {mape(0):.1f}/{mape(H-1):.1f}/{np.mean([mape(h) for h in range(H)]):.1f} | "
        f"WAPE h1/h12/mean {wape(0):.1f}/{wape(H-1):.1f}/{np.mean([wape(h) for h in range(H)]):.1f} | "
        f"nRMSE mean {np.mean([nrmse(h) for h in range(H)]):.1f}"
    )
