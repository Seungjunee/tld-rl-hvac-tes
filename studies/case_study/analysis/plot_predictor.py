"""Fig. 4 of the paper (fig_predictor_4b.pdf): LSTM predicted vs. actual
hourly cooling load over a representative weekday, one panel per source
building (OF/LB/WH/MO), with capacity-normalized nRMSE and R^2 annotated.

The figure is generated at its final printed size (\\textwidth = 7.1667 in),
so the font sizes in rcParams are the sizes on the page; \\includegraphics
must not rescale it. The annotated statistics are asserted against the
values quoted in the paper.

h=1 (next-hour) prediction on the held-out evaluation split. Output:
artifacts/case_study/figures/paper/fig_predictor_4b.pdf (+ png preview in the figures dir).

Run:  python studies/case_study/analysis/plot_predictor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from studies.case_study.core.load_predictor import (
    BUILDING_INPUT_VARIABLES,
    LoadDataset,
    LoadPredictor,
)

# artifacts live in the main checkout; worktrees only carry the repo files
_CANDS = [REPO, Path("/workspaces/sinergym")]
DATA = next(p for p in _CANDS if (p / "artifacts/case_study/exports").exists())
OUT = DATA / "artifacts/case_study/figures/paper_figs_2026-06-13"
MAN = REPO / "artifacts/case_study/figures/paper"
MAN.mkdir(parents=True, exist_ok=True)

DS = {
    "office_5zone": "build_office_observation_dataset",
    "office_lbuilding_8zone": "build_lbuilding_observation_dataset",
    "warehouse_3zone": "build_warehouse_observation_dataset",
    "mediumoffice_15zone": "build_mediumoffice_observation_dataset",
}
QR = {  # rated chiller capacity [kW] (buildings.yaml physical.q_rated)
    "office_5zone": 80.0,
    "office_lbuilding_8zone": 70.0,
    "warehouse_3zone": 57.0,
    "mediumoffice_15zone": 500.0,
}
LAB = {
    "office_5zone": "OF (5-zone)",
    "office_lbuilding_8zone": "LB (8-zone)",
    "warehouse_3zone": "WH (3-zone)",
    "mediumoffice_15zone": "MO (15-zone)",
}
YAXIS = {  # (ymax, tick step) [kW]; standard decimal steps, frame ends on a
    "office_5zone": (75, 25),            # tick, and OF/MO carry extra headroom
    "office_lbuilding_8zone": (40, 10),  # so the legend and the R2/nRMSE box
    "warehouse_3zone": (40, 10),         # sit clear of the curves
    "mediumoffice_15zone": (400, 100),
}
# values quoted in the paper (Section 4.2 and the panel annotations)
EXPECT = {  # (R^2 rounded to 2 dp, nRMSE rounded to 1 dp)
    "office_5zone": (0.92, 4.9),
    "office_lbuilding_8zone": (0.99, 1.0),
    "warehouse_3zone": (0.94, 2.5),
    "mediumoffice_15zone": (0.95, 2.9),
}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "axes.grid": True, "grid.alpha": 0.2, "grid.linewidth": 0.5,
    "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "figure.dpi": 200,
    "pdf.fonttype": 42, "ps.fonttype": 42})
C_ACT, C_PRED = "0.30", "#1f77b4"

fig, axes = plt.subplots(1, 4, figsize=(7.1667, 1.66))
for ax, reg in zip(axes, DS):
    ck = torch.load(f"{DATA}/artifacts/case_study/predictors/{reg}_lstm.pt", map_location="cpu", weights_only=False)
    H, S = int(ck["horizon"]), int(ck["seq_len"])
    csv_path = f"{DATA}/artifacts/case_study/exports/{DS[reg]}/observations_processed.csv"
    ds = LoadDataset(csv_path, BUILDING_INPUT_VARIABLES[reg], S, H)
    df_time = pd.read_csv(csv_path)
    if "hour" in df_time.columns:
        hour_of_day = df_time["hour"].to_numpy().astype(int)
    else:  # cyclic encoding (MO dataset): recover the integer hour
        ang = np.arctan2(df_time["hour_sin"], df_time["hour_cos"])
        hour_of_day = np.rint(ang / (2 * np.pi) * 24).astype(int) % 24
    m = LoadPredictor(len(BUILDING_INPUT_VARIABLES[reg]), ck["hidden_dim"], ck["num_layers"], H)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    n = len(ds)
    val = range(int(n * 0.8), n)
    Pa, Aa = [], []
    with torch.no_grad():
        for i in val:
            x, y = ds[i]
            Pa.append(m(x.unsqueeze(0)).squeeze(0).numpy())  # (H,)
            Aa.append(y.numpy())
    Pa = np.clip(np.array(Pa) * ds.target_std + ds.target_mean, 0, None)  # (n,H) W
    Aa = np.array(Aa) * ds.target_std + ds.target_mean
    # summary stats: R^2 over all horizons; horizon-mean capacity-normalized nRMSE
    qr_w = QR[reg] * 1000.0
    nrmse = np.mean([np.sqrt(np.mean((Pa[:, h] - Aa[:, h]) ** 2)) / qr_w * 100 for h in range(H)])
    r2 = 1 - ((Pa - Aa) ** 2).sum() / ((Aa - Aa.mean()) ** 2).sum()
    assert (round(float(r2), 2), round(float(nrmse), 1)) == EXPECT[reg], \
        f"{reg}: got R2={r2:.3f} nRMSE={nrmse:.2f}, paper says {EXPECT[reg]}"
    # plot the h=1 (next-hour) prediction over one representative weekday (00:00-24:00).
    # 96 samples = 24 h at the 15-min stride. A[k] is the forecast that follows the
    # length-S input window, so its realized clock time is row (val0+k+S); align the
    # window to a midnight starting a day with non-trivial load (skips zero-load days).
    P, A = Pa[:, 0] / 1000.0, Aa[:, 0] / 1000.0  # kW
    val0 = int(n * 0.8)
    amax = float(A.max())
    k0 = 0
    for k in range(len(A) - 96):
        if hour_of_day[val0 + k + S] == 0 and A[k:k + 96].max() > 0.15 * amax:
            k0 = k
            break
    w = slice(k0, k0 + 96)
    t = np.arange(96) * 0.25  # hours of day, 0-24
    # on-peak window (17-20 h), same tint as the operating-profile figure
    ax.axvspan(17, 20, color="#fde9e9", lw=0, zorder=0)
    ax.fill_between(t, 0, A[w], color=C_ACT, alpha=0.22, label="Actual", lw=0)
    ax.plot(t, A[w], color=C_ACT, lw=0.9)
    ax.plot(t, P[w], color=C_PRED, lw=1.3, label="Predicted")
    ax.set_title(LAB[reg])
    ax.set_xlabel("Hour of day")
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ymax, ystep = YAXIS[reg]
    ax.set_ylim(0, ymax)
    ax.set_yticks(range(0, ymax + 1, ystep))
    ax.annotate(
        f"$R^2={r2:.2f}$\nnRMSE ${nrmse:.1f}\\%$",
        xy=(0.97, 0.96), xycoords="axes fraction", ha="right", va="top",
        fontsize=7.5, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.5, alpha=0.9),
    )
    print(f"{reg}: R2={r2:.3f} nRMSE={nrmse:.2f}% (day offset k0={k0})")
axes[0].set_ylabel("Cooling load [kW]")
axes[0].legend(loc="upper left", framealpha=0.9, handlelength=1.1,
               borderpad=0.3, handletextpad=0.5, labelspacing=0.3)
fig.tight_layout(pad=0.25)
fig.savefig(MAN / "fig_predictor_4b.pdf")
fig.savefig(OUT / "fig_predictor_4b.png", dpi=160)
print(f"fig_predictor_4b done -> {MAN}")
