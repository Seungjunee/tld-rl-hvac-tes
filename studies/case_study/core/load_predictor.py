from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Building-specific input variable definitions
# These are the zone-level variables absorbed by the LSTM front-end
# (removed from RL observation via reduce_observation in recipes.yaml)
# ---------------------------------------------------------------------------

BUILDING_INPUT_VARIABLES: Dict[str, List[str]] = {
    "restaurant_3zone": [
        "month_cos",
        "month_sin",
        "is_weekend",
        "dow_cos",
        "dow_sin",
        "hour_cos",
        "hour_sin",
        "minutes",
        "outdoor_temperature",
        "Dining_C_setpoint",
        "occupancy_fraction",
        "Dining_temperature",
        "Kitchen_temperature",
        "CoolingRate",
    ]
    + [f"forecast_{i}_Dry Bulb Temperature" for i in range(1, 13)]
    + [f"forecast_{i}_Relative Humidity" for i in range(1, 13)],
    "office_5zone": [
        "month_cos",
        "month_sin",
        "is_weekend",
        "dow_cos",
        "dow_sin",
        "hour_cos",
        "hour_sin",
        "minutes",
        "outdoor_temperature",
        "Z1C_setpoint",
        "occupancy_fraction",
        "Z1_temperature",
        "Z2_temperature",
        "Z3_temperature",
        "Z4_temperature",
        "Z5_temperature",
        "CoolingRate",
    ]
    + [f"forecast_{i}_Dry Bulb Temperature" for i in range(1, 13)]
    + [f"forecast_{i}_Relative Humidity" for i in range(1, 13)],
    "office_lbuilding_8zone": [
        "month_cos",
        "month_sin",
        "is_weekend",
        "dow_cos",
        "dow_sin",
        "hour_cos",
        "hour_sin",
        "minutes",
        "outdoor_temperature",
        "Z1C_setpoint",
        "occupancy_fraction",
        "Z1_temperature",
        "Z2_temperature",
        "Z3_temperature",
        "Z4_temperature",
        "Z5_temperature",
        "Z6_temperature",
        "Z7_temperature",
        "Z8_temperature",
        "CoolingRate",
    ]
    + [f"forecast_{i}_Dry Bulb Temperature" for i in range(1, 13)]
    + [f"forecast_{i}_Relative Humidity" for i in range(1, 13)],
    "stripmall_10zone": [
        "month_cos",
        "month_sin",
        "is_weekend",
        "dow_cos",
        "dow_sin",
        "hour_cos",
        "hour_sin",
        "minutes",
        "outdoor_temperature",
        "Z1C_setpoint",
        "occupancy_fraction",
    ]
    + [f"Z{i}_temperature" for i in range(1, 11)]
    + ["CoolingRate"]
    + [f"forecast_{i}_Dry Bulb Temperature" for i in range(1, 13)]
    + [f"forecast_{i}_Relative Humidity" for i in range(1, 13)],
    "warehouse_3zone": [
        "month_cos",
        "month_sin",
        "is_weekend",
        "dow_cos",
        "dow_sin",
        "hour_cos",
        "hour_sin",
        "minutes",
        "outdoor_temperature",
        "Office_C_setpoint",
        "occupancy_fraction",
        "Office_temperature",
        "FineStorage_temperature",
        "BulkStorage_temperature",
        "CoolingRate",
    ]
    + [f"forecast_{i}_Dry Bulb Temperature" for i in range(1, 13)]
    + [f"forecast_{i}_Relative Humidity" for i in range(1, 13)],
    # Stand-alone retail (RT): first zone's cooling setpoint and the 24
    # weather-forecast inputs, as the strip mall.
    "retail_4zone": [
        "month_cos",
        "month_sin",
        "is_weekend",
        "dow_cos",
        "dow_sin",
        "hour_cos",
        "hour_sin",
        "minutes",
        "outdoor_temperature",
        "Z1C_setpoint",
        "occupancy_fraction",
    ]
    + [f"Z{i}_temperature" for i in range(1, 5)]
    + ["CoolingRate"]
    + [f"forecast_{i}_Dry Bulb Temperature" for i in range(1, 13)]
    + [f"forecast_{i}_Relative Humidity" for i in range(1, 13)],
    # Quick-service restaurant (QS), the retail layout with 2 zones
    # (Z1 = Dining, whose cooling setpoint follows the 30 / 23.89 C schedule).
    "fastfood_2zone": [
        "month_cos",
        "month_sin",
        "is_weekend",
        "dow_cos",
        "dow_sin",
        "hour_cos",
        "hour_sin",
        "minutes",
        "outdoor_temperature",
        "Z1C_setpoint",
        "occupancy_fraction",
    ]
    + [f"Z{i}_temperature" for i in range(1, 3)]
    + ["CoolingRate"]
    + [f"forecast_{i}_Dry Bulb Temperature" for i in range(1, 13)]
    + [f"forecast_{i}_Relative Humidity" for i in range(1, 13)],
    # MediumOffice 15 conditioned zones — added 2026-05-29.
    # Input dim: 8 (datetime) + 1 (outdoor) + 1 (setpoint) + 1 (occ) + 15 (zone temps)
    #          + 1 (CoolingRate) + 24 (weather forecast) = 51
    "mediumoffice_15zone": [
        "month_cos",
        "month_sin",
        "is_weekend",
        "dow_cos",
        "dow_sin",
        "hour_cos",
        "hour_sin",
        "minutes",
        "outdoor_temperature",
        "Z1C_setpoint",
        "occupancy_fraction",
    ]
    + [f"Z{i}_temperature" for i in range(1, 16)]
    + ["CoolingRate"]
    + [f"forecast_{i}_Dry Bulb Temperature" for i in range(1, 13)]
    + [f"forecast_{i}_Relative Humidity" for i in range(1, 13)],
}


# ---------------------------------------------------------------------------
# LSTM Model
# ---------------------------------------------------------------------------


class LoadPredictor(nn.Module):
    """LSTM front-end that predicts aggregate thermal load trajectory.

    Maps building-side observation sequence -> H-step hourly load forecast.
    Corresponds to g_θ(o1_t) in the paper.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        horizon: int = 24,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_dim) observation sequence.

        Returns:
            (batch, horizon) predicted load trajectory.
        """
        # CoolingRate (Q_load) is physically non-negative, but clamp is applied
        # AFTER denormalization in the wrapper (LoadPredictorWrapper._augment).
        # Pre-2026-06-04 bug: clamp_min(0.0) here on normalized output forced
        # predicted output >= target_mean (since pred_norm=0 denormalizes to mean),
        # creating a target_mean floor and blocking gradient for negative targets.
        # LSTM then learned only mean prediction (flat line at target_mean).
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class LoadDataset(Dataset):
    """Sliding-window dataset for LSTM load prediction training.

    Reads observations_processed.csv, extracts input features as sequences
    and CoolingRate1..96 as targets (averaged to hourly → 24 values).
    """

    # Mapping from DatetimeWrapper variable names to raw CSV columns + transform.
    # is_weekend, dow_cos, dow_sin require year+month+day jointly and are computed
    # in __init__ (precomputed) rather than via a single-column lambda.
    _DATETIME_TRANSFORMS = {
        # (month - 1): MUST match sinergym DatetimeWrapper's encoding
        # (sinergym/utils/wrappers.py, month_cos/sin). Pre-2026-06-10 this
        # used raw month, rotating the inference month feature by one month
        # vs training (June presented as training-May — outside the Jun-Sep
        # training support).
        "month_cos": ("month", lambda x: np.cos(2 * np.pi * (x - 1) / 12)),
        "month_sin": ("month", lambda x: np.sin(2 * np.pi * (x - 1) / 12)),
        "hour_cos": ("hour", lambda x: np.cos(2 * np.pi * x / 24)),
        "hour_sin": ("hour", lambda x: np.sin(2 * np.pi * x / 24)),
    }

    # Mapping from forecast variable patterns to CSV source columns + shift
    # forecast_i_Dry Bulb Temperature → outdoor_temperature shifted by i hours (i*4 steps)
    # setpoint_forecast_i_<var> → <var> repeated (constant/slow-changing)
    # cpu_load_forecast_i → cpu_loading_fraction shifted by i hours (i*4 steps)
    _FORECAST_PATTERNS = [
        (
            r"^forecast_(\d+)_Dry Bulb Temperature$",
            "outdoor_temperature",
            4,
        ),  # hourly, 4 steps/hour
        (r"^cpu_load_forecast_(\d+)$", "cpu_loading_fraction", 4),
        (r"^setpoint_forecast_(\d+)_(.+)$", None, 0),  # setpoint name in group 2
    ]

    @staticmethod
    def _resolve_forecast_var(var: str, df: pd.DataFrame) -> Optional[np.ndarray]:
        import re

        # Weather forecast: forecast_i_Dry Bulb Temperature
        m = re.match(r"^forecast_(\d+)_Dry Bulb Temperature$", var)
        if m:
            i = int(m.group(1))
            if "outdoor_temperature" in df.columns:
                return (
                    df["outdoor_temperature"]
                    .shift(-i * 4)
                    .fillna(method="ffill")
                    .fillna(method="bfill")
                    .values.astype(np.float32)
                )
            return None

        # CPU load forecast: cpu_load_forecast_i
        m = re.match(r"^cpu_load_forecast_(\d+)$", var)
        if m:
            i = int(m.group(1))
            if "cpu_loading_fraction" in df.columns:
                return (
                    df["cpu_loading_fraction"]
                    .shift(-i * 4)
                    .fillna(method="ffill")
                    .fillna(method="bfill")
                    .values.astype(np.float32)
                )
            return None

        # Setpoint forecast: setpoint_forecast_i_<varname>
        m = re.match(r"^setpoint_forecast_(\d+)_(.+)$", var)
        if m:
            setpoint_var = m.group(2)
            if setpoint_var in df.columns:
                # Setpoints are schedule-based; use shifted value
                i = int(m.group(1))
                return (
                    df[setpoint_var]
                    .shift(-i * 4)
                    .fillna(method="ffill")
                    .fillna(method="bfill")
                    .values.astype(np.float32)
                )
            return None

        return None

    def __init__(
        self,
        csv_path: str | Path,
        input_variables: List[str],
        seq_len: int = 24,
        horizon: int = 24,
        simulation_year: int = 1991,
    ):
        df = pd.read_csv(csv_path)

        # Pre-compute weekday-derived features from raw month/day when present.
        # Datasets exported with datetime encoding (e.g. mediumoffice, whose
        # export profile sets datetime: true) already carry is_weekend/dow_cos/
        # dow_sin and have no raw `month`/`day_of_month`; those are picked up by
        # the column-first lookup below, so the raw-time precompute is skipped.
        self._precomputed: Dict[str, np.ndarray] = {}
        if "month" in df.columns and "day_of_month" in df.columns:
            weekday_arr = np.array(
                [
                    date(simulation_year, int(m), int(d)).weekday()
                    for m, d in zip(df["month"], df["day_of_month"])
                ],
                dtype=np.int32,
            )
            self._precomputed = {
                "is_weekend": (weekday_arr >= 5).astype(np.float32),
                "dow_cos": np.cos(2 * np.pi * weekday_arr / 7).astype(np.float32),
                "dow_sin": np.sin(2 * np.pi * weekday_arr / 7).astype(np.float32),
            }

        # Build input features. Existing CSV columns take priority so that
        # datetime-encoded datasets are used as-is; raw-time datasets fall back
        # to the precomputed weekday features and the datetime transforms.
        input_cols = []
        for var in input_variables:
            if var in df.columns:
                input_cols.append(df[var].values.astype(np.float32))
            elif var in self._precomputed:
                input_cols.append(self._precomputed[var])
            elif var in self._DATETIME_TRANSFORMS:
                raw_col, transform = self._DATETIME_TRANSFORMS[var]
                input_cols.append(transform(df[raw_col].values).astype(np.float32))
            else:
                col = self._resolve_forecast_var(var, df)
                if col is not None:
                    input_cols.append(col)
                else:
                    raise KeyError(f"Input variable not found in CSV: {var}")
        self.inputs = np.column_stack(input_cols)

        # Target: first `horizon` hourly averages of CoolingRate (1h = 4 × 15min samples)
        n_samples = horizon * 4
        cr_cols = [f"CoolingRate{i}" for i in range(1, n_samples + 1)]
        missing_cr = [c for c in cr_cols if c not in df.columns]
        if missing_cr:
            raise KeyError(f"CoolingRate columns missing: {missing_cr[:5]}...")
        cr_raw = df[cr_cols].values.astype(np.float32)  # (N, horizon*4)
        # Average every 4 values (15min → 1h)
        self.targets = cr_raw.reshape(len(cr_raw), horizon, 4).mean(
            axis=2
        )  # (N, horizon)

        self.seq_len = seq_len
        self.horizon = horizon

        # Normalize inputs (z-score)
        self.input_mean = self.inputs.mean(axis=0)
        self.input_std = self.inputs.std(axis=0)
        self.input_std[self.input_std < 1e-8] = 1.0
        self.inputs_norm = (self.inputs - self.input_mean) / self.input_std

        # Normalize targets
        self.target_mean = self.targets.mean()
        self.target_std = self.targets.std()
        if self.target_std < 1e-8:
            self.target_std = 1.0
        self.targets_norm = (self.targets - self.target_mean) / self.target_std

    def __len__(self) -> int:
        return len(self.inputs_norm) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.inputs_norm[idx : idx + self.seq_len])
        # Target anchored at the LAST input row (idx+seq_len-1): its shifted
        # CoolingRate{i} columns are that row's future, so y[0] is the hour
        # right after the most recent observation — matching how the wrapper
        # labels load_forecast_1 at inference. Pre-2026-06-10 this used
        # idx+seq_len (one row later), skewing all forecasts by 15 min.
        y = torch.from_numpy(self.targets_norm[idx + self.seq_len - 1])
        return x, y

    def denormalize_target(self, y: np.ndarray) -> np.ndarray:
        return y * self.target_std + self.target_mean


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_predictor(
    csv_path: str | Path,
    input_variables: List[str],
    output_path: str | Path,
    seq_len: int = 24,
    horizon: int = 24,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    val_split: float = 0.2,
    patience: int = 10,
    device: str = "cpu",
) -> Dict:
    """Train a LoadPredictor and save it.

    Returns dict with training metrics.
    """
    dataset = LoadDataset(csv_path, input_variables, seq_len, horizon)
    n = len(dataset)
    n_val = int(n * val_split)
    n_train = n - n_val
    # Chronological split
    train_ds = torch.utils.data.Subset(dataset, range(n_train))
    val_ds = torch.utils.data.Subset(dataset, range(n_train, n))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = LoadPredictor(
        input_dim=len(input_variables),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        horizon=horizon,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs}: train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Save best model + normalization stats
    model.load_state_dict(best_state)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "input_dim": len(input_variables),
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "horizon": horizon,
            "dropout": dropout,
            "input_variables": input_variables,
            "input_mean": dataset.input_mean,
            "input_std": dataset.input_std,
            "target_mean": float(dataset.target_mean),
            "target_std": float(dataset.target_std),
            "seq_len": seq_len,
        },
        output_path,
    )
    print(f"  Model saved to {output_path}")

    return {
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history["train_loss"]),
        "history": history,
        "model": model,
        "dataset": dataset,
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_predictor(
    model_path: str | Path, device: str = "cpu"
) -> Tuple[LoadPredictor, Dict]:
    """Load a trained LoadPredictor from disk."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = LoadPredictor(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        horizon=checkpoint["horizon"],
        dropout=checkpoint.get("dropout", 0.0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


# ---------------------------------------------------------------------------
# Evaluation & Plotting
# ---------------------------------------------------------------------------


def evaluate_and_plot(
    model: LoadPredictor,
    dataset: LoadDataset,
    building_name: str,
    output_dir: str | Path,
    device: str = "cpu",
) -> Dict:
    """Evaluate model on full dataset and generate accuracy plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    all_preds = []
    all_targets = []

    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device))
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())

    preds_norm = np.concatenate(all_preds)
    targets_norm = np.concatenate(all_targets)
    preds = dataset.denormalize_target(preds_norm)
    targets = dataset.denormalize_target(targets_norm)

    # Metrics per horizon
    horizons = list(range(24))
    rmse_per_h = np.sqrt(((preds - targets) ** 2).mean(axis=0))
    mae_per_h = np.abs(preds - targets).mean(axis=0)
    overall_rmse = np.sqrt(((preds - targets) ** 2).mean())
    overall_mae = np.abs(preds - targets).mean()

    # R² per horizon
    ss_res = ((targets - preds) ** 2).sum(axis=0)
    ss_tot = ((targets - targets.mean(axis=0)) ** 2).sum(axis=0)
    r2_per_h = 1 - ss_res / np.maximum(ss_tot, 1e-8)
    overall_r2 = 1 - ((targets - preds) ** 2).sum() / max(
        ((targets - targets.mean()) ** 2).sum(), 1e-8
    )

    # --- Plot 1: Time series overlay (first 672 steps = 1 week) ---
    n_plot = min(672, len(preds))
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"{building_name} — Predicted vs Actual CoolingRate", fontsize=14)

    horizon = targets.shape[1]
    time_idx = sorted({0, horizon // 4, horizon // 2, horizon - 1})
    for idx, (h, ax) in enumerate(zip(time_idx, axes.flat)):
        ax.plot(targets[:n_plot, h], label="Actual", alpha=0.8, linewidth=0.8)
        ax.plot(preds[:n_plot, h], label="Predicted", alpha=0.8, linewidth=0.8)
        ax.set_title(f"{h+1}h ahead (RMSE={rmse_per_h[h]:.0f}, R²={r2_per_h[h]:.3f})")
        ax.set_xlabel("Timestep (15 min)")
        ax.set_ylabel("CoolingRate (W)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{building_name}_timeseries.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: Scatter (1h and last-hour ahead) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for h, ax in zip([0, horizon - 1], axes):
        ax.scatter(targets[:, h], preds[:, h], s=1, alpha=0.3)
        lo = min(targets[:, h].min(), preds[:, h].min())
        hi = max(targets[:, h].max(), preds[:, h].max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set_title(f"{h+1}h ahead (R²={r2_per_h[h]:.3f})")
        ax.set_xlabel("Actual (W)")
        ax.set_ylabel("Predicted (W)")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{building_name} — Scatter: Predicted vs Actual", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / f"{building_name}_scatter.png", dpi=150)
    plt.close(fig)

    # --- Plot 3: RMSE/MAE per horizon ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x_h = np.arange(1, 13)
    ax.bar(x_h - 0.2, rmse_per_h, 0.4, label="RMSE", alpha=0.8)
    ax.bar(x_h + 0.2, mae_per_h, 0.4, label="MAE", alpha=0.8)
    ax.set_xlabel("Forecast Horizon (hours)")
    ax.set_ylabel("Error (W)")
    ax.set_title(
        f"{building_name} — Error by Horizon (Overall RMSE={overall_rmse:.0f})"
    )
    ax.set_xticks(x_h)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / f"{building_name}_error_by_horizon.png", dpi=150)
    plt.close(fig)

    # --- Plot 4: Error distribution ---
    errors = (preds - targets).flatten()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=100, alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="r", linestyle="--")
    ax.set_xlabel("Prediction Error (W)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"{building_name} — Error Distribution (mean={errors.mean():.0f}, std={errors.std():.0f})"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{building_name}_error_distribution.png", dpi=150)
    plt.close(fig)

    last = horizon - 1
    metrics = {
        "overall_rmse": float(overall_rmse),
        "overall_mae": float(overall_mae),
        "overall_r2": float(overall_r2),
        "rmse_1h": float(rmse_per_h[0]),
        f"rmse_{last+1}h": float(rmse_per_h[last]),
        "r2_1h": float(r2_per_h[0]),
        f"r2_{last+1}h": float(r2_per_h[last]),
    }
    print(
        f"  Overall: RMSE={overall_rmse:.0f}  MAE={overall_mae:.0f}  R²={overall_r2:.4f}"
    )
    print(f"  1h ahead: RMSE={rmse_per_h[0]:.0f}  R²={r2_per_h[0]:.4f}")
    print(f"  {last+1}h ahead: RMSE={rmse_per_h[last]:.0f}  R²={r2_per_h[last]:.4f}")
    print(f"  Plots saved to {output_dir}/")
    return metrics
