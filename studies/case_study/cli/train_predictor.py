"""CLI for training and evaluating LSTM load predictors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from studies.case_study.core.load_predictor import (
    BUILDING_INPUT_VARIABLES,
    LoadDataset,
    evaluate_and_plot,
    train_predictor,
)

DATA_PATHS = {
    "restaurant_3zone": "artifacts/case_study/exports/build_restaurant_observation_dataset/observations_processed.csv",
    "office_5zone": "artifacts/case_study/exports/build_office_observation_dataset/observations_processed.csv",
    "office_lbuilding_8zone": "artifacts/case_study/exports/build_lbuilding_observation_dataset/observations_processed.csv",
    "stripmall_10zone": "artifacts/case_study/exports/build_stripmall_observation_dataset/observations_processed.csv",
    "warehouse_3zone": "artifacts/case_study/exports/build_warehouse_observation_dataset/observations_processed.csv",
    "mediumoffice_15zone": "artifacts/case_study/exports/build_mediumoffice_observation_dataset/observations_processed.csv",
    "retail_4zone": "artifacts/case_study/exports/build_retail_observation_dataset/observations_processed.csv",
    "fastfood_2zone": "artifacts/case_study/exports/build_fastfood_observation_dataset/observations_processed.csv",
}

DEFAULT_OUTPUT_DIR = "artifacts/case_study/predictors"
DEFAULT_PLOT_DIR = "artifacts/case_study/figures/predictor_accuracy"


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train LSTM load predictor")
    p.add_argument(
        "--building", required=True, choices=list(BUILDING_INPUT_VARIABLES.keys())
    )
    p.add_argument("--data", default=None, help="Override CSV data path")
    p.add_argument("--output", default=None, help="Override model output path")
    p.add_argument("--plot-dir", default=None, help="Override plot output directory")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=24)
    # H=12 per the paper (supervisory horizon, 12 steps at 60-min = 12 h) and the
    # load_predictor wrapper (recipes use horizon: 12). Was 24 -- a footgun that
    # produced horizon-24 predictors inconsistent with the H=12 convention.
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for LSTM init + shuffling (reproducible predictors)",
    )
    p.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip training, evaluate existing model",
    )
    p.add_argument("--all", action="store_true", help="Train all buildings")
    return p.parse_args(argv)


def _run_building(building: str, args):
    print(f"\n{'='*60}")
    print(f"Building: {building}")
    print(f"{'='*60}")

    # Seed LSTM init + DataLoader shuffling so predictors are reproducible
    # (previously unseeded -> every retrain produced different weights, which
    # propagates nondeterminism into every bank policy trained on top).
    import random

    import numpy as _np
    import torch as _torch

    random.seed(args.seed)
    _np.random.seed(args.seed)
    _torch.manual_seed(args.seed)

    csv_path = args.data or DATA_PATHS[building]
    model_path = args.output or f"{DEFAULT_OUTPUT_DIR}/{building}_lstm.pt"
    plot_dir = args.plot_dir or DEFAULT_PLOT_DIR
    input_vars = BUILDING_INPUT_VARIABLES[building]

    print(f"  Data: {csv_path}")
    print(f"  Input variables ({len(input_vars)}): {input_vars[:5]}...")
    print(f"  Model: {model_path}")

    if not args.evaluate_only:
        print(
            f"\n  Training (epochs={args.epochs}, hidden={args.hidden}, layers={args.layers})..."
        )
        result = train_predictor(
            csv_path=csv_path,
            input_variables=input_vars,
            output_path=model_path,
            seq_len=args.seq_len,
            horizon=args.horizon,
            hidden_dim=args.hidden,
            num_layers=args.layers,
            dropout=args.dropout,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            patience=args.patience,
        )
        model = result["model"]
        dataset = result["dataset"]
    else:
        from studies.case_study.core.load_predictor import load_predictor as load_model

        print(f"  Loading existing model from {model_path}...")
        model, ckpt = load_model(model_path)
        dataset = LoadDataset(csv_path, input_vars, args.seq_len, args.horizon)

    print(f"\n  Evaluating and generating plots...")
    metrics = evaluate_and_plot(model, dataset, building, plot_dir)
    return metrics


def main(argv=None):
    args = _parse_args(argv)

    if args.all:
        buildings = list(BUILDING_INPUT_VARIABLES.keys())
    else:
        buildings = [args.building]

    all_metrics = {}
    for building in buildings:
        metrics = _run_building(building, args)
        all_metrics[building] = metrics

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for b, m in all_metrics.items():
        print(
            f"  {b}: RMSE={m['overall_rmse']:.0f}  R²={m['overall_r2']:.4f}  "
            f"RMSE_1h={m['rmse_1h']:.0f}  R2_1h={m['r2_1h']:.3f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
