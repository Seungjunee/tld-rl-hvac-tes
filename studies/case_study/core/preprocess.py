from __future__ import annotations

from pathlib import Path

import pandas as pd


def preprocess_and_merge(
    obs_path: str | Path,
    info_path: str | Path,
    out_path: str | Path,
    n_steps: int = 96,
) -> Path:
    obs_path = Path(obs_path)
    info_path = Path(info_path)
    out_path = Path(out_path)

    if not obs_path.exists():
        raise FileNotFoundError(f"observations.csv not found: {obs_path}")
    if not info_path.exists():
        raise FileNotFoundError(f"infos.csv not found: {info_path}")

    df_obs = pd.read_csv(obs_path)
    df_info = pd.read_csv(info_path)

    if "comfort_penalty" not in df_info.columns:
        raise KeyError("infos.csv missing comfort_penalty column")

    df_obs = df_obs.copy()
    df_obs["comfort_penalty"] = df_info["comfort_penalty"]

    shifted_frames = []
    targets = ["CoolingRate", "comfort_penalty"]
    for target in targets:
        if target not in df_obs.columns:
            continue

        shifted_frames.append(
            pd.DataFrame(
                {f"{target}{i}": df_obs[target].shift(-i) for i in range(1, n_steps + 1)},
                index=df_obs.index,
            )
        )

    df_final = pd.concat([df_obs, *shifted_frames], axis=1).dropna()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(out_path, index=False)
    return out_path
