from __future__ import annotations

from pathlib import Path

import pandas as pd

from studies.case_study.core.preprocess import preprocess_and_merge


def test_preprocess_and_merge_keeps_schema_and_shift(tmp_path: Path):
    obs_path = tmp_path / "observations.csv"
    info_path = tmp_path / "infos.csv"
    out_path = tmp_path / "observations_processed.csv"

    pd.DataFrame(
        {
            "CoolingRate": [10, 11, 12, 13],
            "dummy": [1, 1, 1, 1],
        }
    ).to_csv(obs_path, index=False)

    pd.DataFrame({"comfort_penalty": [1, 2, 3, 4]}).to_csv(info_path, index=False)

    preprocess_and_merge(obs_path=obs_path, info_path=info_path, out_path=out_path, n_steps=2)

    df = pd.read_csv(out_path)

    assert list(df.columns) == [
        "CoolingRate",
        "dummy",
        "comfort_penalty",
        "CoolingRate1",
        "CoolingRate2",
        "comfort_penalty1",
        "comfort_penalty2",
    ]
    assert len(df) == 2
    assert df["CoolingRate1"].tolist() == [11.0, 12.0]
    assert df["CoolingRate2"].tolist() == [12.0, 13.0]
    assert df["comfort_penalty1"].tolist() == [2.0, 3.0]
    assert df["comfort_penalty2"].tolist() == [3.0, 4.0]
