"""Zero-shot matrix of the further buildings (Fig. 8 data) and S* checks.

- Buildings: the four original buildings + every added building whose RBC
  export and inclusion verdict (analysis/r1_inclusion.py) exist.
- Pair value: savings of each source seed's final model zero-shot on the
  target (<pair>__src<seed>, scripts/run_r1_zs.py), vs the target RBC;
  five-seed mean +- sd (ddof 1). Deployed = the source seed with the highest
  final training reward (the paper's selected source, k*).
- S* per building: on-peak (label 17-20 h) share of the RBC season load.
  Limiting share of a pair = min(S*_src, S*_tgt).
- (i) S* check: every pair with limiting share >= 14 % should have a positive
  five-seed mean; count the violations (new pairs = pairs with an added
  building).
- (ii) among pairs with both S* >= 14 %, the fraction with a positive mean.
- (iii) the full heatmap (five-seed mean, deployed marked).

Usage:
    python -m studies.case_study.analysis.r1_x_matrix
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "artifacts/case_study/exports"
OUT = ROOT / "artifacts/case_study/results"
SF = 900.0 / 3.6e6
SEEDS = [42, 123, 777, 2024, 2025]
PAPER = ["office", "lbuilding", "warehouse", "mediumoffice"]
EXTENSION = ["restaurant", "stripmall", "retail", "fastfood"]
LABEL = {
    "office": "OF",
    "lbuilding": "LB",
    "warehouse": "WH",
    "mediumoffice": "MO",
    "restaurant": "RE",
    "stripmall": "SM",
    "retail": "RT",
    "fastfood": "QS",
}
S_STAR_MIN = 14.0


def _cost(progress: Path) -> float:
    r = pd.read_csv(progress).iloc[-1]
    return float(r["mean_reward_energy_term"]) * int(r["length(timesteps)"]) * SF


def _hour(df):
    ang = np.arctan2(df["hour_sin"], df["hour_cos"])
    return np.rint((ang % (2 * np.pi)) / (2 * np.pi) * 24).astype(int) % 24


def s_star(key: str) -> float:
    df = pd.read_csv(EXP / f"rbc_{key}_smud_tou" / "observations.csv")
    cr = df["CoolingRate"].to_numpy()
    return float(100.0 * cr[np.isin(_hour(df), [17, 18, 19])].sum() / cr.sum())


def deployed_seed(key: str) -> int | None:
    best, arg = -np.inf, None
    for s in SEEDS:
        p = EXP / f"pretrain_{key}_smud_sac_plus" / f"seed_{s}" / "progress.csv"
        if p.exists():
            v = float(pd.read_csv(p).iloc[-1]["mean_reward"])
            if v > best:
                best, arg = v, s
    return arg


def included(key: str) -> bool:
    if key in PAPER:
        return True
    f = OUT / f"x1_inclusion_{key}.json"
    return f.exists() and bool(json.loads(f.read_text())[key]["include"])


def main() -> int:
    blds = PAPER + [
        k for k in EXTENSION if (EXP / f"rbc_{k}_smud_tou").exists() and included(k)
    ]
    ss = {k: s_star(k) for k in blds}
    rbc = {k: _cost(EXP / f"rbc_{k}_smud_tou" / "progress.csv") for k in blds}
    ks = {k: deployed_seed(k) for k in blds}
    pairs = {}
    for src in blds:
        for tgt in blds:
            if src == tgt:
                continue
            case = f"transfer_{src}_to_{tgt}_smud_sacp_zs"
            per = {}
            for s in SEEDS:
                p = EXP / f"{case}__src{s}" / "progress.csv"
                if p.exists():
                    per[s] = 100.0 * (rbc[tgt] - _cost(p)) / rbc[tgt]
            if not per:
                continue
            v = list(per.values())
            pairs[f"{src}->{tgt}"] = {
                "per_seed": per,
                "n": len(v),
                "mean": float(np.mean(v)),
                "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                "deployed_seed": ks[src],
                "deployed": per.get(ks[src]),
                "limiting_share": min(ss[src], ss[tgt]),
                "new": bool({src, tgt} & set(EXTENSION)),
            }
    full = {k: v for k, v in pairs.items() if v["n"] == len(SEEDS)}
    rule_i = [
        k for k, v in full.items() if v["new"] and v["limiting_share"] >= S_STAR_MIN
    ]
    viol_i = [k for k in rule_i if full[k]["mean"] <= 0]
    both = [k for k, v in full.items() if v["limiting_share"] >= S_STAR_MIN]
    res = {
        "buildings": blds,
        "s_star": ss,
        "deployed_seed": ks,
        "pairs": pairs,
        "check_i_new_pairs_limiting_ge_14": {"pairs": rule_i, "violations": viol_i},
        "check_ii_both_ge_14": {
            "pairs": both,
            "positive": [k for k in both if full[k]["mean"] > 0],
            "fraction_positive": (
                (sum(full[k]["mean"] > 0 for k in both) / len(both)) if both else None
            ),
        },
        "incomplete_pairs": sorted(set(pairs) - set(full)),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "x_matrix.json").write_text(json.dumps(res, indent=2))
    rows = [
        {
            "pair": k,
            "src": k.split("->")[0],
            "tgt": k.split("->")[1],
            "mean": v["mean"],
            "sd": v["sd"],
            "deployed": v["deployed"],
            "limiting_share": v["limiting_share"],
            "n": v["n"],
            **{f"s{s}": v["per_seed"].get(s) for s in SEEDS},
        }
        for k, v in pairs.items()
    ]
    pd.DataFrame(rows).to_csv(OUT / "x_matrix_pairs.csv", index=False)
    _heatmap(res, blds)
    print("S*:", {LABEL[k]: round(v, 1) for k, v in ss.items()})
    for k, v in pairs.items():
        a, b = k.split("->")
        sd = f"{v['sd']:.2f}" if v["sd"] is not None else "-"
        dep = f"{v['deployed']:+.2f}" if v["deployed"] is not None else "-"
        print(
            f"{LABEL[a]}->{LABEL[b]}: {v['mean']:+.2f} +- {sd} (n={v['n']}) deployed {dep} lim {v['limiting_share']:.1f}"
        )
    print("check (i):", res["check_i_new_pairs_limiting_ge_14"])
    print("check (ii):", res["check_ii_both_ge_14"]["fraction_positive"])
    return 0


def _heatmap(res: dict, blds: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(blds)
    m = np.full((n, n), np.nan)
    for i, a in enumerate(blds):
        for j, b in enumerate(blds):
            v = res["pairs"].get(f"{a}->{b}")
            if v and v["n"] == len(SEEDS):
                m[i, j] = v["mean"]
    fig, ax = plt.subplots(figsize=(1.0 + 0.8 * n, 0.8 + 0.8 * n))
    lim = np.nanmax(np.abs(m)) if np.isfinite(m).any() else 1.0
    im = ax.imshow(m, cmap="RdBu", vmin=-lim, vmax=lim)
    for i in range(n):
        for j in range(n):
            if np.isfinite(m[i, j]):
                ax.text(j, i, f"{m[i, j]:+.1f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(n), [LABEL[b] for b in blds])
    ax.set_yticks(range(n), [LABEL[b] for b in blds])
    ax.set_xlabel("target")
    ax.set_ylabel("source")
    fig.colorbar(im, ax=ax, label="zero-shot savings vs RBC [%] (5-seed mean)")
    fig.tight_layout()
    fig.savefig(OUT / "x_matrix_heatmap.png", dpi=200)


if __name__ == "__main__":
    sys.exit(main())
