"""Tier-1 shiftability analysis for the case-study redesign (2026-06-23).

Computes, per building (OF/LB/WH), two candidate shiftability metrics M from the
RBC1 episode load series + SMUD tariff:

  Metric (2)  load-price alignment  -- cheap, observation-only:
      PWS  = sum(lam*d) / (mean(lam)*sum(d))   (price-weighted load share; >1 =>
             native load concentrated in expensive hours => more to shift)
      corr = Pearson corr(hourly load d, hourly tariff lam)
      onpeak_share, onmid_share = load fraction in on-peak / on+mid-peak hours

  Metric (3)  oracle potential (lp_benchmark) -- perfect-foresight LP/MILP/sched
      savings of an idealized TES dispatch vs RBC [%]. Most rigorous; folds in
      price, TES capacity, and load timing automatically.

Then relates the existing six-pair TLD-RL zero-shot transfer savings (zs, from the
extracted _final_tables.json) to M_src, M_tgt, |dN| (zone-count gap), and a
source-target load-profile distance, to test the headline claim:
  transfer savings ~ f(M_src, M_tgt), NOT |dN| / distance.

Writes JSON + CSV + a short findings markdown under
artifacts/case_study/figures/case_study_redesign_2026-06-23/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from studies.case_study.analysis import lp_benchmark as lpb

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/case_study/figures/case_study_redesign_2026-06-23"
OUT.mkdir(parents=True, exist_ok=True)

SMUD = lpb.SMUD  # 24-h tariff array
ONPEAK_HOURS = [h for h in range(24) if SMUD[h] == 0.3765]  # 17,18,19
ONMID_HOURS = [h for h in range(24) if SMUD[h] >= 0.2139]  # mid + on

# Zone counts (buildings.yaml) -- for the |dN| gap axis.
ZONES = {"office": 5, "lbuilding": 8, "warehouse": 3}
SHORT = {"office": "OF", "lbuilding": "LB", "warehouse": "WH"}

# Six directed transfer pairs and the canonical TLD-RL zero-shot savings (zs)
# from the extracted results (artifacts/.../_final_tables.json), the deterministic
# 5-seed transfer numbers that feed Table 4 of the paper.
FINAL_TABLES = (
    ROOT / "artifacts/case_study/figures/metric_a_5seed_2026-06-14/_final_tables.json"
)


def daily_load_shape(d_hourly: np.ndarray) -> np.ndarray:
    """Mean 24-h load shape, normalized to unit sum (a profile, not a level)."""
    T = len(d_hourly)
    days = T // 24
    prof = d_hourly[: days * 24].reshape(days, 24).mean(axis=0)
    s = prof.sum()
    return prof / s if s > 0 else prof


def metric2(d: np.ndarray, lam: np.ndarray) -> dict:
    """Load-price alignment metrics from hourly load d and tariff lam."""
    total = float(d.sum())
    pws = float(np.sum(lam * d) / (lam.mean() * total)) if total > 0 else float("nan")
    # hour-of-day index aligned to SMUD (lam already encodes the hour tier)
    onpeak = float(d[np.isin(lam, [0.3765])].sum() / total)
    onmid = float(d[lam >= 0.2139].sum() / total)
    corr = float(np.corrcoef(d, lam)[0, 1])
    return {
        "pws": pws,
        "corr_load_price": corr,
        "onpeak_share": onpeak,
        "onmid_share": onmid,
        "total_load_kwh": total,
    }


def shape_distance(a: np.ndarray, b: np.ndarray) -> dict:
    """Distances between two normalized 24-h load shapes."""
    l1 = float(np.abs(a - b).sum())
    corr = float(np.corrcoef(a, b)[0, 1])
    return {"l1": l1, "one_minus_corr": 1.0 - corr}


def regress(x: np.ndarray, y: np.ndarray) -> dict:
    """OLS y ~ x; return slope, intercept, R^2, Pearson r."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    r = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 else float("nan")
    return {
        "slope": float(coef[1]),
        "intercept": float(coef[0]),
        "r2": r2,
        "pearson_r": r,
    }


def main():
    # ---- per-building metrics ----
    per_building = {}
    shapes = {}
    for b in ("office", "lbuilding", "warehouse"):
        data = lpb.load_rbc_episode(b)
        d, lam = data["d"], data["lam"]
        m2 = metric2(d, lam)
        oracle = lpb.run_building(b, make_plot=False)
        shapes[b] = daily_load_shape(d)
        per_building[b] = {
            "short": SHORT[b],
            "zones": ZONES[b],
            "metric2": m2,
            "metric3_oracle": {
                "savings_lp_pct": oracle["savings_lp_vs_rbc_pct"],
                "savings_milp_pct": oracle["savings_milp_vs_rbc_pct"],
                "savings_schedule_pct": oracle["savings_schedule_vs_rbc_pct"],
                "cost_rbc": oracle["cost_rbc_measured"],
                "G1_replay_rel_err": oracle["G1_replay_rel_err"],
                "G2_bound_holds": oracle["G2_bound_holds_vs_rbc"],
            },
            "daily_load_shape": shapes[b].tolist(),
        }

    # ---- six-pair transfer table ----
    ft = json.load(open(FINAL_TABLES))
    zs = {k: v["zs"] for k, v in ft["transfer"].items()}

    # primary M candidates: (2) PWS, (3) oracle schedule savings %
    def M2(b):
        return per_building[b]["metric2"]["pws"]

    def M3(b):
        return per_building[b]["metric3_oracle"]["savings_schedule_pct"]

    pairs = []
    for key, sav in zs.items():
        src, tgt = key.split("_to_")
        pairs.append(
            {
                "pair": f"{SHORT[src]}->{SHORT[tgt]}",
                "src": src,
                "tgt": tgt,
                "savings_zs": sav,
                "M2_src": M2(src),
                "M2_tgt": M2(tgt),
                "M3_src": M3(src),
                "M3_tgt": M3(tgt),
                "dN": abs(ZONES[src] - ZONES[tgt]),
                "shape_dist_l1": shape_distance(shapes[src], shapes[tgt])["l1"],
                "shape_dist_1mcorr": shape_distance(shapes[src], shapes[tgt])[
                    "one_minus_corr"
                ],
            }
        )

    y = np.array([p["savings_zs"] for p in pairs])
    regs = {}
    for name, xs in [
        ("M2_src", [p["M2_src"] for p in pairs]),
        ("M2_tgt", [p["M2_tgt"] for p in pairs]),
        ("M2_src_x_tgt", [p["M2_src"] * p["M2_tgt"] for p in pairs]),
        ("M3_src", [p["M3_src"] for p in pairs]),
        ("M3_tgt", [p["M3_tgt"] for p in pairs]),
        ("M3_src_plus_tgt", [p["M3_src"] + p["M3_tgt"] for p in pairs]),
        ("dN", [p["dN"] for p in pairs]),
        ("shape_dist_l1", [p["shape_dist_l1"] for p in pairs]),
        ("shape_dist_1mcorr", [p["shape_dist_1mcorr"] for p in pairs]),
    ]:
        regs[name] = regress(np.array(xs), y)

    result = {
        "per_building": per_building,
        "pairs": pairs,
        "regressions_vs_savings_zs": regs,
        "smud": SMUD,
        "onpeak_hours": ONPEAK_HOURS,
    }
    (OUT / "shiftability_metrics.json").write_text(json.dumps(result, indent=2))

    # ---- console summary ----
    print("\n=== Per-building shiftability ===")
    print(
        f"{'B':>3} {'zones':>5} {'PWS(2)':>7} {'corr':>6} {'onpk%':>6} "
        f"{'onmid%':>7} {'LP%(3)':>7} {'MILP%':>6} {'SCHED%':>7}"
    )
    for b, v in per_building.items():
        m2, m3 = v["metric2"], v["metric3_oracle"]
        print(
            f"{v['short']:>3} {v['zones']:>5} {m2['pws']:>7.3f} "
            f"{m2['corr_load_price']:>6.2f} {100*m2['onpeak_share']:>6.1f} "
            f"{100*m2['onmid_share']:>7.1f} {m3['savings_lp_pct']:>7.1f} "
            f"{m3['savings_milp_pct']:>6.1f} {m3['savings_schedule_pct']:>7.1f}"
        )

    print("\n=== Six-pair transfer (zs) vs descriptors ===")
    print(
        f"{'pair':>8} {'zs':>6} {'M2s':>6} {'M2t':>6} {'M3s':>6} {'M3t':>6} "
        f"{'|dN|':>4} {'distL1':>7}"
    )
    for p in pairs:
        print(
            f"{p['pair']:>8} {p['savings_zs']:>6.1f} {p['M2_src']:>6.3f} "
            f"{p['M2_tgt']:>6.3f} {p['M3_src']:>6.1f} {p['M3_tgt']:>6.1f} "
            f"{p['dN']:>4} {p['shape_dist_l1']:>7.3f}"
        )

    print("\n=== Regressions of zs savings on each descriptor (R^2 | r) ===")
    for name, r in regs.items():
        print(
            f"  {name:>18}: R2={r['r2']:+.3f}  r={r['pearson_r']:+.3f}  "
            f"slope={r['slope']:+.3f}"
        )

    print(f"\nwritten: {OUT}/shiftability_metrics.json")


if __name__ == "__main__":
    main()
