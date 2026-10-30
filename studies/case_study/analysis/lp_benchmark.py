"""Idealized-dispatch optimization benchmark.

Computes a perfect-model + perfect-foresight upper bound C* on TES arbitrage
for each building, from logged RBC1 episode data only (no simulation):

  Variant A (lp):   min sum λ[t]·q[t]/COP            (constant-COP LP)
  Variant B (milp): min sum λ[t]·(a·u[t] + b·q[t])   (UC-style affine EIR fit)

subject to the tank SoC balance E[t+1] = E[t] + q[t] - d[t], 0<=E<=Emax,
0<=q<=Qrated (·u), cyclic E[0]=E[T]. Hourly resolution, full cooling season.

Pricing matches the measurement protocol: the same 24-h SMUD array applied to
every day (weekends included).

Gates:
  G1 calibration — replay RBC1's logged thermal dispatch through the affine
     cost model; must reproduce the measured electric cost within ±5%.
  G2 bound — C* <= measured cost of every controller.
  G3 load exogeneity — RBC1 vs SAC total load within 2%.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

ROOT = Path(__file__).resolve().parents[3]

SMUD = [0.155] * 12 + [0.2139] * 5 + [0.3765] * 3 + [0.2139] * 4

BUILDINGS = {
    "office": {
        "epjson": "5ZoneTankFixed_final.epJSON",
        "rbc_case": "rbc_office_smud_tou",
        "sac_export": "pretrain_office_smud_sac_plus",
        "q_rated_kw": 80.0,
        "cop_ref": 3.2,
        "emax_kwh": 290.1,  # 50 m3 x 5 K controllable span
    },
    "lbuilding": {
        "epjson": "LBuilding_final.epJSON",
        "rbc_case": "rbc_lbuilding_smud_tou",
        "sac_export": "pretrain_lbuilding_smud_sac_plus",
        "q_rated_kw": 70.0,
        "cop_ref": 3.2,
        "emax_kwh": 232.1,  # 40 m3 x 5 K
    },
    "warehouse": {
        "epjson": "Warehouse_final.epJSON",
        "rbc_case": "rbc_warehouse_smud_tou",
        "sac_export": "pretrain_warehouse_smud_sac_plus",
        "q_rated_kw": 57.0,
        "cop_ref": 3.2,
        "emax_kwh": 232.1,  # 40 m3 x 5 K
    },
    # --- further commercial buildings ---
    "restaurant": {
        "epjson": "FullServiceRestaurant_r1.epJSON",
        "rbc_case": "rbc_restaurant_smud_tou",
        "sac_export": "pretrain_restaurant_smud_sac_plus",
        "q_rated_kw": 173.0,  # rule sizing
        "cop_ref": 3.2,
        "emax_kwh": 605.5,  # 104.3 m3 x 5 K
    },
    "retail": {
        "epjson": "RetailStandalone_r1.epJSON",
        "rbc_case": "rbc_retail_smud_tou",
        "sac_export": "pretrain_retail_smud_sac_plus",
        "q_rated_kw": 212.4,  # rule sizing
        "cop_ref": 3.2,
        "emax_kwh": 743.1,  # 128.0 m3 x 5 K
    },
    "stripmall": {
        "epjson": "StripMall_r1.epJSON",
        "rbc_case": "rbc_stripmall_smud_tou",
        "sac_export": "pretrain_stripmall_smud_sac_plus",
        "q_rated_kw": 225.0,  # rule sizing
        "cop_ref": 3.2,
        "emax_kwh": 787.3,  # 135.6 m3 x 5 K
    },
    "mediumoffice": {
        "epjson": "MediumOffice_final.epJSON",
        "rbc_case": "rbc_mediumoffice_smud_tou",
        "sac_export": "pretrain_mediumoffice_smud_sac_plus",
        "q_rated_kw": 500.0,
        "cop_ref": 3.2,
        "emax_kwh": 1450.5,  # 250 m3 x 5 K
    },
}

PLR_MIN = 0.1
CHILLER_ON_W = 100.0


def _decode_hour(cos_v: float, sin_v: float) -> int:
    ang = math.atan2(sin_v, cos_v)
    return int(round((ang % (2 * math.pi)) / (2 * math.pi) * 24)) % 24


def load_rbc_episode(building: str, prefer_export: bool = False) -> dict:
    """Hourly aggregates from the latest RBC1 episode observations.csv.

    prefer_export reads the canonical export only: the workspace glob can pick
    up other runs made under the same case id (e.g. held-out weather RBC
    runs), which carry a different season.
    """
    cfg = BUILDINGS[building]
    pattern = f"{ROOT}/artifacts/case_study/workspaces/{cfg['rbc_case']}/*/Eplus*/episode-*/monitor/observations.csv"
    files = (
        []
        if prefer_export
        else sorted(
            glob.glob(pattern), key=lambda p: int(p.split("episode-")[1].split("/")[0])
        )
    )
    if files:
        path = files[-1]
    else:
        # Fallback: the exported episode obs (workspaces are transient and may be
        # pruned; the export observations.csv is the same final RBC episode).
        exp = f"{ROOT}/artifacts/case_study/exports/{cfg['rbc_case']}/observations.csv"
        if not Path(exp).exists():
            raise FileNotFoundError(f"{pattern}  (and no export fallback {exp})")
        path = exp

    with open(path) as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col = {h: i for i, h in enumerate(header)}
        rows = list(reader)

    need = [
        "hour_cos",
        "hour_sin",
        "CoolingRate",
        "ChillerElectricity",
        "EvapCoolingRate",
    ]
    for n in need:
        if n not in col:
            raise KeyError(f"{n} missing in {path}")

    n_hours = len(rows) // 4 + (1 if len(rows) % 4 else 0)
    d = np.zeros(n_hours)  # load kWh-th per hour
    q_rbc = np.zeros(n_hours)  # chiller thermal kWh per hour
    p_rbc = np.zeros(n_hours)  # chiller electric kWh per hour
    u_rbc = np.zeros(n_hours)  # on-fraction per hour
    lam = np.zeros(n_hours)
    eirft_sum, eirft_n = 0.0, 0

    for t in range(n_hours):
        chunk = rows[t * 4 : (t + 1) * 4]
        k = len(chunk)
        cool = sum(float(r[col["CoolingRate"]]) for r in chunk) / k
        evap = sum(float(r[col["EvapCoolingRate"]]) for r in chunk) / k
        elec = sum(float(r[col["ChillerElectricity"]]) for r in chunk) / k
        on = (
            sum(1 for r in chunk if float(r[col["ChillerElectricity"]]) > CHILLER_ON_W)
            / k
        )
        d[t] = cool / 1000.0
        q_rbc[t] = evap / 1000.0
        p_rbc[t] = elec / 1000.0
        u_rbc[t] = on
        h = _decode_hour(
            float(chunk[0][col["hour_cos"]]), float(chunk[0][col["hour_sin"]])
        )
        lam[t] = SMUD[h]
        if "EIRfT" in col:
            for r in chunk:
                if float(r[col["ChillerElectricity"]]) > CHILLER_ON_W:
                    eirft_sum += float(r[col["EIRfT"]])
                    eirft_n += 1

    eirft_mean = eirft_sum / eirft_n if eirft_n else 1.0
    return {
        "path": path,
        "d": d,
        "q_rbc": q_rbc,
        "p_rbc": p_rbc,
        "u_rbc": u_rbc,
        "lam": lam,
        "eirft_mean": eirft_mean,
    }


def eir_curve_coeffs(building: str) -> tuple[float, float, float]:
    cfg = BUILDINGS[building]
    ep = json.load(open(f"{ROOT}/sinergym/data/buildings/{cfg['epjson']}"))
    chiller = list(ep["Chiller:Electric:EIR"].values())[0]
    curve_name = chiller[
        "electric_input_to_cooling_output_ratio_function_of_part_load_ratio_curve_name"
    ]
    for typ in ("Curve:Quadratic", "Curve:Cubic"):
        if curve_name in ep.get(typ, {}):
            c = ep[typ][curve_name]
            return (
                float(c["coefficient1_constant"]),
                float(c["coefficient2_x"]),
                float(c.get("coefficient3_x_2", 0.0)),
            )
    raise KeyError(curve_name)


def affine_fit(building: str, eirft_mean: float) -> tuple[float, float, float]:
    """Fit P_kW(q) ≈ a·u + b·q over PLR in [PLR_MIN, 1]. Returns (a, b)."""
    cfg = BUILDINGS[building]
    c0, c1, c2 = eir_curve_coeffs(building)
    qr, cop = cfg["q_rated_kw"], cfg["cop_ref"]
    plr = np.linspace(PLR_MIN, 1.0, 50)
    p_kw = (qr * eirft_mean / cop) * (c0 + c1 * plr + c2 * plr**2)
    q_kw = plr * qr
    A = np.column_stack([np.ones_like(q_kw), q_kw])
    (a, b), *_ = np.linalg.lstsq(A, p_kw, rcond=None)
    resid = p_kw - (a + b * q_kw)
    r2 = 1 - resid.var() / p_kw.var()
    return float(a), float(b), float(r2)


def solve_lp_const_cop(d, lam, q_rated, emax, cop):
    """Variant A: pure LP with constant COP. Vars: [q(T), E(T)]."""
    T = len(d)
    cost = np.concatenate([lam / cop, np.zeros(T)])
    # SoC balance: E[(t+1)%T] - E[t] - q[t] = -d[t]
    rows, cols, vals = [], [], []
    for t in range(T):
        rows += [t, t, t]
        cols += [T + (t + 1) % T, T + t, t]
        vals += [1.0, -1.0, -1.0]
    A_eq = sparse.coo_matrix((vals, (rows, cols)), shape=(T, 2 * T))
    res = linprog(
        cost,
        A_eq=A_eq.tocsr(),
        b_eq=-d,
        bounds=[(0, q_rated)] * T + [(0, emax)] * T,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(res.message)
    return res.fun, res.x[:T], res.x[T:]


def solve_milp_uc(d, lam, q_rated, emax, a_coef, b_coef, gap=0.005):
    """Variant B: UC-style MILP. Vars: [q(T), E(T), u(T)]."""
    T = len(d)
    cost = np.concatenate([lam * b_coef, np.zeros(T), lam * a_coef])

    rows, cols, vals = [], [], []
    for t in range(T):  # SoC balance (equality, rows 0..T-1)
        rows += [t, t, t]
        cols += [T + (t + 1) % T, T + t, t]
        vals += [1.0, -1.0, -1.0]
    A_eq = sparse.coo_matrix((vals, (rows, cols)), shape=(T, 3 * T)).tocsr()

    rows, cols, vals = [], [], []
    for t in range(T):  # q - Qrated*u <= 0
        rows += [t, t]
        cols += [t, 2 * T + t]
        vals += [1.0, -q_rated]
    for t in range(T):  # PLR_MIN*Qrated*u - q <= 0
        rows += [T + t, T + t]
        cols += [2 * T + t, t]
        vals += [PLR_MIN * q_rated, -1.0]
    A_ub = sparse.coo_matrix((vals, (rows, cols)), shape=(2 * T, 3 * T)).tocsr()

    constraints = [
        LinearConstraint(A_eq, -d, -d),
        LinearConstraint(A_ub, -np.inf, 0.0),
    ]
    integrality = np.concatenate([np.zeros(2 * T), np.ones(T)])
    bounds = Bounds(
        np.zeros(3 * T),
        np.concatenate([np.full(T, q_rated), np.full(T, emax), np.ones(T)]),
    )
    res = milp(
        c=cost,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options={"mip_rel_gap": gap, "time_limit": 600},
    )
    if res.x is None:
        raise RuntimeError(res.message)
    x = res.x
    return res.fun, x[:T], x[T : 2 * T], x[2 * T :]


def solve_milp_schedule(d, lam, q_on, emax, a_coef, b_coef):
    """Variant C: actuator-respecting bound. Chiller is bang-bang at the
    empirical on-rate q_on (kWh-th/h); only the ON/OFF schedule u(t) is
    optimized. Tighter than Variant B: respects that the setpoint+deadband
    actuator cannot modulate q continuously. Vars: [E(T), u(T)]."""
    T = len(d)
    cost = np.concatenate([np.zeros(T), lam * (a_coef + b_coef * q_on)])

    rows, cols, vals = [], [], []
    for t in range(T):  # E[(t+1)%T] - E[t] - q_on*u[t] = -d[t]
        rows += [t, t, t]
        cols += [(t + 1) % T, t, T + t]
        vals += [1.0, -1.0, -q_on]
    A_eq = sparse.coo_matrix((vals, (rows, cols)), shape=(T, 2 * T)).tocsr()

    # u is the per-hour ON-fraction (continuous): the deadband actuator can
    # cycle within an hour, so fractional commitment is physically faithful
    # and keeps the cyclic balance feasible (binary hourly u would require
    # sum(u) = sum(d)/q_on to be integral). This makes Variant C a pure LP.
    res = linprog(
        cost,
        A_eq=A_eq,
        b_eq=-d,
        bounds=[(0, emax)] * T + [(0, 1)] * T,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(res.message)
    return res.fun, res.x[:T], res.x[T:]


def optimal_dispatch(building: str, prefer_export: bool = False) -> dict:
    """Expose the MILP-optimal storage dispatch for open-loop achievability tests.

    Returns the per-hour optimal chiller cooling q*(t) and tank energy e*(t), the
    price/load, and the modeled optimal cost, so an open-loop controller in
    EnergyPlus can be checked against the same model the oracle reports.
    """
    cfg = BUILDINGS[building]
    data = load_rbc_episode(building, prefer_export=prefer_export)
    d, lam = data["d"], data["lam"]
    a_coef, b_coef, _ = affine_fit(building, data["eirft_mean"])
    c_milp, q_mi, e_mi, _ = solve_milp_uc(
        d, lam, cfg["q_rated_kw"], cfg["emax_kwh"], a_coef, b_coef
    )
    return {
        "q_opt": q_mi,
        "e_opt": e_mi,
        "lam": lam,
        "d": d,
        "cost_opt_per_day": c_milp / 122.0,
        "q_rated_kw": cfg["q_rated_kw"],
        "emax_kwh": cfg["emax_kwh"],
        "cop_ref": cfg["cop_ref"],
    }


def run_building(
    building: str, make_plot: bool = True, prefer_export: bool = False
) -> dict:
    cfg = BUILDINGS[building]
    data = load_rbc_episode(building, prefer_export=prefer_export)
    d, lam = data["d"], data["lam"]
    a_coef, b_coef, fit_r2 = affine_fit(building, data["eirft_mean"])

    # Measured RBC1 electric cost (identical accounting to the paper protocol)
    c_rbc_measured = float(np.sum(data["p_rbc"] * lam))

    # G1: replay RBC1's logged thermal dispatch through the affine model
    c_rbc_replay = float(
        np.sum(lam * (a_coef * data["u_rbc"] + b_coef * data["q_rbc"]))
    )
    g1_err = (c_rbc_replay - c_rbc_measured) / c_rbc_measured

    # No-TES reference (constant COP, load-following)
    c_no_tes = float(np.sum(lam * d / cfg["cop_ref"]))

    # Variant A
    c_lp, _q_lp, _e_lp = solve_lp_const_cop(
        d, lam, cfg["q_rated_kw"], cfg["emax_kwh"], cfg["cop_ref"]
    )

    # Variant B
    c_milp, q_mi, e_mi, _u_mi = solve_milp_uc(
        d, lam, cfg["q_rated_kw"], cfg["emax_kwh"], a_coef, b_coef
    )

    # Variant C: empirical chiller-on thermal rate (median over on-hours).
    # Can be infeasible if the empirical q_on cannot meet peak load cyclically
    # within the tank -- then report None (LP/MILP bounds still valid).
    on_mask = data["u_rbc"] > 0.25
    q_on = float(np.median(data["q_rbc"][on_mask] / data["u_rbc"][on_mask]))
    try:
        c_sched, _e_sc, _u_sc = solve_milp_schedule(
            d, lam, q_on, cfg["emax_kwh"], a_coef, b_coef
        )
    except RuntimeError as exc:
        c_sched = None
        print(f"  [warn] schedule-LP infeasible for {building}: {exc}")

    # G3: load exogeneity vs one SAC export
    g3 = None
    sac_obs = f"{ROOT}/artifacts/case_study/exports/{cfg['sac_export']}/seed_42/observations.csv"
    if Path(sac_obs).exists():
        with open(sac_obs) as fh:
            reader = csv.reader(fh)
            header = next(reader)
            ci = header.index("CoolingRate")
            sac_total = sum(float(r[ci]) for r in reader) * 0.25 / 1000.0
        rbc_total = float(np.sum(d))
        g3 = (sac_total - rbc_total) / rbc_total

    out = {
        "building": building,
        "T_hours": len(d),
        "eirft_mean": data["eirft_mean"],
        "affine_fit": {"a_kw": a_coef, "b_kwe_per_kwth": b_coef, "r2": fit_r2},
        "cost_no_tes_constcop": c_no_tes,
        "cost_rbc_measured": c_rbc_measured,
        "cost_rbc_replay_affine": c_rbc_replay,
        "G1_replay_rel_err": g1_err,
        "cost_lp_constcop": c_lp,
        "cost_milp_uc": c_milp,
        "G2_bound_holds_vs_rbc": bool(c_milp <= c_rbc_measured),
        "G3_load_rel_diff_sac_vs_rbc": g3,
        "headroom_rbc_minus_milp": c_rbc_measured - c_milp,
        "savings_milp_vs_rbc_pct": 100 * (c_rbc_measured - c_milp) / c_rbc_measured,
        "savings_lp_vs_rbc_pct": 100 * (c_rbc_measured - c_lp) / c_rbc_measured,
        "q_on_empirical_kw": q_on,
        "cost_milp_schedule": c_sched,
        "savings_schedule_vs_rbc_pct": (
            None
            if c_sched is None
            else 100 * (c_rbc_measured - c_sched) / c_rbc_measured
        ),
    }

    if make_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        T = len(d)
        days = T // 24
        prof = lambda x: x[: days * 24].reshape(days, 24).mean(axis=0)
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        hours = np.arange(24)
        axes[0].plot(hours, prof(d), label="load d[t]")
        axes[0].plot(hours, prof(q_mi), label="q* (MILP)")
        axes[0].plot(hours, prof(data["q_rbc"]), label="q RBC1", ls="--")
        axes[0].set_ylabel("kWh-th/h")
        axes[0].legend()
        axes[1].plot(hours, prof(e_mi), label="E* SoC (MILP)")
        axes[1].set_ylabel("kWh-th")
        axes[1].legend()
        axes[2].step(hours, [SMUD[h] for h in hours], where="post", color="k")
        axes[2].set_ylabel("$/kWh")
        axes[2].set_xlabel("hour of day")
        fig.suptitle(f"{building}: idealized dispatch (daily mean)")
        outdir = Path(f"{ROOT}/artifacts/case_study/figures/lp_benchmark_2026-06-10")
        outdir.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(outdir / f"{building}_dispatch.png", dpi=130)
        plt.close(fig)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--building", choices=[*BUILDINGS, "all"], default="all")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    targets = list(BUILDINGS) if args.building == "all" else [args.building]
    results = {}
    for b in targets:
        print(f"=== {b} ===")
        r = run_building(b, make_plot=not args.no_plot)
        results[b] = r
        print(
            f"  G1 replay err {r['G1_replay_rel_err']:+.1%} | G2 bound {'OK' if r['G2_bound_holds_vs_rbc'] else 'VIOLATED'}"
            f" | G3 load diff {r['G3_load_rel_diff_sac_vs_rbc']:+.1%}"
            if r["G3_load_rel_diff_sac_vs_rbc"] is not None
            else ""
        )
        print(
            f"  C_noTES ${r['cost_no_tes_constcop']:.0f} | C_RBC ${r['cost_rbc_measured']:.0f} | "
            f"C*_LP ${r['cost_lp_constcop']:.0f} ({r['savings_lp_vs_rbc_pct']:+.1f}%) | "
            f"C*_MILP ${r['cost_milp_uc']:.0f} ({r['savings_milp_vs_rbc_pct']:+.1f}%) | "
            f"C*_SCHED ${r['cost_milp_schedule']:.0f} ({r['savings_schedule_vs_rbc_pct']:+.1f}%, q_on={r['q_on_empirical_kw']:.0f}kW)"
        )

    outdir = Path(f"{ROOT}/artifacts/case_study/figures/lp_benchmark_2026-06-10")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "lp_benchmark.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwritten: {outdir}/lp_benchmark.json")


if __name__ == "__main__":
    main()
