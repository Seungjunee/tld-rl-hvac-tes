"""Adoption-cost figure, 1x4 row (paper Fig. 6; replaces the 2x2 fig_adoption_v2).

Semantics (identical to fig_adoption_v2, layout compressed to one row):
  y = cumulative AVERAGE savings vs RBC [%] after k seasons, so every panel
  shares one bounded scale and the deployed policy is a horizontal line.
  - TLD-RL zero-shot (deployed): flat line at the mean over the three incoming
    source-side-selected deployments; band = min-max over those three sources.
    Selection recomputed here from source data only: argmax final training
    mean_reward over the five pretrain_<src>_smud_sac_plus seeds, then the
    deterministic zs episode from exports/transfer_<s>_to_<t>_..._zs__src<seed>.
  - E2E (Scratch) online: per-seed cumulative mean of the first ten training
    seasons of pretrain_<tgt>_smud_sac_e2e (exploration included), 5-seed
    mean +- 1 s.d. band.
  - Open marker: first season where the E2E cumulative average exceeds the
    deployed line (cumulative break-even; occurs only on MO).

Run:  python -m studies.case_study.analysis.fig_adoption_v7
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
# artifacts live in the main checkout; worktrees only carry the repo files
_CANDS = [REPO, Path("/workspaces/sinergym")]
ROOT = next(p for p in _CANDS if (p / "artifacts/case_study/exports").exists())
MAN = REPO / "artifacts/case_study/figures/paper"
MAN.mkdir(parents=True, exist_ok=True)
EXP = ROOT / "artifacts/case_study/exports"
OUT = ROOT / "artifacts/case_study/figures/case_study_redesign_2026-06-23"
SEEDS = ["42", "123", "777", "2024", "2025"]
STEP = 900.0 / 3.6e6
N = 10
X = np.arange(1, N + 1)
B = ["office", "lbuilding", "warehouse", "mediumoffice"]
TITLE = {
    "office": "Office (OF)",
    "lbuilding": "L-Building (LB)",
    "warehouse": "Warehouse (WH)",
    "mediumoffice": "Medium Office (MO)",
}
C_ZS, C_E2E = "#1f77b4", "#d62728"


def _rows(path):
    return [r for r in csv.DictReader(open(path)) if r.get("length(timesteps)")]


def _cost(r):
    return (
        float(r["mean_reward_energy_term"]) * int(float(r["length(timesteps)"])) * STEP
    )


def selection_seed(src: str) -> str:
    """Source-side selection: highest final training mean_reward on the source."""
    finals = {
        s: float(
            _rows(EXP / f"pretrain_{src}_smud_sac_plus" / f"seed_{s}" / "progress.csv")[
                -1
            ]["mean_reward"]
        )
        for s in SEEDS
    }
    return max(finals, key=lambda s: finals[s])


def e2e_cumavg(tgt: str, rbc: float) -> np.ndarray:
    """(5, N) per-seed cumulative-average online savings of from-scratch E2E."""
    curves = []
    for s in SEEDS:
        f = EXP / f"pretrain_{tgt}_smud_sac_e2e" / f"seed_{s}" / "progress.csv"
        sav = [100 * (rbc - _cost(r)) / rbc for r in _rows(f)[:N]]
        curves.append(np.cumsum(sav) / np.arange(1, len(sav) + 1))
    k = min(len(c) for c in curves)
    return np.array([c[:k] for c in curves])


PLATED = {}  # panel -> label strings allowed to plate the zero line
SEL = {b: selection_seed(b) for b in B}
print("selection seeds (sac_plus, final training reward):", SEL)

DATA = {}
for tgt in B:
    rbc = _cost(_rows(EXP / f"rbc_{tgt}_smud_tou" / "progress.csv")[-1])
    zs = []
    for src in B:
        if src == tgt:
            continue
        f = (
            EXP
            / f"transfer_{src}_to_{tgt}_smud_sacp_zs__src{SEL[src]}"
            / "progress.csv"
        )
        zs.append(100 * (rbc - _cost(_rows(f)[-1])) / rbc)
    e2e = e2e_cumavg(tgt, rbc)
    m, sd = e2e.mean(0), e2e.std(0)
    cross = next((k for k in range(len(m)) if m[k] > np.mean(zs)), None)
    DATA[tgt] = dict(
        zs_mean=float(np.mean(zs)),
        zs_lo=min(zs),
        zs_hi=max(zs),
        e2e_m=m,
        e2e_s=sd,
        cross=cross,
    )
    print(
        f"  {tgt:13s} deployed {np.mean(zs):+5.1f} [{min(zs):+5.1f},{max(zs):+5.1f}]"
        f"  E2E s1 {m[0]:+6.1f} s10 {m[-1]:+5.1f}"
        f"  break-even: {'season ' + str(cross + 1) if cross is not None else 'none in 10'}"
    )

# final printed size (\textwidth = 7.1667 in): rcParams sizes are page sizes
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.2,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.5,
        "axes.linewidth": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

YLIM = (-30, 10)
OFF = 0.9  # gap between an anchor and its label band edge (y units; 40-u scale)
# plate pad kept minimal so a plated label cannot cross the panel spine
CHIP = dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="none")

fig, axes = plt.subplots(1, 4, figsize=(7.1667, 1.912), sharey=True)
# geometry first: label boxes are measured in data units with the renderer,
# so the axes must already have their final size and limits
fig.subplots_adjust(left=0.075, right=0.972, top=0.8983, bottom=0.1709, wspace=0.10)
axes[0].set_ylim(*YLIM)
axes[0].set_xlim(0.5, N + 0.5)
_stroke = [pe.withStroke(linewidth=1.6, foreground="white")]


def measure(text):
    """Rendered (width, height) of a chipped label in data units + margin
    (hardcoded band sizes undershot the chip-padded box)."""
    t = axes[0].annotate(
        text,
        xy=(5, 0),
        fontsize=7,
        ha="left",
        va="bottom",
        path_effects=_stroke,
        bbox=CHIP,
    )
    fig.canvas.draw()
    bb = t.get_window_extent()
    (x0, y0), (x1, y1) = axes[0].transData.inverted().transform(
        [[bb.x0, bb.y0], [bb.x1, bb.y1]]
    )
    t.remove()
    return (x1 - x0) + 0.10, (y1 - y0) + 0.30


def band_free(xa, w, lo, hi, e2e, lines, taken):
    """No E2E-curve point, reference line, or placed label inside the band
    (the zero line included)."""
    if lo < YLIM[0] + 0.2 or hi > YLIM[1] - 0.2:
        return False
    ys = np.interp(np.linspace(xa, xa + w, 12), X, e2e)
    if np.any((ys > lo) & (ys < hi)):
        return False
    if any(lo < y < hi for y in lines):
        return False
    return not any(
        xa < tx1 and xa + w > tx0 and lo < thi and hi > tlo
        for tx0, tx1, tlo, thi in taken
    )


def place_right_label(z, w, h, e2e, lines, taken, prefer_up):
    """Vertical side for a right-edge value label: flip when the preferred
    band holds ink; a tightened second pass handles squeezed anchors (the MO
    +1.3 label between its line, the curve tail, and the zero line); the
    white chip covers the final fallback."""
    xa = N + 0.42 - w
    for off, hh in ((OFF, h), (0.55, h - 0.5)):
        for up in (prefer_up, not prefer_up):
            lo = z + off if up else z - off - hh
            if band_free(xa, w, lo, lo + hh, e2e, lines, taken):
                return up, (lo if up else lo + hh), lo, lo + hh, False
    # last resort: accept a band whose only intruder is the zero line - the
    # label then rides a white plate (plated=True); data ink and labels stay
    # untouchable (the MO +1.3 anchor has no zero-free band at this scale)
    nz = [l for l in lines if l != 0.0]
    for off, hh in ((OFF, h), (0.55, h - 0.5)):
        for up in (prefer_up, not prefer_up):
            lo = z + off if up else z - off - hh
            if band_free(xa, w, lo, lo + hh, e2e, nz, taken):
                return up, (lo if up else lo + hh), lo, lo + hh, True
    ya = z + OFF if prefer_up else z - OFF
    lo = ya if prefer_up else ya - h
    return prefer_up, ya, lo, lo + h, True


for ax, tgt in zip(axes, B):
    v = DATA[tgt]
    ax.axhline(0, color="k", lw=0.7, ls=(0, (4, 2.4)), alpha=0.65, zorder=1)
    # deployed line/band span the evaluated seasons [1, N] only (the full-width
    # spans are reserved for the zero reference line)
    ax.fill_between(
        [1, N], v["zs_lo"], v["zs_hi"], color=C_ZS, alpha=0.13, lw=0, zorder=1.5
    )
    (hz,) = ax.plot([1, N], [v["zs_mean"]] * 2, color=C_ZS, lw=1.6, zorder=3)
    (he,) = ax.plot(
        X, v["e2e_m"], color=C_E2E, lw=1.15, marker="o", ms=2.1, mew=0, zorder=4
    )
    ax.fill_between(
        X,
        v["e2e_m"] - v["e2e_s"],
        v["e2e_m"] + v["e2e_s"],
        color=C_E2E,
        alpha=0.14,
        lw=0,
        zorder=2,
    )
    if v["cross"] is not None:
        ax.plot(
            X[v["cross"]],
            v["zs_mean"],
            marker="o",
            ms=4.6,
            mfc="white",
            mec="k",
            mew=0.8,
            zorder=6,
        )
    # end-value labels inside the right edge; the upper anchor is placed
    # first, sides are collision-resolved against the zero line, the deployed
    # line, the curve, and each other (e.g. the MO +1.3 label must not sit
    # across the zero dashes)
    e2e_end = v["e2e_m"][-1]
    taken = []
    blue = (C_ZS, v["zs_mean"], [0.0])
    red = (C_E2E, e2e_end, [0.0, v["zs_mean"]])
    upper, lower = (blue, red) if v["zs_mean"] >= e2e_end else (red, blue)
    for (col, z, lines), prefer in ((upper, True), (lower, False)):
        w, h = measure(f"{z:+.1f}")
        up, ya, lo, hi, plated = place_right_label(
            z, w, h, v["e2e_m"], lines, taken, prefer
        )
        taken.append((N + 0.42 - w, N + 0.42, lo, hi))
        if plated:
            PLATED.setdefault(tgt, set()).add(f"{z:+.1f}")
        ax.annotate(
            f"{z:+.1f}",
            xy=(N + 0.42, ya),
            fontsize=7,
            color=col,
            va="bottom" if up else "top",
            ha="right",
            path_effects=_stroke,
            # halo-only by default: a plate would notch the bands and can
            # cross the spine; only zero-line fallbacks ride one
            bbox=CHIP if plated else None,
            zorder=7,
        )
    ax.set_title(TITLE[tgt])
    ax.set_xlim(0.5, N + 0.5)
    ax.set_xticks([2, 4, 6, 8, 10])
    ax.set_xlabel("Season")
# frame ends on ticks, zero ticked (house convention; ylim = YLIM, set with
# the geometry block above so label measurement sees the final scale)
axes[0].set_yticks([-30, -20, -10, 0, 10])
axes[0].set_ylabel("Cumulative average\nsavings vs. RBC [%]")
axes[0].legend(
    [hz, he],
    ["TLD-RL zero-shot", "E2E (Scratch), online"],
    loc="lower right",
    frameon=True,
    framealpha=0.9,
    edgecolor="0.8",
    borderpad=0.35,
    handlelength=1.7,
    labelspacing=0.28,
    borderaxespad=0.25,
)
# --- label collision audit (dies loudly; a clean pass is the release gate) ---
fig.canvas.draw()
bad = []
for ax, tgt in zip(axes, B):
    v = DATA[tgt]
    inv = ax.transData.inverted()
    xs = np.linspace(*ax.get_xlim(), 300)
    lx = np.linspace(1, N, 300)
    obstacles = [
        (xs, np.zeros_like(xs), "zero line"),
        (lx, np.full_like(lx, v["zs_mean"]), "deployed line"),
        (lx, np.interp(lx, X, v["e2e_m"]), "E2E curve"),
    ]
    boxes = []
    for t in ax.texts:
        bb = t.get_window_extent()
        (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        boxes.append(((x0, x1, y0, y1), t.get_text()))
    for (x0, x1, y0, y1), s in boxes:
        for ox, oy, oname in obstacles:
            if np.any((ox > x0) & (ox < x1) & (oy > y0) & (oy < y1)):
                if oname == "zero line" and s in PLATED.get(tgt, set()):
                    # declared plate over reference ink (tiered rule); an
                    # unplated label on the zero line stays fatal
                    print(f"  note: {tgt} '{s}' plated over the zero line")
                    continue
                bad.append(f"{tgt}: '{s}' intersects {oname}")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (a0, a1, b0, b1), sa = boxes[i]
            (c0, c1, d0, d1), sb = boxes[j]
            if a0 < c1 and a1 > c0 and b0 < d1 and b1 > d0:
                bad.append(f"{tgt}: '{sa}' overlaps '{sb}'")
    leg = ax.get_legend()
    if leg is not None:
        lb = leg.get_window_extent()
        (lx0, ly0), (lx1, ly1) = inv.transform([[lb.x0, lb.y0], [lb.x1, lb.y1]])
        for (x0, x1, y0, y1), s in boxes:
            if x0 < lx1 and x1 > lx0 and y0 < ly1 and y1 > ly0:
                bad.append(f"{tgt}: '{s}' under the legend")
if bad:
    raise SystemExit("label collision audit FAILED:\n  " + "\n  ".join(bad))
print("label collision audit: clean (zero line, deployed line, curve, labels, legend)")

fig.savefig(MAN / "fig_adoption_v3.pdf")
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig_adoption_row.png", dpi=200)
print(f"written {MAN}/fig_adoption_v3.pdf")
