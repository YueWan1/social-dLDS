"""
Supplementary figure: how the 15-operator dyadic dictionary is reduced to the 7
operators interpreted in the main text.

Panel a shows pooled activation rates, panel b shows block-pair coupling
profiles, and panel c tests the matrix-predicted geometry read-outs across
sessions. Panel c does not use behavior annotations. Operator groups are
defined here to match the released manuscript.

Usage:  python3 figures/supp/s4_operator_selection.py
        (from the repository root)
"""
from __future__ import annotations

import glob
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from dlds_release.paths import DERIVED, dyadic_cs_dir, out_dir

# Paths.
D = str(dyadic_cs_dir())
READOUT_NPZ = str(DERIVED / "dyadic" / "readout_reproducibility.npz")
OUT = os.environ.get("PANEL_OUT", str(out_dir("supp")))
os.makedirs(OUT, exist_ok=True)

# Plot constants.
BLK = {"A": slice(0, 14), "B": slice(14, 25), "C": slice(25, 27)}
ORDER = ["A", "B", "C"]
PAIRS = [(r, c) for r in ORDER for c in ORDER]        # f[R,S] = target <- source
# Display names for the feature blocks.
DISP = {"A": "SELF", "B": "DIST", "C": "DIRC"}
PAIR_LBL = [f"{DISP[r]}$\\leftarrow${DISP[c]}" for r, c in PAIRS]
THR = 0.05                                            # engagement threshold on |c|

# 1-indexed operator numbers, matching the manuscript's f_m naming.
SUBSTRATE = [2, 3, 6]
GEOMETRY = [4, 11]
SOCIAL = [9, 15]
EXCLUDED = [7, 8, 10]                                 # engaged but not interpreted
KEPT = SUBSTRATE + GEOMETRY + SOCIAL

# Block pair(s) on which the main text actually reads each operator.  For f_4 this
# is NOT its dominant pair: f_4's largest pi is C<-C (0.301) but the distance
# read-out is taken on B<-B (0.237).  Marking both keeps the figure honest.
READOUT_BLOCKS = {
    2: ["A←A"], 3: ["A←A"], 6: ["A←A"],
    4: ["B←B"], 11: ["C←B"],
    9: ["A←A", "C←B"], 15: ["A←A", "C←B"],
}
PAIR_KEY = [f"{r}←{c}" for r, c in PAIRS]

FAMILY_OF = {}
for m in SUBSTRATE:
    FAMILY_OF[m] = "substrate"
for m in GEOMETRY:
    FAMILY_OF[m] = "geometry"
for m in SOCIAL:
    FAMILY_OF[m] = "social"
for m in EXCLUDED:
    FAMILY_OF[m] = "excluded"

COL = {
    "substrate": "#2e7d8f",
    "geometry": "#c06a2b",
    "social": "#8e4585",
    "excluded": "#9aa0a6",
    "unused": "#d4d6d9",
}

plt.rcParams.update({
    "font.size": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# Load tracked statistics.
def load_operators():
    F = np.load(f"{D}/F_universal.npy")                # (15, 27, 27)
    cs_files = sorted(glob.glob(f"{D}/cs_mouse*.npy"))
    if not cs_files:
        raise FileNotFoundError(f"no cs_mouse*.npy under {D}")
    nact = np.zeros(F.shape[0])
    ntot = 0
    for f in cs_files:
        c = np.load(f)
        nact += (np.abs(c) > THR).sum(axis=1)
        ntot += c.shape[1]
    rate = nact / ntot * 100.0
    return F, rate, ntot, len(cs_files)


def block_profile(Fm):
    raw = np.array([np.abs(Fm[BLK[r], BLK[c]]).mean() for r, c in PAIRS])
    return raw / raw.sum()


# Panel a: activation rates.
def panel_a(ax, rate):
    order = np.argsort(rate)[::-1]                     # descending
    engaged_n = 10
    cut = 0.5 * (rate[order[engaged_n - 1]] + rate[order[engaged_n]])
    gap = rate[order[engaged_n - 1]] / rate[order[engaged_n]]

    ypos = np.arange(len(order))[::-1]
    colors = []
    for i in order:
        m = i + 1
        colors.append(COL[FAMILY_OF[m]] if m in FAMILY_OF else COL["unused"])

    ax.barh(ypos, rate[order], color=colors, height=0.72,
            edgecolor="#333333", linewidth=0.4)
    ax.set_xscale("log")
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"$f_{{{i+1}}}$" for i in order])
    ax.set_xlabel(r"frames with $|c_m|>0.05$ (%, pooled)")
    ax.set_xlim(0.015, 200)

    ax.axvline(cut, color="#b3261e", lw=1.0, ls="--", zorder=5)
    ax.annotate(f"{gap:.1f}$\\times$ gap",
                xy=(cut, ypos[engaged_n - 1] - 0.5),
                xytext=(cut * 2.2, ypos[engaged_n - 1] - 2.6),
                color="#b3261e", fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color="#b3261e", lw=0.7))
    ax.text(cut, 1.005, f"cut {cut:.2f}%", transform=ax.get_xaxis_transform(),
            color="#b3261e", fontsize=7, ha="center", va="bottom")

    ax.text(0.99, 1.02, "10 engaged", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5, color="#333333")
    ax.set_title("a   Activation-rate screen: 15 $\\rightarrow$ 10",
                 loc="left", fontsize=9, fontweight="bold", pad=14)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return cut, gap


# Panel b: block profiles.
def panel_b(ax, F):
    rows = SUBSTRATE + GEOMETRY + SOCIAL + EXCLUDED
    prof = np.array([block_profile(F[m - 1]) for m in rows])

    im = ax.imshow(prof, cmap="Greys", aspect="auto", vmin=0, vmax=prof.max())
    for r, m in enumerate(rows):
        j = int(np.argmax(prof[r]))
        ax.add_patch(Rectangle((j - 0.5, r - 0.5), 1, 1, fill=False,
                               edgecolor=COL[FAMILY_OF[m]], lw=1.8, zorder=4))
        # dashed: block used for the main-text read-out, where it is not the
        # dominant pair (f_4) or where a second block is also read (f_9, f_15)
        for key in READOUT_BLOCKS.get(m, []):
            k = PAIR_KEY.index(key)
            if k == j:
                continue
            ax.add_patch(Rectangle((k - 0.5, r - 0.5), 1, 1, fill=False,
                                   edgecolor=COL[FAMILY_OF[m]], lw=1.4,
                                   ls=(0, (2.2, 1.6)), zorder=4))

    ax.set_xticks(range(len(PAIRS)))
    ax.set_xticklabels(PAIR_LBL, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"$f_{{{m}}}$" for m in rows])
    for r, m in enumerate(rows):
        ax.get_yticklabels()[r].set_color(COL[FAMILY_OF[m]])

    # separators between the family groups
    for boundary in np.cumsum([len(SUBSTRATE), len(GEOMETRY), len(SOCIAL)]) - 0.5:
        ax.axhline(boundary, color="#333333", lw=0.9)
    ax.axhline(len(rows) - len(EXCLUDED) - 0.5, color="#b3261e", lw=1.4)

    labels = [("substrate", 1.0, "substrate"), ("geometry", 3.5, "geometry"),
              ("social", 5.5, "social"), ("not interpreted", 8.0, "excluded")]
    ax.set_xlim(-0.5, 8.5)
    for txt, y, fam in labels:
        ax.text(8.72, y, txt, va="center", ha="left", fontsize=7.5,
                color=COL[fam], clip_on=False)

    ax.set_title("b   Block-pair coupling profile $\\pi$ of the 10 engaged operators",
                 loc="left", fontsize=9, fontweight="bold", pad=8)
    # No x-label: the tick labels already read "target <- source".
    ax.text(0.0, -0.46, "solid: dominant pair    dashed: block read in the main text",
            transform=ax.transAxes, fontsize=6.8, color="#555555", ha="left")
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.26)
    cb.set_label(r"$\pi^{(m)}_{R\leftarrow S}$", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    return prof, rows


# Panel c: screening statistics.
def _compass(ax, m, dat, col, label):
    """Predicted C<-B axis vs measured mean partner bearing, on a compass.

    u_1 is an AXIS, not a vector: its sign is arbitrary, so it is drawn as a full
    line through the origin and the error is the axial one, min(|d|, 180-|d|).
    Using the signed difference would render f_4 as anti-aligned at 176.8 deg
    when its axial error is 3.2 deg, the smallest of the four.
    """
    ang = np.deg2rad(float(dat[f"f{m}_u1_angle_deg"]))
    mu = np.deg2rad(float(dat[f"f{m}_mu_bearing_deg"]))
    R = float(dat[f"f{m}_R"])
    err = float(dat[f"f{m}_axis_error_deg"])

    ax.plot([ang, ang + np.pi], [1, 1], color=col, lw=1.3,
            ls=(0, (3, 1.8)), zorder=3, solid_capstyle="butt")
    ax.annotate("", xy=(mu, R), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#111111", lw=1.5,
                                shrinkA=0, shrinkB=0), zorder=5)

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels([])
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["0", "90", "$\\pm$180", "-90"], fontsize=6.5)
    ax.tick_params(pad=-2)
    ax.grid(color="#dddddd", lw=0.5)
    ax.text(0.5, -0.16, f"{label}\n$\\Delta={err:.1f}^\\circ$,  $R={R:.2f}$",
            transform=ax.transAxes, fontsize=7.5, color=col,
            ha="center", va="top")


def panel_c(axes, dat):
    ax1, ax2, ax3 = axes

    # --- c1: distance read-out, f_4 (kept) vs f_7 (excluded)
    r4 = np.asarray(dat["f4_r_per_session"], dtype=float)
    r7 = np.asarray(dat["f7_r_per_session"], dtype=float)
    r4 = r4[np.isfinite(r4)]
    r7 = r7[np.isfinite(r7)]

    rng = np.random.default_rng(20260719)
    for k, (vals, m, col) in enumerate([(r4, 4, COL["geometry"]),
                                        (r7, 7, COL["excluded"])]):
        x = k + rng.uniform(-0.16, 0.16, size=vals.size)
        ax1.scatter(x, vals, s=7, color=col, alpha=0.65, linewidths=0)
        ax1.hlines(np.median(vals), k - 0.30, k + 0.30,
                   color="#111111", lw=1.6, zorder=6)
        n_pass = int((vals > 0.5).sum())
        ax1.text(k, -1.06, f"{n_pass}/{vals.size}", ha="center", va="bottom",
                 fontsize=7.5, color=col)

    ax1.axhline(0.5, color="#b3261e", ls="--", lw=0.9)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["$f_4$\n(interpreted)", "$f_7$\n(not interpreted)"])
    ax1.set_ylim(-1.15, 1.05)
    ax1.set_ylabel("per-session $r$, distance read-out")
    ax1.set_title("c   Matrix-predicted read-out vs the measured quantity",
                  loc="left", fontsize=9, fontweight="bold", pad=10)
    ax1.text(1.46, 0.5, "$r=0.5$", fontsize=6.8, color="#b3261e",
             ha="right", va="bottom")
    ax1.text(0.5, -0.30, "counts: sessions with $r>0.5$",
             transform=ax1.transAxes, fontsize=6.8, color="#666666",
             va="top", ha="center")
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # --- c2/c3: direction read-out, f_11 (kept) vs f_10 (excluded).
    # The two excluded operators fail differently and no single statistic
    # separates them from the kept set: f_7 fails on the distance read-out
    # (left) while its bearing axis is only 33 deg off, and f_10 has a
    # reproducible direction read-out but its predicted axis misses the
    # measured bearing by 82 deg on diffuse bearings.  Each operator is
    # therefore judged on the read-out it itself claims.
    _compass(ax2, 11, dat, COL["geometry"], "$f_{11}$ (interpreted)")
    _compass(ax3, 10, dat, COL["excluded"], "$f_{10}$ (not interpreted)")
    ax3.text(0.0, -0.52,
             "dashed: predicted DIRC$\\leftarrow$DIST axis    arrow: measured mean "
             "bearing (length $R$)",
             transform=ax3.transAxes, fontsize=6.8, color="#666666",
             va="top", ha="center")


def main():
    F, rate, ntot, n_sess = load_operators()
    print(f"loaded {F.shape[0]} operators, {n_sess} sessions, {ntot} frames")

    have_c = os.path.exists(READOUT_NPZ)
    if not have_c:
        print(f"WARNING {READOUT_NPZ} missing; emitting panels a,b only.")

    if have_c:
        fig = plt.figure(figsize=(7.2, 8.8))
        gs = fig.add_gridspec(3, 4, height_ratios=[1.15, 1.0, 0.85],
                              hspace=0.90, wspace=0.55,
                              left=0.11, right=0.94, top=0.94, bottom=0.08)
        ax_a = fig.add_subplot(gs[0, :])
        ax_b = fig.add_subplot(gs[1, :])
        ax_c1 = fig.add_subplot(gs[2, 0:2])
        ax_c2 = fig.add_subplot(gs[2, 2], projection="polar")
        ax_c3 = fig.add_subplot(gs[2, 3], projection="polar")
    else:
        fig = plt.figure(figsize=(7.2, 6.4))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.55,
                              left=0.11, right=0.94, top=0.93, bottom=0.10)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[1, 0])

    cut, gap = panel_a(ax_a, rate)
    prof, rows = panel_b(ax_b, F)
    if have_c:
        dat = np.load(READOUT_NPZ, allow_pickle=True)
        panel_c((ax_c1, ax_c2, ax_c3), dat)
        print("\n--- panel c ---")
        print(f"  f_4 distance  median r={np.median(dat['f4_r_per_session']):.6f}"
              f"  n={dat['f4_r_per_session'].size}")
        print(f"  f_7 distance  median r={np.median(dat['f7_r_per_session']):.6f}"
              f"  n={dat['f7_r_per_session'].size}")
        for m in (11, 10, 4, 7):
            print(f"  f_{m:<3} u1={float(dat[f'f{m}_u1_angle_deg']):+8.3f}deg"
                  f"  mu={float(dat[f'f{m}_mu_bearing_deg']):+8.3f}deg"
                  f"  axial err={float(dat[f'f{m}_axis_error_deg']):6.3f}deg"
                  f"  signed={float(dat[f'f{m}_axis_error_signed_deg']):+8.3f}deg"
                  f"  R={float(dat[f'f{m}_R']):.6f}")

    stem = "supp_operator_selection" if have_c else "supp_operator_selection_ab"
    p = os.path.join(OUT, f"{stem}.pdf")
    fig.savefig(p, bbox_inches="tight")
    print("saved:", p)
    plt.close(fig)

    # ---- numbers for the caption / prose, printed so they can be checked
    order = np.argsort(rate)[::-1]
    print("\n--- panel a ---")
    for k, i in enumerate(order):
        tag = FAMILY_OF.get(i + 1, "unused")
        print(f"  rank {k+1:>2}  f_{i+1:<3} {rate[i]:8.4f}%   {tag}")
    print(f"  cut = {cut:.6f}%   gap = {gap:.6f}x")
    print("\n--- panel b (dominant pair) ---")
    for r, m in enumerate(rows):
        j = int(np.argmax(prof[r]))
        lbl = f"{PAIRS[j][0]}<-{PAIRS[j][1]}"
        print(f"  f_{m:<3} {lbl:>5}  pi={prof[r][j]:.6f}  {FAMILY_OF[m]}")


if __name__ == "__main__":
    main()
