"""Export the Figure 1c syllable-usage distribution as a standalone vector PDF.

The canvas, margins, axes, threshold, colors, and typography intentionally
match ``fig_dlds_operator_usage_standalone.py`` so the two assets can be placed
side by side in Illustrator without manual realignment.
"""

from __future__ import annotations

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dlds_release.paths import moseq_results, out_dir


KPMS = moseq_results("single")
OUT = out_dir("fig01")
OUTPUT = OUT / "fig_moseq_syllable_usage_standalone.pdf"

SESSIONS = [
    "21_12_10_def6b_3",
    "21_12_2_def6a_1",
    "21_12_2_def6b_2",
    "22_04_26_cage4_0",
    "22_04_26_cage4_1_1",
]
USAGE_MIN = 0.5

BLUE = "#1f6fb2"
GREY = "#c8ccd2"
GREY_DARK = "#566069"
GREY_LINE = "#7e8893"


def load_usage() -> np.ndarray:
    """Return nonzero syllable frame usages in descending order."""
    counts = np.zeros(200, dtype=np.int64)
    with h5py.File(KPMS, "r") as results:
        for sid in SESSIONS:
            syllables = results[sid]["syllable"][:].astype(int)
            counts += np.bincount(syllables[syllables >= 0], minlength=200)
    usage = counts[counts > 0] / counts.sum() * 100.0
    return np.sort(usage)[::-1]


def main() -> None:
    usage = load_usage()
    n_syllables = int(usage.size)
    n_above = int(np.count_nonzero(usage >= USAGE_MIN))
    head_share = float(usage[:n_above].sum())
    tail_share = float(usage[n_above:].sum())

    if (n_above, n_syllables) != (21, 72):
        raise ValueError(
            f"Expected 21/72 syllables at >= {USAGE_MIN}% usage; "
            f"found {n_above}/{n_syllables}"
        )

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(5.6, 3.35))
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.21, top=0.82)

    ranks = np.arange(1, n_syllables + 1)
    ax.bar(ranks[:n_above], usage[:n_above], width=0.94, color=BLUE,
           edgecolor="none", zorder=3)
    ax.bar(ranks[n_above:], usage[n_above:], width=0.94, color=GREY,
           edgecolor="none", zorder=2)

    ax.set_yscale("log")
    ax.set_xlim(0.0, n_syllables + 1.0)
    ax.set_ylim(4e-4, 240)
    ax.set_xticks([1, 10, 20, 30, 40, 50, 60, 70])
    ax.set_xlabel("syllable rank (descending usage)")
    ax.set_ylabel("usage share (%)\n(log scale)")

    yticks = [0.001, 0.01, 0.1, 1, 10, 100]
    ax.set_yticks(yticks)
    ax.set_yticklabels(["0.001", "0.01", "0.1", "1", "10", "100"])

    ax.axhline(USAGE_MIN, color=GREY_LINE, lw=1.1, ls=(0, (4, 3)), zorder=4)
    ax.text(n_syllables - 0.5, USAGE_MIN * 1.12, "0.5% threshold",
            ha="right", va="bottom", fontsize=8.5, color=GREY_DARK)

    # A short aggregate label mirrors the dLDS top-three label without adding
    # an explanatory sentence above the plot.
    bracket_y = 24
    bracket_bottom = 19
    ax.plot([0.54, n_above + 0.46], [bracket_y, bracket_y], color=BLUE,
            lw=1.5, clip_on=False, zorder=5)
    ax.plot([0.54, 0.54], [bracket_bottom, bracket_y], color=BLUE,
            lw=1.5, clip_on=False, zorder=5)
    ax.plot([n_above + 0.46, n_above + 0.46],
            [bracket_bottom, bracket_y], color=BLUE,
            lw=1.5, clip_on=False, zorder=5)
    ax.text((n_above + 1) / 2, 31, f"top 21: {head_share:.3f}%",
            ha="center", va="bottom", fontsize=10.5, color=BLUE,
            fontweight="bold", clip_on=False)

    ax.set_title(
        rf"usage: {n_above}/{n_syllables} $\geq$ {USAGE_MIN:.1f}%",
        fontsize=13, fontweight="bold", color=GREY_DARK, pad=18,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#8c939b")
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(axis="both", colors="#343a40", width=0.9, length=4)
    ax.grid(False)

    fig.savefig(OUTPUT, transparent=True)
    plt.close(fig)
    print(f"wrote {OUTPUT}")
    print(f"top 21 share: {head_share:.6f}%")
    print(f"remaining 51 share: {tail_share:.6f}%")


if __name__ == "__main__":
    main()
