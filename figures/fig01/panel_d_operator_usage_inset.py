"""Export a minimal Figure 1d operator-usage inset as a vector PDF.

Usage is the pooled absolute coefficient mass of each operator,

    100 * sum_t |c_{m,t}| / sum_{j,t} |c_{j,t}|,

computed across the five retained single-mouse sessions.  The chart directly
mirrors the compact Figure 1c syllable-usage inset: ranked bars, a log scale,
the same 0.5% threshold, no tick or axis labels, a blue above-threshold head,
and a grey low-usage tail.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dlds_release.paths import out_dir, single_fit_dir


RUN = single_fit_dir()
OUT = out_dir("fig01")
OUTPUT = OUT / "fig_dlds_operator_usage_standalone.pdf"

SESSIONS = [
    "21_12_10_def6b_3",
    "21_12_2_def6a_1",
    "21_12_2_def6b_2",
    "22_04_26_cage4_0",
    "22_04_26_cage4_1_1",
]
N_OPERATORS = 15
USAGE_MIN = 0.5

BLUE = "#1f6fb2"
GREY = "#c8ccd2"
GREY_LINE = "#7e8893"


def load_usage() -> tuple[np.ndarray, np.ndarray]:
    """Return operator indices and pooled absolute-coefficient usage in rank order."""
    mass = np.zeros(N_OPERATORS, dtype=float)
    for sid in SESSIONS:
        coefficients = np.load(RUN / f"cs_{sid}.npy")
        if coefficients.shape[0] != N_OPERATORS:
            coefficients = coefficients.T
        if coefficients.shape[0] != N_OPERATORS:
            raise ValueError(f"Unexpected coefficient shape for {sid}: {coefficients.shape}")
        mass += np.abs(coefficients).sum(axis=1)

    usage = mass / mass.sum() * 100.0
    order = np.argsort(usage)[::-1]
    return order, usage[order]


def main() -> None:
    order, usage = load_usage()
    above = usage >= USAGE_MIN
    n_above = int(above.sum())
    top_share = float(usage[:n_above].sum())
    tail_share = float(usage[n_above:].sum())

    if n_above != 3:
        raise ValueError(
            f"Expected 3/{N_OPERATORS} operators at >= {USAGE_MIN}% usage; "
            f"found {n_above}/{N_OPERATORS}"
        )
    if not np.array_equal(order[:3], np.array([14, 2, 6])):
        raise ValueError(f"Expected leading operators f14, f2, f6; found {order[:3].tolist()}")

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # Match the compact metadata graphic in panel 1c: the bars carry the
    # argument, while the single heading reports the thresholded count.
    fig, ax = plt.subplots(figsize=(3.1, 1.25))
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.12, top=0.67)

    ranks = np.arange(1, N_OPERATORS + 1)
    ax.bar(ranks[:n_above], usage[:n_above], width=1.0, color=BLUE,
           edgecolor="none", zorder=3)
    ax.bar(ranks[n_above:], usage[n_above:], width=1.0, color=GREY,
           edgecolor="none", zorder=2)

    ax.set_yscale("log")
    ax.set_xlim(0.0, N_OPERATORS + 1)
    ax.set_ylim(usage.min() * 0.6, usage.max() * 3.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axhline(USAGE_MIN, color=GREY_LINE, lw=0.8,
               ls=(0, (3, 2)), zorder=4)

    ax.set_title(
        rf"usage: {n_above}/{N_OPERATORS} $\geq$ {USAGE_MIN:.1f}%",
        fontsize=12, fontweight="bold", color="#566069", pad=9,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b8bec6")
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(axis="both", length=0)
    ax.grid(False)

    # Keep a fixed transparent canvas for clean placement in Illustrator.
    fig.savefig(OUTPUT, transparent=True)
    plt.close(fig)
    print(f"wrote {OUTPUT}")
    print(f"top 3 share: {top_share:.6f}%")
    print(f"remaining 12 share: {tail_share:.6f}%")


if __name__ == "__main__":
    main()
