"""Relevant-only dyadic operator taxonomy for main Figure 4d.

This redesign intentionally omits unused crossed matrices and background/ghost
operators.  It retains the seven operators interpreted in the main text and
groups them in three vertical family columns using the same colors and index
sets as the signed mixer below the panel.

Output: panel_calms21_ov_categories_relevant.pdf
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from dlds_release.paths import dyadic_dictionary, out_dir


DATA = dyadic_dictionary()
OUT = out_dir("fig04")

COL_SUBSTRATE = "#0077BB"  # blue
COL_GEOMETRY = "#009988"  # teal
COL_SOCIAL = "#D55E00"  # vermilion

BLOCKS = {"A": slice(0, 14), "B": slice(14, 25), "C": slice(25, 27)}
ORDER = ("A", "B", "C")
PAIRS = [(target, source) for target in ORDER for source in ORDER]
BOUNDARIES = (13.5, 24.5)
COMPOSITE_OPERATORS = {8, 14}  # zero-indexed f9 and f15
RANK_STYLES = (
    (3.0, "solid"),
    (2.4, (0, (4, 2))),
    (2.0, (0, (1, 1.5))),
)

# Zero-indexed operator IDs.  These are the seven operators retained in the
# main-text interpretation and in the signed mixer below panels c/d.
FAMILIES = (
    {
        "name": "substrate",
        "indices": r"$m \in \{2,3,6\}$",
        "members": (1, 2, 5),
        "color": COL_SUBSTRATE,
    },
    {
        "name": "geometry",
        "indices": r"$n \in \{4,11\}$",
        "members": (3, 10),
        "color": COL_GEOMETRY,
    },
    {
        "name": "social primitive",
        "indices": r"$s \in \{9,15\}$",
        "members": (8, 14),
        "color": COL_SOCIAL,
    },
)


matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 13,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def block_profile(matrix: np.ndarray) -> np.ndarray:
    values = np.array(
        [
            np.abs(matrix[BLOCKS[target], BLOCKS[source]]).mean()
            for target, source in PAIRS
        ]
    )
    return values / values.sum()


def add_block_box(
    ax: plt.Axes,
    pair: tuple[str, str],
    color: str,
    linewidth: float,
    linestyle: str | tuple = "solid",
) -> None:
    target, source = pair
    rows = BLOCKS[target]
    cols = BLOCKS[source]
    ax.add_patch(
        Rectangle(
            (cols.start - 0.5, rows.start - 0.5),
            cols.stop - cols.start,
            rows.stop - rows.start,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=5,
        )
    )


def draw_matrix(
    ax: plt.Axes,
    matrix: np.ndarray,
    operator_index: int,
    profile: np.ndarray,
    vmax: float,
    color: str,
) -> None:
    ax.imshow(
        matrix,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
    )
    for boundary in BOUNDARIES:
        ax.axhline(boundary, color="#777777", linewidth=0.70)
        ax.axvline(boundary, color="#777777", linewidth=0.70)

    ranked = np.argsort(-profile)
    highlight_count = 3 if operator_index in COMPOSITE_OPERATORS else 1
    for rank, ranked_index in enumerate(ranked[:highlight_count]):
        linewidth, linestyle = RANK_STYLES[rank]
        add_block_box(
            ax,
            PAIRS[int(ranked_index)],
            color,
            linewidth,
            linestyle,
        )

    ax.set_xlim(-0.5, 26.5)
    ax.set_ylim(26.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        -0.10,
        0.50,
        rf"$\mathbf{{f}}_{{{operator_index + 1}}}$",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=color,
        clip_on=False,
    )
    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(1.15)


def add_family_header(
    fig: plt.Figure,
    left: float,
    width: float,
    family: dict[str, object],
) -> None:
    """Draw one compact header that directly mirrors a signed-mixer family."""

    ax = fig.add_axes([left, 0.875, width, 0.105])
    ax.set_axis_off()
    color = str(family["color"])
    ax.text(
        0.50,
        0.66,
        str(family["name"]),
        ha="center",
        va="center",
        fontsize=16.5 if family["name"] == "social primitive" else 18,
        fontweight="bold",
        color=color,
        transform=ax.transAxes,
    )
    ax.text(
        0.50,
        0.19,
        str(family["indices"]),
        ha="center",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color=color,
        transform=ax.transAxes,
    )


def main() -> None:
    matrices = np.load(DATA)
    profiles = np.array([block_profile(matrix) for matrix in matrices])
    vmax = float(np.percentile(np.abs(matrices), 99))

    # Three equal columns make the family structure explicit.  Every operator
    # matrix has the same physical size; the 3/2/2 counts therefore remain
    # perceptually honest while the operators, not the labels, dominate.
    fig_width, fig_height = 7.0, 4.30
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")
    column_lefts = (0.018, 0.342, 0.666)
    column_width = 0.316
    matrix_size_in = 1.08
    matrix_width = matrix_size_in / fig_width
    matrix_height = matrix_size_in / fig_height
    y_positions = {
        3: (0.590, 0.315, 0.040),
        2: (0.485, 0.175),
    }

    for left, family in zip(column_lefts, FAMILIES):
        add_family_header(fig, left, column_width, family)
        center_x = left + column_width / 2
        members = tuple(family["members"])
        for bottom, operator_index in zip(y_positions[len(members)], members):
            ax = fig.add_axes(
                [
                    center_x - matrix_width / 2,
                    bottom,
                    matrix_width,
                    matrix_height,
                ]
            )
            draw_matrix(
                ax,
                matrices[operator_index],
                operator_index,
                profiles[operator_index],
                vmax,
                str(family["color"]),
            )

    # Quiet separators reinforce the three-column grammar without adding boxes
    # around the operator inventory.
    guide_ax = fig.add_axes([0, 0, 1, 1], zorder=-1)
    guide_ax.set_axis_off()
    for x in (0.330, 0.654):
        guide_ax.plot(
            [x, x],
            [0.030, 0.855],
            color="#E1E1E1",
            linewidth=0.9,
            transform=guide_ax.transAxes,
            clip_on=False,
        )

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT / "panel_calms21_ov_categories_relevant.pdf",
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
