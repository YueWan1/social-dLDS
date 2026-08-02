"""Published dyadic LOSO stability figure.

The public workflow reads the plotted values shipped in ``derived/``. Readers
therefore reproduce the published figure without rerunning 30 expensive LOSO
dictionary fits.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np

from dlds_release.paths import DERIVED, out_dir


OUT = out_dir("supp")
VALUES = DERIVED / "dyadic" / "loso_stability_values.csv"

BLUE = "#2166AC"
LIGHT_BLUE = "#92C5DE"
RED = "#B2182B"
ORANGE = "#EF8A62"
DARK = "#1A1A1A"
GRID = "#DDDDDD"

STABILITY_CMAP = LinearSegmentedColormap.from_list(
    "stability",
    [
        (0.00, "#B2182B"),
        (0.45, "#EF8A62"),
        (0.80, "#F7F7F7"),
        (0.93, "#92C5DE"),
        (1.00, "#2166AC"),
    ],
)

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_stability() -> dict[str, np.ndarray]:
    if not VALUES.exists():
        raise FileNotFoundError(f"Published LOSO values not found: {VALUES}")

    whole = np.full(30, np.nan)
    per_operator = np.full((30, 15), np.nan)
    cross_slot: dict[int, dict[tuple[int, int], float]] = {}
    with VALUES.open(newline="") as handle:
        for row in csv.DictReader(handle):
            fold = int(row["fold"])
            value = float(row["value"])
            if row["record"] == "whole_dictionary":
                whole[fold - 1] = value
            elif row["record"] == "same_slot":
                operator = int(row["consensus_operator"])
                per_operator[fold - 1, operator - 1] = value
            elif row["record"] == "cross_slot":
                pair = (
                    int(row["consensus_operator"]),
                    int(row["fold_operator"]),
                )
                cross_slot.setdefault(fold, {})[pair] = value

    if np.isnan(whole).any() or np.isnan(per_operator).any():
        raise ValueError(f"Incomplete published LOSO table: {VALUES}")

    diagnostics = []
    for fold, operators in ((23, (6, 10)), (24, (2, 6)), (25, (6, 10))):
        values = cross_slot.get(fold, {})
        matrix = np.array(
            [[values[(row, column)] for column in operators] for row in operators]
        )
        diagnostics.append(
            {"fold": fold, "operators": np.asarray(operators), "matrix": matrix}
        )

    assert np.isclose(whole.mean(), 0.99530, atol=5e-5)
    assert np.isclose(whole.min(), 0.96418, atol=5e-5)
    assert np.isclose(per_operator[:, 5].mean(), 0.94103, atol=5e-5)
    assert np.isclose(per_operator[:, 5].min(), 0.38936, atol=5e-5)

    return {
        "fold_ids": np.arange(1, 31),
        "whole": whole,
        "per_operator": per_operator,
        "diagnostics": diagnostics,
    }


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=0.9, length=3.5)


def draw_stability_matrix(
    ax_top: plt.Axes,
    ax_heat: plt.Axes,
    ax_right: plt.Axes,
    cax: plt.Axes,
    data: dict[str, np.ndarray],
) -> None:
    values = data["per_operator"]
    whole = data["whole"]
    folds = data["fold_ids"]
    x = np.arange(15)

    operator_mean = values.mean(axis=0)
    operator_min = values.min(axis=0)
    ax_top.axhspan(0.95, 1.01, color="#EAF3F8", zorder=0)
    ax_top.plot(
        x,
        operator_mean,
        color=BLUE,
        marker="o",
        markersize=3.8,
        linewidth=1.1,
        label="mean",
    )
    ax_top.plot(
        x,
        operator_min,
        color=RED,
        marker="v",
        markersize=4.2,
        linewidth=1.0,
        label="minimum",
    )
    ax_top.set_xlim(-0.5, 14.5)
    ax_top.set_ylim(0.35, 1.015)
    ax_top.set_yticks([0.4, 0.7, 1.0])
    ax_top.set_xticks([])
    ax_top.set_ylabel("operator\ncosine")
    ax_top.legend(frameon=False, ncol=2, loc="lower left")
    ax_top.set_title(
        "Fold-by-operator stability",
        loc="left",
        fontweight="bold",
        pad=7,
    )
    style_axis(ax_top)

    image = ax_heat.imshow(
        values,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        cmap=STABILITY_CMAP,
        vmin=0.35,
        vmax=1.0,
    )
    low_cells = np.argwhere(values < 0.80)
    for fold_index, operator_index in low_cells:
        ax_heat.add_patch(
            Rectangle(
                (operator_index - 0.5, fold_index - 0.5),
                1,
                1,
                fill=False,
                edgecolor=DARK,
                linewidth=0.75,
            )
        )
        ax_heat.text(
            operator_index,
            fold_index,
            f"{values[fold_index, operator_index]:.2f}",
            ha="center",
            va="center",
            fontsize=6.2,
            color="white" if values[fold_index, operator_index] < 0.58 else DARK,
            fontweight="bold",
        )

    ax_heat.set_xticks(
        x,
        [rf"$f_{{{operator}}}$" for operator in range(1, 16)],
    )
    shown_folds = [1, 5, 10, 15, 20, 23, 24, 25, 30]
    ax_heat.set_yticks(
        [fold - 1 for fold in shown_folds],
        [str(fold) for fold in shown_folds],
    )
    for tick, fold in zip(ax_heat.get_yticklabels(), shown_folds):
        if fold in (23, 24, 25):
            tick.set_color(RED)
            tick.set_fontweight("bold")
    for operator in (2, 6, 10):
        tick = ax_heat.get_xticklabels()[operator - 1]
        tick.set_color(RED)
        tick.set_fontweight("bold")
    ax_heat.set_xlabel("operator slot after sign alignment")
    ax_heat.set_ylabel("LOSO fold")
    ax_heat.tick_params(length=0)

    colorbar = plt.colorbar(image, cax=cax, orientation="vertical")
    colorbar.set_ticks([0.4, 0.6, 0.8, 0.95, 1.0])
    colorbar.set_label("fold-to-consensus cosine", fontsize=8.5)
    colorbar.ax.tick_params(labelsize=7.5, length=2.5)

    y = np.arange(30)
    ax_right.scatter(
        whole,
        y,
        color=BLUE,
        s=18,
        edgecolor="white",
        linewidth=0.3,
        zorder=3,
    )
    for fold in (23, 24, 25):
        ax_right.scatter(
            whole[fold - 1],
            fold - 1,
            color=RED,
            s=28,
            edgecolor=DARK,
            linewidth=0.4,
            zorder=4,
        )
    ax_right.axvline(whole.mean(), color=RED, linestyle="--", linewidth=1.0)
    ax_right.set_xlim(0.958, 1.002)
    ax_right.set_ylim(29.5, -0.5)
    ax_right.set_xticks([0.96, 0.98, 1.0])
    ax_right.set_yticks([])
    ax_right.set_xlabel("whole-dictionary\ncosine")
    ax_right.set_title(
        f"mean {whole.mean():.3f}\nmin {whole.min():.3f}",
        fontsize=8.5,
        pad=5,
    )
    ax_right.grid(axis="x", color=GRID, linewidth=0.6)
    style_axis(ax_right)


def draw_mixing_diagnostics(
    axes: list[plt.Axes],
    cax: plt.Axes,
    diagnostics: list[dict[str, np.ndarray]],
) -> None:
    last_image = None
    for index, (ax, diagnostic) in enumerate(zip(axes, diagnostics)):
        matrix = diagnostic["matrix"]
        operators = diagnostic["operators"]
        last_image = ax.imshow(
            matrix,
            cmap=STABILITY_CMAP,
            vmin=0.35,
            vmax=1.0,
            aspect="equal",
            interpolation="nearest",
        )
        for row in range(2):
            for col in range(2):
                value = matrix[row, col]
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="white" if value < 0.58 else DARK,
                )
        labels = [rf"$f_{{{operator}}}$" for operator in operators]
        ax.set_xticks([0, 1], labels)
        ax.set_yticks([0, 1], labels if index == 0 else [])
        ax.set_xlabel("fold operator", labelpad=2)
        if index == 0:
            ax.set_ylabel("consensus\noperator", labelpad=4)
        ax.set_title(
            f"fold {diagnostic['fold']}: "
            + "/".join(f"$f_{{{operator}}}$" for operator in operators),
            fontsize=9.5,
            fontweight="bold",
            pad=5,
        )
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color(DARK)

    colorbar = plt.colorbar(last_image, cax=cax, orientation="horizontal")
    colorbar.set_ticks([0.4, 0.6, 0.8, 1.0])
    colorbar.set_label("absolute cross-slot cosine", fontsize=8.5, labelpad=2)
    colorbar.ax.tick_params(labelsize=7.5, length=2.5)


def write_values(data: dict[str, np.ndarray]) -> Path:
    path = OUT / "loso_stability_values.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["record", "fold", "consensus_operator", "fold_operator", "value"]
        )
        for fold, whole in zip(data["fold_ids"], data["whole"]):
            writer.writerow(["whole_dictionary", fold, "", "", whole])
        for fold, values in zip(data["fold_ids"], data["per_operator"]):
            for operator, value in enumerate(values, start=1):
                writer.writerow(["same_slot", fold, operator, operator, value])
        for diagnostic in data["diagnostics"]:
            for row, consensus_operator in enumerate(diagnostic["operators"]):
                for col, fold_operator in enumerate(diagnostic["operators"]):
                    writer.writerow(
                        [
                            "cross_slot",
                            diagnostic["fold"],
                            consensus_operator,
                            fold_operator,
                            diagnostic["matrix"][row, col],
                        ]
                    )
    return path


def main() -> None:
    data = load_stability()

    fig = plt.figure(figsize=(11.8, 7.2))
    outer = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        left=0.075,
        right=0.975,
        top=0.88,
        bottom=0.095,
        width_ratios=[2.45, 1.0],
        wspace=0.34,
    )

    matrix_grid = gridspec.GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=outer[0],
        height_ratios=[1.0, 4.8],
        width_ratios=[5.2, 1.0],
        hspace=0.10,
        wspace=0.15,
    )
    ax_top = fig.add_subplot(matrix_grid[0, 0])
    cax_a = fig.add_subplot(matrix_grid[0, 1])
    ax_heat = fig.add_subplot(matrix_grid[1, 0])
    ax_right = fig.add_subplot(matrix_grid[1, 1])
    draw_stability_matrix(ax_top, ax_heat, ax_right, cax_a, data)

    mixing_grid = gridspec.GridSpecFromSubplotSpec(
        4,
        1,
        subplot_spec=outer[1],
        height_ratios=[1.0, 1.0, 1.0, 0.13],
        hspace=0.64,
    )
    mixing_axes = [fig.add_subplot(mixing_grid[index]) for index in range(3)]
    cax_b = fig.add_subplot(mixing_grid[3])
    draw_mixing_diagnostics(mixing_axes, cax_b, data["diagnostics"])

    a_position = ax_top.get_position()
    b_position = mixing_axes[0].get_position()
    fig.text(
        a_position.x0 - 0.055,
        a_position.y1 + 0.018,
        "a",
        fontsize=14,
        fontweight="bold",
        va="top",
    )
    fig.text(
        b_position.x0 - 0.055,
        b_position.y1 + 0.060,
        "b",
        fontsize=14,
        fontweight="bold",
        va="top",
    )
    fig.text(
        b_position.x0,
        b_position.y1 + 0.060,
        "Localized cross-slot mixing",
        fontsize=11,
        fontweight="bold",
        va="top",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT / "SUPP_loso_stability.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    csv_path = write_values(data)

    print(f"Saved {pdf_path}")
    print(f"Saved {csv_path}")
    print(
        f"Whole dictionary: mean={data['whole'].mean():.6f}, "
        f"min={data['whole'].min():.6f}"
    )
    f6 = data["per_operator"][:, 5]
    print(f"f6: mean={f6.mean():.6f}, min={f6.min():.6f}")


if __name__ == "__main__":
    main()
