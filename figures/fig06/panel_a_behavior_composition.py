"""Figure 6a: absolute operator composition and social remainder.

Only the two panels placed in the current paper are emitted, both as PDF.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dlds_release.paths import dyadic_cs_dir, feature27_dir, out_dir


RUN = dyadic_cs_dir()
FEATURES = feature27_dir()
OUT = out_dir("fig06")

BEHAVIORS = ("attack", "invest", "mount", "other")
BEHAVIOR_COLORS = ("#B22222", "#D98A00", "#2E7D32", "#777777")
DISPLAY_ORDER = (3, 0, 1, 2)

# (zero-based slot, display label, color), ordered by stack position.
OPERATORS = (
    (2, r"$f_3$", "#8e44ad"),
    (1, r"$f_2$", "#ff7f0e"),
    (5, r"$f_6$", "#1f77b4"),
    (3, r"$f_4$", "#bcbd22"),
    (10, r"$f_{11}$", "#e377c2"),
    (8, r"$f_9$", "#17becf"),
    (14, r"$f_{15}$", "#8c564b"),
)

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.linewidth": 0.9,
    }
)


def behavior_means() -> tuple[np.ndarray, np.ndarray]:
    """Return mean |c| by operator/behavior and behavior frame counts."""
    slots = [slot for slot, _, _ in OPERATORS]
    totals = np.zeros((len(slots), 4))
    counts = np.zeros(4)

    for session in range(1, 71):
        coefficients_path = RUN / f"cs_mouse{session:03d}.npy"
        labels_path = FEATURES / f"cleaned_label_mouse{session:03d}.npy"
        if not (coefficients_path.exists() and labels_path.exists()):
            continue
        coefficients = np.load(coefficients_path)
        labels = np.load(labels_path)
        length = min(coefficients.shape[1], labels.size)
        selected = np.abs(coefficients[slots, :length])
        labels = labels[:length]
        for behavior in range(4):
            mask = labels == behavior
            if mask.any():
                totals[:, behavior] += selected[:, mask].sum(axis=1)
                counts[behavior] += mask.sum()

    if not counts.all():
        raise RuntimeError(f"Missing one or more behavior classes; counts={counts}")
    return totals / counts, counts


def style_x_axis(axis: plt.Axes, counts: np.ndarray) -> None:
    rates = counts / counts.sum()
    names = [BEHAVIORS[index] for index in DISPLAY_ORDER]
    colors = [BEHAVIOR_COLORS[index] for index in DISPLAY_ORDER]
    axis.set_xticks(
        np.arange(4),
        [f"{name}\n{100 * rates[index]:.1f}%" for index, name in enumerate(names)],
        fontsize=12,
    )
    for tick, color in zip(axis.get_xticklabels(), colors):
        tick.set_color(color)
        tick.set_fontweight("bold")
    axis.tick_params(axis="y", labelsize=11)
    axis.spines[["top", "right"]].set_visible(False)


def absolute_composition(means: np.ndarray, counts: np.ndarray) -> None:
    means = means[:, DISPLAY_ORDER]
    counts = counts[list(DISPLAY_ORDER)]
    figure, axis = plt.subplots(figsize=(4.6, 4.0))
    x = np.arange(4)
    bottom = np.zeros(4)

    for row, (_, label, color) in zip(means, OPERATORS):
        axis.bar(
            x,
            row,
            bottom=bottom,
            width=0.72,
            color=color,
            edgecolor="white",
            linewidth=0.7,
        )
        for behavior, height in enumerate(row):
            if height >= 0.026:
                axis.text(
                    behavior,
                    bottom[behavior] + height / 2,
                    label,
                    ha="center",
                    va="center",
                    fontsize=11.5,
                    color="white" if color in ("#8e44ad", "#8c564b") else "black",
                    fontweight="bold",
                )
        bottom += row

    for behavior, total in enumerate(bottom):
        axis.text(
            behavior,
            total + 0.012,
            rf"$\Sigma$={total:.2f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
        )

    style_x_axis(axis, counts)
    axis.set_ylim(0, 1.22)
    axis.set_ylabel(r"mean $|c|$ (absolute)", fontsize=12.5)
    axis.set_title("Absolute operator composition", fontsize=13.5, loc="left", pad=7)
    figure.subplots_adjust(left=0.18, right=0.98, top=0.89, bottom=0.18)
    output = OUT / "fig_main6a_composition_absolute.pdf"
    figure.savefig(output, facecolor="white")
    plt.close(figure)
    print("wrote", output)


def social_remainder(means: np.ndarray, counts: np.ndarray) -> None:
    # Remove the shared substrate f2/f3/f6 and the distance operator f4.
    social_operators = OPERATORS[4:]
    social = means[4:, DISPLAY_ORDER]
    shares = social / social.sum(axis=0, keepdims=True)
    counts = counts[list(DISPLAY_ORDER)]

    figure, axis = plt.subplots(figsize=(4.6, 4.0))
    x = np.arange(4)
    bottom = np.zeros(4)
    for row, (_, label, color) in zip(shares, social_operators):
        axis.bar(
            x,
            row,
            bottom=bottom,
            width=0.72,
            color=color,
            edgecolor="white",
            linewidth=0.7,
        )
        for behavior, height in enumerate(row):
            if height >= 0.055:
                axis.text(
                    behavior,
                    bottom[behavior] + height / 2,
                    f"{label} {100 * height:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=11.5 if height >= 0.12 else 9.5,
                    color="white" if color == "#8c564b" else "black",
                    fontweight="bold",
                )
            bottom[behavior] += height

    style_x_axis(axis, counts)
    axis.set_ylim(0, 1)
    axis.set_ylabel("share of remaining operators", fontsize=12.5)
    axis.set_title("Social remainder composition", fontsize=13.5, loc="left", pad=7)
    figure.subplots_adjust(left=0.20, right=0.98, top=0.89, bottom=0.18)
    output = OUT / "fig_main6a_composition_ratio.pdf"
    figure.savefig(output, facecolor="white")
    plt.close(figure)
    print("wrote", output)


def main() -> None:
    means, counts = behavior_means()
    absolute_composition(means, counts)
    social_remainder(means, counts)


if __name__ == "__main__":
    main()
