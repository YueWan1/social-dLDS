"""Build the temporary syllable-row artwork consumed by Figure 2b.

The selection/statistics are computed by
``panel_b_operator_syllable_selectivity.py``.  This script only changes the
output layout: each signed operator phase becomes one standalone row containing
only its three selected syllables, and the viridis time scale becomes a separate
compact visual key. Operator labels, phase descriptions, and connecting arrows
are intentionally omitted from the rows.

This is an imported helper. The public runner calls
``panel_b_syllable_components.py``, which keeps these intermediates in a
temporary directory and publishes only the five final vertical PDFs and their
shared colorbar.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import numpy as np


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


HERE = Path(__file__).resolve().parent
# Loaded by path so this helper and the selectivity calculations share one
# implementation without turning figures/ into an installed package.
SOURCE_PATH = HERE / "panel_b_operator_syllable_selectivity.py"
PHASE_SLUGS = ["op14", "op2_positive", "op2_negative", "op6_positive", "op6_negative"]


def load_source_module():
    spec = spec_from_file_location("figure2b_current_source", SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load plotting source: {SOURCE_PATH}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_rows(source):
    data = source.load()
    omega, present = source.syll_omega_and_present(data)
    candidates = [
        syllable
        for syllable in range(source.NSYL)
        if present[syllable] >= source.MIN_SESS and syllable in omega
    ]

    rows = []
    for slot, sign, label, strength in source.PHASES:
        _, phase_omega, _ = source.onion(
            data,
            lambda kp, cs, z, sl=slot, sg=sign: source.active_mask(cs, sl, sg),
        )
        selectivity = source.selectivity(data, slot, sign, candidates)
        ranked = sorted(
            [syllable for syllable in candidates if np.isfinite(selectivity[syllable])],
            key=lambda syllable: -selectivity[syllable],
        )
        top = [
            syllable
            for syllable in ranked
            if selectivity[syllable] >= source.ZTHR
        ][: source.NTOP]

        syllables = []
        for syllable in top:
            trajectory, _, n_onsets = source.onion(
                data,
                lambda kp, cs, z, selected=syllable: z == selected,
            )
            syllables.append(
                {
                    "syllable": syllable,
                    "trajectory": (
                        trajectory if n_onsets >= source.MIN_ONSETS else None
                    ),
                    "z": selectivity[syllable],
                    "omega": omega[syllable],
                    "n_onsets": n_onsets,
                }
            )

        rows.append(
            {
                "slot": slot,
                "sign": sign,
                "label": label,
                "strength": strength,
                "phase_omega": phase_omega,
                "syllables": syllables,
            }
        )
    return rows


def draw_row(source, row):
    # A full manuscript-width source canvas keeps 12--14 pt labels readable
    # after Illustrator placement instead of relying on raster resolution.
    fig = plt.figure(figsize=(7.0, 2.05))
    grid = fig.add_gridspec(
        1,
        3,
        left=0.018,
        right=0.992,
        bottom=0.045,
        top=0.985,
        wspace=0.065,
    )

    for index in range(source.NTOP):
        cell = grid[0, index].subgridspec(
            2, 1, height_ratios=[0.62, 0.38], hspace=0.0
        )
        trajectory_ax = fig.add_subplot(cell[0, 0])
        text_ax = fig.add_subplot(cell[1, 0])
        text_ax.axis("off")

        if index >= len(row["syllables"]):
            trajectory_ax.axis("off")
            continue

        item = row["syllables"][index]
        source.draw(trajectory_ax, item["trajectory"], "", "#1a5e2a")
        for spine in trajectory_ax.spines.values():
            spine.set_visible(False)

        weight = (
            "bold"
            if item["syllable"] in source.FIG1_EXAMPLES
            else "normal"
        )
        text_ax.text(
            0.5,
            0.84,
            f"syll {item['syllable']}",
            ha="center",
            va="center",
            transform=text_ax.transAxes,
            fontsize=18,
            color="#1a5e2a",
            fontweight=weight,
        )
        text_ax.text(
            0.5,
            0.50,
            rf"$z={item['z']:.1f}$",
            ha="center",
            va="center",
            transform=text_ax.transAxes,
            fontsize=16,
            color="#1a5e2a",
            fontweight=weight,
        )
        text_ax.text(
            0.5,
            0.16,
            rf"$\omega={item['omega']:+.0f}^\circ$/s",
            ha="center",
            va="center",
            transform=text_ax.transAxes,
            fontsize=16,
            color="#1a5e2a",
            fontweight=weight,
        )
    return fig


def save_figure(fig, stem, output_dir):
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(
        pdf_path,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)
    return pdf_path


def draw_time_colorbar(source):
    fig = plt.figure(figsize=(5.0, 1.15))
    fig.text(
        0.5,
        0.91,
        "Time from onset",
        ha="center",
        va="top",
        fontsize=20,
        fontweight="bold",
        color="#222222",
    )
    colorbar_ax = fig.add_axes([0.08, 0.34, 0.84, 0.25])
    start_ms = -source.PRE / source.FPS * 1000
    end_ms = source.POST / source.FPS * 1000
    colorbar = ColorbarBase(
        colorbar_ax,
        cmap=plt.colormaps[source.SYL_CMAP_NAME],
        norm=Normalize(vmin=start_ms, vmax=end_ms),
        orientation="horizontal",
        ticks=[start_ms, 0.0, end_ms],
    )
    colorbar.set_ticklabels(
        [f"{start_ms:.0f}", "0", f"{end_ms:.0f}"]
    )
    colorbar.ax.tick_params(axis="x", labelsize=16, length=3, pad=2)
    colorbar.outline.set_linewidth(0.8)
    colorbar.ax.set_xlabel("ms", fontsize=16, labelpad=2)
    return fig


def write_summary(rows, outputs, output_dir):
    manifest_lines = ["phase\tpdf"]
    for row, pdf_path in zip(rows, outputs):
        manifest_lines.append(f"{row['label']}\t{pdf_path.name}")
    manifest_lines.append("time_colorbar\ttime_from_onset_colorbar.pdf")
    (output_dir / "manifest.tsv").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )

    selection_lines = [
        "phase\tstrength\tphase_omega_deg_s\trank\tsyllable\t"
        "selectivity_z\tsyllable_omega_deg_s\tn_onsets"
    ]
    for row in rows:
        for rank, item in enumerate(row["syllables"], start=1):
            selection_lines.append(
                "\t".join(
                    [
                        row["label"],
                        row["strength"],
                        f"{row['phase_omega']:.6f}",
                        str(rank),
                        str(item["syllable"]),
                        f"{item['z']:.6f}",
                        f"{item['omega']:.6f}",
                        str(item["n_onsets"]),
                    ]
                )
            )
    (output_dir / "selection_summary.tsv").write_text(
        "\n".join(selection_lines) + "\n", encoding="utf-8"
    )


def build(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    source = load_source_module()
    rows = compute_rows(source)

    outputs = []
    for slug, row in zip(PHASE_SLUGS, rows):
        outputs.append(save_figure(draw_row(source, row), f"row_{slug}", output_dir))

    save_figure(draw_time_colorbar(source), "time_from_onset_colorbar", output_dir)
    write_summary(rows, outputs, output_dir)
