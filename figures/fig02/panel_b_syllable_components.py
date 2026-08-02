#!/usr/bin/env python3
"""Reproduce the five compact syllable components used in Figure 2b.

The source trajectory artwork remains vector graphics.  Its original labels are
cropped away and re-typeset at a deliberately large size so that they remain
legible after the narrow vertical panels are placed in the composite figure.

The temporary horizontal rows and selection table are built automatically;
only the five vertical PDFs and their shared colorbar are published.

Needs a TeX installation (pdflatex, with standalone/sansmath/DejaVuSans) and
`pdfinfo` on PATH, which is why it shells out rather than drawing in
matplotlib: the labels are typeset, not plotted.

Run from the repository root:

    python3 figures/fig02/panel_b_step2_vertical_components.py
"""

from __future__ import annotations

import csv
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from dlds_release.paths import out_dir
from figures.fig02 import syllable_rows


COMPONENT_DIR = out_dir("fig02", "figure2b_syllable_components")

PHASE_TO_STEM = {
    "op14": "op14",
    "op2$+$": "op2_positive",
    "op2$-$": "op2_negative",
    "op6$+$": "op6_positive",
    "op6$-$": "op6_negative",
}


def pdf_page_size(pdf_path: Path) -> tuple[float, float]:
    info = subprocess.run(
        ["pdfinfo", str(pdf_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    match = re.search(r"Page size:\s+([0-9.]+) x ([0-9.]+) pts", info)
    if not match:
        raise RuntimeError(f"Could not read page size from {pdf_path}")
    return float(match.group(1)), float(match.group(2))


def load_rows(directory: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    with (directory / "selection_summary.tsv").open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            grouped.setdefault(row["phase"], []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["rank"]))
    return grouped


def latex_panel(
    source_pdf: Path,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    row: dict[str, str],
) -> str:
    syllable = int(row["syllable"])
    z_score = float(row["selectivity_z"])
    omega = round(float(row["syllable_omega_deg_s"]))
    if omega < 0:
        omega_tex = f"-{abs(omega)}"
    elif omega > 0:
        omega_tex = f"+{omega}"
    else:
        omega_tex = "0"
    source = str(source_pdf).replace("\\", "/")
    return rf"""
\begin{{minipage}}{{50mm}}
  \centering
  \includegraphics[
    page=1,
    viewport={x0:.3f} {y0:.3f} {x1:.3f} {y1:.3f},
    clip,
    width=45mm
  ]{{\detokenize{{{source}}}}}\par
  \vspace{{-1.2mm}}
  {{\labelstyle\fontsize{{20}}{{20.5}}\selectfont\bfseries syll~{syllable}\par}}
  \vspace{{-0.4mm}}
  {{\labelstyle\fontsize{{17}}{{17.5}}\selectfont\sansmath $z = {z_score:.1f}$\par}}
  \vspace{{-0.6mm}}
  {{\labelstyle\fontsize{{20}}{{20.5}}\selectfont\bfseries\boldmath
    $\omega = {omega_tex}^\circ\!/\mathrm{{s}}$\par}}
\end{{minipage}}"""


def make_tex(source_pdf: Path, rows: list[dict[str, str]]) -> str:
    width, height = pdf_page_size(source_pdf)
    panel_width = width / 3
    # Preserve the trajectory while excluding labels from the source panels.
    trajectory_bottom = height * 0.43
    panels = []
    for index, row in enumerate(rows):
        panels.append(
            latex_panel(
                source_pdf,
                index * panel_width,
                trajectory_bottom,
                (index + 1) * panel_width,
                height,
                row,
            )
        )
    separator = "\n\\par\\vspace{-1.0mm}\n"
    body = separator.join(panels)
    return rf"""\documentclass[border=0pt]{{standalone}}
\usepackage{{graphicx}}
\usepackage{{xcolor}}
\usepackage{{amsmath}}
\usepackage{{sansmath}}
\usepackage{{DejaVuSans}}
\definecolor{{labelgreen}}{{HTML}}{{1A5E2A}}
\newcommand{{\labelstyle}}{{\color{{labelgreen}}\sffamily}}
\setlength{{\parindent}}{{0pt}}
\begin{{document}}
\begin{{minipage}}{{50mm}}
\centering
{body}
\end{{minipage}}
\end{{document}}
"""


def main() -> None:
    COMPONENT_DIR.mkdir(parents=True, exist_ok=True)
    made = []
    with tempfile.TemporaryDirectory(prefix="social-dlds-fig02-") as tmp:
        build_dir = Path(tmp)
        row_dir = build_dir / "rows"
        syllable_rows.build(row_dir)
        grouped = load_rows(row_dir)
        for phase, stem in PHASE_TO_STEM.items():
            source_pdf = row_dir / f"row_{stem}.pdf"
            tex_path = build_dir / f"vertical_{stem}.tex"
            tex_path.write_text(make_tex(source_pdf, grouped[phase]))
            subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    f"-output-directory={build_dir}",
                    str(tex_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            built_pdf = build_dir / f"vertical_{stem}.pdf"
            output_pdf = COMPONENT_DIR / built_pdf.name
            shutil.copy2(built_pdf, output_pdf)
            made.append(output_pdf)
        colorbar = COMPONENT_DIR / "time_from_onset_colorbar.pdf"
        shutil.copy2(row_dir / colorbar.name, colorbar)
        made.append(colorbar)
    print("\n".join(str(path) for path in made))


if __name__ == "__main__":
    main()
