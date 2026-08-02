# Figure reproduction

Run all published data panels:

```bash
bash reproduce.sh figures
```

Or run one group:

```bash
bash reproduce.sh figures fig03
bash reproduce.sh figures fig06 supp assemble
```

`run_figures.sh` lists the scripts and execution order. Outputs are written
under `out/<figure>/`.

Each asset has one output format:

- PDF for plots and vector panel sources.
- PNG only for raster filmstrips and the Supplementary S9 raster composite.
- NumPy/CSV/TSV for numerical results.

Figure 3 and Supplementary S6 include scripted composition. Other targets emit
the panel files used in the paper layouts. See
[`../docs/FIGURE_INDEX.md`](../docs/FIGURE_INDEX.md).
