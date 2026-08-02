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

`run_figures.sh` lists the scripts and execution order. Outputs, including the
Figure 3 and Supplementary S6 composites, are written under the configured
`out_root`.

Figure 1a uses a 30-second skeleton-overlay clip generated from the public
single-mouse source data. The batch command creates the MP4 automatically when
it is absent.

Panels that require non-distributed keypoint-MoSeq results or cleaned keypoints
are reported as `SKIP`. The final summary lists every skipped panel and still
returns success when the remaining requested panels complete. Unknown
optional-artifact names are configuration errors and fail the run.

Each asset has one output format:

- PDF for plots and vector panel sources.
- PNG only for raster filmstrips and the Supplementary S9 raster composite.
- NumPy/CSV/TSV for numerical results.

Figure 3 and Supplementary S6 include scripted composition. Other targets emit
the panel files used in the paper layouts. See
[`../docs/FIGURE_INDEX.md`](../docs/FIGURE_INDEX.md).
