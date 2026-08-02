# Figure index

`run_figures.sh` lists every generated panel. Outputs are written under
`out/`.

```bash
bash reproduce.sh analysis
bash reproduce.sh figures
```

| Target | Source |
| --- | --- |
| Figure 1 | `figures/fig01/` |
| Figure 2 | `figures/fig02/` |
| Figure 3 | `figures/fig03/` |
| Figure 4 | `figures/fig04/` |
| Figure 5 | `figures/fig05/` |
| Figure 6 | `figures/fig06/` |
| Supplementary figures | `figures/supp/` |

Use a target name to regenerate one group, for example:

```bash
bash reproduce.sh figures fig05
bash reproduce.sh figures supp assemble
```

Figure 3 and Supplementary Figure S6 include scripted composition. Other
targets emit the panel files used in the paper layouts. Numerical analyses are
listed in [`../analysis/README.md`](../analysis/README.md).
