# Statistical analyses

These scripts recompute the numerical results reported in the Methods, Results,
Table S2, and Supplementary Figure S5:

| Script | Result |
| --- | --- |
| `operator_screening.py` | DIST and DIRC operator screening |
| `f4_distance_deciles.py` | Graded distance response of `f_4` |
| `f9_foreshortening.py` | Projected body lengths across `f_9` phases |
| `signed_selectivity.py` | Sign-resolved operator selectivity |
| `syllable_selectivity.py` | Syllable by behavior selectivity |
| `geometry_gating.py` | Session-stratified geometry-gate odds ratios |
| `loso_stability.py` | LOSO dictionary stability |

Run all analyses from the repository root:

```bash
bash reproduce.sh analysis
```

Regenerated files are written to `out/analysis/`. The tracked copies under
`derived/` contain the values used for the paper figures.

## Conventions

The dyadic dictionary uses zero-based array slots, while the manuscript labels
operators from one:

| Manuscript | Slot |
| --- | --- |
| `f_2` | 1 |
| `f_3` | 2 |
| `f_4` | 3 |
| `f_6` | 5 |
| `f_9` | 8 |
| `f_11` | 10 |
| `f_15` | 14 |

Single-mouse names such as `op14`, `op2`, and `op6` are already zero-based and
refer to a different dictionary.

CalMS21 supplies 70 feature and label files, but coefficient analyses use 69
sessions. Session 36 is shorter than the fitting snippet length and has no
`cs_mouse036.npy`.

Behavior labels are `0/1/2/3 = attack/investigation/mount/other`. FEATURE27
uses the blocks `SELF = 0:14`, `DIST = 14:25`, and `DIRC = 25:27`.

Unless a script states otherwise, an operator is active when `|c| > 0.05`.
The geometry-gating analysis uses the stricter condition `c_4 = 0`.
