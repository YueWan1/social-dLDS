# Feature preprocessing

The public pipeline converts the two source datasets into one NumPy matrix per
session. Both matrices use shape `(features, frames)`.

## FEATURE16: single mouse

Input: the open-field 2D data from
[Zenodo 15171335](https://doi.org/10.5281/zenodo.15171335).

```bash
bash features/fetch_zenodo_kpms.sh
bash reproduce.sh features-single
```

The builder centers eight keypoints on the animal and rotates them into its
heading frame. Flattened x/y coordinates give 16 dimensions:

| dimensions | keypoint |
| --- | --- |
| 0–1 | spine4 |
| 2–3 | spine3 |
| 4–5 | spine2 |
| 6–7 | spine1 |
| 8–9 | head |
| 10–11 | nose |
| 12–13 | right ear |
| 14–15 | left ear |

All ten open-field sessions are exported as separate FEATURE16 files.
`SESSION_QC.csv` marks the five heading-anchor QC passes used for the paper's
dLDS fit. Output goes to `<results_root>/feature_inputs_kpmoseq/`.

## FEATURE27: two mice

Input: CalMS21 Task 1 from
[CaltechDATA](https://doi.org/10.22002/D1.1991).

```bash
bash features/fetch_calms21.sh
bash reproduce.sh convert-calms21
bash reproduce.sh features-dyadic
```

`calms21_to_npy.py` first splits the public JSON into per-session keypoints,
confidence scores and labels. It moves time from the first axis to the last and
checks the public schema, shapes, labels and coordinate orientation.

The feature builder then produces:

| dimensions | block | contents |
| --- | --- | --- |
| 0–13 | SELF | resident pose in its heading frame |
| 14–24 | DIST | centroid, matched-keypoint and head/body/tail distances |
| 25–26 | DIRC | partner centroid in the resident frame |

The published smoothing values are the defaults:
`--kp-sigma 1.0 --pose-sigma 0.5`. Output goes to
`<results_root>/feature_inputs_feature27_kp1p0_pose0p5/`.

`mouse036` has only 96 frames. It is converted to FEATURE27 but was too short
for the paper's 200-frame fitting snippets, so the deposited coefficients cover
69 of the 70 sessions.

## Paths and overwrite policy

Raw downloads and converted CalMS21 arrays use `raw_data_root`; feature outputs
use `results_root`. Configure both in `paths.yml` or with the environment
variables documented in the root README.

The builders do not replace existing feature exports unless `--overwrite` is
passed explicitly.
