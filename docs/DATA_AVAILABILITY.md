# Data availability

## Public source datasets

The release does not redistribute third-party raw data.

- Single mouse: keypoint-MoSeq open-field 2D, Zenodo record 15171335,
  DOI `10.5281/zenodo.15171335`.
- Dyadic behavior: CalMS21 Task 1, CaltechDATA,
  DOI `10.22002/D1.1991`.

The download and conversion commands are:

```bash
bash features/fetch_zenodo_kpms.sh
bash features/fetch_calms21.sh
bash reproduce.sh convert-calms21
bash reproduce.sh features-single
bash reproduce.sh features-dyadic
```

## Companion data release

The companion Zenodo record contains:

| Bundle | Contents |
| --- | --- |
| `derived` | Published dLDS dictionaries, per-session coefficients and the 30 small LOSO fold dictionaries |
| `features` | Published FEATURE16 and FEATURE27 matrices |
| `moseq` | Published keypoint-MoSeq `results.h5` files |
| `keypoints` | Cleaned CalMS21 keypoints used by plotted trajectories |

```bash
bash features/fetch_derived_bundle.sh all
```

The compact keypoint-MoSeq AR dictionary used by Figure 1c ships in
`derived/single_mouse/kpmoseq_single_Ab.npy`; the 1.86 GB training checkpoint is
not required.

Before publication, set the Zenodo record ID in
`features/fetch_derived_bundle.sh` and add its DOI to the manuscript Data
Availability statement.

## Local paths

Copy `paths.example.yml` to `paths.yml` and set:

```text
raw_data_root
data_root
results_root
out_root
```

`paths.yml`, downloaded data and `out/` are excluded from Git.
