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

## Companion paper-fit archive

The initial-submission companion archive is:

| File | Contents |
| --- | --- |
| `dlds_derived_models.tar.zst` | Published dLDS dictionaries, 69 dyadic and 5 single-mouse coefficient arrays, fit records, and the 30 LOSO fold dictionaries |

Download:
[Google Drive](https://drive.google.com/file/d/1wBcfj0d4gs-eSbJVkumU--NDH6VPbbH8/view?usp=drive_link)

SHA-256:

```text
bd38eac427d81e071b556a5680d80ad2a1668699751d4c74dcb459ef439bea1d
```

Extract the archive into the configured `data_root`:

```bash
tar --zstd -xf dlds_derived_models.tar.zst \
  -C /path/to/social-dlds-data
```

The compact keypoint-MoSeq AR dictionary used by Figure 1c ships in
`derived/single_mouse/kpmoseq_single_Ab.npy`; the 1.86 GB training checkpoint is
not required.

The archive contains author-generated model outputs, not copies of the public
source datasets. FEATURE16 and FEATURE27 can be rebuilt with the commands above.
The permanent release will use a versioned Zenodo record; the corresponding
download script accepts its record ID through `SOCIAL_DLDS_ZENODO_RECORD`.

The source records do not currently state a redistribution licence for their
coordinate data. Their raw data, repackaged feature matrices and cleaned
keypoints are therefore not included in this public archive.

## Local paths

Copy `paths.example.yml` to `paths.yml` and set:

```text
raw_data_root
data_root
results_root
out_root
```

`paths.yml`, downloaded data and `out/` are excluded from Git.
