# social-dLDS

Code for:

> **Decomposed Linear Dynamical Systems (dLDS) reveal interpretable temporal
> structure in mouse social behavior**
>
> Yue Wan, Mikio Aoi, Adam Charles

This release does three things:

```text
public data -> FEATURE16 / FEATURE27
one feature session -> Julia dLDS -> Fs.npy / cs.npy
published Fs / cs -> statistical analyses -> paper data panels
```

## Setup

Python 3.11 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp paths.example.yml paths.yml
# edit paths.yml
```

## 1. Build features from the public datasets

Single-mouse data come from keypoint-MoSeq open-field 2D
([Zenodo 15171335](https://doi.org/10.5281/zenodo.15171335)); dyadic data come
from CalMS21 Task 1 ([CaltechDATA](https://doi.org/10.22002/D1.1991)).

```bash
# FEATURE16, one file per open-field session plus the paper QC table
bash features/fetch_zenodo_kpms.sh
bash reproduce.sh features-single

# FEATURE27, one file per CalMS21 session
bash features/fetch_calms21.sh
bash reproduce.sh convert-calms21
bash reproduce.sh features-dyadic
```

The exact coordinates and output directories are documented in
[`features/README.md`](features/README.md).

## 2. Run one Julia dLDS fit

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
bash reproduce.sh fit-one \
  --input /path/to/FEATURE27_mouse001.npy \
  --output /path/to/run_seed0 \
  --seed 0
```

This command runs one seed for one session and writes `Fs.npy`, `cs.npy` and
`params.txt`. The workflow was validated with Julia 1.12.1, including a small
fit/inference run. See
[`julia/README.md`](julia/README.md).

## 3. Recompute the analyses and paper figures

Download the deposited paper fits and supporting files, then run:

```bash
python -m pip install -e '.[moseq]'  # required by three MoSeq panels
bash features/fetch_derived_bundle.sh all
bash reproduce.sh analysis
bash reproduce.sh figures
```

Analyses and figures are written under `out/`; tracked values in `derived/`
remain unchanged. See the [`analysis index`](analysis/README.md) and
[`figure index`](docs/FIGURE_INDEX.md).

## Layout

```text
features/       public-data download, conversion and FEATURE16/27
julia/          dLDS implementation and one-session example
analysis/       statistics and null tests reported in the paper
figures/        published main/supplementary data panels
derived/        small canonical values used by published panels
dlds_release/   shared path and plotting helpers
docs/           data availability and figure index
scripts/        environment checks
```

Python code is MIT licensed; `julia/` is GPL-3.0-or-later. See
[`LICENSE`](LICENSE) and [`CITATION.cff`](CITATION.cff). Data sources and
release files are listed in [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md).
