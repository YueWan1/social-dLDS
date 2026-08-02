# Julia dLDS example

This directory contains the dLDS implementation and one public fitting
example. The example fits one FEATURE16 or FEATURE27 session with one random
seed and writes:

```text
Fs.npy       operator dictionary, shape (M, D, D)
cs.npy       per-frame coefficients, shape (M, T - 1)
params.txt   input path, seed and hyperparameters
```

## Set up Julia

The release uses Julia 1.12.1.

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
```

The full Julia workflow was run and validated with Julia 1.12.1, including a
small fit and inference run through the public driver.

## Fit one session

First create either FEATURE16 or FEATURE27 data with the scripts under
`features/`. Then pass one `.npy` file and a new output directory:

```bash
julia --project=julia julia/drivers/fit_one_session.jl \
  --input /path/to/FEATURE27_mouse001.npy \
  --output /path/to/run_seed0 \
  --seed 0
```

The command uses a small demonstration fit: 10 operators, 150 iterations and
150 snippets. These values are declared together at the top of the driver.

Use the deposited paper `Fs` and `cs` files for analysis and figure
reproduction.

## Implementation

The paper uses the no-observation path: the feature vector is the model state,
`fit_no_obs_model` learns the operator dictionary, and
`infer_no_obs_state` estimates the coefficients. Features are normalized per
session by dimension-wise standard deviation and then by the 99th percentile
of the absolute values, matching the paper pipeline.

This directory is licensed under GNU GPL v3 or later. It derives from Sai
Koukuntla's dLDS implementation; provenance and modifications are recorded in
the source headers and `julia/LICENSE`.
