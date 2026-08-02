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
  --seed 0 \
  --profile demo
```

The default `demo` profile uses 10 operators, 150 iterations and 150 snippets.
Two additional profiles apply the paper hyperparameters to the same
one-session driver:

| Profile | M | L1 | Smooth | Iterations | Snippets |
| --- | ---: | ---: | ---: | ---: | ---: |
| `demo` | 10 | 0.30 | 0.10 | 150 | 150 |
| `single` | 15 | 0.65 | 0.30 | 300 | 300 |
| `dyadic` | 15 | 0.40 | 0.15 | 300 | 350 |

The profiles select hyperparameters, not the production orchestration. The
published single-mouse dictionary was fitted jointly across five sessions; the
published dyadic dictionary was formed from 30 completed LOSO fits. Their exact
fit records are
[`../derived/single_mouse/params.txt`](../derived/single_mouse/params.txt) and
[`../derived/dyadic/F_universal_params.txt`](../derived/dyadic/F_universal_params.txt).
Use the companion paper `Fs` and `cs` files for analysis and figure
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
