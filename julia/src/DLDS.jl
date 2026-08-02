# This file is part of DLDS, Copyright (C) Sai Koukuntla.
# Licensed under the GNU General Public License v3 or later; see ../LICENSE.
# Upstream: https://github.com/saikoukunt/dLDS
# Unmodified from upstream.

module DLDS

using LinearAlgebra
using Distributions
using Random
using Statistics
using ProximalOperators
using ProximalAlgorithms
using GLMNet
using Base.Threads
using Distributed

export fit_full_model, fit_no_obs_model, infer_full_state, infer_no_obs_state
export update_c!, update_D!, update_F!, update_X!
export InitDistribution, init_matrix

export calculate_latent_recon_error!, step_dynamics!, calculate_delta_F
export sample_snippets, worker_update_c, update_c_parallel!

include("./matrix_utils.jl")
include("./model.jl")
include("./sample_trials.jl")
include("./fit_infer.jl")

end # module DLDS
