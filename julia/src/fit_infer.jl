# This file is part of DLDS, Copyright (C) Sai Koukuntla.
#
# DLDS is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. See ../LICENSE for the full text.
#
# MODIFIED for the accompanying paper (Wan, Aoi and Charles), 2025-2026.
# Upstream: https://github.com/saikoukunt/dLDS
# Change: adds an early-stop rule to `fit_no_obs_model` (abort when dF stays
# above `early_stop_dF_thresh` for `early_stop_patience` iterations while the
# reconstruction error is at least `early_stop_recon_thresh`), and changes the
# `F_lr_decay` default from 0.99995 to 0.996. The release also removes a
# duplicate `c_l1_coeff_decay` update; the paper uses a decay of 1, so this
# correction does not alter any reported fit.

function initialize_separated_F!(
	F::AbstractArray{T, 3},
	max_corr::T;
	max_attempts::Int = 10_000,
) where {T <: AbstractFloat}
	max_corr == zero(T) && return F
	zero(T) < max_corr <= one(T) ||
		throw(ArgumentError("F_init_max_corr must be in [0, 1]"))
	max_corr == one(T) && return F

	num_motifs = size(F, 1)
	num_entries = size(F, 2) * size(F, 3)
	num_entries > 1 ||
		throw(ArgumentError("correlation requires at least two entries per motif"))

	means = Vector{T}(undef, num_motifs)
	centered_norms_sq = Vector{T}(undef, num_motifs)
	max_corr_sq = max_corr * max_corr

	for motif in axes(F, 1)
		attempt = 0
		while true
			candidate = @view F[motif, :, :]
			candidate_mean = sum(candidate) / num_entries
			candidate_norm_sq = zero(T)
			@inbounds for col in axes(F, 3)
				@simd for row in axes(F, 2)
					centered = F[motif, row, col] - candidate_mean
					candidate_norm_sq += centered * centered
				end
			end

			separated = candidate_norm_sq > zero(T)
			if separated
				@inbounds for previous in 1:(motif-1)
					centered_dot = zero(T)
					for col in axes(F, 3)
						@simd for row in axes(F, 2)
							centered_dot +=
								(F[motif, row, col] - candidate_mean) *
								(F[previous, row, col] - means[previous])
						end
					end
					if centered_dot * centered_dot >
					   max_corr_sq * candidate_norm_sq * centered_norms_sq[previous]
						separated = false
						break
					end
				end
			end

			if separated
				means[motif] = candidate_mean
				centered_norms_sq[motif] = candidate_norm_sq
				break
			end

			attempt += 1
			attempt <= max_attempts || error(
				"could not initialize motif $motif below correlation $max_corr " *
				"after $max_attempts attempts",
			)
			randn!(candidate)
		end
	end

	return F
end

function perturb_stalled_F!(
	F::AbstractArray{T, 3},
	F_old::AbstractArray{T, 3},
	stalled_iterations::Int,
	perturb_threshold::T,
	noise_sigma::T,
	normalize_F::Bool,
) where {T <: AbstractFloat}
	delta_F = calculate_delta_F(F, F_old)
	stalled_iterations =
		delta_F <= perturb_threshold ? stalled_iterations + 1 : 0

	if stalled_iterations >= 5
		randn!(F_old)
		@. F += noise_sigma * F_old
		if normalize_F
			for motif in axes(F, 1)
				normalize_matrix!(@view(F[motif, :, :]))
			end
		end
		return 0, delta_F, true
	end

	return stalled_iterations, delta_F, false
end

function fit_full_model(
	Y::AbstractMatrix{T},
	num_latents::Int,
	num_motifs::Int;
	random_seed::Int = 0,
	max_iter::Int = 3000,
	recon_threshold::T = T(1e-3),
	x_l1_coeff::T = zero(T),
	c_l1_coeff::T = zero(T),
	c_l1_coeff_decay::T = T(1),
	c_smooth_coeff::T = zero(T),
	c_fista_tol::T = T(1e-8),
	c_fista_max_iter::Int = 3000,
	D_lr::T = T(30),
	D_sign_coeff::T = zero(T),
	D_frobenius_coeff::T = zero(T),
	F_lr_init::T = T(30),
	F_normalize_matrix::Bool = true,
	F_normalize_gradient::Bool = false,
	F_perturb_threshold::T = T(1e-5),
	F_noise_sigma::T = T(0.1),
	F_init_max_corr::T = zero(T),
	F_lr_decay::T = T(0.996),
	verbose::Bool = true,
) where {T <: AbstractFloat}
	"""
	train_dLDS()

	"""
	num_observations::Int = size(Y, 1)
	num_timepoints::Int = size(Y, 2)

	# Initialize/pre-allocated model parameters and state
	D::tMatrix{T} = init_matrix(
		InitDistribution.Sparse(),
		(num_observations, num_latents),
		random_seed;
		k = 4,
	)
	F::Array{T, 3} = init_matrix(
		InitDistribution.Normal(),
		(num_motifs, num_latents, num_latents),
		random_seed,
	)
	initialize_separated_F!(F, F_init_max_corr)
	c::Matrix{T} = init_matrix(
		InitDistribution.Normal(),
		(num_motifs, num_timepoints - 1),
		random_seed,
	)
	X::Matrix{T} = Matrix{T}(undef, num_latents, num_timepoints)

	F_lr::T = F_lr_init
	i::Int = 1
	latent_recon_err::Vector{T} = zeros(T, num_timepoints)
	data_recon_err = Inf

	# Pre-allocate for intermediate results
	data_prediction::Matrix{T} = similar(Y)
	FX_prod::Matrix{T} = Matrix{T}(undef, num_latents, num_motifs) # for update_c!
	FX_prod_gram::Matrix{T} = Matrix{T}(undef, num_motifs, num_motifs)
	gradient_sum::Array{T, 3} = similar(F)                          # for update f! 
	temp_gradient::Matrix{T} = Matrix{T}(undef, num_latents, num_latents)
	x_hat_next::Vector{T} = Vector{T}(undef, num_latents)
	update_F_residuals::Vector{T} = Vector{T}(undef, num_latents)
	F_old::Array{T, 3} = similar(F)
	F_stalled_iterations::Int = 0

	while (data_recon_err > recon_threshold) && (i <= max_iter)
		copyto!(F_old, F)
		update_X!(X, D, Y, lambda_l1 = x_l1_coeff)

		c_l1_coeff *= c_l1_coeff_decay
		if i > 1
			update_c!(
				c,
				FX_prod,
				FX_prod_gram,
				X,
				F,
				smooth_coeff = c_smooth_coeff,
				l1_coeff = c_l1_coeff;
				tol = c_fista_tol,
				max_iter = c_fista_max_iter,
			)
		end

		update_D!(
			D,
			D_lr,
			X,
			Y;
			sign_coeff = D_sign_coeff,
			frobenius_coeff = D_frobenius_coeff,
		)

		update_F!(
			F,
			gradient_sum,
			temp_gradient,
			x_hat_next,
				update_F_residuals,
				X,
				c,
				F_lr;
				normalize_F = F_normalize_matrix,
			)
		F_lr *= F_lr_decay
		F_stalled_iterations, _, F_perturbed = perturb_stalled_F!(
			F,
			F_old,
			F_stalled_iterations,
			F_perturb_threshold,
			F_noise_sigma,
			F_normalize_matrix,
		)

		data_recon_err = calculate_data_recon_error!(data_prediction, Y, D, X)
		latent_recon_err[i] = calculate_latent_recon_error!(x_hat_next, F, X, c)

		if verbose
			F_perturbed && println("Perturbed stalled dynamics motifs")
			println(
				"Iter $(i): Data Rec. Error: $(data_recon_err), Latent Rec. Error: $(latent_recon_err[i]) ",
			)
		end
		i += 1
	end

	return D, F, X, c, latent_recon_err
end

function fit_no_obs_model(
	X::Vector{<:AbstractMatrix{T}},
	num_motifs::Int;
	samples_per_snippet::Int = 200,
	num_snippets::Int = 50,
	random_seed::Int = 0,
	max_iter::Int = 5000,
	recon_threshold::T = T(1e-5),
	c_l1_coeff::T = T(0.2),
	c_l1_coeff_decay::T = T(1),
	c_smooth_coeff::T = T(0.4),
	c_fista_tol::T = T(1e-8),
	c_fista_max_iter::Int = 1000,
	F_lr_init::T = T(1),
	F_normalize_matrix::Bool = true,
	F_decorr_coeff::T = T(0.0),
	F_perturb_threshold::T = T(1e-5),
	F_noise_sigma::T = T(0.1),
	F_init_max_corr::T = zero(T),
	F_lr_decay::T = T(0.996),
	verbose::Bool = true,
	early_stop_dF_thresh::T = T(3.5),
	early_stop_recon_thresh::T = T(1.0),
	early_stop_patience::Int = 10,
) where {T <: AbstractFloat}
	num_latents::Int = size(X[1], 1)
	num_timepoints::Int = size(X[1], 2)

	# Initialize model parameters and state
	F::Array{T, 3} = init_matrix(
		InitDistribution.Normal(),
		(num_motifs, num_latents, num_latents),
		random_seed,
	)
	initialize_separated_F!(F, F_init_max_corr)

	c = Vector{Matrix{T}}(undef, num_snippets)
	for i in 1:num_snippets
		c[i] = zeros(num_motifs, samples_per_snippet - 1)
	end

	F_lr::T = F_lr_init
	i::Int = 1
	latent_recon_err = Inf
	dF_history = T[]

	gradient_sum::Array{T, 3} = similar(F)                          # for update f! 
	temp_gradient::Matrix{T} = Matrix{T}(undef, num_latents, num_latents)
	x_hat_next::Vector{T} = Vector{T}(undef, num_latents)
	update_F_residuals::Vector{T} = Vector{T}(undef, num_latents)
	F_old::Array{T, 3} = similar(F)
	F_stalled_iterations::Int = 0

	while (i <= max_iter)
		copyto!(F_old, F)
		X_snippets, _ = sample_snippets(X, num_snippets, samples_per_snippet)

		update_c_parallel!(
			c,
			X_snippets,
			F;
			smooth_coeff = c_smooth_coeff,
			l1_coeff = c_l1_coeff,
			max_iter = c_fista_max_iter,
			tol = c_fista_tol,
		)

		update_F!(
			F,
			gradient_sum,
			temp_gradient,
			x_hat_next,
				update_F_residuals,
				X_snippets,
				c,
				F_lr;
				normalize_F = F_normalize_matrix,
				decorr_coeff = F_decorr_coeff,
		)
		c_l1_coeff *= c_l1_coeff_decay
		F_lr *= F_lr_decay
		F_stalled_iterations, dF, F_perturbed = perturb_stalled_F!(
			F,
			F_old,
			F_stalled_iterations,
			F_perturb_threshold,
			F_noise_sigma,
			F_normalize_matrix,
		)

		if verbose
			latent_recon_err =
				calculate_latent_recon_error!(x_hat_next, F, X_snippets, c)
			F_perturbed && println("Perturbed stalled dynamics motifs")
			println("Iter $(i): Rec. Error: $(latent_recon_err),  dF: $(dF) ")

			# Early stopping: if dF is stuck near a high plateau AND recon error
			# is large, the run has likely diverged — abort without saving.
			push!(dF_history, T(dF))
			if length(dF_history) >= early_stop_patience
				recent = dF_history[end-early_stop_patience+1:end]
				if all(v -> v >= early_stop_dF_thresh, recent) &&
				   latent_recon_err >= early_stop_recon_thresh
					println("Early stop: dF stuck ≥ $(early_stop_dF_thresh) for ",
					        "$(early_stop_patience) iters and recon error ",
					        "$(latent_recon_err) ≥ $(early_stop_recon_thresh). ",
					        "Aborting run.")
					error("EARLY_STOP: diverged run — caller should discard output")
				end
			end
		end
		i += 1
	end

	return F
end

function infer_no_obs_state(
	F::AbstractArray{T, 3},
	X::AbstractMatrix{T};
	c_l1_coeff::T = zero(T),
	c_smooth_coeff::T = zero(T),
	c_fista_tol::T = T(1e-8),
	c_fista_max_iter::Int = 1000,
	random_seed::Int = 0,
) where {T <: AbstractFloat}
	num_motifs = size(F, 1)
	num_timepoints = size(X, 2)
	num_latents = size(X, 1)

	c::Matrix{T} = init_matrix(
		InitDistribution.Normal(),
		(num_motifs, num_timepoints - 1),
		random_seed,
	)

	FX_prod::Matrix{T} = Matrix{T}(undef, num_latents, num_motifs)
	FX_prod_gram::Matrix{T} = Matrix{T}(undef, num_motifs, num_motifs)
	update_c!(
		c,
		FX_prod,
		FX_prod_gram,
		X,
		F,
		smooth_coeff = c_smooth_coeff,
		l1_coeff = c_l1_coeff;
		tol = c_fista_tol,
		max_iter = c_fista_max_iter,
	)

	return c
end

function infer_full_state(
	D::AbstractMatrix{T},
	F::AbstractArray{T, 3},
	Y::AbstractMatrix{T};
	x_l1_coeff::T = zero(T),
	c_l1_coeff::T = zero(T),
	c_smooth_coeff::T = zero(T),
	c_fista_tol::T = T(1e-8),
	c_fista_max_iter::Int = 1000,
) where {T <: AbstractFloat}
	X::Matrix{T} = Matrix{T}(undef, num_latents, num_timepoints)
	update_X!(X, D, Y, lambda_l1 = x_l1_coeff)

	c = infer_no_obs_state(
		F,
		X;
		c_l1_coeff = c_l1_coeff,
		c_smooth_coeff = c_smooth_coeff,
		c_fista_tol = c_fista_tol,
		c_fista_max_iter = c_fista_max_iter,
	)

	return X, c
end

function calculate_data_recon_error!(
	prediction::AbstractMatrix{T},
	Y::AbstractMatrix{T},
	D::AbstractMatrix{T},
	X::AbstractMatrix{T},
) where {T <: AbstractFloat}
	mul!(prediction, D, X)
	@. prediction .= Y - prediction     # residual, reusing array to avoid extra allocation

	return dot(prediction, prediction) / length(Y)
end

function calculate_latent_recon_error!(
	x_hat_next::AbstractVector{T},
	F::AbstractArray{T, 3},
	X::Vector{<:AbstractMatrix{T}},
	c::Vector{<:AbstractMatrix{T}},
) where {T <: AbstractFloat}
	total_error::T = zero(T)

	for trial in axes(c, 1)
		for t in 1:(size(X[trial], 2)-1)
			step_dynamics!(x_hat_next, @view(X[trial][:, t]), @view(c[trial][:, t]), F)
			x_hat_next .= @view(X[trial][:, t+1]) .- x_hat_next    # residual, reusing array to avoid extra allocation

			total_error += dot(x_hat_next, x_hat_next)
		end
	end

	return total_error / sum([sum(X[trial] .^ 2) for trial in axes(c, 1)])
end

function calculate_delta_F(
	F_new::AbstractArray{T, 3},
	F_old::AbstractArray{T, 3},
) where {T <: AbstractFloat}
	delta = zero(T)
	num_motifs = size(F_old, 1)

	@inbounds for motif in axes(F_old, 1)
		change_norm_sq = zero(T)
		new_norm_sq = zero(T)
		for col in axes(F_old, 3)
			@simd for row in axes(F_old, 2)
				new_value = F_new[motif, row, col]
				change = F_old[motif, row, col] - new_value
				change_norm_sq += change * change
				new_norm_sq += new_value * new_value
			end
		end
		delta += change_norm_sq / new_norm_sq
	end

	return delta / num_motifs
end
