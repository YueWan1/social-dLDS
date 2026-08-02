# This file is part of DLDS, Copyright (C) Sai Koukuntla.
#
# DLDS is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. See ../LICENSE for the full text.
#
# MODIFIED for the accompanying paper (Wan, Aoi and Charles), 2025-2026.
# Upstream: https://github.com/saikoukunt/dLDS
# Change: whitespace only, and removal of an empty `sample_trials_weighted`
# stub. No behavioural difference.

function sample_snippets(
    data::AbstractVector{<:AbstractMatrix{T}},
    num_snippets::Int,
    samples_per_snippet::Int,
) where {T<:AbstractFloat}
    snippets = Vector{Matrix{Float64}}(undef, num_snippets)
    indices = Vector{Vector{Int}}(undef, num_snippets)

    trial_inds =
        num_snippets == size(data, 1) ? (1:num_snippets) :
        rand(1:size(data, 1), num_snippets)

    for (i, trial_id) in enumerate(trial_inds)
        trial_length = size(data[trial_id], 2)
        if trial_length <= samples_per_snippet
            snippets[i] = data[trial_id]
        else
            t_start = rand(1:(trial_length - samples_per_snippet))
            t_end = t_start + samples_per_snippet - 1
            snippets[i] = @view(data[trial_id][:, t_start:t_end])
            indices[i] = [trial_id, t_start, t_end]
        end
    end

    return snippets, indices
end
