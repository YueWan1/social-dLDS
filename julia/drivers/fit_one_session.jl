"""Fit one dLDS model to one FEATURE16 or FEATURE27 session."""

import Pkg
const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(PROJECT_ROOT; io = devnull)

using Distributed
using DLDS
using NPZ
using Statistics

const WORKERS = 4
const MOTIFS = 10
const L1 = 0.30
const SMOOTH = 0.10
const F_LR = 2.0
const F_LR_DECAY = 0.996
const DECORR = 0.16
const ITERATIONS = 150
const SNIPPETS = 150
const SNIPPET_LENGTH = 200


function usage()
    println("""
Usage:
  julia --project=julia julia/drivers/fit_one_session.jl \\
      --input FEATURE.npy --output RUN_DIR [--seed 0] [--overwrite]
""")
end


function parse_cli(args)
    values = Dict{String, String}("seed" => "0")
    overwrite = false
    index = 1
    while index <= length(args)
        option = args[index]
        if option in ("-h", "--help")
            usage()
            exit(0)
        elseif option == "--overwrite"
            overwrite = true
            index += 1
        elseif option in ("--input", "--output", "--seed")
            index == length(args) && error("Missing value after $option")
            values[option[3:end]] = args[index + 1]
            index += 2
        else
            error("Unknown option: $option")
        end
    end
    haskey(values, "input") || error("--input is required")
    haskey(values, "output") || error("--output is required")
    return (
        input = abspath(values["input"]),
        output = abspath(values["output"]),
        seed = parse(Int, values["seed"]),
        overwrite = overwrite,
    )
end


function load_features(path)
    isfile(path) || error("Feature file not found: $path")
    data = Float64.(npzread(path))
    ndims(data) == 2 || error("Expected a 2-D feature matrix")
    if size(data, 1) ∉ (16, 27) && size(data, 2) in (16, 27)
        data = permutedims(data)
    end
    size(data, 1) in (16, 27) ||
        error("Expected shape (16,T) or (27,T); got $(size(data))")
    all(isfinite, data) || error("Input contains NaN or Inf")

    data ./= max.(std(data, dims = 2), 1e-3)
    data ./= max(quantile(abs.(data[:]), 0.99), 1e-6)
    return data
end


function main()
    cfg = parse_cli(ARGS)
    data = load_features(cfg.input)
    size(data, 2) > SNIPPET_LENGTH ||
        error("Input needs more than $SNIPPET_LENGTH frames")

    outputs = [joinpath(cfg.output, name) for name in ("Fs.npy", "cs.npy", "params.txt")]
    !cfg.overwrite && any(isfile, outputs) &&
        error("Output exists; choose a new directory or pass --overwrite")
    mkpath(cfg.output)

    addprocs(max(0, WORKERS - nworkers()); exeflags = "--project=$(PROJECT_ROOT)")
    Distributed.remotecall_eval(Main, workers(), :(using DLDS))

    operators = fit_no_obs_model(
        [data],
        MOTIFS;
        samples_per_snippet = SNIPPET_LENGTH,
        num_snippets = SNIPPETS,
        random_seed = cfg.seed,
        max_iter = ITERATIONS,
        c_l1_coeff = L1,
        c_smooth_coeff = SMOOTH,
        F_lr_init = F_LR,
        F_lr_decay = F_LR_DECAY,
        F_decorr_coeff = DECORR,
    )
    coefficients = infer_no_obs_state(
        operators,
        data;
        c_l1_coeff = L1,
        c_smooth_coeff = SMOOTH,
        random_seed = cfg.seed,
    )
    all(isfinite, operators) || error("Fit produced non-finite operators")
    all(isfinite, coefficients) || error("Inference produced non-finite coefficients")

    npzwrite(outputs[1], operators)
    npzwrite(outputs[2], coefficients)
    open(outputs[3], "w") do io
        println(io, "input = ", cfg.input)
        println(io, "feature_dim = ", size(data, 1))
        println(io, "frames = ", size(data, 2))
        println(io, "seed = ", cfg.seed)
        println(io, "workers = ", WORKERS)
        println(io, "motifs = ", MOTIFS)
        println(io, "l1 = ", L1)
        println(io, "smooth = ", SMOOTH)
        println(io, "f_lr = ", F_LR)
        println(io, "f_lr_decay = ", F_LR_DECAY)
        println(io, "decorr = ", DECORR)
        println(io, "iterations = ", ITERATIONS)
        println(io, "snippets = ", SNIPPETS)
        println(io, "snippet_length = ", SNIPPET_LENGTH)
    end

    println("Saved ", outputs[1], " ", size(operators))
    println("Saved ", outputs[2], " ", size(coefficients))
end


main()
