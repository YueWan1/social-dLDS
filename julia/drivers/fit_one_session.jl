"""Fit one dLDS model to one FEATURE16 or FEATURE27 session."""

import Pkg
const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(PROJECT_ROOT; io = devnull)

using Distributed
using DLDS
using NPZ
using Statistics

const WORKERS = 4
const PROFILES = Dict(
    "demo" => (
        motifs = 10,
        l1 = 0.30,
        smooth = 0.10,
        F_lr = 2.0,
        F_lr_decay = 0.996,
        decorr = 0.16,
        iterations = 150,
        snippets = 150,
        snippet_length = 200,
    ),
    "single" => (
        motifs = 15,
        l1 = 0.65,
        smooth = 0.30,
        F_lr = 2.0,
        F_lr_decay = 0.996,
        decorr = 0.16,
        iterations = 300,
        snippets = 300,
        snippet_length = 200,
    ),
    "dyadic" => (
        motifs = 15,
        l1 = 0.40,
        smooth = 0.15,
        F_lr = 2.0,
        F_lr_decay = 0.997,
        decorr = 0.10,
        iterations = 300,
        snippets = 350,
        snippet_length = 200,
    ),
)


function usage()
    println("""
Usage:
  julia --project=julia julia/drivers/fit_one_session.jl \\
      --input FEATURE.npy --output RUN_DIR \\
      [--seed 0] [--profile demo|single|dyadic] [--overwrite]
""")
end


function parse_cli(args)
    values = Dict{String, String}("seed" => "0", "profile" => "demo")
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
        elseif option in ("--input", "--output", "--seed", "--profile")
            index == length(args) && error("Missing value after $option")
            values[option[3:end]] = args[index + 1]
            index += 2
        else
            error("Unknown option: $option")
        end
    end
    haskey(values, "input") || error("--input is required")
    haskey(values, "output") || error("--output is required")
    haskey(PROFILES, values["profile"]) ||
        error("--profile must be one of: demo, single, dyadic")
    return (
        input = abspath(values["input"]),
        output = abspath(values["output"]),
        seed = parse(Int, values["seed"]),
        profile = values["profile"],
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
    hp = PROFILES[cfg.profile]
    data = load_features(cfg.input)
    size(data, 2) > hp.snippet_length ||
        error("Input needs more than $(hp.snippet_length) frames")

    outputs = [joinpath(cfg.output, name) for name in ("Fs.npy", "cs.npy", "params.txt")]
    !cfg.overwrite && any(isfile, outputs) &&
        error("Output exists; choose a new directory or pass --overwrite")
    mkpath(cfg.output)

    addprocs(max(0, WORKERS - nworkers()); exeflags = "--project=$(PROJECT_ROOT)")
    Distributed.remotecall_eval(Main, workers(), :(using DLDS))

    operators = fit_no_obs_model(
        [data],
        hp.motifs;
        samples_per_snippet = hp.snippet_length,
        num_snippets = hp.snippets,
        random_seed = cfg.seed,
        max_iter = hp.iterations,
        c_l1_coeff = hp.l1,
        c_smooth_coeff = hp.smooth,
        F_lr_init = hp.F_lr,
        F_lr_decay = hp.F_lr_decay,
        F_decorr_coeff = hp.decorr,
    )
    coefficients = infer_no_obs_state(
        operators,
        data;
        c_l1_coeff = hp.l1,
        c_smooth_coeff = hp.smooth,
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
        println(io, "profile = ", cfg.profile)
        println(io, "workers = ", WORKERS)
        println(io, "motifs = ", hp.motifs)
        println(io, "l1 = ", hp.l1)
        println(io, "smooth = ", hp.smooth)
        println(io, "f_lr = ", hp.F_lr)
        println(io, "f_lr_decay = ", hp.F_lr_decay)
        println(io, "decorr = ", hp.decorr)
        println(io, "iterations = ", hp.iterations)
        println(io, "snippets = ", hp.snippets)
        println(io, "snippet_length = ", hp.snippet_length)
    end

    println("Saved ", outputs[1], " ", size(operators))
    println("Saved ", outputs[2], " ", size(coefficients))
end


main()
