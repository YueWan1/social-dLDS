#!/usr/bin/env bash
#
# Public entry point for the release workflows: build features, run one dLDS
# fit, recompute the downstream analyses, and redraw the published data panels.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

PYTHON="${PYTHON:-python3}"
JULIA="${JULIA:-julia}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

resolve_path() {
    "$PYTHON" -c "from dlds_release.paths import $1; print($1)"
}

export SOCIAL_DLDS_ROOT="${SOCIAL_DLDS_ROOT:-$(resolve_path ROOT)}"
export SOCIAL_DLDS_RAW_ROOT="${SOCIAL_DLDS_RAW_ROOT:-$(resolve_path RAW_ROOT)}"
export SOCIAL_DLDS_RESULTS_ROOT="${SOCIAL_DLDS_RESULTS_ROOT:-$(resolve_path RESULTS_ROOT)}"
export SOCIAL_DLDS_OUT="${SOCIAL_DLDS_OUT:-$(resolve_path OUT)}"

configure_matplotlib() {
    export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/social-dlds-matplotlib}"
    mkdir -p "$MPLCONFIGDIR"
}

usage() {
    cat <<'EOF'
social-dLDS reproduction workflow

Usage: bash reproduce.sh <target> [target options]

  paths              print the four resolved paths
  check              validate dependencies and deposited artifacts
  convert-calms21    convert the public CalMS21 JSON to per-session arrays
  features-single    build the published 16-D single-mouse features
  features-dyadic    build the published 27-D dyadic features
  features           run both feature builders
  fit-one            fit one feature file and write Fs.npy + cs.npy
  analysis           recompute downstream statistics and null tests
  figures [targets]  redraw panels; targets are fig01..fig06, supp, assemble

Environment overrides: PYTHON, JULIA, SOCIAL_DLDS_ROOT,
SOCIAL_DLDS_RAW_ROOT, SOCIAL_DLDS_RESULTS_ROOT, SOCIAL_DLDS_OUT.
EOF
}

target="${1:-help}"
[ $# -eq 0 ] || shift

case "$target" in
    help|-h|--help)
        usage
        ;;
    paths)
        printf 'data_root:    %s\n' "$SOCIAL_DLDS_ROOT"
        printf 'raw_data_root:%s\n' " $SOCIAL_DLDS_RAW_ROOT"
        printf 'results_root: %s\n' "$SOCIAL_DLDS_RESULTS_ROOT"
        printf 'out_root:     %s\n' "$SOCIAL_DLDS_OUT"
        ;;
    check)
        configure_matplotlib
        "$PYTHON" scripts/check_env.py "$@"
        ;;
    convert-calms21)
        "$PYTHON" features/calms21_to_npy.py "$@"
        ;;
    features-single)
        "$PYTHON" features/build_feature16_single_mouse.py "$@"
        ;;
    features-dyadic)
        "$PYTHON" features/build_feature27_dyadic.py "$@"
        ;;
    features)
        [ $# -eq 0 ] || {
            echo "'features' uses the published defaults and takes no options; run a specific feature target to customize it." >&2
            exit 2
        }
        "$PYTHON" features/build_feature16_single_mouse.py
        "$PYTHON" features/build_feature27_dyadic.py
        ;;
    fit-one)
        "$JULIA" --project=julia julia/drivers/fit_one_session.jl "$@"
        ;;
    analysis)
        configure_matplotlib
        [ $# -eq 0 ] || {
            echo "'analysis' takes no options; run an individual script to customize it." >&2
            exit 2
        }
        "$PYTHON" analysis/operator_screening.py
        "$PYTHON" figures/supp/s6_pose_readout.py
        "$PYTHON" figures/supp/s6_geometry_readout.py
        "$PYTHON" analysis/f4_distance_deciles.py
        "$PYTHON" analysis/f9_foreshortening.py
        "$PYTHON" analysis/signed_selectivity.py
        "$PYTHON" analysis/syllable_selectivity.py
        "$PYTHON" analysis/geometry_gating.py
        "$PYTHON" analysis/loso_stability.py
        ;;
    figures)
        configure_matplotlib
        bash run_figures.sh "$@"
        ;;
    *)
        echo "unknown target: $target" >&2
        usage >&2
        exit 2
        ;;
esac
