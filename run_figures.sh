#!/usr/bin/env bash
#
# Redraw the data-derived panels placed in the paper.
#
#   bash run_figures.sh              # everything
#   bash run_figures.sh fig05        # one figure
#   bash run_figures.sh fig03 supp   # several
#
# Output goes to out/<figure>/. Nothing is written back into the data root.
#
# Commands are explicit so the publication panel set is easy to audit. Pure
# text labels, schematics, design previews and internal diagnostics are absent.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python3}"
# Interpreter that has the optional `moseq` extra. Three rendering panels
# import keypoint-MoSeq; other MoSeq-dependent panels only read results.h5.
MOSEQ_PYTHON="${MOSEQ_PYTHON:-$PYTHON}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/social-dlds-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

failed=()

run() {
    echo "--> $*"
    "$PYTHON" "$@" || { echo "    FAILED: $1"; failed+=("$1"); }
}

optional_available() {
    "$PYTHON" scripts/check_optional_input.py "$1" >/dev/null 2>&1
}

check_optional() {
    local requirements="$1"
    local script="$2"
    local artifact
    local IFS=,
    for artifact in $requirements; do
        if ! optional_available "$artifact"; then
            echo "--- SKIP $script (optional artifact '$artifact' is not distributed; see docs/DATA_AVAILABILITY.md)"
            return 1
        fi
    done
}

run_if_available() {
    local requirements="$1"
    shift
    if check_optional "$requirements" "$1"; then
        run "$@"
    fi
}

run_moseq() {
    local requirements="$1"
    shift
    if ! check_optional "$requirements" "$1"; then
        return
    fi
    local jax_platforms="${JAX_PLATFORMS:-cpu}"
    local -a jax_env=(env "JAX_PLATFORMS=$jax_platforms")
    if [ "$jax_platforms" = cpu ]; then
        jax_env+=("JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1")
    fi
    if "${jax_env[@]}" "$MOSEQ_PYTHON" \
        -c 'import keypoint_moseq' >/dev/null 2>&1; then
        echo "--> $*"
        "${jax_env[@]}" "$MOSEQ_PYTHON" "$@" \
            || { echo "    FAILED: $1"; failed+=("$1"); }
    else
        echo "--- SKIP $1 (needs keypoint-MoSeq: pip install -e '.[moseq]')"
    fi
}

fig01() {
    run figures/fig01/panel_a_filmstrip_grid.py
    run_moseq moseq_single figures/fig01/panel_c_moseq_model_and_dictionary.py
    run_if_available moseq_single figures/fig01/panel_c_syllable_usage_inset.py
    run figures/fig01/panel_d_dlds_engine.py
    run figures/fig01/panel_d_operator_usage_inset.py
    run_moseq moseq_single figures/fig01/panel_e_clip_strip.py
    run_if_available moseq_single figures/fig01/panel_f_clip_timeseries.py
}

fig02() {
    run figures/fig02/panel_a_geometry_components.py
    run_if_available moseq_single figures/fig02/panel_b_syllable_components.py
}

fig03() {
    run_if_available moseq_single figures/fig03/panel_a_shared_space_hero.py
    run_if_available moseq_single figures/fig03/panels_bc_syllable_subspace.py
    run_if_available moseq_single figures/fig03/panel_d_omega_sorted.py
}

fig04() {
    run_moseq moseq_dyadic,preprocessed_kp_clean figures/fig04/panel_b_syllable_dictionary_28.py
    run figures/fig04/panel_d_operator_overview.py
}

# One run emits exactly the eleven placed sub-panels.
fig05() { run figures/fig05/make_all_panels.py; }

fig06() {
    run figures/fig06/panel_a_behavior_composition.py
    run figures/fig06/panel_b_signed_selectivity.py
    run figures/fig06/panel_c_operator_gate.py
    run_if_available moseq_dyadic figures/fig06/panel_d_coverage_curve.py
    run_if_available moseq_dyadic figures/fig06/panel_e_decoder_ladders.py
}

supp() {
    run figures/supp/s1_matched_pipeline_null.py
    run figures/supp/s4_operator_selection.py
    run figures/supp/s5_loso_stability.py
    run figures/supp/s6_pose_readout.py
    run figures/supp/s6_geometry_readout.py
    run figures/supp/s7_selectivity_tie.py
    run figures/supp/s8_tracked_gate.py
    run_if_available moseq_dyadic figures/supp/s9_decoder_4class.py
}

# Figure 3 and Supplementary S6 have scripted composition. The other main
# layouts were assembled in Illustrator; this release regenerates their
# data-derived source panels, not typography-only assembly fragments.
assemble() {
    local build_dir
    build_dir="$(mktemp -d)"
    mkdir -p out/fig03 out/supp
    local -a fig03_inputs=(
        out/fig03/fig3a_shared_space.pdf
        out/fig03/fig3b_syll21_3d.pdf
        out/fig03/fig3c_syll21_op2_op6.pdf
        out/fig03/fig3d_omega_sorted.pdf
    )
    local missing_fig03=0
    local input
    optional_available moseq_single || missing_fig03=1
    for input in "${fig03_inputs[@]}"; do
        [ -f "$input" ] || missing_fig03=1
    done
    if [ "$missing_fig03" -eq 1 ]; then
        echo "--- SKIP figures/fig03/compose.tex (Figure 3 panel inputs are absent)"
    else
        echo "--> pdflatex figures/fig03/compose.tex"
        pdflatex -interaction=nonstopmode -halt-on-error \
            -jobname=fig03_compose \
            -output-directory="$build_dir" figures/fig03/compose.tex >/dev/null \
            || { echo "    FAILED: fig03_compose.tex"; failed+=(fig03_compose.tex); }
        if [ -f "$build_dir/fig03_compose.pdf" ]; then
            cp "$build_dir/fig03_compose.pdf" out/fig03/fig03_compose.pdf
        fi
    fi
    echo "--> pdflatex figures/supp/compose_s6.tex"
    pdflatex -interaction=nonstopmode -halt-on-error \
        -jobname=SUPP_dyadic_readout_v2 \
        -output-directory="$build_dir" figures/supp/compose_s6.tex >/dev/null \
        || { echo "    FAILED: compose_s6.tex"; failed+=(supp_s6); }
    if [ -f "$build_dir/SUPP_dyadic_readout_v2.pdf" ]; then
        cp "$build_dir/SUPP_dyadic_readout_v2.pdf" out/supp/SUPP_dyadic_readout_v2.pdf
    fi
    rm -rf "$build_dir"
}

TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(fig01 fig02 fig03 fig04 fig05 fig06 supp assemble)

for t in "${TARGETS[@]}"; do
    case "$t" in
        fig01|fig02|fig03|fig04|fig05|fig06|supp|assemble) echo; echo "===== $t ====="; "$t" ;;
        *) echo "unknown target: $t" >&2; echo "valid: fig01..fig06 supp assemble" >&2; exit 2 ;;
    esac
done

echo
if [ ${#failed[@]} -eq 0 ]; then
    echo "all requested panels completed"
else
    echo "${#failed[@]} failed:"
    printf '  %s\n' "${failed[@]}"
    echo "A missing input is the usual cause; run 'bash reproduce.sh check' and see docs/FIGURE_INDEX.md."
    exit 1
fi
