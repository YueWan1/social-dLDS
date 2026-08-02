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
# Interpreter that has the optional `moseq` extra. Three panels need
# keypoint-MoSeq; they are skipped rather than failed when it is absent.
MOSEQ_PYTHON="${MOSEQ_PYTHON:-$PYTHON}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/social-dlds-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

failed=()

run() {
    echo "--> $*"
    "$PYTHON" "$@" || { echo "    FAILED: $1"; failed+=("$1"); }
}

run_moseq() {
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
    run_moseq figures/fig01/panel_c_moseq_model_and_dictionary.py
    run figures/fig01/panel_c_syllable_usage_inset.py
    run figures/fig01/panel_d_dlds_engine.py
    run figures/fig01/panel_d_operator_usage_inset.py
    run_moseq figures/fig01/panel_e_clip_strip.py
    run figures/fig01/panel_f_clip_timeseries.py
}

fig02() {
    run figures/fig02/panel_a_geometry_components.py
    run figures/fig02/panel_b_syllable_components.py
}

fig03() {
    run figures/fig03/panel_a_shared_space_hero.py
    run figures/fig03/panels_bc_syllable_subspace.py
    run figures/fig03/panel_d_omega_sorted.py
}

fig04() {
    run_moseq figures/fig04/panel_b_syllable_dictionary_28.py
    run figures/fig04/panel_d_operator_overview.py
}

# One run emits exactly the eleven placed sub-panels.
fig05() { run figures/fig05/make_all_panels.py; }

fig06() {
    run figures/fig06/panel_a_behavior_composition.py
    run figures/fig06/panel_b_signed_selectivity.py
    run figures/fig06/panel_c_operator_gate.py
    run figures/fig06/panel_d_coverage_curve.py
    run figures/fig06/panel_e_decoder_ladders.py      # slowest, LOSO logistic regressions
}

supp() {
    run figures/supp/s1_matched_pipeline_null.py
    run figures/supp/s4_operator_selection.py
    run figures/supp/s5_loso_stability.py
    run figures/supp/s6_pose_readout.py
    run figures/supp/s6_geometry_readout.py
    run figures/supp/s7_selectivity_tie.py
    run figures/supp/s8_tracked_gate.py
    run figures/supp/s9_decoder_4class.py             # slow
}

# Figure 3 and Supplementary S6 have scripted composition. The other main
# layouts were assembled in Illustrator; this release regenerates their
# data-derived source panels, not typography-only assembly fragments.
assemble() {
    local build_dir
    build_dir="$(mktemp -d)"
    echo "--> pdflatex figures/fig03/compose.tex"
    mkdir -p out/fig03 out/supp
    pdflatex -interaction=nonstopmode -halt-on-error \
        -jobname=fig03_compose \
        -output-directory="$build_dir" figures/fig03/compose.tex >/dev/null \
        || { echo "    FAILED: fig03_compose.tex"; failed+=(fig03_compose.tex); }
    if [ -f "$build_dir/fig03_compose.pdf" ]; then
        cp "$build_dir/fig03_compose.pdf" out/fig03/fig03_compose.pdf
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
    echo "A missing input is the usual cause; docs/FIGURE_INDEX.md lists panel inputs."
    exit 1
fi
