#!/usr/bin/env bash
#
# Redraw the data-derived panels placed in the paper.
#
#   bash run_figures.sh              # everything
#   bash run_figures.sh fig05        # one figure
#   bash run_figures.sh fig03 supp   # several
#
# Output goes to the configured out_root. Nothing is written back into the
# data root.
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

OUT_ROOT="$("$PYTHON" -c 'from dlds_release.paths import OUT; print(OUT)')" || exit 1
FIG01_CLIP="$("$PYTHON" -c '
from dlds_release.paths import RESULTS_ROOT
print(RESULTS_ROOT / "kpms_repro/clip_video/21_12_2_def6a_1_fr75369_76269/clip_skeleton.mp4")
')" || exit 1

failed=()
skipped=()

run() {
    echo "--> $*"
    "$PYTHON" "$@" || { echo "    FAILED: $1"; failed+=("$1"); }
}

record_skip() {
    local script="$1"
    local reason="$2"
    echo "--- SKIP $script ($reason)"
    skipped+=("$script")
}

check_optional() {
    local requirements="$1"
    local script="$2"
    local artifact
    local output
    local status
    local IFS=,
    for artifact in $requirements; do
        output="$("$PYTHON" scripts/check_optional_input.py "$artifact" 2>&1)"
        status=$?
        case "$status" in
            0) ;;
            1)
                record_skip "$script" \
                    "optional artifact '$artifact' is not distributed; see docs/DATA_AVAILABILITY.md"
                return 1
                ;;
            *)
                echo "    FAILED: invalid optional-artifact guard for $script"
                [ -z "$output" ] || printf '      %s\n' "$output"
                failed+=("$script")
                return 2
                ;;
        esac
    done
    return 0
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
        record_skip "$1" "needs keypoint-MoSeq: pip install -e '.[moseq]'"
    fi
}

ensure_fig01_clip() {
    if [ -s "$FIG01_CLIP" ]; then
        echo "--> using existing $FIG01_CLIP"
        return 0
    fi
    echo "--> SKIP_FRAMES=1 $PYTHON -m dlds_release.kpms_clip_video_frames"
    SKIP_FRAMES=1 "$PYTHON" -m dlds_release.kpms_clip_video_frames \
        || {
            echo "    FAILED: dlds_release.kpms_clip_video_frames"
            failed+=("dlds_release.kpms_clip_video_frames")
            return 1
        }
}

fig01() {
    if ensure_fig01_clip; then
        run figures/fig01/panel_a_filmstrip_grid.py
    fi
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
    local fig03_out="$OUT_ROOT/fig03"
    local supp_out="$OUT_ROOT/supp"
    mkdir -p "$fig03_out" "$supp_out"
    local -a fig03_inputs=(
        "$fig03_out/fig3a_shared_space.pdf"
        "$fig03_out/fig3b_syll21_3d.pdf"
        "$fig03_out/fig3c_syll21_op2_op6.pdf"
        "$fig03_out/fig3d_omega_sorted.pdf"
    )
    local missing_fig03=0
    local optional_status
    local input
    check_optional moseq_single figures/fig03/compose.tex
    optional_status=$?
    [ "$optional_status" -eq 0 ] || missing_fig03=1
    for input in "${fig03_inputs[@]}"; do
        [ -f "$input" ] || missing_fig03=1
    done
    if [ "$missing_fig03" -eq 1 ]; then
        if [ "$optional_status" -eq 0 ]; then
            record_skip figures/fig03/compose.tex "Figure 3 panel inputs are absent"
        fi
    else
        echo "--> pdflatex figures/fig03/compose.tex"
        pdflatex -interaction=nonstopmode -halt-on-error \
            -jobname=fig03_compose \
            -output-directory="$build_dir" \
            "\\def\\PanelRoot{\\detokenize{$fig03_out/}}\\input{figures/fig03/compose.tex}" \
            >/dev/null \
            || { echo "    FAILED: fig03_compose.tex"; failed+=(fig03_compose.tex); }
        if [ -f "$build_dir/fig03_compose.pdf" ]; then
            cp "$build_dir/fig03_compose.pdf" "$fig03_out/fig03_compose.pdf"
        fi
    fi
    local -a s6_inputs=(
        "$supp_out/dyadic_fig2a_matrix_pose_refined.pdf"
        "$supp_out/dyadic_supp_pose_alignment_nulls.pdf"
        "$supp_out/dyadic_supp_crossblock_validation_refined.pdf"
    )
    local missing_s6=0
    for input in "${s6_inputs[@]}"; do
        [ -f "$input" ] || missing_s6=1
    done
    if [ "$missing_s6" -eq 1 ]; then
        echo "    FAILED: figures/supp/compose_s6.tex (panel inputs are absent from $supp_out)"
        failed+=(figures/supp/compose_s6.tex)
    else
        echo "--> pdflatex figures/supp/compose_s6.tex"
        pdflatex -interaction=nonstopmode -halt-on-error \
            -jobname=SUPP_dyadic_readout_v2 \
            -output-directory="$build_dir" \
            "\\def\\PanelRoot{\\detokenize{$supp_out/}}\\input{figures/supp/compose_s6.tex}" \
            >/dev/null \
            || { echo "    FAILED: compose_s6.tex"; failed+=(supp_s6); }
        if [ -f "$build_dir/SUPP_dyadic_readout_v2.pdf" ]; then
            cp "$build_dir/SUPP_dyadic_readout_v2.pdf" \
                "$supp_out/SUPP_dyadic_readout_v2.pdf"
        fi
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
if [ ${#skipped[@]} -gt 0 ]; then
    echo "${#skipped[@]} skipped:"
    printf '  %s\n' "${skipped[@]}"
fi
if [ ${#failed[@]} -eq 0 ] && [ ${#skipped[@]} -eq 0 ]; then
    echo "all requested panels completed"
elif [ ${#failed[@]} -eq 0 ]; then
    echo "requested figure workflow completed with ${#skipped[@]} skipped"
else
    echo "${#failed[@]} failed:"
    printf '  %s\n' "${failed[@]}"
    echo "A missing input is the usual cause; run 'bash reproduce.sh check' and see docs/FIGURE_INDEX.md."
    exit 1
fi
