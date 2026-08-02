"""Recompute the single-mouse geometry circular-shift null.

Every circular-shift replicate reruns the same orientation rule used for the
observed statistic before evaluating alignment:

  1. independently circular-shift each session's coefficient time series;
  2. recompute the pooled positive- and negative-phase pose means;
  3. orient each leading eigenvector toward the shifted positive-phase mean;
  4. rebuild d_pred and d_emp; and
  5. calculate |cos(d_pred, d_emp)|.

The null uses all nonzero lags as its sampling support. Monte Carlo p-values
use the finite-sample correction (b + 1) / (N + 1), where b is the number of
null statistics greater than or equal to the observed statistic.

All-shift pose sums are precomputed with FFT circular correlations.  A
run-time assertion checks them against direct ``np.roll`` calculations.

Run from the repository root:

    python3 figures/supp/s1_matched_pipeline_null.py

The default 100,000 replicates take a few minutes; ``--n-null`` shortens a quick
local check but changes the resolution of the reported p-value.

Outputs are written to ``<out_root>/supp/matched_pipeline_null/``.  The
subdirectory is kept because the manuscript includes the figure by the relative
path ``matched_pipeline_null/matched_pipeline_circular_shift_null.pdf``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dlds_release.paths import ROOT, feature16_dir, out_dir, single_fit_dir


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)


# The accessors carry the hyperparameter stamp of the published single-mouse
# fit, so this null cannot silently be evaluated against a neighbouring sweep.
RUN = single_fit_dir()
FEATURE_DIR = feature16_dir()
DEFAULT_OUTPUT_DIR = out_dir("supp", "matched_pipeline_null")

SESSIONS = {
    "21_12_10_def6b_3": "FEATURE16_kpmoseq_21_12_10_def6b_3.npy",
    "21_12_2_def6a_1": "FEATURE16_kpmoseq_21_12_2_def6a_1.npy",
    "21_12_2_def6b_2": "FEATURE16_kpmoseq_21_12_2_def6b_2.npy",
    "22_04_26_cage4_0": "FEATURE16_kpmoseq_22_04_26_cage4_0.npy",
    "22_04_26_cage4_1_1": "FEATURE16_kpmoseq_22_04_26_cage4_1_1.npy",
}

THRESHOLD = 0.05
OPERATORS = ((2, "op2", "strong turn"), (6, "op6", "weaker turn"))
DEFAULT_N_NULL = 100_000
DEFAULT_SEED = 20260718
NULL_COLOR = "#5d8f8a"
OBSERVED_COLOR = "#c62828"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-null",
        type=int,
        default=DEFAULT_N_NULL,
        help=f"Monte Carlo replicates (default: {DEFAULT_N_NULL:,}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: matched_pipeline_null next to script).",
    )
    args = parser.parse_args()
    if args.n_null < 1:
        parser.error("--n-null must be positive")
    return args


def normalize_features(data: np.ndarray) -> np.ndarray:
    """Match the per-session normalization in the current figure code."""
    features = data.copy().astype(float)
    scale = np.maximum(np.std(features, axis=1, keepdims=True), 1e-3)
    features /= scale
    q99 = max(float(np.quantile(np.abs(features), 0.99)), 1e-6)
    features /= q99
    return features


def load_data() -> tuple[np.ndarray, list[dict[str, object]], np.ndarray]:
    operators = np.load(RUN / "Fs.npy")
    sessions: list[dict[str, object]] = []
    pooled_pose_sum = np.zeros(16, dtype=float)
    pooled_frames = 0

    for session_id, feature_name in SESSIONS.items():
        coefficients = np.load(RUN / f"cs_{session_id}.npy")
        raw_pose = np.load(FEATURE_DIR / feature_name)[:, : coefficients.shape[1]]
        pose = normalize_features(raw_pose)
        if pose.shape != (16, coefficients.shape[1]):
            raise ValueError(
                f"Shape mismatch for {session_id}: pose={pose.shape}, "
                f"coefficients={coefficients.shape}"
            )
        if not np.isfinite(pose).all() or not np.isfinite(coefficients).all():
            raise ValueError(f"Non-finite values found in session {session_id}")

        sessions.append(
            {
                "id": session_id,
                "coefficients": coefficients,
                "pose": pose,
            }
        )
        pooled_pose_sum += pose.sum(axis=1)
        pooled_frames += pose.shape[1]

    return operators, sessions, pooled_pose_sum / pooled_frames


def leading_real_eigenvectors(
    operator_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eig(operator_matrix)
    order = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    v1 = eigenvectors[:, 0].real
    v2 = eigenvectors[:, 1].real
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return eigenvalues, v1, v2


def circular_shift_pose_sums(
    pose: np.ndarray,
    mask: np.ndarray,
    session_id: str,
    mask_name: str,
) -> np.ndarray:
    """Return pose sums for every lag, exactly matching np.roll(mask, lag)."""
    n_frames = mask.size
    pose_fft = np.fft.rfft(pose, axis=1)
    mask_fft = np.fft.rfft(mask.astype(float))
    sums = np.fft.irfft(
        pose_fft * np.conj(mask_fft)[None, :],
        n=n_frames,
        axis=1,
    ).real.T

    # Validate the FFT orientation and numerical accuracy against direct rolls.
    check_lags = sorted({0, 1, n_frames // 3, n_frames - 1})
    direct = np.stack(
        [pose @ np.roll(mask.astype(float), lag) for lag in check_lags],
        axis=0,
    )
    error = float(np.max(np.abs(sums[check_lags] - direct)))
    tolerance = 1e-8 * max(1.0, float(np.max(np.abs(direct))))
    if error > tolerance:
        raise AssertionError(
            f"FFT shift check failed for {session_id}/{mask_name}: "
            f"max error={error:.3g}, tolerance={tolerance:.3g}"
        )
    return sums


def orient_and_align(
    mean_positive: np.ndarray,
    mean_negative: np.ndarray,
    mean_pose: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Apply the observed sign convention and compute one alignment."""
    reference = mean_positive - mean_pose
    sign1 = 1.0 if float(v1 @ reference) + 1e-12 >= 0.0 else -1.0
    sign2 = 1.0 if float(v2 @ reference) + 1e-12 >= 0.0 else -1.0
    d_pred = sign1 * v1 + sign2 * v2
    d_pred /= np.linalg.norm(d_pred)
    d_emp = (mean_positive - mean_negative) / 2.0
    empirical_norm = float(np.linalg.norm(d_emp))
    signed_cosine = float(d_pred @ d_emp) / (empirical_norm + 1e-12)
    signed_cosine = float(np.clip(signed_cosine, -1.0, 1.0))
    return {
        "d_pred": d_pred,
        "d_emp": d_emp,
        "sign1": sign1,
        "sign2": sign2,
        "signed_cosine": signed_cosine,
        "abs_cosine": abs(signed_cosine),
        "signed_angle_deg": float(np.degrees(np.arccos(signed_cosine))),
        "acute_angle_deg": float(np.degrees(np.arccos(abs(signed_cosine)))),
        "empirical_norm": empirical_norm,
    }


def vectorized_null_alignment(
    mean_positive: np.ndarray,
    mean_negative: np.ndarray,
    mean_pose: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute matched-pipeline statistics for sampled shifts."""
    reference = mean_positive - mean_pose[None, :]
    sign1 = np.where(reference @ v1 + 1e-12 >= 0.0, 1.0, -1.0)
    sign2 = np.where(reference @ v2 + 1e-12 >= 0.0, 1.0, -1.0)
    d_pred = sign1[:, None] * v1[None, :] + sign2[:, None] * v2[None, :]
    d_pred /= np.linalg.norm(d_pred, axis=1, keepdims=True)

    d_emp = (mean_positive - mean_negative) / 2.0
    d_emp_norm = np.linalg.norm(d_emp, axis=1)
    if np.any(d_emp_norm <= 1e-12):
        raise ValueError("A circular-shift replicate produced a zero d_emp vector")

    matched = np.abs(np.einsum("ij,ij->i", d_pred, d_emp) / d_emp_norm)
    return (
        np.clip(matched, 0.0, 1.0),
        sign1.astype(np.int8),
        sign2.astype(np.int8),
    )


def summarize_null(null: np.ndarray, observed: float) -> dict[str, object]:
    exceedances = int(np.count_nonzero(null >= observed))
    denominator = int(null.size + 1)
    numerator = exceedances + 1
    return {
        "exceedances": exceedances,
        "p_numerator": numerator,
        "p_denominator": denominator,
        "p_mc": numerator / denominator,
        "mean": float(np.mean(null)),
        "sd": float(np.std(null, ddof=1)),
        "q025": float(np.quantile(null, 0.025)),
        "q50": float(np.quantile(null, 0.5)),
        "q975": float(np.quantile(null, 0.975)),
        "maximum": float(np.max(null)),
    }


def format_p(summary: dict[str, object]) -> str:
    return (
        f"{summary['p_numerator']}/{summary['p_denominator']} = "
        f"{float(summary['p_mc']):.8g}"
    )


def compute_operator_null(
    slot: int,
    name: str,
    role: str,
    operator_matrices: np.ndarray,
    sessions: list[dict[str, object]],
    mean_pose: np.ndarray,
    shifts: dict[str, np.ndarray],
) -> dict[str, object]:
    eigenvalues, v1, v2 = leading_real_eigenvectors(operator_matrices[slot])
    n_null = next(iter(shifts.values())).size
    positive_sum_null = np.zeros((n_null, 16), dtype=float)
    negative_sum_null = np.zeros((n_null, 16), dtype=float)
    positive_sum_observed = np.zeros(16, dtype=float)
    negative_sum_observed = np.zeros(16, dtype=float)
    n_positive = 0
    n_negative = 0

    print(f"[{name}] precomputing all-lag pose sums ...", flush=True)
    for session in sessions:
        session_id = str(session["id"])
        coefficients = np.asarray(session["coefficients"])[slot]
        pose = np.asarray(session["pose"])
        positive = coefficients > THRESHOLD
        negative = coefficients < -THRESHOLD
        if not positive.any() or not negative.any():
            raise ValueError(
                f"{name}/{session_id} lacks one signed phase at threshold {THRESHOLD}"
            )

        positive_sums = circular_shift_pose_sums(
            pose, positive, session_id, f"{name}_positive"
        )
        negative_sums = circular_shift_pose_sums(
            pose, negative, session_id, f"{name}_negative"
        )
        sampled_lags = shifts[session_id]
        positive_sum_null += positive_sums[sampled_lags]
        negative_sum_null += negative_sums[sampled_lags]
        positive_sum_observed += positive_sums[0]
        negative_sum_observed += negative_sums[0]
        n_positive += int(positive.sum())
        n_negative += int(negative.sum())

    mean_positive_observed = positive_sum_observed / n_positive
    mean_negative_observed = negative_sum_observed / n_negative
    observed = orient_and_align(
        mean_positive_observed,
        mean_negative_observed,
        mean_pose,
        v1,
        v2,
    )
    mean_positive_null = positive_sum_null / n_positive
    mean_negative_null = negative_sum_null / n_negative
    matched, sign1, sign2 = vectorized_null_alignment(
        mean_positive_null,
        mean_negative_null,
        mean_pose,
        v1,
        v2,
    )
    matched_summary = summarize_null(matched, float(observed["abs_cosine"]))

    sign_counts = {
        f"{a:+d},{b:+d}": int(np.count_nonzero((sign1 == a) & (sign2 == b)))
        for a in (-1, 1)
        for b in (-1, 1)
    }
    result = {
        "slot": slot,
        "name": name,
        "role": role,
        "leading_eigenvalues_real": [
            float(eigenvalues[0].real),
            float(eigenvalues[1].real),
        ],
        "leading_eigenvalues_imag": [
            float(eigenvalues[0].imag),
            float(eigenvalues[1].imag),
        ],
        "n_positive_frames": n_positive,
        "n_negative_frames": n_negative,
        "observed": {
            key: value
            for key, value in observed.items()
            if key not in {"d_pred", "d_emp"}
        },
        "observed_d_pred": np.asarray(observed["d_pred"]).tolist(),
        "observed_d_emp": np.asarray(observed["d_emp"]).tolist(),
        "matched": matched_summary,
        "null_sign_pattern_counts": sign_counts,
        "matched_null": matched,
        "sign1": sign1,
        "sign2": sign2,
    }
    print(
        f"[{name}] observed |cos|={float(observed['abs_cosine']):.6f}, "
        f"acute angle={float(observed['acute_angle_deg']):.4f} deg; "
        f"matched b={matched_summary['exceedances']}/{n_null}, "
        f"p_MC={format_p(matched_summary)}",
        flush=True,
    )
    return result


def save_publication_figure(
    results: list[dict[str, object]], output_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.3, 3.05))
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.22, top=0.79, wspace=0.34)

    ax = axes[0]
    ax.text(
        0.5,
        0.60,
        "n/a",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=18,
        color="#777777",
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.35,
        "near-continuous positive hold\n(no signed phase contrast)",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="#666666",
    )
    ax.set_title("op14: baseline / forward", fontweight="bold", pad=7)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    for ax, result in zip(axes[1:], results):
        null = np.asarray(result["matched_null"])
        observed = float(result["observed"]["abs_cosine"])
        summary = result["matched"]
        ax.hist(
            null,
            bins=np.linspace(0.0, 1.0, 41),
            color=NULL_COLOR,
            edgecolor="white",
            linewidth=0.35,
        )
        ax.axvline(observed, color=OBSERVED_COLOR, lw=2.2, zorder=5)
        ax.set_xlim(0.0, 1.015)
        ax.set_xlabel(r"$|\cos(d_{\rm pred}^{(b)},d_{\rm emp}^{(b)})|$")
        ax.set_ylabel("circular-shift replicates")
        ax.set_title(
            f"{result['name']}: {result['role']}",
            fontweight="bold",
            pad=7,
        )
        ax.text(
            0.03,
            0.97,
            "observed |cos| = "
            f"{observed:.3f}\n"
            f"b = {summary['exceedances']:,} / {null.size:,}\n"
            r"$p_{\rm MC}$ = "
            f"{float(summary['p_mc']):.3g}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Matched-pipeline within-session circular-shift null",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )
    fig.savefig(
        output_dir / "matched_pipeline_circular_shift_null.pdf",
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)


def save_tables_and_archive(
    results: list[dict[str, object]],
    shifts: dict[str, np.ndarray],
    n_null: int,
    seed: int,
    output_dir: Path,
) -> None:
    json_results = []
    for result in results:
        clean = {
            key: value
            for key, value in result.items()
            if key
            not in {"matched_null", "sign1", "sign2"}
        }
        clean["matched"]["bonferroni_p_two_operators"] = min(
            1.0, 2.0 * float(clean["matched"]["p_mc"])
        )
        json_results.append(clean)

    payload = {
        "analysis": "single-mouse operator geometry matched-pipeline circular-shift null",
        "fit": str(RUN.relative_to(ROOT)),
        "feature_directory": str(FEATURE_DIR.relative_to(ROOT)),
        "sessions": list(SESSIONS),
        "operators_tested": [2, 6],
        "phase_threshold": THRESHOLD,
        "n_null": n_null,
        "seed": seed,
        "shift_sampling": (
            "one independent uniform lag from integers 1,...,T_s-1 for each "
            "session and replicate; zero lag excluded"
        ),
        "statistic": "abs(cos(d_pred, d_emp))",
        "matched_pipeline": (
            "each replicate recomputes shifted mu_pos and mu_neg, reorients "
            "both leading eigenvectors toward shifted mu_pos - pooled_mu, then "
            "rebuilds d_pred and d_emp"
        ),
        "p_value": "(number of null statistics >= observed + 1) / (N + 1)",
        "p_resolution": 1.0 / (n_null + 1),
        "results": json_results,
    }
    with (output_dir / "matched_pipeline_null_results.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    fieldnames = [
        "operator",
        "observed_abs_cosine",
        "observed_acute_angle_deg",
        "n_positive_frames",
        "n_negative_frames",
        "matched_exceedances",
        "n_null",
        "matched_p_numerator",
        "matched_p_denominator",
        "matched_p_mc",
        "matched_p_bonferroni_two",
        "matched_null_mean",
        "matched_null_sd",
        "matched_null_q025",
        "matched_null_q50",
        "matched_null_q975",
        "matched_null_max",
    ]
    with (output_dir / "matched_pipeline_null_results.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for result in json_results:
            observed = result["observed"]
            matched = result["matched"]
            writer.writerow(
                {
                    "operator": result["name"],
                    "observed_abs_cosine": observed["abs_cosine"],
                    "observed_acute_angle_deg": observed["acute_angle_deg"],
                    "n_positive_frames": result["n_positive_frames"],
                    "n_negative_frames": result["n_negative_frames"],
                    "matched_exceedances": matched["exceedances"],
                    "n_null": n_null,
                    "matched_p_numerator": matched["p_numerator"],
                    "matched_p_denominator": matched["p_denominator"],
                    "matched_p_mc": matched["p_mc"],
                    "matched_p_bonferroni_two": matched[
                        "bonferroni_p_two_operators"
                    ],
                    "matched_null_mean": matched["mean"],
                    "matched_null_sd": matched["sd"],
                    "matched_null_q025": matched["q025"],
                    "matched_null_q50": matched["q50"],
                    "matched_null_q975": matched["q975"],
                    "matched_null_max": matched["maximum"],
                }
            )

    archive: dict[str, np.ndarray] = {
        "n_null": np.asarray(n_null),
        "seed": np.asarray(seed),
    }
    for session_id, sampled_shifts in shifts.items():
        archive[f"shifts__{session_id}"] = sampled_shifts
    for result in results:
        name = str(result["name"])
        archive[f"{name}__matched_null"] = np.asarray(result["matched_null"])
        archive[f"{name}__sign1"] = np.asarray(result["sign1"])
        archive[f"{name}__sign2"] = np.asarray(result["sign2"])
        archive[f"{name}__observed_d_pred"] = np.asarray(
            result["observed_d_pred"]
        )
        archive[f"{name}__observed_d_emp"] = np.asarray(result["observed_d_emp"])
    np.savez_compressed(output_dir / "matched_pipeline_null_distributions.npz", **archive)

    readme = (
        "Matched-pipeline circular-shift validation\n"
        "==========================================\n\n"
        f"Replicates: {n_null:,}\n"
        f"Seed: {seed}\n"
        f"Phase threshold: |c| > {THRESHOLD}\n"
        "Shift support: all nonzero lags 1,...,T_s-1, sampled independently "
        "within each session.\n"
        "Statistic: |cos(d_pred, d_emp)|.\n"
        "Matched null: shifted phase means are recomputed, both leading "
        "eigenvectors are re-oriented toward the shifted positive-phase mean, "
        "and d_pred/d_emp are rebuilt in every replicate.\n"
        "Monte Carlo p: (b + 1)/(N + 1), where b counts null statistics >= "
        "the observed statistic.\n"
    )
    (output_dir / "README.txt").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading current fit and five sessions from {ROOT} ...", flush=True)
    operator_matrices, sessions, mean_pose = load_data()
    rng = np.random.default_rng(args.seed)
    shifts = {
        str(session["id"]): rng.integers(
            1,
            np.asarray(session["coefficients"]).shape[1],
            size=args.n_null,
            dtype=np.int64,
        )
        for session in sessions
    }
    print(
        f"Sampled {args.n_null:,} matched sets of independent nonzero "
        f"within-session lags (seed={args.seed}).",
        flush=True,
    )

    results = [
        compute_operator_null(
            slot,
            name,
            role,
            operator_matrices,
            sessions,
            mean_pose,
            shifts,
        )
        for slot, name, role in OPERATORS
    ]
    save_publication_figure(results, output_dir)
    save_tables_and_archive(
        results,
        shifts,
        args.n_null,
        args.seed,
        output_dir,
    )
    print(f"Wrote corrected null outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
