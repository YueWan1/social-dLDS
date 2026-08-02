"""Generate the cross-block validation panel in Supplementary Figure S6.

The panel tests three partner-direction operators and the ``f_4`` distance
operator across dyadic sessions. Null distributions use per-session circular
shifts.
"""

import matplotlib.pyplot as plt
import numpy as np

from dlds_release.paths import (
    dyadic_cs_dir,
    dyadic_dictionary,
    feature27_dir,
    out_dir,
)


plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
    }
)

RUN_DIR = dyadic_cs_dir()
FEAT_DIR = feature27_dir()
FIG_DIR = out_dir("supp")

A_IDX = list(range(0, 14))
B_IDX = list(range(14, 25))
C_IDX = list(range(25, 27))
BLOCKS = {"A": A_IDX, "B": B_IDX, "C": C_IDX}
ACTIVATION_THRESHOLD = 0.05

DIRECTION_OPERATORS = [
    {"name": "f_11", "slot": 10, "source": "B"},
    {"name": "f_9", "slot": 8, "source": "B"},
    {"name": "f_15", "slot": 14, "source": "B"},
]
POSITIVE_COLORS = {"f_11": "#1f77b4", "f_9": "#9467bd", "f_15": "#d62728"}
HISTOGRAM_COLORS = {
    "f_11": "#9bb5d9",
    "f_9": "#c0a8d8",
    "f_15": "#f1a5a5",
    "f_4": "#9bb5d9",
}
SCATTER_COLOR = "#5e6266"
FIT_COLOR = "#d62728"
F4_SLOT = 3


def normalize_features(features):
    features = features.astype(float, copy=True)
    features /= np.maximum(np.std(features, axis=1, keepdims=True), 1e-3)
    features /= max(np.quantile(np.abs(features), 0.99), 1e-6)
    return features


def load_sessions():
    sessions = []
    for session_id in range(1, 71):
        coefficient_path = RUN_DIR / f"cs_mouse{session_id:03d}.npy"
        feature_path = FEAT_DIR / f"FEATURE27_mouse{session_id:03d}.npy"
        if not (coefficient_path.exists() and feature_path.exists()):
            continue

        coefficients = np.load(coefficient_path)
        features = np.load(feature_path)
        if features.shape[0] != 27:
            features = features.T
        features = normalize_features(features)
        length = min(coefficients.shape[1], features.shape[1])
        sessions.append((coefficients[:, :length], features[:, :length]))

    if not sessions:
        raise FileNotFoundError(
            "No matching cs_mouse*.npy and FEATURE27_mouse*.npy sessions were found."
        )
    print(f"Loaded {len(sessions)} dyadic sessions")
    return sessions


def direction_correlations(sessions, operator, u1, v1, offsets=None):
    source_indices = BLOCKS[operator["source"]]
    correlations = []
    for index, (coefficients, features) in enumerate(sessions):
        coefficient = coefficients[operator["slot"]]
        if offsets is not None:
            coefficient = np.roll(coefficient, offsets[index])
        response = u1 @ features[C_IDX, :]
        prediction = coefficient * (v1 @ features[source_indices, :])
        if response.std() > 1e-9 and prediction.std() > 1e-9:
            correlations.append(np.corrcoef(response, prediction)[0, 1])
    return np.asarray(correlations)


def compute_direction_readout(dictionary, sessions, operator):
    source_indices = BLOCKS[operator["source"]]
    matrix = dictionary[operator["slot"]][np.ix_(C_IDX, source_indices)]
    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    u1 = u[:, 0]
    v1 = vt[0, :]
    rank1_fraction = singular_values[0] ** 2 / max(
        np.sum(singular_values**2), 1e-12
    )

    session_correlations = direction_correlations(
        sessions, operator, u1, v1
    )
    pooled_response = []
    pooled_prediction = []
    positive_coupling = 0.0
    positive_count = 0

    for coefficients, features in sessions:
        coefficient = coefficients[operator["slot"]]
        response = u1 @ features[C_IDX, :]
        prediction = coefficient * (v1 @ features[source_indices, :])
        if response.std() > 1e-9 and prediction.std() > 1e-9:
            pooled_response.append(response)
            pooled_prediction.append(prediction)

        positive = coefficient > ACTIVATION_THRESHOLD
        positive_coupling += float(prediction[positive].sum())
        positive_count += int(positive.sum())

    # Fix the arbitrary SVD sign using the positive-coefficient phase.
    if positive_coupling / max(positive_count, 1) < 0:
        pooled_response = [-values for values in pooled_response]
        pooled_prediction = [-values for values in pooled_prediction]

    # Use one shift schedule so operator nulls are directly comparable.
    rng = np.random.default_rng(20260601)
    null_medians = np.empty(1000)
    for permutation in range(null_medians.size):
        offsets = [
            int(rng.integers(1, coefficients.shape[1]))
            for coefficients, _ in sessions
        ]
        null_medians[permutation] = np.median(
            direction_correlations(sessions, operator, u1, v1, offsets)
        )
    observed_median = float(np.median(session_correlations))
    null_mean = float(null_medians.mean())
    null_sd = float(null_medians.std(ddof=1))

    return {
        "rank1_fraction": float(rank1_fraction),
        "session_correlations": session_correlations,
        "response": np.concatenate(pooled_response),
        "prediction": np.concatenate(pooled_prediction),
        "null_medians": null_medians,
        "z_score": (observed_median - null_mean) / max(null_sd, 1e-12),
    }


def compute_distance_readout(dictionary, sessions):
    matrix = dictionary[F4_SLOT][np.ix_(B_IDX, B_IDX)]
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    direction = eigenvectors[:, np.argmax(np.abs(eigenvalues))].real

    session_correlations = []
    pooled_coefficient = []
    pooled_projection = []
    for coefficients, features in sessions:
        coefficient = coefficients[F4_SLOT]
        projection = direction @ features[B_IDX, :]
        if coefficient.std() < 1e-9:
            continue
        session_correlations.append(np.corrcoef(coefficient, projection)[0, 1])
        pooled_coefficient.append(coefficient)
        pooled_projection.append(projection)

    session_correlations = np.asarray(session_correlations)
    if np.median(session_correlations) < 0:
        session_correlations *= -1
        pooled_projection = [-values for values in pooled_projection]

    return {
        "session_correlations": session_correlations,
        "coefficient": np.concatenate(pooled_coefficient),
        "projection": np.concatenate(pooled_projection),
    }


def plot_reproducibility(ax, name, correlations, color, z_score=None):
    ax.hist(correlations, bins=18, color=color, edgecolor="white", linewidth=0.35)
    median = float(np.median(correlations))
    ax.axvline(median, color="black", lw=1.2)
    annotation = rf"median $r={median:.2f}$"
    if z_score is not None:
        annotation += "\n" + rf"$z_{{null}}={z_score:.1f}$"
    ax.text(
        0.97,
        0.95,
        annotation,
        transform=ax.transAxes,
        fontsize=5.5,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "#dddddd"},
    )
    label = name.replace("_", "_{") + "}"
    ax.set_title(rf"$\mathbf{{{label}}}$", fontsize=7.3, fontweight="bold", pad=2)
    ax.set_xlabel("per-session Pearson $r$", fontsize=5.5, labelpad=1.5)
    ax.tick_params(labelsize=5.0, pad=1.0)
    ax.spines[["top", "right"]].set_visible(False)


def subsample_pair(x, y, rng, limit=12_000):
    count = min(limit, len(x))
    indices = rng.choice(len(x), count, replace=False)
    return x[indices], y[indices]


def plot_framewise_agreement(
    ax, x, y, color, rng, annotation, xlabel, ylabel=""
):
    shown_x, shown_y = subsample_pair(x, y, rng)
    slope, intercept = np.polyfit(shown_x, shown_y, 1)
    pooled_r = float(np.corrcoef(x, y)[0, 1])
    fit_x = np.linspace(shown_x.min(), shown_x.max(), 150)

    ax.scatter(
        shown_x,
        shown_y,
        s=1.0,
        color=SCATTER_COLOR,
        alpha=0.12,
        edgecolors="none",
        rasterized=True,
    )
    ax.plot(fit_x, slope * fit_x + intercept, color=color, lw=1.25)
    ax.axhline(0, color="#bbbbbb", lw=0.45)
    ax.axvline(0, color="#bbbbbb", lw=0.45)
    ax.text(
        0.03,
        0.97,
        rf"$r_{{pool}}={pooled_r:.2f}$" + annotation,
        transform=ax.transAxes,
        fontsize=5.4,
        color=color,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "#dddddd"},
    )
    ax.set_xlabel(xlabel, fontsize=5.2, labelpad=1.5)
    ax.set_ylabel(ylabel, fontsize=5.2, labelpad=1.5)
    ax.tick_params(labelsize=5.0, pad=1.0)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    sessions = load_sessions()
    dictionary = np.load(dyadic_dictionary())
    direction_data = {
        operator["name"]: compute_direction_readout(dictionary, sessions, operator)
        for operator in DIRECTION_OPERATORS
    }
    distance_data = compute_distance_readout(dictionary, sessions)
    z_scores = {
        name: values["z_score"] for name, values in direction_data.items()
    }

    for name, values in direction_data.items():
        print(
            f"{name}: rank-1 fraction={values['rank1_fraction']:.3f}, "
            f"median r={np.median(values['session_correlations']):+.3f}"
        )
    print(
        "f_4: "
        f"median r={np.median(distance_data['session_correlations']):+.3f}"
    )

    np.savez_compressed(
        out_dir("analysis") / "dyadic_crossblock_validation.npz",
        **{
            f"{name}_{key}": values[key]
            for name, values in direction_data.items()
            for key in ("session_correlations", "null_medians", "z_score")
        },
        f_4_session_correlations=distance_data["session_correlations"],
    )

    figure, axes = plt.subplots(2, 4, figsize=(6.9, 3.15))
    plt.subplots_adjust(
        left=0.065,
        right=0.995,
        bottom=0.14,
        top=0.84,
        wspace=0.28,
        hspace=0.52,
    )

    for column, operator in enumerate(DIRECTION_OPERATORS):
        name = operator["name"]
        values = direction_data[name]
        plot_reproducibility(
            axes[0, column],
            name,
            values["session_correlations"],
            HISTOGRAM_COLORS[name],
            z_scores[name],
        )
    plot_reproducibility(
        axes[0, 3],
        "f_4",
        distance_data["session_correlations"],
        HISTOGRAM_COLORS["f_4"],
    )

    rng = np.random.default_rng(42)
    for column, operator in enumerate(DIRECTION_OPERATORS):
        name = operator["name"]
        values = direction_data[name]
        plot_framewise_agreement(
            axes[1, column],
            values["prediction"],
            values["response"],
            POSITIVE_COLORS[name],
            rng,
            "\n" + rf"$z_{{null}}={z_scores[name]:.1f}$",
            r"$c_m(t)[\mathbf{v}_1^\top\mathbf{x}^B(t)]$",
        )
    plot_framewise_agreement(
        axes[1, 3],
        distance_data["projection"],
        distance_data["coefficient"],
        FIT_COLOR,
        rng,
        "",
        r"$\mathbf{v}_1^\top\mathbf{x}^B(t)$",
    )

    figure.text(
        0.015,
        0.70,
        "ACROSS-SESSION\nREPRODUCIBILITY",
        rotation=90,
        ha="center",
        va="center",
        fontsize=6.4,
        fontweight="bold",
        color="#555555",
    )
    figure.text(
        0.015,
        0.275,
        "FRAMEWISE\nAGREEMENT",
        rotation=90,
        ha="center",
        va="center",
        fontsize=6.4,
        fontweight="bold",
        color="#555555",
    )
    figure.suptitle(
        "Cross-block partner-geometry validation",
        fontsize=8.6,
        fontweight="bold",
        y=0.975,
    )

    output_path = FIG_DIR / "dyadic_supp_crossblock_validation_refined.pdf"
    figure.savefig(output_path, facecolor="white")
    plt.close(figure)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
