"""Export only the data-derived geometry components placed in Figure 2."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dlds_release.paths import feature16_dir, out_dir, single_fit_dir


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)

RUN, FEAT = single_fit_dir(), feature16_dir()
OUT = out_dir("fig02", "figure2_geometry_components")
SESSIONS = {
    "21_12_10_def6b_3": "FEATURE16_kpmoseq_21_12_10_def6b_3.npy",
    "21_12_2_def6a_1": "FEATURE16_kpmoseq_21_12_2_def6a_1.npy",
    "21_12_2_def6b_2": "FEATURE16_kpmoseq_21_12_2_def6b_2.npy",
    "22_04_26_cage4_0": "FEATURE16_kpmoseq_22_04_26_cage4_0.npy",
    "22_04_26_cage4_1_1": "FEATURE16_kpmoseq_22_04_26_cage4_1_1.npy",
}
OPS = (
    (14, "op14", "baseline / forward", "rank-1"),
    (2, "op2", "turn (strong)", "rank-2"),
    (6, "op6", "turn (weaker)", "rank-2"),
)
THETA, MAX_DISPLAY_FRAMES = 0.05, 2200
KP_NAMES = ("spine4", "spine3", "spine2", "spine1", "head", "nose", "r_ear", "l_ear")
KP_COLORS = {
    "spine4": "#16a085", "spine3": "#1abc9c", "spine2": "#27ae60",
    "spine1": "#f1c40f", "head": "#e67e22", "nose": "#e74c3c",
    "r_ear": "#f39c12", "l_ear": "#d35400",
}
BONES = (
    ("spine4", "spine3"), ("spine3", "spine2"), ("spine2", "spine1"),
    ("spine1", "head"), ("head", "nose"), ("head", "r_ear"), ("head", "l_ear"),
)
PRED_COLOR, POS_COLOR, NEG_COLOR, EMP_COLOR = "crimson", "#c0392b", "#1f6fb2", "#111111"


def normalize_features(data: np.ndarray) -> np.ndarray:
    features = data.astype(float, copy=True)
    features /= np.maximum(features.std(axis=1, keepdims=True), 1e-3)
    features /= max(float(np.quantile(np.abs(features), 0.99)), 1e-6)
    return features


def vec16_to_display(vector: np.ndarray) -> np.ndarray:
    points = vector.reshape(8, 2)
    return np.column_stack((-points[:, 1], points[:, 0]))


def load_current_fit():
    operators = np.load(RUN / "Fs.npy")
    coefficients, normalized, raw = [], [], []
    for session_id, feature_name in SESSIONS.items():
        cs = np.load(RUN / f"cs_{session_id}.npy")
        features = np.load(FEAT / feature_name)[:, : cs.shape[1]]
        coefficients.append(cs)
        raw.append(features)
        normalized.append(normalize_features(features))
    return operators, coefficients, normalized, raw


def compute_operator_quantities(slot, operators, coefficients, normalized, raw):
    c_all = np.concatenate(coefficients, axis=1)
    xn_all = np.concatenate(normalized, axis=1)
    xr_all = np.concatenate(raw, axis=1)
    mean_pose, raw_std = xn_all.mean(axis=1), xr_all.std(axis=1)

    eigenvalues, eigenvectors = np.linalg.eig(operators[slot])
    order = np.argsort(-np.abs(eigenvalues))
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    c = c_all[slot]
    positive, negative = c > THETA, c < -THETA
    mean_pos_n, mean_pos_raw = xn_all[:, positive].mean(axis=1), xr_all[:, positive].mean(axis=1)
    mean_neg_n = xn_all[:, negative].mean(axis=1) if negative.any() else None
    mean_neg_raw = xr_all[:, negative].mean(axis=1) if negative.any() else None

    reference = mean_pos_n - mean_pose
    v1, v2 = eigenvectors[:, 0].real, eigenvectors[:, 1].real
    for vector in (v1, v2):
        vector *= np.sign(float(vector @ reference) + 1e-12)
        vector /= np.linalg.norm(vector)

    bidirectional = mean_neg_n is not None
    if bidirectional:
        d_emp = (mean_pos_n - mean_neg_n) / 2
        d_pred = v1 + v2
        d_pred /= np.linalg.norm(d_pred)
        d_raw = (v1 + v2) * raw_std
    else:
        d_emp, d_pred, d_raw = mean_pos_n - mean_pose, v1.copy(), v1 * raw_std

    e1 = v1.copy()
    e2 = v2 - float(v2 @ e1) * e1
    e2 /= np.linalg.norm(e2) + 1e-12
    d_emp_norm = np.linalg.norm(d_emp)
    cosine = float(d_pred @ d_emp) / (np.linalg.norm(d_pred) * d_emp_norm + 1e-12)
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    in_plane = float(np.linalg.norm([d_emp @ e1, d_emp @ e2]) / (d_emp_norm + 1e-12))

    active = np.flatnonzero(np.abs(c) > THETA)
    projection = (xn_all[:, active] - mean_pose[:, None]).T @ d_pred
    correlation = float(np.corrcoef(np.abs(c[active]), np.abs(projection))[0, 1])
    return {
        "slot": slot, "eigenvalues": eigenvalues, "c": c,
        "mean_pos_raw": mean_pos_raw, "mean_neg_raw": mean_neg_raw,
        "v1": v1, "v2": v2, "d_pred": d_pred, "d_emp": d_emp, "d_raw": d_raw,
        "e1": e1, "e2": e2, "angle": angle, "in_plane_fraction": in_plane,
        "magnitude_correlation": correlation, "bidirectional": bidirectional,
        "n_positive": int(positive.sum()), "n_negative": int(negative.sum()),
        "active_indices": active, "xn_all": xn_all, "mean_normalized_pose": mean_pose,
    }


def component_figure(width=2.65, height=2.45):
    figure, axis = plt.subplots(figsize=(width, height))
    figure.subplots_adjust(left=0.19, right=0.97, bottom=0.18, top=0.86)
    return figure, axis


def vector_arrow(
    axis, vector, *, color, width, style="-|>", alpha=1.0,
    mutation=8, zorder=None, opposite=False,
):
    options = {
        "arrowstyle": style, "lw": width, "color": color,
        "alpha": alpha, "mutation_scale": mutation,
    }
    if zorder is not None:
        options["zorder"] = zorder
    axis.annotate("", xy=vector, xytext=(0, 0), arrowprops=options)
    if opposite:
        options = {**options, "lw": width / 2, "alpha": 0.35}
        axis.annotate("", xy=-vector, xytext=(0, 0), arrowprops=options)


def empty_axis(
    axis, message, detail, *, message_y=0.57, detail_y=0.37,
    message_size=18, detail_size=11, detail_color="#777777",
    detail_va="center",
):
    axis.text(
        0.5, message_y, message, transform=axis.transAxes, ha="center", va="center",
        fontsize=message_size, color="#888888", fontweight="bold",
    )
    axis.text(
        0.5, detail_y, detail, transform=axis.transAxes, ha="center", va=detail_va,
        fontsize=detail_size, color=detail_color,
    )
    axis.set(xticks=[], yticks=[])
    for spine in axis.spines.values():
        spine.set_visible(False)


def save_component(figure, stem, manifest, operator, component):
    pdf = OUT / f"{stem}.pdf"
    options = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.04}
    figure.savefig(pdf, **options)
    plt.close(figure)
    manifest.append((stem, operator, component, pdf.name))


def project_to_plane(q, vector):
    return np.array([float(vector @ q["e1"]), float(vector @ q["e2"])])


def draw_spectrum(q, name, role, rank):
    figure, axis = component_figure()
    values = q["eigenvalues"]
    theta = np.linspace(0, 2 * np.pi, 240)
    axis.plot(np.cos(theta), np.sin(theta), color="#bbbbbb", lw=0.8, ls="--")
    axis.axhline(0, color="#eeeeee", lw=0.6)
    axis.axvline(0, color="#eeeeee", lw=0.6)
    axis.scatter(
        values.real, values.imag, s=18, color="#34495e",
        edgecolors="white", linewidths=0.4, zorder=4,
    )
    count = 1 if rank == "rank-1" else 2
    axis.scatter(
        values.real[:count], values.imag[:count], s=78, facecolors="none",
        edgecolors=PRED_COLOR, linewidths=1.35, zorder=5,
    )
    axis.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), aspect="equal")
    axis.set_xlabel("Re", fontsize=12)
    axis.set_ylabel("Im", fontsize=12)
    axis.set_title(f"{name}: {role} [{rank}]", fontsize=13, fontweight="bold", pad=4)
    axis.tick_params(labelsize=11)
    lambdas = "  ".join(f"{value.real:+.2f}" for value in values[:count])
    axis.text(
        0.5, 0.03, rf"$\lambda$: {lambdas}", transform=axis.transAxes,
        fontsize=11, ha="center", color=PRED_COLOR,
    )
    return figure


def configure_pose_axes(axis, limit):
    axis.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal", xticks=[], yticks=[])
    axis.annotate("", xy=(0, limit * 0.9), xytext=(0, 0), arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "#dddddd"})
    axis.text(0.03, limit * 0.9, "fwd", fontsize=14, color="#bbbbbb")
    axis.annotate("", xy=(-limit * 0.9, 0), xytext=(0, 0), arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "#dddddd"})
    axis.text(-limit * 0.88, limit * 0.04, "left", fontsize=14, color="#bbbbbb", ha="right")
    for spine in axis.spines.values():
        spine.set_visible(False)


def draw_pose(q, name, phase):
    figure, axis = plt.subplots(figsize=(3.6, 3.3))
    figure.subplots_adjust(left=0.035, right=0.985, bottom=0.045, top=0.86)
    positive = phase == "positive"
    pose = q["mean_pos_raw"] if positive else q["mean_neg_raw"]
    sign, color, phase_label = (1.0, POS_COLOR, "+c") if positive else (-1.0, NEG_COLOR, "-c")
    count = q["n_positive"] if positive else q["n_negative"]
    active_total = q["n_positive"] + q["n_negative"]
    axis.set_title(f"{name}: empirical pose {phase_label}", fontsize=19, fontweight="bold", pad=7)
    if pose is None:
        empty_axis(
            axis, "n/a", "single-signed operator",
            message_y=0.55, detail_y=0.38, message_size=24, detail_size=15,
            detail_color="#888888", detail_va="baseline",
        )
        return figure

    display_pose = vec16_to_display(pose)
    display_action = vec16_to_display(sign * q["d_raw"])
    limit = 1.1 * float(np.max(np.abs(display_pose)))
    configure_pose_axes(axis, limit)
    positive_limit = 1.1 * float(np.max(np.abs(vec16_to_display(q["mean_pos_raw"]))))
    arrow_scale = 0.24 * positive_limit / (float(np.max(np.abs(vec16_to_display(q["d_raw"])))) + 1e-9)
    points = dict(zip(KP_NAMES, display_pose))
    for first, second in BONES:
        axis.plot(
            [points[first][0], points[second][0]], [points[first][1], points[second][1]],
            color="#9aa0a3", lw=2.4, alpha=0.85, zorder=2,
        )
    for keypoint, position, action in zip(KP_NAMES, display_pose, display_action):
        axis.scatter(*position, s=36, color=KP_COLORS[keypoint], edgecolors="white", linewidths=0.8, zorder=5)
        axis.annotate(
            "", xy=position + arrow_scale * action, xytext=position,
            arrowprops={"arrowstyle": "-|>", "lw": 1.4, "color": color, "mutation_scale": 10},
            zorder=6,
        )
    axis.text(0.02, 0.02, rf"${'+' if positive else '-'}d_{{\rm pred}}$", transform=axis.transAxes, fontsize=16, color=color, va="bottom")
    axis.text(
        0.98, 0.02, f"{phase_label}: {100 * count / max(active_total, 1):.1f}%",
        transform=axis.transAxes, fontsize=15, color="#555555", ha="right", va="bottom",
    )
    return figure


def draw_active_frames(q, name, indices):
    figure = plt.figure(figsize=(4.25, 2.85))
    axis = figure.add_axes([0.11, 0.18, 0.53, 0.70])
    delta = q["xn_all"][:, indices] - q["mean_normalized_pose"][:, None]
    x_coord, y_coord = delta.T @ q["e1"], delta.T @ q["e2"]
    coefficients = q["c"][indices]
    vmax = max(float(np.max(np.abs(coefficients))), 1e-6)
    axis.scatter(
        x_coord, y_coord, c=coefficients, cmap="coolwarm", s=3, alpha=0.28,
        vmin=-vmax, vmax=vmax, linewidths=0, rasterized=True,
    )

    v1, v2, d_pred, d_emp = (
        project_to_plane(q, q[key]) for key in ("v1", "v2", "d_pred", "d_emp")
    )
    for vector, label in [(v1, "$v_1$")] + ([(v2, "$v_2$")] if q["bidirectional"] else []):
        vector_arrow(axis, vector, color="#777777", width=0.95, style="->", mutation=7, zorder=5)
        axis.annotate(label, xy=vector, xytext=(3, 3), textcoords="offset points", fontsize=11, color="#666666", zorder=8)
    vector_arrow(axis, d_pred, color=PRED_COLOR, width=1.6, mutation=8, zorder=6)
    if q["bidirectional"]:
        vector_arrow(axis, -d_pred, color=PRED_COLOR, width=0.8, alpha=0.35, mutation=7, zorder=6)
    vector_arrow(axis, d_emp, color=EMP_COLOR, width=1.2, mutation=8, zorder=7)
    axis.annotate(r"$d_{\rm pred}$", xy=d_pred, xytext=(4, 5), textcoords="offset points", fontsize=11, color=PRED_COLOR, fontweight="bold", zorder=9)
    axis.annotate(r"$d_{\rm emp}$", xy=d_emp, xytext=(4, -10), textcoords="offset points", fontsize=11, color=EMP_COLOR, fontweight="bold", zorder=9)
    axis.axhline(0, color="#bbbbbb", lw=0.45)
    axis.axvline(0, color="#bbbbbb", lw=0.45)

    extent = float(np.percentile(np.hypot(x_coord, y_coord), 95))
    limit = 1.08 * max(
        extent, np.linalg.norm(v1), np.linalg.norm(v2) if q["bidirectional"] else 0,
        np.linalg.norm(d_pred), np.linalg.norm(d_emp), 0.2,
    )
    axis.set(
        xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal",
        xlabel="eigenplane axis 1", ylabel="axis 2",
    )
    axis.xaxis.label.set_size(12)
    axis.yaxis.label.set_size(12)
    axis.tick_params(labelsize=11)
    figure.suptitle(f"{name}: matrix prediction + empirical frames", fontsize=14, fontweight="bold", y=0.98)

    vector_values = (
        rf"$d_{{pred}}=({d_pred[0]:+.2f},{d_pred[1]:+.2f}),\ |d|={np.linalg.norm(q['d_pred']):.2f}$"
        "\n"
        rf"$d_{{emp}}=({d_emp[0]:+.2f},{d_emp[1]:+.2f}),\ |d|={np.linalg.norm(q['d_emp']):.2f}$"
    )
    if q["bidirectional"]:
        annotation = (
            vector_values + "\n"
            rf"$\angle(d_{{pred}},d_{{emp}})={q['angle']:.1f}^\circ$" + "\n"
            rf"$d_{{emp}}$ in plane: {q['in_plane_fraction'] * 100:.0f}%" + "\n"
            rf"corr$(|c|,|d|)={q['magnitude_correlation']:.2f}$"
        )
        edge_color = "#2e7d32"
    else:
        annotation, edge_color = vector_values + "\n" + "baseline / hold; no signed contrast", "#777777"
    figure.text(
        0.68, 0.79, annotation, fontsize=11, va="top",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": edge_color, "alpha": 0.92},
    )
    figure.text(0.68, 0.20, "color: c (blue -, red +)", fontsize=10.5, color="#666666", ha="left", va="bottom")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    return figure


def write_manifest(manifest, quantities):
    rows = ["stem\toperator\tcomponent\tpdf"] + ["\t".join(row) for row in manifest]
    (OUT / "manifest.tsv").write_text("\n".join(rows) + "\n")
    summary = [
        "Figure 2 single-mouse geometry component export",
        f"fit: {RUN.name}",
        f"phase threshold: |c| > {THETA}",
        f"maximum displayed active frames per operator: {MAX_DISPLAY_FRAMES}",
        "",
        "operator\tpositive_active_fraction\tnegative_active_fraction\t"
        "d_pred_plane_1\td_pred_plane_2\td_pred_norm\t"
        "d_emp_plane_1\td_emp_plane_2\td_emp_norm\t"
        "angle_deg\tin_plane_fraction\tcorr_abs_c_abs_d",
    ]
    for _, name, _, _ in OPS:
        q = quantities[name]
        active_total = q["n_positive"] + q["n_negative"]
        d_pred, d_emp = project_to_plane(q, q["d_pred"]), project_to_plane(q, q["d_emp"])
        values = [
            name,
            f"{q['n_positive'] / max(active_total, 1):.6f}",
            f"{q['n_negative'] / max(active_total, 1):.6f}",
            f"{d_pred[0]:.6f}", f"{d_pred[1]:.6f}", f"{np.linalg.norm(q['d_pred']):.6f}",
            f"{d_emp[0]:.6f}", f"{d_emp[1]:.6f}", f"{np.linalg.norm(q['d_emp']):.6f}",
            f"{q['angle']:.6f}", f"{q['in_plane_fraction']:.6f}",
            f"{q['magnitude_correlation']:.6f}",
        ]
        summary.append("\t".join(values))
    (OUT / "statistics.txt").write_text("\n".join(summary) + "\n")


def main():
    operators, coefficients, normalized, raw = load_current_fit()
    quantities = {
        name: compute_operator_quantities(slot, operators, coefficients, normalized, raw)
        for slot, name, _, _ in OPS
    }
    rng = np.random.default_rng(0)
    display_indices = {}
    for _, name, _, _ in OPS:
        active = quantities[name]["active_indices"]
        display_indices[name] = (
            rng.choice(active, MAX_DISPLAY_FRAMES, replace=False)
            if active.size > MAX_DISPLAY_FRAMES else active
        )

    manifest = []
    for _, name, role, rank in OPS:
        q = quantities[name]
        components = [
            ("01_eigenvalue_spectrum", "eigenvalue spectrum", draw_spectrum(q, name, role, rank)),
            ("03_empirical_pose_positive", "empirical +c mean pose", draw_pose(q, name, "positive")),
            ("05_empirical_active_frames", "matrix prediction + active-frame empirical overlay", draw_active_frames(q, name, display_indices[name])),
        ]
        if name != "op14":
            components.insert(
                2,
                ("04_empirical_pose_negative", "empirical -c mean pose", draw_pose(q, name, "negative")),
            )
        for suffix, description, figure in components:
            save_component(figure, f"{name}_{suffix}", manifest, name, description)
    write_manifest(manifest, quantities)
    print(f"Wrote {len(manifest)} published PDF components to:\n{OUT}")
    for _, name, _, _ in OPS:
        q = quantities[name]
        print(
            f"  {name}: angle={q['angle']:.1f} deg, "
            f"in-plane={q['in_plane_fraction'] * 100:.0f}%, "
            f"corr={q['magnitude_correlation']:.2f}"
        )


if __name__ == "__main__":
    main()
