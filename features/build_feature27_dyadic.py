#!/usr/bin/env python3
"""Build the paper's 27-D resident-centered CalMS21 features.

Rows 0-13 encode resident pose, 14-24 encode social distances and 25-26
encode intruder direction. See ``features/README.md`` for the row-level
definition. The published smoothing values are the command defaults.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from dlds_release.paths import RAW_ROOT, results_path


PARTS = (
    "nose",
    "left_ear",
    "right_ear",
    "neck",
    "left_hip",
    "right_hip",
    "tail_base",
)
PART_INDEX = {name: index for index, name in enumerate(PARTS)}
HEAD = [PART_INDEX[name] for name in ("left_ear", "nose", "right_ear", "neck")]
BODY = [PART_INDEX[name] for name in ("neck", "left_hip", "tail_base", "right_hip")]
RAW_NPY_DIR = RAW_ROOT / "calms21_npy"
RAW_STEM = "annotator-id_0__task1_train_mouse{tag}_task1_annotator1__{field}.npy"
CONF_THRESHOLD, BAD_FRAME_FRACTION = 0.5, 0.6


def build_pair_features_27(
    res_kp: np.ndarray,
    intr_kp: np.ndarray,
    eps: float = 1e-8,
    pose_sigma: float | None = 2.0,
) -> np.ndarray:
    """Convert two ``(2, 7, T)`` keypoint arrays to one ``(27, T)`` matrix."""
    if res_kp.shape != intr_kp.shape or res_kp.shape[:2] != (2, 7):
        raise ValueError(f"Expected two (2, 7, T) arrays, got {res_kp.shape} and {intr_kp.shape}")

    # Work as (time, part, xy).
    resident = res_kp.transpose(2, 1, 0)
    intruder = intr_kp.transpose(2, 1, 0)
    frames = resident.shape[0]
    res_centroid = np.nanmean(resident, axis=1)
    intr_centroid = np.nanmean(intruder, axis=1)

    def distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum((first - second) ** 2, axis=-1) + eps)

    heading = resident[:, PART_INDEX["neck"]] - res_centroid
    heading /= np.linalg.norm(heading, axis=1, keepdims=True) + eps
    cosine, sine = heading.T

    def to_ego(vectors: np.ndarray) -> np.ndarray:
        x, y = vectors[..., 0], vectors[..., 1]
        shape = (frames,) + (1,) * (x.ndim - 1)
        c, s = cosine.reshape(shape), sine.reshape(shape)
        return np.stack((c * x + s * y, -s * x + c * y), axis=-1)

    pose = to_ego(resident - res_centroid[:, None]).reshape(frames, 14).T
    centroid_distance = distance(res_centroid, intr_centroid)[None]
    paired_distances = distance(resident, intruder).T

    intruder_head = np.nanmean(intruder[:, HEAD], axis=1)
    intruder_body = np.nanmean(intruder[:, BODY], axis=1)
    resident_nose = resident[:, PART_INDEX["nose"]]
    social_distances = np.vstack(
        (
            distance(resident_nose, intruder_head),
            distance(resident_nose, intruder_body),
            distance(resident_nose, intruder[:, PART_INDEX["tail_base"]]),
        )
    )
    direction = to_ego(intr_centroid - res_centroid).T

    features = np.vstack(
        (pose, centroid_distance, paired_distances, social_distances, direction)
    )
    if pose_sigma is not None and pose_sigma > 0:
        features[:14] = gaussian_filter1d(
            features[:14], sigma=pose_sigma, axis=1, mode="nearest"
        )
    if features.shape != (27, frames):
        raise RuntimeError(f"Expected (27, T), got {features.shape}")
    return features


def qc_confidence(
    keypoints: np.ndarray,
    scores: np.ndarray,
    label: np.ndarray | None = None,
    conf_thresh: float = CONF_THRESHOLD,
    bad_frac: float = BAD_FRAME_FRACTION,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Interpolate low-confidence samples and return the majority-bad frame mask."""
    clean = keypoints.astype(float).copy()
    bad_mask = (scores < conf_thresh).mean(axis=(0, 1)) > bad_frac
    time = np.arange(keypoints.shape[-1])

    for mouse in range(2):
        for part in range(7):
            bad = scores[mouse, part] < conf_thresh
            good = ~bad
            if bad.any() and good.sum() >= 2:
                for xy in range(2):
                    clean[mouse, xy, part, bad] = np.interp(
                        time[bad], time[good], clean[mouse, xy, part, good]
                    )

    if label is None:
        return clean, None, bad_mask
    label_clean = np.asarray(label).copy()
    good = ~bad_mask
    if bad_mask.any() and good.sum() >= 2:
        label_clean[bad_mask] = np.rint(
            np.interp(time[bad_mask], time[good], label_clean[good].astype(float))
        ).astype(label_clean.dtype)
    return clean, label_clean, bad_mask


def smooth_keypoints_simple(
    keypoints: np.ndarray, sigma: float | None = 2.0
) -> np.ndarray:
    """Gaussian-smooth ``(2, 2, 7, T)`` keypoints along time."""
    if keypoints.shape[:3] != (2, 2, 7):
        raise ValueError(f"Expected (2, 2, 7, T), got {keypoints.shape}")
    if sigma is None or sigma <= 0:
        return keypoints.copy()
    return gaussian_filter1d(keypoints, sigma=sigma, axis=-1, mode="nearest")


def _raw_path(mouse_id: int, field: str) -> Path:
    return RAW_NPY_DIR / RAW_STEM.format(tag=f"{mouse_id:03d}", field=field)


def export_session(
    mouse_id: int,
    *,
    kp_sigma: float,
    pose_sigma: float,
    out_dir: Path,
    overwrite: bool,
) -> dict:
    """Build and save one session."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{mouse_id:03d}"
    outputs = {
        "feature": out_dir / f"FEATURE27_mouse{tag}.npy",
        "label": out_dir / f"cleaned_label_mouse{tag}.npy",
        "mask": out_dir / f"bad_mask_mouse{tag}.npy",
    }
    if not overwrite and all(path.exists() for path in outputs.values()):
        feature, bad_mask = np.load(outputs["feature"]), np.load(outputs["mask"])
        return _summary(mouse_id, feature, bad_mask, skipped=True)

    keypoints = np.load(_raw_path(mouse_id, "keypoints"))
    scores = np.load(_raw_path(mouse_id, "scores"))
    labels = np.load(_raw_path(mouse_id, "annotations"))
    keypoints, labels, bad_mask = qc_confidence(keypoints, scores, labels)
    keypoints = smooth_keypoints_simple(keypoints, kp_sigma)
    feature = build_pair_features_27(
        keypoints[0], keypoints[1], pose_sigma=pose_sigma
    )
    arrays = {"feature": feature, "label": labels, "mask": bad_mask}
    for name, path in outputs.items():
        np.save(path, arrays[name])
    return _summary(mouse_id, feature, bad_mask, skipped=False)


def _summary(
    mouse_id: int, feature: np.ndarray, bad_mask: np.ndarray, *, skipped: bool
) -> dict:
    return {
        "mouse_id": mouse_id,
        "n_frames": int(feature.shape[1]),
        "n_dim": int(feature.shape[0]),
        "n_bad_frames": int(bad_mask.sum()),
        "skipped_existing": skipped,
    }


def parse_session_ids(raw: str, *, exclude: set[int] | None = None) -> list[int]:
    """Parse comma-separated ids and inclusive ranges such as ``1:10,20``."""
    raw = raw.strip() or "1:70"
    ids: set[int] = set()
    for token in raw.split(","):
        if not token.strip():
            continue
        if ":" not in token:
            ids.add(int(token))
            continue
        start, stop = map(int, token.split(":", 1))
        step = 1 if stop >= start else -1
        ids.update(range(start, stop + step, step))
    return sorted(ids - (exclude or set()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kp-sigma", type=float, default=1.0)
    parser.add_argument("--pose-sigma", type=float, default=0.5)
    parser.add_argument("--session-ids", default="1:70")
    parser.add_argument("--exclude-36", action="store_true")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_ids = parse_session_ids(
        args.session_ids, exclude={36} if args.exclude_36 else set()
    )
    if not session_ids:
        raise SystemExit("No session ids selected")

    kp_tag = str(args.kp_sigma).replace(".", "p")
    pose_tag = str(args.pose_sigma).replace(".", "p")
    output = args.out_dir or results_path(
        f"feature_inputs_feature27_kp{kp_tag}_pose{pose_tag}"
    )
    existing = list(output.glob("FEATURE27_mouse*.npy"))
    if existing and not (args.overwrite or args.reuse_existing):
        raise SystemExit(
            f"{output} already contains {len(existing)} feature files; use a new "
            "results_root, --reuse-existing or --overwrite."
        )

    rows = []
    for mouse_id in session_ids:
        row = export_session(
            mouse_id,
            kp_sigma=args.kp_sigma,
            pose_sigma=args.pose_sigma,
            out_dir=output,
            overwrite=args.overwrite,
        )
        rows.append(row)
        state = "reused" if row["skipped_existing"] else "saved"
        print(
            f"mouse{mouse_id:03d}: {state} shape=(27,{row['n_frames']}) "
            f"bad_frames={row['n_bad_frames']}"
        )

    params = {
        "kp_sigma": args.kp_sigma,
        "pose_sigma": args.pose_sigma,
        "session_ids": session_ids,
        "exclude_36": args.exclude_36,
        "confidence_threshold": CONF_THRESHOLD,
        "bad_frame_fraction": BAD_FRAME_FRACTION,
        "source_raw_root": str(RAW_NPY_DIR),
        "output_dir": str(output),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "export_params.json").write_text(json.dumps(params, indent=2))
    (output / "session_summary.json").write_text(json.dumps(rows, indent=2))
    print(f"Saved {len(rows)} sessions to {output}")


if __name__ == "__main__":
    main()
