"""Build the paper's 16-D single-mouse features from DeepLabCut tracking.

Eight keypoints are centered, rotated so nose-to-spine4 points along +x, then
flattened as part-major ``(x, y)`` pairs. Cleaning and QC match the feature
inputs used for the paper; see ``features/README.md`` for the full definition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dlds_release.paths import RAW_ROOT, results_path


DATA_DIR = RAW_ROOT / "keypoint_moseq_zenodo_15171335/open_field_2D/videos"
OUT_DIR = results_path("feature_inputs_kpmoseq")
RAW_PARTS = (
    "tail", "spine4", "spine3", "spine2", "spine1",
    "head", "nose", "right ear", "left ear",
)
USE_PARTS = (
    "spine4", "spine3", "spine2", "spine1",
    "head", "nose", "right ear", "left ear",
)
USE_INDICES = [RAW_PARTS.index(part) for part in USE_PARTS]
OUTLIER_SCALE = 6.0
CONF_THRESHOLD = 0.5
LONG_GAP_FRAMES = 30
HEADING_ANCHOR_QC = 5.0


def load_dlc_h5(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return coordinates ``(T, 9, 2)`` and confidences ``(T, 9)``."""
    frame = pd.read_hdf(path)
    scorer = "dummyscorer"
    coordinates = np.stack(
        [
            np.column_stack(
                (frame[(scorer, part, "x")], frame[(scorer, part, "y")])
            )
            for part in RAW_PARTS
        ],
        axis=1,
    )
    confidence = np.column_stack(
        [frame[(scorer, part, "likelihood")] for part in RAW_PARTS]
    )
    return coordinates, confidence


def mark_outliers(
    coordinates: np.ndarray,
    confidence: np.ndarray,
    scale: float = OUTLIER_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero confidence outside median distance + ``scale`` times MAD."""
    distances = np.linalg.norm(
        coordinates - np.median(coordinates, axis=1, keepdims=True), axis=-1
    )
    median = np.median(distances, axis=0)
    threshold = median + scale * np.median(np.abs(distances - median), axis=0)
    outliers = distances > threshold
    cleaned = confidence.copy()
    cleaned[outliers] = 0.0
    return cleaned, outliers


def interp_bad_frames(
    coordinates: np.ndarray,
    confidence: np.ndarray,
    thr: float = CONF_THRESHOLD,
) -> np.ndarray:
    """Interpolate low-confidence or non-finite keypoint positions over time."""
    output = coordinates.copy()
    time = np.arange(len(output))
    for keypoint in range(output.shape[1]):
        bad = (confidence[:, keypoint] < thr) | np.isnan(
            output[:, keypoint]
        ).any(axis=1)
        good = ~bad
        if bad.any() and good.any():
            for xy in range(2):
                output[bad, keypoint, xy] = np.interp(
                    time[bad], time[good], output[good, keypoint, xy]
                )
    return output


def egocentric_align(
    coordinates: np.ndarray,
    anterior_idxs: list[int],
    posterior_idxs: list[int],
) -> np.ndarray:
    """Center each frame and rotate its anterior-posterior axis to +x."""
    centroid = np.median(coordinates, axis=1)
    forward = (
        coordinates[:, anterior_idxs].mean(axis=1)
        - coordinates[:, posterior_idxs].mean(axis=1)
    )
    heading = np.arctan2(forward[:, 1], forward[:, 0])
    cosine, sine = np.cos(-heading), np.sin(-heading)
    rotation = np.stack(
        (
            np.stack((cosine, -sine), axis=1),
            np.stack((sine, cosine), axis=1),
        ),
        axis=1,
    )
    aligned = np.einsum(
        "tij,tkj->tki", rotation, coordinates - centroid[:, None]
    )
    return aligned


def long_gap_fraction(
    bad_mask: np.ndarray, min_gap: int = LONG_GAP_FRAMES
) -> float:
    """Fraction of frames inside bad runs longer than ``min_gap``."""
    edges = np.diff(np.r_[False, bad_mask, False].astype(int))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return float(lengths[lengths > min_gap].sum() / len(bad_mask))


def process_session(path: Path, verbose: bool = True) -> dict:
    coordinates, confidence = load_dlc_h5(path)
    confidence, _ = mark_outliers(coordinates, confidence)
    coordinates = coordinates[:, USE_INDICES]
    confidence = confidence[:, USE_INDICES]
    interpolated = interp_bad_frames(coordinates, confidence)

    bad = confidence < CONF_THRESHOLD
    anchor = {name: USE_PARTS.index(name) for name in ("nose", "spine4")}
    qc = {
        f"{name}_low_pct": 100 * float(bad[:, index].mean())
        for name, index in anchor.items()
    }
    qc.update(
        {
            f"{name}_long_pct": 100 * long_gap_fraction(bad[:, index])
            for name, index in anchor.items()
        }
    )
    qc["pass_qc"] = all(
        qc[f"{name}_long_pct"] <= HEADING_ANCHOR_QC for name in anchor
    )

    aligned = egocentric_align(
        interpolated, [anchor["nose"]], [anchor["spine4"]]
    )
    features = aligned.reshape(len(aligned), -1).T
    if verbose:
        state = "PASS" if qc["pass_qc"] else "DROP"
        print(
            f"{path.name[:38]:38s} T={len(aligned):6d} {state:4s} "
            f"nose={qc['nose_long_pct']:.2f}% spine4={qc['spine4_long_pct']:.2f}%"
        )
    return {"features": features, "qc": qc}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _session_id(path: Path) -> str:
    return path.stem.split("DLC", 1)[0].removesuffix(".top.ir").rstrip(".")


def main() -> None:
    args = parse_args()
    data_dir, output = args.data_dir.resolve(), args.out_dir.resolve()
    sessions = sorted(data_dir.glob("*.h5"))
    if not sessions:
        raise SystemExit(
            f"No DeepLabCut .h5 files in {data_dir}; run "
            "'bash features/fetch_zenodo_kpms.sh' or pass --data-dir."
        )
    existing = list(output.glob("FEATURE16_kpmoseq_*.npy"))
    if existing and not args.overwrite:
        raise SystemExit(
            f"{output} already contains {len(existing)} feature files; use a "
            "new results_root or pass --overwrite."
        )

    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sessions:
        result = process_session(path)
        session_id = _session_id(path)
        stem = f"FEATURE16_kpmoseq_{session_id}"
        np.save(output / f"{stem}.npy", result["features"])
        rows.append(
            {"session_id": session_id, "T": result["features"].shape[1], **result["qc"]}
        )

    qc = pd.DataFrame(rows)
    qc.to_csv(output / "SESSION_QC.csv", index=False)
    passed = qc["pass_qc"]
    print(
        f"Pass: {int(passed.sum())} sessions; drop: {int((~passed).sum())}; "
        f"QC table: {output / 'SESSION_QC.csv'}"
    )
    for row in qc.loc[~passed].itertuples():
        print(
            f"drop {row.session_id}: nose={row.nose_long_pct:.2f}% "
            f"spine4={row.spine4_long_pct:.2f}%"
        )


if __name__ == "__main__":
    main()
