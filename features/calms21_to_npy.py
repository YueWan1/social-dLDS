#!/usr/bin/env python3
"""Convert CalMS21 Task-1 JSON to the arrays consumed by FEATURE27.

The public JSON stores time first; the feature builder expects time last:
keypoints ``(2, 2, 7, T)``, scores ``(2, 7, T)`` and annotations ``(T,)``.
Schema, naming and verification details are documented in ``features/README.md``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from dlds_release.paths import RAW_ROOT


GROUP = "annotator-id_0"
SEQUENCE_RE = re.compile(r"^task1/train/mouse(\d{3})_task1_annotator1$")
SESSION_IDS = tuple(range(1, 71))
N_MICE, N_XY, N_PARTS = 2, 2, 7
IMAGE_W, IMAGE_H = 1024, 570
VOCAB = {"attack": 0, "investigation": 1, "mount": 2, "other": 3}
FIELDS = ("keypoints", "scores", "annotations")


class SchemaError(RuntimeError):
    """Input does not match the CalMS21 Task-1 training schema."""


def raw_download_root() -> Path:
    return RAW_ROOT / "calms21"


def default_json() -> Path | None:
    root = raw_download_root()
    hits = sorted(root.rglob("calms21_task1_train.json")) if root.exists() else []
    return hits[0] if hits else None


def default_out_dir() -> Path:
    return RAW_ROOT / "calms21_npy"


def stem_for(session_id: int) -> str:
    return (
        f"{GROUP}__task1_train_mouse{session_id:03d}_task1_annotator1__"
    )


def parse_session_id(sequence_key: str) -> int:
    match = SEQUENCE_RE.fullmatch(sequence_key)
    if match is None:
        raise SchemaError(
            f"Unexpected sequence key {sequence_key!r}; expected the Task-1 train split."
        )
    return int(match.group(1))


def _check_orientation(keypoints: np.ndarray, sequence_key: str) -> None:
    """Use the 1024x570 frame asymmetry to catch an x/y transposition."""
    with np.errstate(invalid="ignore"):
        x99 = float(np.nanpercentile(keypoints[:, :, 0], 99))
        y99 = float(np.nanpercentile(keypoints[:, :, 1], 99))
    if x99 > IMAGE_W * 1.05 or y99 > IMAGE_H * 1.05:
        raise SchemaError(
            f"{sequence_key}: coordinates exceed the {IMAGE_W}x{IMAGE_H} frame "
            f"(x99={x99:.0f}, y99={y99:.0f}); check axis order."
        )


def convert_sequence(
    sequence_key: str,
    payload: dict,
    *,
    check_orientation: bool = True,
) -> dict[str, np.ndarray]:
    """Convert one sequence without changing values."""
    missing = set(FIELDS) - payload.keys()
    if missing:
        raise SchemaError(f"{sequence_key}: missing {sorted(missing)}")

    keypoints = np.asarray(payload["keypoints"], dtype=np.float64)
    scores = np.asarray(payload["scores"], dtype=np.float64)
    annotations = np.asarray(payload["annotations"])
    expected_tail = (N_MICE, N_XY, N_PARTS)
    if keypoints.ndim != 4 or keypoints.shape[1:] != expected_tail:
        raise SchemaError(
            f"{sequence_key}: keypoints {keypoints.shape}, expected (T, 2, 2, 7)"
        )

    frames = keypoints.shape[0]
    if scores.shape != (frames, N_MICE, N_PARTS):
        raise SchemaError(
            f"{sequence_key}: scores {scores.shape}, expected ({frames}, 2, 7)"
        )
    if annotations.shape != (frames,):
        raise SchemaError(
            f"{sequence_key}: annotations {annotations.shape}, expected ({frames},)"
        )
    if not np.isfinite(scores).all() or not np.all((0 <= scores) & (scores <= 1)):
        raise SchemaError(f"{sequence_key}: confidence scores must be finite and in [0, 1]")
    if not set(np.unique(annotations)) <= set(VOCAB.values()):
        raise SchemaError(f"{sequence_key}: annotation ids are outside {VOCAB}")

    metadata_vocab = (payload.get("metadata") or {}).get("vocab")
    if metadata_vocab is not None and dict(metadata_vocab) != VOCAB:
        raise SchemaError(f"{sequence_key}: metadata vocabulary differs from {VOCAB}")
    if check_orientation:
        _check_orientation(keypoints, sequence_key)

    return {
        "keypoints": np.ascontiguousarray(keypoints.transpose(1, 2, 3, 0)),
        "scores": np.ascontiguousarray(scores.transpose(1, 2, 0)),
        "annotations": np.ascontiguousarray(annotations, dtype=np.int64),
    }


def convert_json(
    json_path: Path,
    out_dir: Path,
    *,
    expect_sessions: tuple[int, ...] | None = SESSION_IDS,
    check_orientation: bool = True,
    overwrite: bool = False,
    verbose: bool = True,
) -> dict[int, int]:
    """Convert the complete train split and return ``{session_id: frames}``."""
    if verbose:
        print(f"Reading {json_path} (requires several GB of RAM)", flush=True)
    with json_path.open() as handle:
        blob = json.load(handle)
    if GROUP not in blob:
        raise SchemaError(f"Top-level key {GROUP!r} not found")

    out_dir.mkdir(parents=True, exist_ok=True)
    frame_counts: dict[int, int] = {}
    for sequence_key, payload in sorted(blob[GROUP].items()):
        session_id = parse_session_id(sequence_key)
        if session_id in frame_counts:
            raise SchemaError(f"Session {session_id:03d} appears twice")
        arrays = convert_sequence(
            sequence_key, payload, check_orientation=check_orientation
        )
        for field, array in arrays.items():
            output = out_dir / f"{stem_for(session_id)}{field}.npy"
            if overwrite or not output.exists():
                np.save(output, array)
        frame_counts[session_id] = len(arrays["annotations"])
        if verbose:
            print(f"mouse{session_id:03d}: T={frame_counts[session_id]}")

    if expect_sessions is not None and tuple(sorted(frame_counts)) != expect_sessions:
        raise SchemaError(
            f"Expected sessions 001-070; found {sorted(frame_counts)}"
        )
    if verbose:
        print(f"Wrote {3 * len(frame_counts)} arrays to {out_dir}")
    return frame_counts


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="Path to calms21_task1_train.json")
    parser.add_argument("--out-dir", type=Path, help="Default: <raw_data_root>/calms21_npy")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--no-orientation-check", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    json_path = args.json or default_json()
    if json_path is None or not json_path.exists():
        raise SystemExit(
            f"CalMS21 train JSON not found under {raw_download_root()}; "
            "run 'bash features/fetch_calms21.sh' or pass --json."
        )
    convert_json(
        json_path,
        args.out_dir or default_out_dir(),
        expect_sessions=None if args.allow_partial else SESSION_IDS,
        check_orientation=not args.no_orientation_check,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
