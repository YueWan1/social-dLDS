#!/usr/bin/env python3
"""Return success when an optional paper artifact is available."""

from __future__ import annotations

import sys

from dlds_release.paths import moseq_results, preprocessed_dir


def available(name: str) -> bool:
    try:
        if name == "moseq_single":
            return moseq_results("single").is_file()
        if name == "moseq_dyadic":
            return moseq_results("dyadic").is_file()

        root = preprocessed_dir()
        if name == "preprocessed_res_kp":
            return any(root.glob("mouse*/res_kp.npy"))
        if name == "preprocessed_kp_clean":
            return any(root.glob("mouse*/kp_clean.npy"))
    except FileNotFoundError:
        return False

    raise ValueError(f"unknown optional input: {name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_optional_input.py ARTIFACT")
    raise SystemExit(0 if available(sys.argv[1]) else 1)
