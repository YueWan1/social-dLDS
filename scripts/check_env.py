#!/usr/bin/env python3
"""Check dependencies, configured paths, and tracked derived artifacts."""

import importlib
import importlib.util
import shutil
import sys
from pathlib import Path

ok = True
print("interpreter :", sys.executable)
print("python      :", sys.version.split()[0])
print()

# (module, distribution name, what breaks without it)
CORE = [
    ("numpy",        "numpy",        "everything"),
    ("scipy",        "scipy",        "statistics, signal, linalg"),
    ("pandas",       "pandas",       "DeepLabCut .h5 and the CSV tables"),
    ("matplotlib",   "matplotlib",   "every panel"),
    ("sklearn",      "scikit-learn", "the LOSO decoders (Fig 6e, S9)"),
    ("h5py",         "h5py",         "keypoint-MoSeq results.h5"),
    ("tables",       "pytables",     "pandas.read_hdf on the DLC .h5 -- NOT optional"),
    ("seaborn",      "seaborn",      "Figure 3 panels"),
    ("PIL",          "pillow",       "Figure 1a grid and Supplementary S9 row stitch"),
    ("cv2",          "opencv",       "clip decoding for Fig 1a / 1e"),
]
print("dependencies")
for mod, dist, why in CORE:
    try:
        m = importlib.import_module(mod)
        print("  OK      %-11s %s" % (mod, getattr(m, "__version__", "?")))
    except Exception as exc:
        ok = False
        print("  MISSING %-11s install %-13s (%s) [%s]" % (mod, dist, why, type(exc).__name__))

# Optional: only three panels need it, so its absence is not a failure.
if importlib.util.find_spec("keypoint_moseq") is not None:
    print("  present keypoint_moseq  (not imported; avoids initializing JAX)")
else:
    print("  absent  keypoint_moseq  -> Fig 1c, 1e, 4b skip; pip install -e '.[moseq]'")
print()

print("package modules")
for mod in ("dlds_release.paths", "dlds_release.plotting",
            "dlds_release.empirical_wireframe", "dlds_release.lds_topchain",
            "dlds_release.kpms_clip_video_frames"):
    try:
        importlib.import_module(mod)
        print("  OK      %s" % mod)
    except Exception as exc:
        # Reported rather than hidden: kpms_clip_video_frames loads the feature
        # builder out of features/ by path, so it is the one package module
        # that can fail because of where the repository sits.
        print("  FAILED  %s  (%s: %s)" % (mod, type(exc).__name__, exc))
        ok = False
print()

from dlds_release import paths
print("roots")
print("  REPO :", paths.REPO)
print("  ROOT :", paths.ROOT, "" if paths.ROOT.exists() else " <- DOES NOT EXIST (see paths.yml)")
print("  RAW  :", paths.RAW_ROOT)
print("  RESULTS:", paths.RESULTS_ROOT)
print("  OUT  :", paths.OUT)
print()

print("data root contents")
PROBES = [("dyadic coefficients", paths.dyadic_cs_dir),
          ("single-mouse fit", paths.single_fit_dir),
          ("FEATURE27", paths.feature27_dir),
          ("FEATURE16", paths.feature16_dir),
          ("cleaned keypoints", paths.preprocessed_dir),
          ("MoSeq single", lambda: paths.moseq_results("single")),
          ("MoSeq dyadic", lambda: paths.moseq_results("dyadic"))]
for name, fn in PROBES:
    try:
        print("  OK      %-20s %s" % (name, fn()))
    except FileNotFoundError as exc:
        hint = [l.strip() for l in str(exc).splitlines() if "obtain it with" in l]
        print("  absent  %-20s %s" % (name, hint[0] if hint else "not downloaded"))
print()

print("derived/ artifacts (these ship in git and must load)")
import numpy as np
for name, getter, want in (("dyadic F_universal.npy", paths.dyadic_dictionary, (15, 27, 27)),
                           ("single-mouse Fs.npy", paths.single_dictionary, (15, 16, 16))):
    try:
        arr = np.load(getter())
        good = tuple(arr.shape) == want
        ok = ok and good
        print("  %s %-24s shape %s%s" % ("OK     " if good else "WRONG  ", name, arr.shape,
                                         "" if good else "  expected %s" % (want,)))
    except Exception as exc:
        ok = False
        print("  FAILED  %-24s %s" % (name, exc))
for rel in ("dyadic/signed_selectivity_ztable.npz",
            "dyadic/syllable_behavior_selectivity.npz",
            "dyadic/readout_reproducibility.npz"):
    try:
        with np.load(paths.DERIVED / rel) as z:
            print("  OK      %-24s %d arrays" % (Path(rel).name, len(z.files)))
    except Exception as exc:
        ok = False
        print("  FAILED  %-24s %s" % (Path(rel).name, exc))
for rel in ("dyadic/loso_stability_values.csv", "single_mouse/SESSION_QC.csv"):
    p = paths.DERIVED / rel
    if p.exists():
        print("  OK      %-24s %d lines" % (Path(rel).name, len(p.read_text().splitlines())))
    else:
        ok = False
        print("  MISSING %-24s %s" % (Path(rel).name, p))
print()

print("external tools")
for tool, why in (("pdflatex", "bash run_figures.sh assemble, fig02 panel b step 2"),
                  ("pdfinfo", "fig02 panel b step 2"),
                  ("julia", "bash reproduce.sh fit-one")):
    where = shutil.which(tool)
    print("  %s %-11s %s" % ("OK     " if where else "absent ", tool, where or "(" + why + ")"))
print()

print("check-env:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
