"""Path resolution for the social-dLDS release.

Every public entry point imports paths from this module.  Four locations are
kept separate because they have different lifecycles:

``REPO``
    This repository.  Holds code and the small derived artifacts under
    ``derived/`` that ship in git.

``ROOT``
    The *data root* where the paper's Zenodo bundles are unpacked.  The bundles
    preserve the original ``dLDS/results/`` layout.

``RAW_ROOT``
    Third-party source datasets downloaded from CaltechDATA and Zenodo.

``RESULTS_ROOT``
    Feature matrices, fitted models and other analysis artifacts.  It defaults
    to ``ROOT / "dLDS/results"`` for the deposited-artifact workflow.  Set it
    to a separate writable directory for a full refit so the deposited files
    remain untouched.

``OUT``
    Regenerated figures and recomputed tables.  This is disposable output and
    is never used as a model-input directory.

Resolution order for each root: environment variable, then ``paths.yml``,
then a default inside the repository.

    SOCIAL_DLDS_ROOT          -> ROOT
    SOCIAL_DLDS_RAW_ROOT      -> RAW_ROOT
    SOCIAL_DLDS_RESULTS_ROOT  -> RESULTS_ROOT
    SOCIAL_DLDS_OUT           -> OUT

The named accessors below (``dyadic_cs_dir()``, ``moseq_results()``, ...) are the
preferred way to reach the canonical artifacts; they carry the exact
hyperparameter-stamped directory names used for the published fits, so a script
cannot accidentally pick up a neighbouring sweep.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "REPO",
    "ROOT",
    "RAW_ROOT",
    "RESULTS_ROOT",
    "OUT",
    "DERIVED",
    "require",
    "results_path",
    "dyadic_cs_dir",
    "dyadic_dictionary",
    "single_fit_dir",
    "single_dictionary",
    "moseq_results",
    "feature27_dir",
    "feature16_dir",
    "preprocessed_dir",
    "out_dir",
]

REPO = Path(__file__).resolve().parents[1]

# Published fit locations relative to RESULTS_ROOT. Changing these stamps
# selects a different fitted model for every downstream analysis and figure.

_DYADIC_STAMP = (
    "M15_l1_0.4_smooth_0.15_Flr_2.0_FlrDecay_0.997_decorr_0.1_D27_iter300_snips350"
)
_SINGLE_STAMP = (
    "M15_l1_0.65_smooth_0.3_Flr_2.0_FlrDecay_0.996_decorr_0.16_D16_iter300_snips300"
)

_REL = {
    "dyadic_cs": f"universal_feature27_kp1p0_pose0p5_meanF30_cs/{_DYADIC_STAMP}/infer_l1_0.4_smooth_0.15",
    "single_fit": f"kpmoseq_feature16_joint/{_SINGLE_STAMP}",
    "moseq_single": "kpms_repro/sweep_single/k05000/kpms_project/2026_06_13-13_58_43/results.h5",
    "moseq_dyadic": "kpms_repro_calms21/sweep/k3e04/kpms_project/2026_06_07-01_49_11/results.h5",
    "feature27": "feature_inputs_feature27_kp1p0_pose0p5",
    "feature16": "feature_inputs_kpmoseq",
    "preprocessed": "preprocessed",
}

_MOSEQ_RUN_DIR = {
    "single": "kpms_repro/sweep_single/k05000",
    "dyadic": "kpms_repro_calms21/sweep/k3e04",
}

_DRIVE_MODELS = (
    "https://drive.google.com/file/d/"
    "1wBcfj0d4gs-eSbJVkumU--NDH6VPbbH8/view?usp=drive_link"
)

# Concrete next steps for missing inputs. Some paper artifacts cannot currently
# be redistributed; saying so is preferable to pointing to a nonexistent file.
_PROVENANCE = {
    "dyadic_cs": (
        f"download dlds_derived_models.tar.zst from {_DRIVE_MODELS} and extract "
        "it into data_root (see docs/DATA_AVAILABILITY.md)"
    ),
    "single_fit": (
        f"download dlds_derived_models.tar.zst from {_DRIVE_MODELS} and extract "
        "it into data_root (see docs/DATA_AVAILABILITY.md)"
    ),
    "moseq_single": (
        "not currently distributed; exact MoSeq-dependent panels require the "
        "paper results.h5 (see docs/DATA_AVAILABILITY.md)"
    ),
    "moseq_dyadic": (
        "not currently distributed; exact MoSeq-dependent panels require the "
        "paper results.h5 (see docs/DATA_AVAILABILITY.md)"
    ),
    "feature27": (
        "bash features/fetch_calms21.sh; bash reproduce.sh convert-calms21; "
        "bash reproduce.sh features-dyadic"
    ),
    "feature16": (
        "bash features/fetch_zenodo_kpms.sh; "
        "bash reproduce.sh features-single"
    ),
    "preprocessed": (
        "not currently distributed; cleaned-keypoint tasks are skipped "
        "(see docs/DATA_AVAILABILITY.md)"
    ),
}


def _config() -> dict:
    """Read paths.yml if it exists.

    Parsed by hand rather than with PyYAML: this module is imported by every
    script including the ones that run in the bare-python environment, and a
    four-key flat file does not justify a dependency.
    """
    cfg_file = REPO / "paths.yml"
    if not cfg_file.exists():
        return {}
    out = {}
    for line in cfg_file.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.strip().strip("'\"")
        if value:
            out[key.strip()] = value
    return out


_CFG = _config()


def _root(env_var: str, cfg_key: str, default: Path) -> Path:
    if value := os.environ.get(env_var):
        return Path(value).expanduser().resolve()
    if value := _CFG.get(cfg_key):
        path = Path(value).expanduser()
        # Relative entries belong to the configuration file, not to whichever
        # directory happened to be current when a script was launched.
        return (REPO / path).resolve() if not path.is_absolute() else path.resolve()
    return default.resolve()


ROOT = _root("SOCIAL_DLDS_ROOT", "data_root", REPO / "data_root")
RAW_ROOT = _root("SOCIAL_DLDS_RAW_ROOT", "raw_data_root", ROOT / "data")
RESULTS_ROOT = _root(
    "SOCIAL_DLDS_RESULTS_ROOT",
    "results_root",
    ROOT / "dLDS" / "results",
)
OUT = _root("SOCIAL_DLDS_OUT", "out_root", REPO / "out")
DERIVED = REPO / "derived"


def require(path: Path, what: str = "") -> Path:
    """Return ``path``, or fail with a message that says how to obtain it.

    A missing input is the most common way a reproduction attempt stalls, and a
    bare FileNotFoundError three frames deep does not tell the reader whether
    they skipped a download or mis-set a root.
    """
    path = Path(path)
    if path.exists():
        return path
    hint = _PROVENANCE.get(what, "")
    lines = [
        f"Required input not found: {path}",
        f"  data root (ROOT) = {ROOT}",
        f"  results root     = {RESULTS_ROOT}",
    ]
    if hint:
        lines.append(f"  next step:         {hint}")
    if not ROOT.exists():
        lines.append(
            "  ROOT itself does not exist. Set it in paths.yml "
            "(copy paths.example.yml) or export SOCIAL_DLDS_ROOT."
        )
    raise FileNotFoundError("\n".join(lines))


def results_path(*parts: str) -> Path:
    """Return a path under the configured results tree without creating it."""
    return RESULTS_ROOT.joinpath(*parts)


# Named accessors


def dyadic_cs_dir() -> Path:
    """Per-session dyadic coefficients: F_universal.npy + 69 cs_mouse<NNN>.npy.

    mouse036 is genuinely absent (T=96 frames, shorter than the 200-frame
    snippet length), so this directory holds 69 sessions, not 70.
    """
    return require(RESULTS_ROOT / _REL["dyadic_cs"], "dyadic_cs")


def dyadic_dictionary() -> Path:
    """The dyadic operator dictionary, (15, 27, 27).

    Ships in the repository under ``derived/dyadic/`` so the operator-geometry
    claims are checkable without any download; falls back to the data root.
    """
    shipped = DERIVED / "dyadic" / "F_universal.npy"
    if shipped.exists():
        return shipped
    return require(dyadic_cs_dir() / "F_universal.npy", "dyadic_cs")


def single_fit_dir() -> Path:
    """Single-mouse joint fit: Fs.npy + 5 cs_<session>.npy."""
    return require(RESULTS_ROOT / _REL["single_fit"], "single_fit")


def single_dictionary() -> Path:
    """The single-mouse operator dictionary, (15, 16, 16)."""
    shipped = DERIVED / "single_mouse" / "Fs.npy"
    if shipped.exists():
        return shipped
    return require(single_fit_dir() / "Fs.npy", "single_fit")


def moseq_results(dataset: str) -> Path:
    """keypoint-MoSeq syllable labels for the published fits.

    ``dataset='single'`` is the kappa=5000 / 21-substantive-syllable model;
    ``dataset='dyadic'`` is the kappa=3e4 / 28-substantive-syllable model.
    """
    if dataset not in ("single", "dyadic"):
        raise ValueError(f"dataset must be 'single' or 'dyadic', got {dataset!r}")
    key = f"moseq_{dataset}"
    canonical = RESULTS_ROOT / _REL[key]
    if canonical.exists():
        return canonical

    # A fresh keypoint-MoSeq fit uses a timestamped model directory. The sweep
    # driver records that name next to the project so downstream code does not
    # need a source edit after every refit.
    run_dir = RESULTS_ROOT / _MOSEQ_RUN_DIR[dataset]
    model_name_file = run_dir / "MODEL_NAME.txt"
    if model_name_file.exists():
        model_name = model_name_file.read_text().strip()
        generated = run_dir / "kpms_project" / model_name / "results.h5"
        return require(generated, key)
    return require(canonical, key)


def feature27_dir() -> Path:
    """Dyadic 27-D features: SELF(14) + DIST(11) + DIRC(2), kp_sigma 1.0 / pose_sigma 0.5."""
    return require(RESULTS_ROOT / _REL["feature27"], "feature27")


def feature16_dir() -> Path:
    """Single-mouse 16-D features, plus SESSION_QC.csv (5 of 10 sessions pass)."""
    return require(RESULTS_ROOT / _REL["feature16"], "feature16")


def preprocessed_dir() -> Path:
    """Cleaned CalMS21 keypoints, per session."""
    return require(RESULTS_ROOT / _REL["preprocessed"], "preprocessed")


def out_dir(*parts: str) -> Path:
    """Create and return an output directory under OUT."""
    d = OUT.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d
