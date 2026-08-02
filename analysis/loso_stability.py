"""Recompute the LOSO dictionary stability values in Supplementary Figure S5."""

import csv
import json

import numpy as np

from dlds_release.paths import ROOT, dyadic_dictionary, out_dir, require

LOSO_DIR = (ROOT / "dLDS/results/loso_feature27_kp1p0_pose0p5"
            / "M15_l1_0.4_smooth_0.15_Flr_2.0_FlrDecay_0.997_decorr_0.1_D27_iter300_snips350")

HEADLINE = {"f_4": 3, "f_9": 8, "f_11": 10, "f_15": 14}

OUTLIER_Z = 2.0
MIXING_DIAGNOSTICS = {23: (6, 10), 24: (2, 6), 25: (6, 10)}


def sign_align(folds):
    """Align each fold to fold 0 under the operator sign ambiguity."""
    ref = folds[0]
    out = folds.copy()
    for f in range(len(out)):
        for k in range(out.shape[1]):
            if np.sum(out[f, k] * ref[k]) < 0:
                out[f, k] *= -1
    return out


def cosine(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def write_stability_table(ids, folds, consensus):
    """Write every numerical value consumed by Supplementary Figure S5."""
    destination = out_dir("analysis") / "loso_stability_values.csv"
    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["record", "fold", "consensus_operator", "fold_operator", "value"]
        )
        for fold_id, fold in zip(ids, folds):
            writer.writerow(
                ["whole_dictionary", fold_id, "", "", cosine(consensus, fold)]
            )
        for fold_id, fold in zip(ids, folds):
            for operator in range(folds.shape[1]):
                writer.writerow(
                    [
                        "same_slot",
                        fold_id,
                        operator + 1,
                        operator + 1,
                        cosine(consensus[operator], fold[operator]),
                    ]
                )
        fold_by_id = dict(zip(ids, folds))
        for fold_id, operators in MIXING_DIAGNOSTICS.items():
            fold = fold_by_id[fold_id]
            for consensus_operator in operators:
                for fold_operator in operators:
                    value = abs(
                        cosine(
                            consensus[consensus_operator - 1],
                            fold[fold_operator - 1],
                        )
                    )
                    writer.writerow(
                        [
                            "cross_slot",
                            fold_id,
                            consensus_operator,
                            fold_operator,
                            value,
                        ]
                    )
    return destination


def main():
    paths = sorted(LOSO_DIR.glob("fold_test*/Fs.npy"))
    if not paths:
        require(LOSO_DIR / "fold_test001" / "Fs.npy", "dyadic_loso_analysis")
    ids = [int(p.parent.name.replace("fold_test", "")) for p in paths]
    folds = sign_align(np.array([np.load(p) for p in paths]))
    print(f"{len(folds)} folds: {ids[0]}..{ids[-1]}, "
          f"contiguous: {ids == list(range(ids[0], ids[-1] + 1))}")

    released = np.load(dyadic_dictionary())
    consensus = folds.mean(0)
    table_path = write_stability_table(ids, folds, consensus)
    max_diff = float(np.abs(released - consensus).max())
    print(f"\nreleased F_universal vs recomputed mean: max|diff| = {max_diff:.2e} "
          f"({'reproduces' if max_diff < 1e-12 else 'DOES NOT REPRODUCE'})")
    print(f"wrote S5 numerical source: {table_path}")

    dist = np.array([np.linalg.norm(f - consensus) / np.linalg.norm(consensus)
                     for f in folds])
    z = (dist - dist.mean()) / dist.std()
    outliers = [i for i in range(len(folds)) if abs(z[i]) > OUTLIER_Z]
    print(f"\ndistance from consensus: mean {dist.mean():.3f}, sd {dist.std():.3f}")
    if outliers:
        print(f"outlier folds (|z| > {OUTLIER_Z}):")
        for i in outliers:
            print(f"  fold {ids[i]:03d}   {dist[i]:.3f}   z = {z[i]:+.1f}")
    else:
        print("no outlier folds")

    if not outliers:
        return

    keep = [i for i in range(len(folds)) if i not in outliers]
    reduced = folds[keep].mean(0)
    print(f"\nleave the outliers out (n = {len(keep)}) and compare to the released dictionary:")
    print(f"  overall cosine          {cosine(released, reduced):.4f}")
    print(f"  relative difference     {np.linalg.norm(released - reduced) / np.linalg.norm(released):.4f}")
    per_op = sorted((cosine(released[k], reduced[k]), k + 1) for k in range(folds.shape[1]))
    print(f"  worst operator          f_{per_op[0][1]} at {per_op[0][0]:.4f}")
    print("  operators the claims rest on:")
    for name, k in HEADLINE.items():
        print(f"    {name:<5} {cosine(released[k], reduced[k]):.4f}")

    dest = out_dir("analysis") / "loso_fold_robustness.json"
    dest.write_text(json.dumps({
        "n_folds": len(folds),
        "fold_ids": ids,
        "consensus_reproduces": max_diff < 1e-12,
        "max_abs_diff_vs_released": max_diff,
        "distance_from_consensus": dist.tolist(),
        "outlier_fold_ids": [ids[i] for i in outliers],
        "without_outliers": {
            "n_folds": len(keep),
            "overall_cosine": cosine(released, reduced),
            "per_operator_cosine": {f"f_{k+1}": cosine(released[k], reduced[k])
                                    for k in range(folds.shape[1])},
        },
    }, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
