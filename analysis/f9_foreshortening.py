"""Measure resident body lengths across ``f_9`` phases and behavior labels."""

import json

import numpy as np

from dlds_release.paths import dyadic_cs_dir, feature27_dir, out_dir, preprocessed_dir

F9_SLOT = 8
THR = 0.05

# CalMS21 keypoint indices
NOSE, NECK, TAIL = 0, 3, 6

BEHAVIOR = {0: "attack", 1: "investigation", 2: "mount", 3: "other"}

def load():
    """Pool frame-aligned coefficients, keypoint distances, and labels."""
    cs_dir, feat_dir, pre_dir = dyadic_cs_dir(), feature27_dir(), preprocessed_dir()
    c9, neck_tail, nose_neck, label = [], [], [], []
    for sid in range(1, 71):
        cp = cs_dir / f"cs_mouse{sid:03d}.npy"
        kp = pre_dir / f"mouse{sid:03d}" / "res_kp.npy"
        lp = feat_dir / f"cleaned_label_mouse{sid:03d}.npy"
        if not (cp.exists() and kp.exists() and lp.exists()):
            continue
        cs, k, lb = np.load(cp), np.load(kp), np.load(lp)
        T = min(cs.shape[1], k.shape[2], len(lb))
        c9.append(cs[F9_SLOT, :T])
        neck_tail.append(np.linalg.norm(k[:, NECK, :T] - k[:, TAIL, :T], axis=0))
        nose_neck.append(np.linalg.norm(k[:, NOSE, :T] - k[:, NECK, :T], axis=0))
        label.append(lb[:T])
    return (np.concatenate(c9), np.concatenate(neck_tail),
            np.concatenate(nose_neck), np.concatenate(label))


def describe(values):
    q25, q75 = np.percentile(values, [25, 75])
    return {"n": int(values.size), "mean": float(values.mean()),
            "median": float(np.median(values)), "iqr": [float(q25), float(q75)]}


def row(name, mask, neck_tail, nose_neck, results):
    nt, nn = neck_tail[mask], nose_neck[mask]
    results[name] = {"neck_tail": describe(nt), "nose_neck": describe(nn)}
    print(f"  {name:<16} n={mask.sum():>8,}   "
          f"neck-tail {nt.mean():6.1f}   nose-neck {nn.mean():5.1f}")


def main():
    c9, neck_tail, nose_neck, label = load()
    results = {}

    print(f"{len(c9):,} frames, {len(np.unique(label))} behavior labels\n")
    print(f"all frames:  neck-tail {neck_tail.mean():.1f}   nose-neck {nose_neck.mean():.1f}\n")

    print("by f_9 sign/activity")
    row("positive_active", c9 > THR, neck_tail, nose_neck, results)
    row("negative_active", c9 < -THR, neck_tail, nose_neck, results)
    row("silent", np.abs(c9) <= THR, neck_tail, nose_neck, results)

    print("\nby behavior label")
    for code in (2, 3, 1, 0):
        row(BEHAVIOR[code], label == code, neck_tail, nose_neck, results)

    print("\nwithin mount")
    mount = label == 2
    row("mount_positive", mount & (c9 > THR), neck_tail, nose_neck, results)
    row("mount_silent", mount & (np.abs(c9) <= THR), neck_tail, nose_neck, results)

    by_label = {BEHAVIOR[c]: neck_tail[label == c].mean() for c in range(4)}
    most = min(by_label, key=by_label.get)
    print(f"\nmost foreshortened label: {most} "
          f"({by_label[most]:.1f} px)")
    print("  " + "  ".join(f"{k} {v:.1f}" for k, v in
                           sorted(by_label.items(), key=lambda kv: kv[1])))

    dest = out_dir("analysis") / "f9_foreshortening.json"
    dest.write_text(json.dumps(
        {"threshold": THR,
         "results": results,
         "neck_tail_by_label": by_label,
         "most_foreshortened": most},
        indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
