"""Recompute the distance-decile and projection results for ``f_4``."""

import json

import numpy as np

from dlds_release.paths import dyadic_cs_dir, dyadic_dictionary, feature27_dir, out_dir

F4_SLOT = 3
DIST_SLICE = slice(14, 25)
CENT_DIST_DIM = 14
N_BINS = 10

ACTIVE_THR = 0.05


def normalize_features(features):
    normalized = features.astype(float, copy=True)
    normalized /= np.maximum(np.std(normalized, axis=1, keepdims=True), 1e-3)
    normalized /= max(np.quantile(np.abs(normalized), 0.99), 1e-6)
    return normalized


def load_sessions():
    """Load frame-aligned coefficients and distance features."""
    cs_dir, feat_dir = dyadic_cs_dir(), feature27_dir()
    out = []
    for sid in range(1, 71):
        cp = cs_dir / f"cs_mouse{sid:03d}.npy"
        fp = feat_dir / f"FEATURE27_mouse{sid:03d}.npy"
        if not (cp.exists() and fp.exists()):
            continue
        cs, raw_features = np.load(cp), np.load(fp)
        if raw_features.shape[0] != 27:
            raw_features = raw_features.T
        features = normalize_features(raw_features)
        T = min(cs.shape[1], raw_features.shape[1])
        out.append(
            (
                sid,
                cs[F4_SLOT, :T],
                features[DIST_SLICE, :T],
                raw_features[CENT_DIST_DIM, :T],
            )
        )
    return out


def decile_means(abs_c, dist, n_bins=N_BINS):
    """Mean |c_4| in each distance decile, cut on quantiles of `dist`."""
    edges = np.quantile(dist, np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.inf
    idx = np.searchsorted(edges, dist, side="right") - 1
    idx = np.clip(idx, 0, n_bins - 1)
    return np.array([abs_c[idx == b].mean() if np.any(idx == b) else np.nan
                     for b in range(n_bins)])


def leading_projection_r(sessions):
    """Median per-session correlation between c_4 and the leading distance projection.

    The projection is the leading eigenvector of f_4's DIST-DIST sub-block.
    Its global sign is fixed so that the median correlation is positive.
    """
    F = np.load(dyadic_dictionary())
    idx = np.arange(DIST_SLICE.start, DIST_SLICE.stop)
    w, V = np.linalg.eig(F[F4_SLOT][np.ix_(idx, idx)])
    v1 = V[:, np.argsort(np.abs(w))[::-1][0]].real

    rs = []
    for _, c4, dist_block, _ in sessions:
        if c4.std() < 1e-9:
            continue
        rs.append(float(np.corrcoef(c4, v1 @ dist_block)[0, 1]))
    rs = np.array(rs)
    if np.median(rs) < 0:
        rs = -rs
    return rs


def main():
    sessions = load_sessions()
    print(f"{len(sessions)} sessions\n")

    c_all = np.concatenate([s[1] for s in sessions])
    dist_all_frames = np.concatenate([s[3] for s in sessions])
    active = np.abs(c_all) > ACTIVE_THR
    abs_c_all, dist_all = np.abs(c_all[active]), dist_all_frames[active]

    print(f"all frames           : {len(c_all):,}")
    print(f"active (|c_4|>{ACTIVE_THR})   : {active.sum():,}  "
          f"({100 * active.mean():.0f}%)")
    print(f"c_4 negative on      : {100 * np.mean(c_all[active] < 0):.1f}% of active frames\n")

    pooled = decile_means(abs_c_all, dist_all)

    # Equal-weight sessions prevent long recordings from dominating.
    per_session = []
    for _, c4, _, cd in sessions:
        m = np.abs(c4) > ACTIVE_THR
        if m.sum() >= 10 * N_BINS:
            per_session.append(decile_means(np.abs(c4[m]), cd[m]))
    within = np.nanmean(np.array(per_session), axis=0)

    print("mean |c_4| by decile of centroid distance")
    print("  decile   pooled cuts   per-session cuts")
    for b in range(N_BINS):
        print(f"  {b + 1:>4}     {pooled[b]:>10.3f}   {within[b]:>14.3f}")
    print()

    for tag, arr in (("pooled cuts", pooled), ("per-session cuts", within)):
        near, far = arr[0], arr[-1]
        monotone = bool(np.all(np.diff(arr) > 0))
        print(f"{tag:>18}: nearest {near:.2f} -> farthest {far:.2f}   "
              f"monotone: {'yes' if monotone else 'NO'}")

    rs = leading_projection_r(sessions)
    print(f"\nmedian per-session r(c_4, leading distance projection) = {np.median(rs):.3f} "
          f"over {len(rs)} sessions")

    dest = out_dir("analysis") / "f4_distance_deciles.json"
    dest.write_text(json.dumps({
        "n_sessions": len(sessions),
        "n_frames_all": int(len(c_all)),
        "n_frames_active": int(active.sum()),
        "active_threshold": ACTIVE_THR,
        "c4_negative_fraction_among_active": float(np.mean(c_all[active] < 0)),
        "decile_means_pooled_cuts": pooled.tolist(),
        "decile_means_per_session_cuts": within.tolist(),
        "median_r_leading_projection": float(np.median(rs)),
    }, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
