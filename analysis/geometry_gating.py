"""Compute session-stratified odds ratios within ``f_4``-silent frames."""

import argparse
import json

import numpy as np

from dlds_release.paths import dyadic_cs_dir, feature27_dir, out_dir

SLOT_F4 = 3
BEHAVIOR = {0: "attack", 1: "investigation", 2: "mount", 3: "other"}

TARGETS = [("f_9", 8, 2), ("f_15", 14, 0)]

MIN_FRAMES = 50
MIN_MARGIN = 20
SILENT_ATOL = 1e-12


def session_tables(slot, behavior_code, thr, margin_rule):
    """Yield (session_id, a, b, c, d) for every session that passes the filter.

    a = recruited and behavior        b = recruited and not behavior
    c = not recruited and behavior    d = not recruited and not behavior

    All cells use the f_4-silent frames; recruitment is c > thr.
    """
    cs_dir, feat_dir = dyadic_cs_dir(), feature27_dir()
    for sid in range(1, 71):
        cp = cs_dir / f"cs_mouse{sid:03d}.npy"
        lp = feat_dir / f"cleaned_label_mouse{sid:03d}.npy"
        if not (cp.exists() and lp.exists()):
            continue
        cs, lb = np.load(cp), np.load(lp)
        T = min(cs.shape[1], len(lb))
        cs, lb = cs[:, :T], lb[:T]

        silent = np.abs(cs[SLOT_F4]) <= SILENT_ATOL
        if silent.sum() < MIN_FRAMES:
            continue

        recruited = cs[slot] > thr
        is_beh = lb == behavior_code

        a = int(np.sum(silent & recruited & is_beh))
        b = int(np.sum(silent & recruited & ~is_beh))
        c = int(np.sum(silent & ~recruited & is_beh))
        d = int(np.sum(silent & ~recruited & ~is_beh))

        margin = {
            "rows": min(a + b, c + d),
            "all4": min(a + b, c + d, a + c, b + d),
            "cols": min(a + c, b + d),
        }[margin_rule]
        if margin < MIN_MARGIN:
            continue
        yield sid, a, b, c, d


def mantel_haenszel(tables):
    """MH odds ratio with the Robins-Breslow-Greenland standard error.

    The confidence interval is reported with the point estimate.
    """
    num = den = 0.0
    for _, a, b, c, d in tables:
        n = a + b + c + d
        num += a * d / n
        den += b * c / n
    if den == 0:
        return float("inf"), (float("nan"), float("nan"))
    or_mh = num / den

    # Robins, Breslow and Greenland (1986), variance of log(OR_MH).
    s_pr = s_pspr = s_qs = 0.0
    for _, a, b, c, d in tables:
        n = a + b + c + d
        p = (a + d) / n
        q = (b + c) / n
        r = a * d / n
        s = b * c / n
        s_pr += p * r
        s_pspr += p * s + q * r
        s_qs += q * s
    var = s_pr / (2 * num**2) + s_pspr / (2 * num * den) + s_qs / (2 * den**2)
    se = np.sqrt(var)
    lo, hi = np.exp(np.log(or_mh) - 1.96 * se), np.exp(np.log(or_mh) + 1.96 * se)
    return or_mh, (lo, hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thr", type=float, default=0.05,
                    help="recruitment threshold on the signed coefficient (matches the rest of the analysis)")
    ap.add_argument("--margin-rule", choices=("rows", "all4", "cols"), default="rows",
                    help="which marginal totals the >=20 rule applies to")
    args = ap.parse_args()

    print(f"recruitment c > {args.thr} (positive phase)   f_4 silent c_4 = 0")
    print(f"session filter: >= {MIN_FRAMES} f_4-silent frames and >= {MIN_MARGIN} in each "
      f"{args.margin_rule} margin\n")

    summary = {}

    for name, slot, code in TARGETS:
        tables = list(session_tables(slot, code, args.thr, args.margin_rule))
        or_mh, (lo, hi) = mantel_haenszel(tables)
        beh = BEHAVIOR[code]

        # The pooled estimate shows the effect of session stratification.
        A = sum(t[1] for t in tables); B = sum(t[2] for t in tables)
        C = sum(t[3] for t in tables); D = sum(t[4] for t in tables)
        pooled = (A * D) / (B * C) if B * C else float("inf")

        print(f"=== {name} / {beh} ===")
        print(f"  sessions passing filter : {len(tables):3d}")
        print(f"  Mantel-Haenszel OR      : {or_mh:.2f}   95% CI [{lo:.2f}, {hi:.2f}]")
        print(f"  pooled OR, same frames  : {pooled:.2f}")
        print(f"  counts  a={A} b={B} c={C} d={D}")
        print()

        summary[name] = {
            "behavior": beh, "n_sessions": len(tables), "or_mh": or_mh,
            "ci95": [lo, hi], "or_pooled": pooled,
            "counts": {"a": A, "b": B, "c": C, "d": D},
            "per_session": [{"session": s, "a": a, "b": b, "c": c, "d": d}
                            for s, a, b, c, d in tables],
        }

    dest = out_dir("analysis") / "mantel_haenszel_gating.json"
    dest.write_text(json.dumps(
        {"threshold": args.thr,
         "f4_silence": f"abs(c_4) <= {SILENT_ATOL}",
         "margin_rule": args.margin_rule, "recruitment": "signed positive phase c > thr",
         "min_frames": MIN_FRAMES, "min_margin": MIN_MARGIN, "results": summary},
        indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
