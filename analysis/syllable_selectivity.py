"""Compute dyadic syllable selectivity for Supplementary Figure S7."""
import numpy as np
import h5py

from dlds_release.paths import feature27_dir, moseq_results, out_dir

RES = moseq_results('dyadic')
LBL = feature27_dir()
OUT = out_dir('analysis')
NSYL = 100
NBEH = 4
BEH_NAME = {0: 'attack', 1: 'invest', 2: 'mount', 3: 'other'}
N_PERM = 1000


def load():
    sess = []
    with h5py.File(RES, 'r') as f:
        for k in sorted(f.keys()):
            z = f[k]['syllable'][:].astype(int)
            sid = k.replace('mouse', '')
            lp = LBL / f'cleaned_label_mouse{sid}.npy'
            if not lp.exists():
                continue
            lb = np.load(lp).astype(int)
            T = min(len(z), len(lb))
            z, lb = z[:T], lb[:T]
            if np.any((z < 0) | (z >= NSYL)):
                raise ValueError(
                    f"{k}: syllable IDs must lie in [0, {NSYL - 1}]"
                )
            if np.any((lb < 0) | (lb >= NBEH)):
                raise ValueError(f"{k}: behavior labels must lie in [0, {NBEH - 1}]")
            sess.append((z, lb))
    return sess


def contingency(sess, shifts=None):
    cnt = np.zeros((NSYL, NBEH))
    for i, (z, lb) in enumerate(sess):
        l = lb if shifts is None else np.roll(lb, shifts[i])
        np.add.at(cnt, (z, l), 1)
    return cnt


def main():
    sess = load()
    print(f'{len(sess)} sessions')
    cnt = contingency(sess)
    usage = cnt.sum(1)
    Pbk = cnt / np.maximum(usage[:, None], 1)            # P(b|k)
    base = cnt.sum(0) / cnt.sum()                        # P(b)
    enrich = Pbk / np.maximum(base[None, :], 1e-9)
    print('base rates:', {BEH_NAME[b]: round(base[b], 3) for b in range(NBEH)})

    rng = np.random.default_rng(20260615)
    null_mean = np.zeros((NSYL, NBEH)); null_M2 = np.zeros((NSYL, NBEH))
    for p in range(N_PERM):
        sh = [int(rng.integers(1, len(z))) for z, _ in sess]
        cp = contingency(sess, sh)
        Pp = cp / np.maximum(cp.sum(1)[:, None], 1)
        d = Pp - null_mean; null_mean += d / (p + 1); null_M2 += d * (Pp - null_mean)
    null_sd = np.sqrt(null_M2 / (N_PERM - 1))
    z = (Pbk - null_mean) / (null_sd + 1e-12)

    sub = np.where(usage >= 0.005 * usage.sum())[0]
    print(f'\n{len(sub)} substantive syllables (usage >= 0.5%). '
          f'Per behavior, top syllables by z (attack/invest/mount):')
    for b in [0, 2, 1]:                                   # attack, mount, invest
        order = sub[np.argsort(-z[sub, b])]
        print(f'\n=== {BEH_NAME[b]} (base {base[b]*100:.1f}%) ===')
        print(f'{"syll":>4} {"usage%":>6} {"P(b|k)%":>8} {"enrich":>6} {"z":>7} {">50%":>5}')
        for k in order[:8]:
            print(f'{k:4d} {100*usage[k]/usage.sum():6.2f} {100*Pbk[k,b]:8.1f} '
                  f'{enrich[k,b]:6.1f} {z[k,b]:+7.1f} {"YES" if Pbk[k,b]>0.5 else "":>5}')

    np.savez(OUT / 'syllable_behavior_selectivity.npz',
             cnt=cnt, usage=usage, Pbk=Pbk, base=base, enrich=enrich, z=z, sub=sub)
    print(f'\nsaved {OUT}/syllable_behavior_selectivity.npz')


if __name__ == '__main__':
    main()
