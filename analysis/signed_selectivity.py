"""Compute sign-resolved operator selectivity for Figure 6b and Table S2."""
import numpy as np

from dlds_release.paths import dyadic_cs_dir, feature27_dir, out_dir

CS_DIR = dyadic_cs_dir()
LBL_DIR = feature27_dir()
OUT = out_dir('analysis')

BEH = [('attack', 0), ('invest', 1), ('mount', 2)]
OPS = [('f_2', 1), ('f_3', 2), ('f_6', 5), ('f_9', 8), ('f_15', 14)]
N_PERM = 1000

print('Loading sessions + labels ...')
sessions = []
for sid in range(1, 71):
    cp = CS_DIR / f'cs_mouse{sid:03d}.npy'; lp = LBL_DIR / f'cleaned_label_mouse{sid:03d}.npy'
    if not (cp.exists() and lp.exists()):
        continue
    cs = np.load(cp); lb = np.load(lp)
    T = min(cs.shape[1], len(lb))
    sessions.append((cs[:, :T], lb[:T]))
print(f'  {len(sessions)} sessions')


def pooled_stats(slot, shifts=None):
    """Pool signed, positive, and negative amplitudes by behavior."""
    s_signed = {b: 0.0 for _, b in BEH}; s_pos = {b: 0.0 for _, b in BEH}
    s_neg = {b: 0.0 for _, b in BEH}; n = {b: 0 for _, b in BEH}
    for i, (cs, lb) in enumerate(sessions):
        c = cs[slot] if shifts is None else np.roll(cs[slot], shifts[i])
        for _, b in BEH:
            m = lb == b
            if not m.any():
                continue
            cc = c[m]
            s_signed[b] += cc.sum()
            s_pos[b] += np.where(cc > 0, cc, 0.0).sum()
            s_neg[b] += np.where(cc < 0, -cc, 0.0).sum()
            n[b] += int(m.sum())
    return {b: (s_signed[b] / max(n[b], 1), s_pos[b] / max(n[b], 1), s_neg[b] / max(n[b], 1))
            for _, b in BEH}


def run(slot):
    rng = np.random.default_rng(424242 + slot)
    obs = pooled_stats(slot)
    null = {b: [np.zeros(N_PERM) for _ in range(3)] for _, b in BEH}
    for p in range(N_PERM):
        sh = [int(rng.integers(1, cs.shape[1])) for cs, _ in sessions]
        a = pooled_stats(slot, sh)
        for _, b in BEH:
            for k in range(3):
                null[b][k][p] = a[b][k]
    z = {}
    for _, b in BEH:
        for k, nm in enumerate(['signed', 'pos', 'neg']):
            mu = null[b][k].mean(); sd = null[b][k].std(ddof=1)
            z[(b, nm)] = (obs[b][k] - mu) / (sd + 1e-12)
    return obs, z


def star(z):
    a = abs(z); return '***' if a > 10 else '**' if a > 5 else '*' if a > 3 else ''


print('\nComputing pooled sign-resolved selectivity (N=1000) ...')
save = {}
for nm, slot in OPS:
    obs, z = run(slot)
    for bn, b in BEH:
        for k, mk in enumerate(['signed', 'pos', 'neg']):
            save[f'z_{nm}_{mk}_{bn}'] = z[(b, mk)]
            save[f'obs_{nm}_{mk}_{bn}'] = obs[b][k]

for mk in ['signed', 'pos', 'neg']:
    print(f'\n=== {mk} z (pooled) ===')
    print(f'{"op":>5} | {"attack":>11} {"invest":>11} {"mount":>11}')
    for nm, slot in OPS:
        cells = [f'{save[f"z_{nm}_{mk}_{bn}"]:+6.2f}{star(save[f"z_{nm}_{mk}_{bn}"]):>3}' for bn, _ in BEH]
        print(f'{nm:>5} | ' + ' '.join(c.rjust(11) for c in cells))

np.savez(OUT / 'ztable.npz', **save)
print(f'\nSaved {OUT}/ztable.npz')
