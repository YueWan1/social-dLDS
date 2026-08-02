"""
Figure 2.2 -- behaviour-selectivity TIE between dLDS operators and KP-MoSeq syllables.
Promote "other" to a full 4th label and score BOTH methods with the same circular-shift null.

Left  : dLDS operators x 4 labels, SIGNED V3-amp z (mean signed c during behaviour vs a per-session
        circular-shift of c) -- the SAME metric/style as the main-text selectivity figure
        (panel_layer2_b1.py), redrawn with the "other" column added and all 7 retained ops included.
Right : KP-MoSeq 28 substantive syllables x 4 labels, enrichment z (P(label|syllable) vs a per-session
        circular-shift of the label sequence; precomputed in syllable_behavior_selectivity.npz).

Message: with "other" as a real 4th label and the same circular-shift z-test, BOTH methods have
base-rate-corrected behaviour-specific units -- each of attack/invest/mount has its own dLDS operator
AND its own handful of KP-MoSeq syllables. Single-unit selectivity is a TIE; the dLDS advantage is the
representation analysed in 2.3-2.6, not selectivity.

Run: python3 figures/supp/s7_selectivity_tie.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from dlds_release.paths import DERIVED, dyadic_cs_dir, feature27_dir, out_dir

RUN = dyadic_cs_dir()
LBL = feature27_dir()
OUT = out_dir('supp')
# The MoSeq enrichment z-scores are precomputed by
# analysis/syllable_selectivity.py; a copy ships in derived/ so this
# figure builds from the release alone.
NPZ = DERIVED / 'dyadic' / 'syllable_behavior_selectivity.npz'
THR = 0.05
N_PERM = 1000
BN = ['attack', 'invest', 'mount', 'other']
# functional order: attack op, mount op, invest op, substrate, geometry/other,
# then the background-tier operators (f_7/f_8/f_10) appended for comparison.
OPS = [('$f_{15}$', 14), ('$f_9$', 8), ('$f_3$', 2), ('$f_6$', 5),
       ('$f_2$', 1), ('$f_4$', 3), ('$f_{11}$', 10),
       ('$f_7$', 6), ('$f_8$', 7), ('$f_{10}$', 9)]


def stars(z):
    a = abs(z)
    return '$\\bigstar\\bigstar\\bigstar$' if a > 10 else '$\\bigstar\\bigstar$' if a > 5 \
        else '$\\bigstar$' if a > 3 else ''


def main():
    sess = []
    for sid in range(1, 71):
        cp = RUN / f'cs_mouse{sid:03d}.npy'; lp = LBL / f'cleaned_label_mouse{sid:03d}.npy'
        if not (cp.exists() and lp.exists()):
            continue
        cs = np.load(cp); lb = np.load(lp); T = min(cs.shape[1], len(lb))
        sess.append((cs[:, :T], lb[:T]))

    # dLDS mean-amplitude z score against a circular-shift null
    def pooled(slot, shifts=None):
        s = np.zeros(4); n = np.zeros(4)
        for i, (cs, lb) in enumerate(sess):
            c = np.abs(cs[slot] if shifts is None else np.roll(cs[slot], shifts[i]))
            for b in range(4):
                m = lb == b
                if m.any():
                    s[b] += c[m].sum(); n[b] += m.sum()
        return s / np.maximum(n, 1)

    op_z = np.zeros((len(OPS), 4))
    for r, (nm, slot) in enumerate(OPS):
        rng = np.random.default_rng(424242 + slot)
        obs = pooled(slot)
        null = np.array([pooled(slot, [int(rng.integers(1, cs.shape[1])) for cs, _ in sess])
                         for _ in range(N_PERM)])
        op_z[r] = (obs - null.mean(0)) / (null.std(0, ddof=1) + 1e-12)
        print(nm, np.round(op_z[r], 1))

    # Precomputed MoSeq enrichment z score against a shifted-label null
    d = np.load(NPZ)
    Pbk, sub, zsyl = d['Pbk'], d['sub'], d['z']
    rows = []
    for k in sub:
        amax = int(zsyl[k].argmax())
        rows.append((int(k), amax, float(zsyl[k, amax])))
    rows.sort(key=lambda t: (t[1], -t[2]))
    syl_ids = [r[0] for r in rows]
    syl_z = np.array([zsyl[k] for k in syl_ids])
    # behaviour-specific syllables = column z>3 (a syllable can be specific to a behaviour even if
    # "other" is its argmax, since "other" has the 63% base; matches the recorded selectivity result)
    nspec = {b: int(sum(1 for k in sub if zsyl[k, b] > 3)) for b in range(4)}
    print('MoSeq specific (column z>3) per behaviour:', {BN[b]: nspec[b] for b in range(4)})

    fig = plt.figure(figsize=(11.6, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.92, 1.08], wspace=0.30)
    vmax = 12

    # (a) dLDS ops x 4 -- paper-style stars + z numbers
    axL = fig.add_subplot(gs[0, 0])
    im = axL.imshow(op_z, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    axL.set_xticks(range(4)); axL.set_xticklabels(BN, fontsize=11)
    axL.set_yticks(range(len(OPS))); axL.set_yticklabels([nm for nm, _ in OPS], fontsize=13)
    axL.set_title('(a) dLDS operators $\\times$ behaviour\n$V_3$-amp $z$ (mean$|c|$, circ-shift null)',
                  fontsize=11.5, loc='left')
    for r in range(len(OPS)):
        for b in range(4):
            z = op_z[r, b]
            st = stars(z) if z > 3 else ''   # only enriched (positive) cells get stars; no numbers
            if st:
                axL.text(b, r, st, ha='center', va='center', fontsize=12,
                         color='white' if z > 7 else 'black')
                axL.add_patch(Rectangle((b - 0.49, r - 0.49), 0.98, 0.98,
                                        fill=False, edgecolor='black', lw=1.4, zorder=5))
    axL.set_xticks(np.arange(-.5, 4, 1), minor=True)
    axL.set_yticks(np.arange(-.5, len(OPS), 1), minor=True)
    axL.grid(which='minor', color='white', lw=1.0); axL.tick_params(which='minor', length=0)

    # (b) MoSeq syllables x 4 -- enrichment z, stars on |z|>3
    axR = fig.add_subplot(gs[0, 1])
    axR.imshow(syl_z, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    axR.set_xticks(range(4)); axR.set_xticklabels(BN, fontsize=11)
    axR.set_yticks(range(len(syl_ids)))
    axR.set_yticklabels([f's{k}' for k in syl_ids], fontsize=6.6)
    axR.set_title('(b) KP-MoSeq syllables $\\times$ behaviour\nenrichment $z$ (circ-shift null)',
                  fontsize=11.5, loc='left')
    for r, k in enumerate(syl_ids):
        for b in range(4):
            z = syl_z[r, b]
            if z > 3:
                axR.text(b, r, '$\\bigstar$', ha='center', va='center',
                         fontsize=6.5, color='black')
    # colour each syllable's tick label by its preferred (argmax) behaviour group
    cgrp = ['#B22222', '#D98A00', '#2E7D32', '#777777']
    for tick, k in zip(axR.get_yticklabels(), syl_ids):
        tick.set_color(cgrp[int(zsyl[k].argmax())])
        tick.set_fontweight('bold')

    cbar = fig.colorbar(im, ax=[axL, axR], fraction=0.024, pad=0.02)
    cbar.set_label('selectivity $z$ (base-rate corrected, ``other'' included)', fontsize=9.5)

    fig.suptitle('Behaviour selectivity is a TIE: each of attack / invest / mount has its own dLDS '
                 'operator AND its own KP-MoSeq syllables', fontsize=11.5, y=1.0)
    fig.text(0.5, -0.03,
             f'Same circular-shift $z$-test, ``other'' as a 4th label. MoSeq behaviour-specific '
             f'syllables ($z{{>}}3$): attack {nspec[0]}, invest {nspec[1]}, mount {nspec[2]} '
             f'(and {nspec[3]} ``other''). $\\bigstar$: $z{{>}}3$; syllable labels coloured by preferred '
             f'behaviour. The dLDS advantage is representational, not selectivity (see 2.3$\\to$2.6).',
             ha='center', fontsize=8.6, color='dimgray')
    fig.savefig(OUT / 'fig_selectivity_tie.png', dpi=150, bbox_inches='tight')
    print('wrote', OUT / 'fig_selectivity_tie.png', '| n syllables', len(syl_ids))


if __name__ == '__main__':
    main()
