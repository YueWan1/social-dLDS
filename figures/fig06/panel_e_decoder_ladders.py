"""Generate the one-vs-rest LOSO decoder ladders for Figure 6e.

Each behavior compares constrained dLDS operator sets with behavior-specific
and substantive keypoint-MoSeq syllables. The four standalone PDFs use the same
rungs and axis limits as the four-class comparison in Supplementary Figure S9.
"""
import os
import numpy as np, h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from dlds_release.paths import (
    DERIVED,
    dyadic_cs_dir,
    feature27_dir,
    moseq_results,
    out_dir,
)

RUN = dyadic_cs_dir()
FEAT = feature27_dir()
KPMS = moseq_results('dyadic')
NPZ = DERIVED / 'dyadic' / 'syllable_behavior_selectivity.npz'
OUT = out_dir('fig06')

F = dict(f2=1, f3=2, f4=3, f6=5, f7=6, f8=7, f9=8, f10=9, f11=10, f15=14)
ALL10 = [1, 2, 3, 5, 6, 7, 8, 9, 10, 14]
ANALYSED7 = [F['f2'], F['f3'], F['f4'], F['f6'], F['f9'], F['f11'], F['f15']]   # idx 1,2,3,5,8,10,14
d = np.load(NPZ, allow_pickle=True); SUB = [int(s) for s in d['sub']]; Zsel = d['z']

sess = []
with h5py.File(KPMS, 'r') as f:
    kk = set(f.keys())
    for sid in range(1, 71):
        s3 = f'{sid:03d}'; cp = RUN / f'cs_mouse{s3}.npy'; lp = FEAT / f'cleaned_label_mouse{s3}.npy'
        key = f'mouse{s3}'
        if cp.exists() and lp.exists() and key in kk:
            cs = np.load(cp); lb = np.load(lp).astype(int); z = f[key]['syllable'][:].astype(int)
            T = min(cs.shape[1], len(lb), len(z)); sess.append((cs[:, :T], lb[:T], z[:T]))
n = len(sess)
GRP = np.concatenate([np.full(len(lb), i) for i, (_, lb, _) in enumerate(sess)])
LAB = np.concatenate([lb for (_, lb, _) in sess])
ZZ = np.concatenate([z for (_, _, z) in sess])
print(f'sessions {n}; substantive (>=0.5%) syllables = {len(SUB)}; labels 0/1/2/3 = atk/inv/mnt/oth')


def dfeat(slots):
    return np.concatenate([cs[slots].T for (cs, _, _) in sess])


def mfeat(cols):
    return np.stack([(ZZ == s).astype(float) for s in cols], 1)


def loso(X, bcode, elig):
    """ONE-VS-REST LOSO: fit a binary logistic on (y == bcode); AUC of its posterior vs (y == bcode)."""
    y = (LAB == bcode).astype(int)
    aucs = []
    for held in elig:
        tr = np.isin(GRP, [e for e in elig if e != held]); te = (GRP == held)
        if y[te].sum() in (0, te.sum()) or y[tr].sum() == 0:
            continue
        c = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=500, class_weight='balanced')).fit(X[tr], y[tr])
        p = c.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs) / np.sqrt(max(len(aucs), 1))), len(aucs)


DBLUE, ORANGE, MOSEQ = '#16365c', '#c55a11', '#6f6f6f'
# Font sizes account for reduction in the final 2-by-2 layout.
FS_OP = 17.5
FS_SYLL = 15.0
FS_DIAM, FS_TICK, FS_AXIS = 17.0, 14.5, 15.5
FS_TITLE, FS_LEG, FS_CHANCE = 18.0, 13.5, 12.5

# behaviour -> (label code, eligibility min frac, constrained dLDS combos [(label, slots)])
# Use the same ladder rungs as the four-class decoder comparison.
CFG = {
    'Attack': dict(code=0, minfrac=0.005, fname='attack', legendloc='upper left', diamonds=['7'], ladder=True,
                   d=[('$f_9$', [F['f9']]),
                      ('$f_{15}$', [F['f15']]),
                      ('$f_9,f_{15}$', [F['f9'], F['f15']]),
                      ('$f_{15},f_9,f_{11}$', [F['f15'], F['f9'], F['f11']]),
                      ('$f_4,f_{11},f_9,f_{15}$', [F['f4'], F['f11'], F['f9'], F['f15']])]),
    'Mount': dict(code=2, minfrac=0.01, fname='mount', legendloc='upper left', diamonds=['7'], ladder=True,
                  d=[('$f_{15}$', [F['f15']]),
                     ('$f_9$', [F['f9']]),
                     ('$f_9,f_{15}$', [F['f9'], F['f15']]),
                     ('$f_9,f_{15},f_{11}$', [F['f9'], F['f15'], F['f11']]),
                     ('$f_9,f_{15},f_4,f_{11}$', [F['f9'], F['f15'], F['f4'], F['f11']])]),
    'Investigation': dict(code=1, minfrac=0.02, fname='investigation', legendloc='center left',
                          diamonds=['7'], ladder=True,
                          d=[('$f_2,f_3,f_6$', [F['f2'], F['f3'], F['f6']]),
                             ('$f_2,f_3,f_6,f_4,f_{11}$',
                              [F['f2'], F['f3'], F['f6'], F['f4'], F['f11']])]),
    'Other': dict(code=3, minfrac=0.05, fname='other', legendloc='center left',
                  diamonds=['7'], ladder=True,
                  d=[('$f_4$', [F['f4']]),
                     ('$f_4,f_{11}$', [F['f4'], F['f11']]),
                     ('$f_2,f_3,f_6$', [F['f2'], F['f3'], F['f6']])]),
}

ONLY = os.environ.get('ONLY')   # set ONLY=Investigation to regenerate just that panel (faster)
for bname, cfg in CFG.items():
    if ONLY and bname != ONLY:
        continue
    bcode = cfg['code']
    frac = np.array([(lb == bcode).mean() for (_, lb, _) in sess])
    elig = sorted(i for i in range(n) if frac[i] >= cfg['minfrac'])
    spec = sorted(s for s in SUB if Zsel[s, bcode] > 2)
    print(f'\n=== {bname} (label {bcode}); elig {len(elig)} sess; {len(spec)} specific syllables {spec} ===')

    dlds = []
    for lab, slots in cfg['d']:
        a, se, _ = loso(dfeat(slots), bcode, elig); dlds.append((lab, len(slots), a, se))
        print(f'  dLDS  {lab:26s} {len(slots):2d}u  AUC {a:.3f}')
    a7, se7, _ = loso(dfeat(ANALYSED7), bcode, elig)
    d7 = ('7 analysed ops', 7, a7, se7)
    print(f'  dLDS  {"7 analysed ops":26s}  7u  AUC {a7:.3f}')
    diamonds = [d7]
    if '10' in cfg.get('diamonds', ['7', '10']):
        a10, se10, _ = loso(dfeat(ALL10), bcode, elig)
        d10 = ('all 10 operators', 10, a10, se10); diamonds.append(d10)
        print(f'  dLDS  {"all 10 operators":26s} 10u  AUC {a10:.3f}')
    moseq = []
    for lab, cols in [(f'{len(spec)} specific syll', spec), ('28 substantive syll', SUB)]:
        if not cols:
            continue
        a, se, _ = loso(mfeat(cols), bcode, elig); moseq.append((lab, len(cols), a, se))
        print(f'  MoSeq {lab:26s} {len(cols):2d}u  AUC {a:.3f}')

    # ---- figure ----  (ladder=step-index x for all; tick label = number of units)
    ladder = cfg.get('ladder', False)
    # A moderately wide aspect ratio keeps the 2x2 main-figure grid compact
    # while preserving room for the rung annotations.
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.axhline(0.5, ls=':', color='#888', lw=1.0)

    if ladder:                                   # one rung per set, in the cfg['d'] order, then all-7, then MoSeq
        xs_d = list(range(1, len(dlds) + 1))
        xs_diam = [len(dlds) + 1 + i for i in range(len(diamonds))]
        xs_m = [len(dlds) + 1 + len(diamonds) + i for i in range(len(moseq))]
    else:                                        # x = number of units; nudge coincident MoSeq marker right
        dlds_units = {u for _, u, _, _ in dlds} | {dd[1] for dd in diamonds}
        mdisp = lambda u: u * 1.30 if u in dlds_units else u
        xs_d = [u for _, u, _, _ in dlds]
        xs_diam = [dd[1] for dd in diamonds]
        xs_m = [mdisp(u) for _, u, _, _ in moseq]

    dy = [a for _, _, a, _ in dlds]; my = [a for _, _, a, _ in moseq]
    line_xy = sorted(zip(xs_d + xs_diam, dy + [dd[2] for dd in diamonds]))
    ax.plot([p[0] for p in line_xy], [p[1] for p in line_xy], '-', color=DBLUE, lw=1.9, zorder=2, alpha=0.55)
    ax.errorbar(xs_d, dy, yerr=[e for _, _, _, e in dlds], fmt='o', color=DBLUE, ms=8.0, capsize=4,
                mec='white', mew=1.1, zorder=5)
    for xx, dd in zip(xs_diam, diamonds):
        ax.errorbar([xx], [dd[2]], yerr=[dd[3]], fmt='D', color=ORANGE, ms=11.0, capsize=4,
                    mec='white', mew=1.4, zorder=6)
    ax.plot(xs_m, my, '--', color=MOSEQ, lw=1.9, zorder=2, alpha=0.7)
    ax.errorbar(xs_m, my, yerr=[e for _, _, _, e in moseq], fmt='s', color=MOSEQ, ms=7.5, capsize=4,
                mec='white', mew=1.1, zorder=5)

    # Labels identify the feature recipe only.  Numeric AUC annotations are
    # intentionally omitted because the y-axis ticks already report magnitude.
    spec_units = {len(spec)} if spec else set()
    for i, (xx, (lab, u, a, _)) in enumerate(zip(xs_d, dlds)):
        yoff = (20 if i % 2 == 0 else 46) if ladder else (38 if u in spec_units else 20)
        crowded_last = (i == len(dlds) - 1
                        and abs(a - diamonds[0][2]) < 0.05)
        ax.annotate(lab, (xx, a), textcoords='offset points',
                    xytext=(-8 if crowded_last else 0,
                            22 if crowded_last else yoff),
                    ha='right' if crowded_last else 'center',
                    va='bottom', fontsize=FS_OP, color=DBLUE)
    if len(diamonds) == 1:
        dd = diamonds[0]; xx = xs_diam[0]
        crowded_diamond = abs(dd[2] - dlds[-1][2]) < 0.05
        noff = 28 if dd[2] < 0.82 else 20
        ax.annotate('all 7 ops', (xx, dd[2]), textcoords='offset points',
                    xytext=(8 if crowded_diamond else 0,
                            22 if crowded_diamond else noff),
                    ha='left' if crowded_diamond else 'center',
                    va='bottom', fontsize=FS_DIAM, color=ORANGE,
                    fontweight='bold')
    else:
        (d7, x7), (d10, x10) = (diamonds[0], xs_diam[0]), (diamonds[1], xs_diam[1])
        ax.annotate('7 ops', (x7, d7[2]), textcoords='offset points', xytext=(0, 18),
                    ha='center', va='bottom', fontsize=FS_DIAM, color=ORANGE, fontweight='bold')
        ax.annotate('all 10 ops', (x10, d10[2]), textcoords='offset points', xytext=(0, -18),
                    ha='center', va='top', fontsize=FS_DIAM, color=ORANGE, fontweight='bold')
    for m_i, (xx, (lab, u, a, _)) in enumerate(zip(xs_m, moseq)):
        # Keep low-AUC labels above their marker so the 0.5 chance line never
        # runs through the text; otherwise alternate the pair for separation.
        above = (a < 0.56 or m_i % 2 == 1)
        ax.annotate(lab, (xx, a), textcoords='offset points',
                    xytext=(0, 18 if above else -16),
                    ha='right' if above else 'center',
                    va='bottom' if above else 'top',
                    fontsize=FS_SYLL, color=MOSEQ)

    if ladder:
        allx = xs_d + xs_diam + xs_m
        allu = [u for _, u, _, _ in dlds] + [dd[1] for dd in diamonds] + [u for _, u, _, _ in moseq]
        ax.set_xticks(allx); ax.set_xticklabels([str(u) for u in allu])
        ax.set_xlim(0.5, max(allx) + 0.7)
        ax.set_xlabel('decoder rung  (tick $=$ number of units)', fontsize=FS_AXIS)
    else:
        ax.set_xscale('log')
        XT = [1, 2, 3, 5, 7, 10, 28]
        ax.set_xticks(XT); ax.set_xticklabels([str(t) for t in XT])
        ax.set_xlim(0.8, 45)
        ax.set_xlabel('number of units in the decoder (log scale)', fontsize=FS_AXIS)
    ax.set_ylim(0.45, 0.94)
    ax.set_ylabel('one-vs-rest decoding AUC (LOSO)', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    from matplotlib.lines import Line2D
    diam_label = 'dLDS all-7 operators' if len(diamonds) == 1 else 'dLDS 7 / all-10 operators'
    leg = [Line2D([0], [0], marker='o', color=DBLUE, lw=1.9, label='dLDS operators'),
           Line2D([0], [0], marker='D', color=ORANGE, lw=0, label=diam_label),
           Line2D([0], [0], marker='s', color=MOSEQ, lw=1.9, ls='--', label='keypoint-MoSeq syllables')]
    # One legend is enough for the final 2x2 assembly; retaining it only in
    # Investigation prevents repeated boxes from covering the enlarged labels.
    if bname == 'Investigation':
        ax.legend(handles=leg, loc='center left', fontsize=FS_LEG,
                  framealpha=0.92)
    ax.set_title(bname, fontsize=FS_TITLE, loc='left')
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    # Fixed page geometry keeps the four PDF canvases aligned.
    fig.subplots_adjust(left=0.155, right=0.985, bottom=0.175, top=0.885)
    if bname == 'Investigation':
        out = OUT / 'fig_decoder_investigation_binary.pdf'
    else:
        ax.set_xlabel('')
        fig.subplots_adjust(left=0.155, right=0.985, bottom=0.175, top=0.885)
        out = OUT / f'fig_decoder_{cfg["fname"]}_binary_nox.pdf'
    fig.savefig(out, facecolor='white')
    plt.close(fig)
    print('  wrote', out)
