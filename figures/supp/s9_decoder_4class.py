"""Reproduce the published four-panel Supplementary Figure S9 decoder row."""

import numpy as np, h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from dlds_release.paths import (
    DERIVED, dyadic_cs_dir, feature27_dir, moseq_results, out_dir
)

RUN = dyadic_cs_dir()
FEAT = feature27_dir()
KPMS = moseq_results('dyadic')
OUT = out_dir('supp')
# Canonical selectivity z-scores deposited with the release.
NPZ = DERIVED / 'dyadic' / 'syllable_behavior_selectivity.npz'

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


def loso4(X, bcode, elig):
    """4-CLASS LOSO: fit multinomial logistic on all 4 labels; AUC of class-bcode posterior vs (y==bcode)."""
    aucs = []
    for held in elig:
        tr = np.isin(GRP, [e for e in elig if e != held]); te = (GRP == held)
        ytr, yte = LAB[tr], LAB[te]
        if len(np.unique(ytr)) < 4:                      # need all 4 classes to train the 4-class model
            continue
        m = (yte == bcode)
        if m.sum() == 0 or m.sum() == te.sum():
            continue
        c = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=500, class_weight='balanced')).fit(X[tr], ytr)
        cols = list(c.named_steps['logisticregression'].classes_)
        p = c.predict_proba(X[te])[:, cols.index(bcode)]
        aucs.append(roc_auc_score(m.astype(int), p))
    return float(np.mean(aucs)), float(np.std(aucs) / np.sqrt(max(len(aucs), 1))), len(aucs)


DBLUE, ORANGE, MOSEQ = '#16365c', '#c55a11', '#6f6f6f'

# behaviour -> (label code, eligibility min frac, constrained dLDS combos [(label, slots)])
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

rendered = {}
for bname, cfg in CFG.items():
    bcode = cfg['code']
    frac = np.array([(lb == bcode).mean() for (_, lb, _) in sess])
    elig = sorted(i for i in range(n) if frac[i] >= cfg['minfrac'])
    spec = sorted(s for s in SUB if Zsel[s, bcode] > 2)
    print(f'\n=== {bname} (label {bcode}); elig {len(elig)} sess; {len(spec)} specific syllables {spec} ===')

    dlds = []
    for lab, slots in cfg['d']:
        a, se, _ = loso4(dfeat(slots), bcode, elig); dlds.append((lab, len(slots), a, se))
        print(f'  dLDS  {lab:26s} {len(slots):2d}u  AUC {a:.3f}')
    a7, se7, _ = loso4(dfeat(ANALYSED7), bcode, elig)
    d7 = ('7 analysed ops', 7, a7, se7)
    print(f'  dLDS  {"7 analysed ops":26s}  7u  AUC {a7:.3f}')
    diamonds = [d7]
    if '10' in cfg.get('diamonds', ['7', '10']):
        a10, se10, _ = loso4(dfeat(ALL10), bcode, elig)
        d10 = ('all 10 operators', 10, a10, se10); diamonds.append(d10)
        print(f'  dLDS  {"all 10 operators":26s} 10u  AUC {a10:.3f}')
    moseq = []
    for lab, cols in [(f'{len(spec)} specific syll', spec), ('28 substantive syll', SUB)]:
        if not cols:
            continue
        a, se, _ = loso4(mfeat(cols), bcode, elig); moseq.append((lab, len(cols), a, se))
        print(f'  MoSeq {lab:26s} {len(cols):2d}u  AUC {a:.3f}')

    # ---- figure ----  (ladder=step-index x for attack/mount; else AUC vs #units log x)
    ladder = cfg.get('ladder', False)
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
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
    ax.plot([p[0] for p in line_xy], [p[1] for p in line_xy], '-', color=DBLUE, lw=1.6, zorder=2, alpha=0.55)
    ax.errorbar(xs_d, dy, yerr=[e for _, _, _, e in dlds], fmt='o', color=DBLUE, ms=9, capsize=3,
                mec='white', mew=1.0, zorder=5)
    for xx, dd in zip(xs_diam, diamonds):
        ax.errorbar([xx], [dd[2]], yerr=[dd[3]], fmt='D', color=ORANGE, ms=12, capsize=3,
                    mec='white', mew=1.3, zorder=6)
    ax.plot(xs_m, my, '--', color=MOSEQ, lw=1.6, zorder=2, alpha=0.7)
    ax.errorbar(xs_m, my, yerr=[e for _, _, _, e in moseq], fmt='s', color=MOSEQ, ms=8, capsize=3,
                mec='white', mew=1.0, zorder=5)

    # labels: name ABOVE the marker, AUC value to the RIGHT
    spec_units = {len(spec)} if spec else set()
    for i, (xx, (lab, u, a, _)) in enumerate(zip(xs_d, dlds)):
        yoff = (13 if i % 2 == 0 else 28) if ladder else (25 if u in spec_units else 13)
        ax.annotate(lab, (xx, a), textcoords='offset points', xytext=(0, yoff),
                    ha='center', va='bottom', fontsize=8.4, color=DBLUE)
        ax.annotate(f'{a:.2f}', (xx, a), textcoords='offset points', xytext=(7, -1),
                    ha='left', va='center', fontsize=8.2, color=DBLUE, fontweight='bold')
    if len(diamonds) == 1:
        dd = diamonds[0]; xx = xs_diam[0]
        ax.annotate('all 7 ops', (xx, dd[2]), textcoords='offset points', xytext=(0, 14),
                    ha='center', va='bottom', fontsize=8.6, color=ORANGE, fontweight='bold')
        ax.annotate(f'{dd[2]:.2f}', (xx, dd[2]), textcoords='offset points', xytext=(7, -1),
                    ha='left', va='center', fontsize=8.2, color=ORANGE, fontweight='bold')
    else:
        (d7, x7), (d10, x10) = (diamonds[0], xs_diam[0]), (diamonds[1], xs_diam[1])
        ax.annotate('7 ops', (x7, d7[2]), textcoords='offset points', xytext=(0, 14),
                    ha='center', va='bottom', fontsize=8.6, color=ORANGE, fontweight='bold')
        ax.annotate(f'{d7[2]:.2f}', (x7, d7[2]), textcoords='offset points', xytext=(-7, -1),
                    ha='right', va='center', fontsize=8.2, color=ORANGE, fontweight='bold')
        ax.annotate('all 10 ops', (x10, d10[2]), textcoords='offset points', xytext=(0, -14),
                    ha='center', va='top', fontsize=8.6, color=ORANGE, fontweight='bold')
        ax.annotate(f'{d10[2]:.2f}', (x10, d10[2]), textcoords='offset points', xytext=(7, -1),
                    ha='left', va='center', fontsize=8.2, color=ORANGE, fontweight='bold')
    for xx, (lab, u, a, _) in zip(xs_m, moseq):
        ax.annotate(lab, (xx, a), textcoords='offset points', xytext=(0, -13),
                    ha='center', va='top', fontsize=8.4, color=MOSEQ)
        ax.annotate(f'{a:.2f}', (xx, a), textcoords='offset points', xytext=(7, 1),
                    ha='left', va='center', fontsize=8.2, color=MOSEQ, fontweight='bold')

    if ladder:
        allx = xs_d + xs_diam + xs_m
        allu = [u for _, u, _, _ in dlds] + [dd[1] for dd in diamonds] + [u for _, u, _, _ in moseq]
        ax.set_xticks(allx); ax.set_xticklabels([str(u) for u in allu])
        ax.set_xlim(0.5, max(allx) + 0.7)
        ax.text(max(allx) + 0.35, 0.508, 'chance', fontsize=8.5, color='#888', ha='right', va='bottom')
        ax.set_xlabel('operator / KP-MoSeq set, in ladder order   (tick $=$ number of units)', fontsize=9.5)
    else:
        ax.set_xscale('log')
        XT = [1, 2, 3, 5, 7, 10, 28]
        ax.set_xticks(XT); ax.set_xticklabels([str(t) for t in XT])
        ax.set_xlim(0.8, 45)
        ax.text(42, 0.508, 'chance', fontsize=8.5, color='#888', ha='right', va='bottom')
        ax.set_xlabel('number of units in the decoder (log scale)', fontsize=10)
    ax.set_ylim(0.45, 0.93)
    ax.set_ylabel(f'{bname.lower()} AUC — 4-class decoder (LOSO)', fontsize=10)
    from matplotlib.lines import Line2D
    diam_label = 'dLDS all-7 operators' if len(diamonds) == 1 else 'dLDS 7 / all-10 operators'
    leg = [Line2D([0], [0], marker='o', color=DBLUE, lw=1.6, label='dLDS operators'),
           Line2D([0], [0], marker='D', color=ORANGE, lw=0, label=diam_label),
           Line2D([0], [0], marker='s', color=MOSEQ, lw=1.6, ls='--', label='KP-MoSeq syllables')]
    ax.legend(handles=leg, loc=cfg.get('legendloc', 'lower right'), fontsize=8.4, framealpha=0.92)
    ax.set_title(bname, fontsize=11, loc='left')
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.set_dpi(180)
    fig.canvas.draw()
    rendered[cfg["fname"]] = np.asarray(fig.canvas.buffer_rgba()).copy()[64:]
    plt.close(fig)


order = ["investigation", "mount", "attack", "other"]
titles = ["(a) Investigation (4-class AUC)", "(b) Mount (4-class AUC)",
          "(c) Attack (4-class AUC)", "(d) Other (4-class AUC)"]
images = [rendered[name] for name in order]
ratios = [image.shape[1] / image.shape[0] for image in images]
height = 4.3
figure, axes = plt.subplots(
    1, 4,
    figsize=(sum(ratios) * height, height * 1.08),
    gridspec_kw={"width_ratios": ratios, "wspace": 0.04},
)
for axis, image, title in zip(axes, images, titles):
    axis.imshow(image)
    axis.axis("off")
    axis.set_title(title, fontsize=11, loc="left")
output = OUT / "fig_decoder_row_combined.png"
figure.savefig(output, dpi=145, bbox_inches="tight")
plt.close(figure)
print("wrote", output)
