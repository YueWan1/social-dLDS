"""Reproduce Figure 6c.

What panel c has to show: what the geometry gate does to the behavioural
specificity of f_9 and f_15.  So each mask is drawn as the full composition of
the four behaviour labels, not as a single probability: the reader sees the
target behaviour grow AND what it grows out of.

One row per operator, five masks per row:
    all frames | operator alone | geometry alone | operator x geometry | operator x far
Output: panelc_v4_operator_gate.pdf
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'font.family': 'sans-serif', 'font.size': 8,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
})

from dlds_release.paths import dyadic_cs_dir, feature27_dir, out_dir

RUN_DIR = dyadic_cs_dir()
FEAT_DIR = feature27_dir()
OUT = out_dir('fig06')
THR = 0.05

C, L, D, TH = [], [], [], []
for sid in range(1, 71):
    cp = RUN_DIR / f'cs_mouse{sid:03d}.npy'
    xp = FEAT_DIR / f'FEATURE27_mouse{sid:03d}.npy'
    lp = FEAT_DIR / f'cleaned_label_mouse{sid:03d}.npy'
    if not (cp.exists() and xp.exists() and lp.exists()):
        continue
    cs = np.load(cp); xr = np.load(xp); lb = np.load(lp).ravel()
    xr = xr if xr.shape[0] == 27 else xr.T
    T = min(cs.shape[1], xr.shape[1], lb.size)
    C.append(cs[:, :T]); L.append(lb[:T]); D.append(xr[14, :T])
    TH.append(np.degrees(np.arctan2(xr[26, :T], xr[25, :T])))
C = np.concatenate(C, 1); L = np.concatenate(L)
D = np.concatenate(D); TH = np.concatenate(TH)
# the gate is stated in the model's own currency: f_4's magnitude IS the distance
# read-out (Fig. 5b), so f_4 silent = the animals are close, and a large |c_4| =
# they are far apart.  No external mask on the tracked features is used.
c4 = np.abs(C[3])
Q2 = np.quantile(c4, 2/3)
near, faroff = c4 <= 1e-9, c4 > Q2
ALL = np.ones(L.size, bool)
print(f'f_4 silent: n={near.sum():,}, mean centroid distance {D[near].mean():.0f} px')
print(f'f_4 top third (|c_4|>{Q2:.2f}): n={faroff.sum():,}, mean distance {D[faroff].mean():.0f} px')

# behaviour colours follow panel a
BCOL = {0: '#c0392b', 1: '#e79a3c', 2: '#2c8c3c', 3: '#c9ccd0'}
BNAME = {0: 'attack', 1: 'investigation', 2: 'mount', 3: 'other'}

ROWS = [
    dict(op=r'$f_9$', slot=8, beh=2, geo=near,
         rows=[('all frames', ALL, 'ref'),
               (r'$c_9>0.05$', None, 'op'),
               (r'$f_4$ silent  (partner near)', near, 'geo'),
               (r'$c_9>0.05$  &  $f_4$ silent', None, 'both'),
               (r'$c_9>0.05$  &  $|c_4|$ top third  (far)', None, 'anti')]),
    dict(op=r'$f_{15}$', slot=14, beh=0, geo=near,
         rows=[('all frames', ALL, 'ref'),
               (r'$c_{15}>0.05$', None, 'op'),
               (r'$f_4$ silent  (partner near)', near, 'geo'),
               (r'$c_{15}>0.05$  &  $f_4$ silent', None, 'both'),
               (r'$c_{15}>0.05$  &  $|c_4|$ top third  (far)', None, 'anti')]),
]

fig, axes = plt.subplots(2, 1, figsize=(5.4, 3.9), gridspec_kw=dict(hspace=0.55))
for ax, r in zip(axes, ROWS):
    act = C[r['slot']] > THR
    masks = []
    for lab, m, kind in r['rows']:
        mm = {'ref': m, 'geo': m, 'op': act, 'both': act & r['geo'], 'anti': act & faroff}[kind]
        masks.append((lab, mm, kind))
    ys = np.arange(len(masks))[::-1]
    for y, (lab, m, kind) in zip(ys, masks):
        left = 0.0
        for k in (2, 0, 1, 3) if r['beh'] == 2 else (0, 2, 1, 3):
            v = 100 * np.mean(L[m] == k)
            ax.barh(y, v, left=left, height=0.62, color=BCOL[k],
                    edgecolor='white', linewidth=0.6, zorder=3,
                    label=BNAME[k] if y == ys[0] else None)
            left += v
        tgt = 100 * np.mean(L[m] == r['beh'])
        ax.text(-1.5, y, lab, ha='right', va='center', fontsize=6.4)
        ax.text(101.5, y, f'{tgt:.0f}%  ', ha='left', va='center', fontsize=7.2,
                fontweight='bold', color=BCOL[r['beh']])
        ax.text(101.5, y - 0.34, f'n={m.sum():,}', ha='left', va='center',
                fontsize=5.0, color='#9aa0a6')
    ax.set_xlim(0, 100); ax.set_ylim(-0.7, len(masks) - 0.3)
    ax.set_yticks([]); ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis='x', labelsize=6.2)
    ax.set_title(f'{r["op"]}: {BNAME[r["beh"]]} specificity alone and composed with $f_4$',
                 fontsize=7.6, loc='left', pad=3)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
axes[0].legend(frameon=False, fontsize=6.2, ncol=4, loc='lower left',
               bbox_to_anchor=(0.0, 1.16), handlelength=0.9, columnspacing=1.1,
               handletextpad=0.4)
axes[1].set_xlabel('share of frames in the mask [%]', fontsize=7.2)
fig.savefig(OUT / 'panelc_v4_operator_gate.pdf', dpi=300,
            bbox_inches='tight', transparent=True)
print('wrote panelc_v4_operator_gate.pdf')
