"""Supplementary figure: the geometry gate defined on the tracked features.

Main-text Fig. 6c states the gate in the dictionary's own currency (f_4 silent
vs loud).  This supplement repeats it on the measured bearing and distance, and
adds the finer joint dependence that tertiles cannot show.

  a  label composition under the tracked-feature masks (same grammar as Fig. 6c)
  b  P(behaviour | operator recruited) over a 5x5 bearing-by-distance grid

Bins, thresholds and the 40-frame cell minimum follow the Fig. 6c gating script
(figures/fig06/panel_c_operator_gate.py); keep the two in step so the tertile
cuts stay identical. Writes supp_tracked_gate.pdf under <out_root>/supp/.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from dlds_release.paths import dyadic_cs_dir, feature27_dir, out_dir

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'font.family': 'sans-serif', 'font.size': 8,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
})

RUN_DIR = dyadic_cs_dir()
FEAT_DIR = feature27_dir()
OUT = out_dir('supp')
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
AH = np.quantile(np.abs(TH), 1/3); CL = np.quantile(D, 1/3); FR = np.quantile(D, 2/3)
ahead, close, far = np.abs(TH) <= AH, D <= CL, D > FR
ALL = np.ones(L.size, bool)
print(f'thresholds: ahead <= {AH:.1f} deg, close <= {CL:.1f} px, far > {FR:.1f} px')

BCOL = {0: '#c0392b', 1: '#e79a3c', 2: '#2c8c3c', 3: '#c9ccd0'}
BNAME = {0: 'attack', 1: 'investigation', 2: 'mount', 3: 'other'}
B_EDGES = np.array([-180, -108, -36, 36, 108, 180])
D_EDGES = np.array([0, 90, 140, 200, 300, 720])
HEAT = LinearSegmentedColormap.from_list('heat', ['#f5f5f5', '#ffe08a', '#f59e0b', '#b91c1c'])

ROWS = [
    dict(op=r'$f_9$', slot=8, beh=2, geo=ahead & close, glab='partner ahead & close'),
    dict(op=r'$f_{15}$', slot=14, beh=0, geo=close, glab='partner close'),
]

fig = plt.figure(figsize=(6.6, 6.0))
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.25], hspace=0.62, wspace=0.35)

# Panel a: composition under tracked-feature masks
for i, r in enumerate(ROWS):
    ax = fig.add_subplot(gs[i, :])
    act = C[r['slot']] > THR
    masks = [('all frames', ALL),
             (rf'$c_{{{r["op"][3:-1] or ""}}}$'.replace('$$', '') if False else
              (r'$c_9>0.05$' if r['slot'] == 8 else r'$c_{15}>0.05$'), act),
             (r.__getitem__('glab'), r['geo']),
             ((r'$c_9>0.05$' if r['slot'] == 8 else r'$c_{15}>0.05$') + r'  &  ' + r['glab'],
              act & r['geo']),
             ((r'$c_9>0.05$' if r['slot'] == 8 else r'$c_{15}>0.05$') + r'  &  partner far',
              act & far)]
    ys = np.arange(len(masks))[::-1]
    for y, (lab, m) in zip(ys, masks):
        left = 0.0
        for k in ((2, 0, 1, 3) if r['beh'] == 2 else (0, 2, 1, 3)):
            v = 100 * np.mean(L[m] == k)
            ax.barh(y, v, left=left, height=0.6, color=BCOL[k], edgecolor='white',
                    linewidth=0.6, zorder=3, label=BNAME[k] if (y == ys[0] and i == 0) else None)
            left += v
        ax.text(-1.5, y, lab, ha='right', va='center', fontsize=6.4)
        ax.text(101.5, y, f'{100*np.mean(L[m]==r["beh"]):.0f}%', ha='left', va='center',
                fontsize=7.2, fontweight='bold', color=BCOL[r['beh']])
        ax.text(101.5, y - 0.34, f'n={m.sum():,}', ha='left', va='center',
                fontsize=5.0, color='#9aa0a6')
    ax.set_xlim(0, 100); ax.set_ylim(-0.7, len(masks) - 0.3)
    ax.set_yticks([]); ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis='x', labelsize=6.2)
    ax.set_title(f'{r["op"]}: {BNAME[r["beh"]]} specificity under the tracked geometry',
                 fontsize=7.6, loc='left', pad=3)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    if i == 1:
        ax.set_xlabel('share of frames in the mask [%]', fontsize=7.2)
    if i == 0:
        ax.legend(frameon=False, fontsize=6.2, ncol=4, loc='lower left',
                  bbox_to_anchor=(0.0, 1.22), handlelength=0.9, columnspacing=1.1,
                  handletextpad=0.4)
        ax.text(-0.30, 1.55, 'a', transform=ax.transAxes, fontsize=11, fontweight='bold',
                va='top', ha='left')

# Panel b: joint dependence
for i, r in enumerate(ROWS):
    ax = fig.add_subplot(gs[2, i])
    act = C[r['slot']] > THR
    bi = np.digitize(TH[act], B_EDGES) - 1
    di = np.digitize(D[act], D_EDGES) - 1
    yb = L[act] == r['beh']
    cnt = np.zeros((5, 5)); hit = np.zeros((5, 5))
    for k in range(int(act.sum())):
        if 0 <= bi[k] < 5 and 0 <= di[k] < 5:
            cnt[di[k], bi[k]] += 1; hit[di[k], bi[k]] += yb[k]
    grid = np.where(cnt >= 40, 100.0 * hit / np.maximum(cnt, 1), np.nan)
    ax.set_facecolor('#dddddd')
    ax.imshow(grid, origin='lower', aspect='auto', cmap=HEAT, vmin=0,
              vmax=np.nanmax(grid) if np.isfinite(grid).any() else 1)
    for ii in range(5):
        for jj in range(5):
            if np.isfinite(grid[ii, jj]):
                ax.text(jj, ii, f'{grid[ii, jj]:.0f}', ha='center', va='center',
                        fontsize=6.0,
                        color='white' if grid[ii, jj] > 0.6 * np.nanmax(grid) else '#333')
            else:
                ax.text(jj, ii, 'n/a', ha='center', va='center', fontsize=5.0,
                        color='#888', style='italic')
    ax.set_xticks(range(5)); ax.set_xticklabels(['-144', '-72', '0', '72', '144'], fontsize=5.8)
    ax.set_yticks(range(5))
    ax.set_yticklabels(['0-90', '90-140', '140-200', '200-300', '300-720'], fontsize=5.6)
    ax.set_xlabel('partner bearing [deg]  (0 = ahead)', fontsize=6.6, labelpad=1)
    ax.set_ylabel('centroid distance [px]', fontsize=6.6, labelpad=1)
    ax.set_title(f'P({BNAME[r["beh"]]}) | {r["op"]} recruited', fontsize=7.4, loc='left', pad=3)
    ax.tick_params(length=2, pad=1)
    if i == 0:
        ax.text(-0.34, 1.20, 'b', transform=ax.transAxes, fontsize=11, fontweight='bold',
                va='top', ha='left')

fig.savefig(OUT / 'supp_tracked_gate.pdf', dpi=300, bbox_inches='tight')
print('wrote supp_tracked_gate.pdf')
