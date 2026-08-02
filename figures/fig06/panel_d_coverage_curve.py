"""Figure 6d: cumulative coverage of investigation frames.

dLDS operators are added greedily using |c| > 0.05. Keypoint-MoSeq syllables
are ordered by usage. Outputs are written to the Figure 6 stage directory.
"""
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.5,
    'axes.linewidth': 0.9, 'xtick.major.width': 0.9, 'ytick.major.width': 0.9,
})

from dlds_release.paths import (
    DERIVED,
    dyadic_cs_dir,
    feature27_dir,
    moseq_results,
    out_dir,
)

RUN = dyadic_cs_dir()
FEAT = feature27_dir()
RES = moseq_results('dyadic')
# Only the substantive-syllable list is taken from this table, and it ships in
# the repository; the recomputed copy is used when it is present.
NPZ = DERIVED / 'dyadic' / 'syllable_behavior_selectivity.npz'
OUT = out_dir('fig06')
F2, F3, F6 = 1, 2, 5
THR = 0.05

DLDS_COL = '#1f6fb2'
MOSEQ_COL = '#7f7f7f'
ACC = '#d62728'

# Substantive syllables with at least 0.5% usage
d = np.load(NPZ)
SUB = sorted(int(s) for s in d['sub'])

# Load aligned syllable labels and substrate coefficients
SYv, C2v, C3v, C6v = [], [], [], []
with h5py.File(RES, 'r') as f:
    hk = {k.replace('mouse', ''): k for k in f.keys()}
    for sid in range(1, 71):
        s3 = f'{sid:03d}'
        cp = RUN / f'cs_mouse{s3}.npy'; lp = FEAT / f'cleaned_label_mouse{s3}.npy'
        if s3 not in hk or not (cp.exists() and lp.exists()):
            continue
        cs = np.load(cp); lb = np.load(lp); syl = f[hk[s3]]['syllable'][:].astype(int)
        T = min(cs.shape[1], len(lb), len(syl))
        m = lb[:T] == 1
        SYv.append(syl[:T][m]); C2v.append(cs[F2][:T][m])
        C3v.append(cs[F3][:T][m]); C6v.append(cs[F6][:T][m])
SY = np.concatenate(SYv)
C2 = np.concatenate(C2v); C3 = np.concatenate(C3v); C6 = np.concatenate(C6v)
N = SY.size
print(f'investigation frames: {N}')

# Greedy cumulative dLDS coverage
ops = {'f_2': np.abs(C2) > THR, 'f_3': np.abs(C3) > THR, 'f_6': np.abs(C6) > THR}
remaining = list(ops)
covered = np.zeros(N, bool)
dlds_order, dlds_cum = [], []
while remaining:
    best, best_gain = None, -1
    for k in remaining:
        gain = int((ops[k] & ~covered).sum())
        if gain > best_gain:
            best_gain, best = gain, k
    covered |= ops[best]
    dlds_order.append(best); dlds_cum.append(100 * covered.mean())
    remaining.remove(best)
dlds_unexpressed = int((~covered).sum())
print('dLDS greedy order:', dlds_order)
print('dLDS cumulative coverage %:', [f'{v:.3f}' for v in dlds_cum])
print(f'dLDS unexpressed frames: {dlds_unexpressed} ({100*dlds_unexpressed/N:.3f}%)')

# Cumulative keypoint-MoSeq coverage ranked by usage
cnt = np.array(sorted((int((SY == s).sum()) for s in SUB), reverse=True), dtype=float)
moseq_cum = 100 * np.cumsum(cnt) / N           # 28 points; plateaus below 100 by the rare overflow
moseq_ceiling = float(moseq_cum[-1])
n95 = int(np.searchsorted(moseq_cum, 95.0) + 1)
n50 = int(np.searchsorted(moseq_cum, 50.0) + 1)
overflow = 100.0 - moseq_ceiling
print(f'MoSeq-28 ceiling {moseq_ceiling:.2f}%  (overflow {overflow:.2f}%)  '
      f'| 50% needs {n50} syll, 95% needs {n95} syll')

# Coverage curves.
fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.6))
plt.subplots_adjust(left=0.105, right=0.97, top=0.92, bottom=0.125)

xm = np.arange(1, len(moseq_cum) + 1)
ax.plot(xm, moseq_cum, '-o', color=MOSEQ_COL, ms=3.4, lw=1.6, zorder=3,
        label='keypoint-MoSeq syllables (cumulative, by usage)')
xd = np.arange(1, len(dlds_cum) + 1)
ax.plot(xd, dlds_cum, '-o', color=DLDS_COL, ms=5.5, lw=2.0, zorder=6,
        label='dLDS substrate operators (cumulative)')

# label each dLDS point with the operator it ADDS (placed above, in the top margin)
lbls = [rf'${dlds_order[0].replace("_","_{")+"}"}$'] + \
       [rf'$+{o.replace("_","_{")+"}"}$' for o in dlds_order[1:]]
for x, y, t in zip(xd, dlds_cum, lbls):
    ax.annotate(t, (x, y), textcoords='offset points', xytext=(0, 9),
                fontsize=9, color=DLDS_COL, fontweight='bold', ha='center', va='bottom')

# dLDS endpoint headline (clear lower-left area, arrow up to the 3-operator point)
ax.annotate(f'3 operators reach {dlds_cum[-1]:.2f}%\n'
            f'(only {dlds_unexpressed} of {N:,} frames unexpressed;\n'
            f'each point adds one operator)',
            (xd[-1], dlds_cum[-1]), textcoords='offset points', xytext=(34, -38),
            fontsize=9, color=DLDS_COL, ha='left', va='center', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=DLDS_COL, lw=1.1))

# 95% reference + where MoSeq crosses it (95% readout lives on the y-axis, see set_yticks below)
ax.axhline(95, color='#bbbbbb', lw=1.0, ls='--', zorder=1)
ax.plot([n95, n95], [0, moseq_cum[n95 - 1]], color=MOSEQ_COL, lw=1.0, ls=':', zorder=2)
ax.annotate(f'keypoint-MoSeq: {n95} of 28\nsyllables for 95%',
            (n95, 95), textcoords='offset points', xytext=(-8, -58),
            fontsize=8.6, color='#555', ha='right',
            arrowprops=dict(arrowstyle='->', color='#999', lw=0.9))

# MoSeq ceiling + the 3.6% it cannot express (far right, in the margin ABOVE the grey curve)
ax.axhline(moseq_ceiling, color=MOSEQ_COL, lw=0.9, ls=(0, (2, 2)), zorder=1, alpha=0.6)
ax.annotate('', xy=(28.7, 100), xytext=(28.7, moseq_ceiling),
            arrowprops=dict(arrowstyle='<->', color=ACC, lw=1.4))
ax.text(28.3, 104.0,
        f'{overflow:.1f}%: the 28 syllables\ncannot express it',
        fontsize=8.4, color=ACC, ha='right', va='center')
# ceiling value labelled at the right (too close to 95% to be a second left tick)
ax.text(28.95, moseq_ceiling, f'{moseq_ceiling:.1f}%', fontsize=7.8,
        color='#888', ha='left', va='center')

ax.set_xlim(0.3, 30.3)
ax.set_ylim(0, 108)
# put the 95% readout on the y-axis with the other ticks (clear of the blue points)
ax.set_yticks([0, 20, 40, 60, 80, 95, 100])
ax.set_yticklabels(['0', '20', '40', '60', '80', '95', '100'])
for _t in ax.get_yticklabels():
    if _t.get_text() == '95':
        _t.set_color('#777')
ax.set_xlabel('number of units (operators / syllables, cumulative)')
ax.set_ylabel('investigation frames expressed (cumulative %)')
ax.set_title('A few continuous operators express what the discrete code fragments',
             fontsize=10.5, loc='left')
ax.legend(loc='lower right', frameon=True, framealpha=0.95)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.grid(axis='y', alpha=0.18)

# Main Figure 6 panel c: a half-width source with typography enlarged before
# rasterization.  The statistics and annotations are unchanged; only the
# physical canvas and print-scale readability differ from the reusable source.
fig.set_size_inches(5.4, 3.7, forward=True)
ax.set_title('Continuous operators cover investigation parsimoniously',
             fontsize=12.5, loc='left')
ax.set_xlabel('number of units', fontsize=11.5)
ax.set_ylabel('investigation frames expressed (%)', fontsize=11.5)
ax.tick_params(axis='both', labelsize=9.5)
for item in ax.texts:
    txt = item.get_text()
    if txt.startswith('3 operators reach'):
        item.set_text(f'3 operators: {dlds_cum[-1]:.2f}% coverage\n'
                      f'({dlds_unexpressed} / {N:,} frames unexpressed)')
        item.set_fontsize(9.5)
        item.set_position((34, -34))
    elif txt.startswith('keypoint-MoSeq:'):
        item.set_text(f'{n95}/28 syllables for 95%')
        item.set_fontsize(9)
        item.set_position((-8, -45))
    elif 'the 28 syllables' in txt:
        item.set_text(f'{overflow:.1f}% outside the\n28-syllable set')
        item.set_fontsize(9)
    elif txt.endswith('%'):
        item.set_fontsize(8.5)
    else:
        item.set_fontsize(9)
legend = ax.get_legend()
legend.get_texts()[0].set_text('keypoint-MoSeq syllables')
legend.get_texts()[1].set_text('dLDS substrate operators')
for item in legend.get_texts():
    item.set_fontsize(9.5)
fig.subplots_adjust(left=0.14, right=0.965, top=0.90, bottom=0.16)
main_out = OUT / 'fig_main6c_invest_coverage_curve.pdf'
fig.savefig(main_out, facecolor='white', bbox_inches='tight')
print('wrote', main_out)
plt.close(fig)
