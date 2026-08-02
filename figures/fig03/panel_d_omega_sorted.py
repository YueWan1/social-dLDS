"""Reproduce Figure 3d and its full-dictionary supplementary panel."""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr

from dlds_release import kpms_clip_video_frames as kcv
from dlds_release.paths import moseq_results, out_dir, single_fit_dir

OUT = out_dir('fig03')
RUN = single_fit_dir()
KPMS = moseq_results('single')
SESS = ['21_12_10_def6b_3', '21_12_2_def6a_1', '21_12_2_def6b_2',
        '22_04_26_cage4_0', '22_04_26_cage4_1_1']
USAGE_MIN = 0.005
FPS = 30.0
NOSE = kcv.USE_PARTS.index('nose')
SP4 = kcv.USE_PARTS.index('spine4')
COL = {'op14': '#8e44ad', 'op2+': '#e74c3c', 'op2-': '#f39c12',
       'op6+': '#27ae60', 'op6-': '#16a085'}
TURNS = ['op2-', 'op6-', 'op6+', 'op2+']
PHASES = ['op2+', 'op2-', 'op6+', 'op6-']


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


Zs, C14, C2, C6, OM = [], [], [], [], []
with h5py.File(KPMS, 'r') as f:
    for s in SESS:
        cs = np.load(RUN / f'cs_{s}.npy')
        z = f[s]['syllable'][:].astype(int)
        kp = kcv.load_clean_pixel_kp(s)
        th = np.arctan2(kp[:, NOSE, 1] - kp[:, SP4, 1], kp[:, NOSE, 0] - kp[:, SP4, 0])
        om = np.r_[0, wrap(np.diff(th))] * FPS * 180 / np.pi
        T = min(cs.shape[1], len(z), len(om))
        Zs.append(z[:T])
        C14.append(np.abs(cs[14, :T]))
        C2.append(cs[2, :T])
        C6.append(cs[6, :T])
        OM.append(om[:T])
Z = np.concatenate(Zs)
c14 = np.concatenate(C14)
c2 = np.concatenate(C2)
c6 = np.concatenate(C6)
omega = np.concatenate(OM)

u, cnt = np.unique(Z, return_counts=True)
usage = {int(a): b / len(Z) for a, b in zip(u, cnt)}
nframes = {int(a): int(b) for a, b in zip(u, cnt)}
allsyl = sorted(usage)                                   # every observed syllable
sub = [s for s in usage if usage[s] >= USAGE_MIN]        # the substantive 21 used in the panel

rec, om_mean = {}, {}
for s in allsyl:
    m = Z == s
    rec[s] = dict(op14=c14[m].mean(),
                  **{'op2+': np.maximum(c2[m], 0).mean(), 'op2-': np.maximum(-c2[m], 0).mean(),
                     'op6+': np.maximum(c6[m], 0).mean(), 'op6-': np.maximum(-c6[m], 0).mean()})
    om_mean[s] = float(omega[m].mean())

turn = {s: (rec[s]['op2+'] - rec[s]['op2-']) - (rec[s]['op6+'] - rec[s]['op6-']) for s in allsyl}

order_omega = sorted(sub, key=lambda s: om_mean[s])
rho, p_rho = spearmanr(
    [om_mean[s] for s in sub],
    [turn[s] for s in sub],
)

print(f'{len(sub)} substantive syllables, {len(Z):,} frames')
print(f'turn coordinate versus mean omega: rho={rho:+.3f}, p={p_rho:.2e}')

def draw_main_panel(order):
    """Stacked signed-turn recipe, one bar per syllable, ordered by measured omega.

    Tick labels carry both the syllable id and the omega value that sets the order,
    so the ordering variable is visible without a separate trace.
    """
    N = len(order)
    xs = np.arange(N)
    fig, a = plt.subplots(figsize=(7.0, 3.4))
    bottom = np.zeros(N)
    for ph in TURNS:
        h = np.array([rec[s][ph] for s in order])
        a.bar(xs, h, bottom=bottom, width=0.84, color=COL[ph],
              edgecolor='white', linewidth=0.3, zorder=3)
        bottom += h

    a.set_xlim(-0.7, N - 0.3)
    a.set_xticks(xs)
    # two rows: syllable id on top, the measured omega that orders it underneath.
    # one decimal, because rounding to integers collapses distinct syllables onto
    # duplicate labels (-8.3/-7.9, +43.2/+43.5)
    a.set_xticklabels([f's{int(s)}\n{om_mean[s]:+.1f}' for s in order],
                      fontsize=6.0)
    a.tick_params(axis='x', pad=1, length=2)
    a.tick_params(axis='y', labelsize=7.8)
    a.set_ylabel('signed-turn $|c|$ (op14 baseline removed)', fontsize=9.2)
    for sp in ('top', 'right'):
        a.spines[sp].set_visible(False)
    # row key at the left margin, so the two tick rows are self-explanatory
    # without a separate axis label
    a.text(-0.012, -0.028, 'syllable', transform=a.transAxes, fontsize=6.0,
           ha='right', va='top', color='0.35')
    a.text(-0.012, -0.080, '$\\omega$ (deg s$^{-1}$)',
           transform=a.transAxes, fontsize=6.0, ha='right', va='top', color='0.35')

    handles = [Patch(color=COL[ph], label=ph.replace('op2+', 'op2$+$').replace('op2-', 'op2$-$')
                     .replace('op6+', 'op6$+$').replace('op6-', 'op6$-$')) for ph in PHASES]
    a.legend(handles=handles, ncol=4, fontsize=8.2, loc='lower center',
             bbox_to_anchor=(0.5, 1.03), frameon=False)
    out = OUT / 'fig3d_omega_sorted.pdf'
    fig.savefig(out, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print('wrote', out)


draw_main_panel(order_omega)

# ── supplementary panel: the full dictionary down to a usable sample size ──
# >= 100 frames keeps 34 of 72 syllables and 99.7% of all frames, dropping only the
# 2-to-99-frame syllables whose means are pure noise (they are what stretch the ALL
# panel's y-axis to 0.58 and its omega range to -347..+450 deg/s).  The point of this
# panel is that the sparse syllables slot into the same omega gradient as the 21
# substantive ones, so they are kept visible rather than merged away.
sub100 = [s for s in allsyl if nframes[s] >= 100]
order100 = sorted(sub100, key=lambda s: om_mean[s])
cov100 = 100 * sum(usage[s] for s in sub100)
rho100, _ = spearmanr([om_mean[s] for s in order100], [turn[s] for s in order100])

N = len(order100)
xs = np.arange(N)
is_sub = np.array([s in set(sub) for s in order100])
figS, a = plt.subplots(figsize=(11.0, 4.6))

# shade the left-turn (omega < 0) half so the sign split reads without a gridline hunt
split = float(np.searchsorted([om_mean[s] for s in order100], 0.0)) - 0.5
a.axvspan(-0.7, split, color='#f2f5f8', zorder=0)
a.axvline(split, color='0.72', lw=0.9, ls=(0, (4, 3)), zorder=1)

bottom = np.zeros(N)
for ph in TURNS:
    h = np.array([rec[s][ph] for s in order100])
    # sparse-tail bars are drawn lighter: same data, lower confidence
    a.bar(xs[is_sub], h[is_sub], bottom=bottom[is_sub], width=0.82, color=COL[ph],
          edgecolor='white', linewidth=0.35, zorder=3)
    a.bar(xs[~is_sub], h[~is_sub], bottom=bottom[~is_sub], width=0.82, color=COL[ph],
          edgecolor='white', linewidth=0.35, alpha=0.5, zorder=3)
    bottom += h

a.set_xlim(-0.7, N - 0.3)
a.set_ylim(0, bottom.max() * 1.06)
a.set_xticks(xs)
a.set_xticklabels([f's{int(s)}\n{om_mean[s]:+.1f}' for s in order100],
                  fontsize=5.8, rotation=90)
a.tick_params(axis='x', pad=1.5, length=2)
a.tick_params(axis='y', labelsize=8)
for t, s in zip(a.get_xticklabels(), order100):
    if s not in set(sub):
        t.set_color('0.55')
a.set_ylabel('signed-turn $|c|$ (op14 baseline removed)', fontsize=9.5)
for sp in ('top', 'right'):
    a.spines[sp].set_visible(False)
a.text(-0.014, -0.020, 'syllable', transform=a.transAxes, fontsize=5.8,
       ha='right', va='top', color='0.35')
a.text(-0.014, -0.058, '$\\omega$ (deg s$^{-1}$)', transform=a.transAxes, fontsize=5.8,
       ha='right', va='top', color='0.35')

# name the two halves rather than leaving the reader to decode the sign
a.text(split / 2 / N * 0.99, 0.965, 'left turn  ($\\omega < 0$)', transform=a.transAxes,
       fontsize=8, color='0.45', ha='center', va='top')
a.text((split + N) / 2 / N, 0.965, 'right turn  ($\\omega > 0$)', transform=a.transAxes,
       fontsize=8, color='0.45', ha='center', va='top')

handles = [Patch(color=COL[ph], label=ph.replace('op2+', 'op2$+$').replace('op2-', 'op2$-$')
                 .replace('op6+', 'op6$+$').replace('op6-', 'op6$-$')) for ph in PHASES]
handles.append(Patch(facecolor='0.55', alpha=0.5, label='usage $<$ 0.5% (lighter bars)'))
a.legend(handles=handles, ncol=5, fontsize=8, loc='lower center',
         bbox_to_anchor=(0.5, 1.02), frameon=False)
a.text(0.5, 1.115, f'{N} syllables with $\\geq$100 frames ({cov100:.1f}% of all frames), '
       f'ordered by measured $\\omega$   ($\\rho = {rho100:.3f}$)',
       transform=a.transAxes, fontsize=8.4, style='italic', color='0.30',
       ha='center', va='bottom')

outS = OUT / 'fig3d_omega_sorted_full_dictionary.pdf'
figS.savefig(outS, bbox_inches='tight', pad_inches=0.03)
plt.close(figS)
print('wrote', outS)
