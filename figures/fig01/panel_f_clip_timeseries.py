"""
Figure 5 (worked example): the three worked-example syllables (0/21/47) as continuous
dLDS operator coefficients, decorr=0.16 fit (op14=slot14 forward / op2=slot2 strong turn /
op6=slot6 weak turn).

(a) fig5a_clip_timeseries_largefont.pdf : a real ~7 s clip
    (session 22_04_26_cage4_1_1) -- the three
    operator coefficient traces c14/c2/c6 as lines, with the keypoint-MoSeq syllable strip below;
    syl 0 / 21 / 47 are highlighted. When MoSeq sits in syl 0 -> op2>0, syl 21 -> op2<0
    (same operator, opposite sign), syl 47 -> op2~0 with only the op14 baseline.
Run Figure 1f:
  python3 figures/fig01/panel_f_clip_timeseries.py
"""
import numpy as np, h5py
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, ConnectionPatch, Patch
from matplotlib.transforms import Bbox

from dlds_release.paths import moseq_results, out_dir, single_fit_dir

RUN = single_fit_dir()
KPMS = moseq_results('single')
OUT = out_dir('fig01')
SESS = ['21_12_10_def6b_3','21_12_2_def6a_1','21_12_2_def6b_2','22_04_26_cage4_0','22_04_26_cage4_1_1']
FPS = 30.0
OPS = [('op14 (forward baseline)', 14, '#8e44ad'),
       ('op2 (strong turn)',       2,  '#1f6fb2'),
       ('op6 (weak turn)',         6,  '#2e7d32')]
# shared absolute-coefficient y-scale for ALL stacked recipe bars (these per-frame clip panels AND the
# average recipe bars in fig5_fingerprints.py) so the two are directly comparable; covers every max
# stacked op14+|op2|+|op6| (clip op2 1.05 / clip op6 0.92 / fingerprint op2 0.86 / op6 0.82).
RECIPE_YMAX = 1.1
# op2-case worked syllables = op2-SPECIFIC syllables only (syl 47 is the forward baseline, NOT op2-specific,
# so it is no longer coloured). 0 & 90 are both op2+ (one signed operator, two MoSeq syllables), 21 is op2-.
WORK = {0: ('#c0392b', 'syl 0'), 90: ('#566573', 'syl 90'), 21: ('#2471a3', 'syl 21')}

C, Z = {}, {}
with h5py.File(KPMS, 'r') as f:
    for sid in SESS:
        z = f[sid]['syllable'][:].astype(int); cs = np.load(RUN/f'cs_{sid}.npy')
        L = min(cs.shape[1], len(z)); C[sid] = cs[:, :L]; Z[sid] = z[:L]

SYLL_LABEL_FS = 18.0
SYLL_LABEL_EFFECTS = [pe.withStroke(linewidth=2.0, foreground="white")]
CLIP_FIGSIZE, CLIP_DPI = (10.5, 8.7), 300
CLIP_OUTPUT_BBOX = Bbox.from_bounds(0, 0, *CLIP_FIGSIZE)


def make_clip(sid, a, w, workmap, outname, title):
    cs = C[sid][:, a:a+w]; z = Z[sid][a:a+w]; t = np.arange(w)/FPS
    # contiguous occurrences of the worked (coloured) syllables (>=3 frames)
    occs = []; i = 0
    while i < w:
        s = int(z[i]); j = i
        while j < w and int(z[j]) == s:
            j += 1
        if s in workmap and (j - i) >= 3:
            occs.append((s, i, j))
        i = j
    M = max(1, len(occs))

    fig = plt.figure(figsize=CLIP_FIGSIZE)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.35], hspace=0.48,
                          left=0.115, right=0.98, top=0.975, bottom=0.105)
    gst = gs[0].subgridspec(4, 1, height_ratios=[1, 1, 1, 0.55], hspace=0.20)
    axes = [fig.add_subplot(gst[0])]
    for k in range(1, 4):
        axes.append(fig.add_subplot(gst[k], sharex=axes[0]))
    # Coefficient traces
    for ax, (name, slot, col) in zip(axes[:3], OPS):
        ax.axhline(0, color='0.8', lw=0.7)
        ax.plot(t, cs[slot], color=col, lw=1.3)
        ax.set_ylabel('')
        meaning = name.split('(', 1)[1].rstrip(')')
        ax.text(0.01, 0.84, meaning, transform=ax.transAxes, color=col,
                fontsize=18, fontweight='normal', va='top')
        for s, (sc, _) in workmap.items():
            for k in np.where(z == s)[0]:
                ax.axvspan(k/FPS, (k+1)/FPS, color=sc, alpha=0.14, lw=0)
        ax.tick_params(labelsize=17, width=1.3, length=5)
        plt.setp(ax.get_xticklabels(), visible=False)
    # Syllable strip
    axs = axes[3]
    sw = np.where(np.diff(z) != 0)[0] + 1
    bounds = np.r_[0, sw, w]
    for u, v in zip(bounds[:-1], bounds[1:]):
        s = int(z[u])
        color = workmap[s][0] if s in workmap else '0.9'
        axs.add_patch(Rectangle((u/FPS, 0), (v-u)/FPS, 1, color=color, lw=0))
        if s in workmap or v - u >= 12:
            axs.text((u + v) / (2 * FPS), 0.5, str(s), ha='center', va='center',
                     fontsize=SYLL_LABEL_FS, color='black',
                     fontweight='bold' if s in workmap else 'normal',
                     path_effects=SYLL_LABEL_EFFECTS)
    axs.set_xlim(0, w/FPS); axs.set_ylim(0, 1); axs.set_yticks([])
    axs.set_xlabel('time (s)', fontsize=18)
    axs.tick_params(labelsize=17, width=1.3, length=5)
    axs.set_ylabel('')
    axs.text(0.0, 0.5, 'syllable', transform=axs.transAxes,
             fontsize=18, fontweight='bold', rotation=0,
             ha='right', va='center', clip_on=False)

    # ---- per-occurrence DYNAMIC recipe: stacked op14 + op2+/op2-/op6+/op6- bar PER FRAME
    #      (op14 baseline at the bottom). FLAT 1.5.2 colours (same key as the average recipe bars in
    #      the fingerprint column); the x-axis (frame) already encodes time, so colour is constant. ----
    BASE = {'op14': '#8e44ad', 'op2+': '#e74c3c', 'op2-': '#f39c12', 'op6+': '#27ae60', 'op6-': '#16a085'}
    PH = ['op14', 'op2-', 'op6-', 'op6+', 'op2+']   # bottom->top: op14 baseline then the signed turns

    ymax = RECIPE_YMAX
    # panel widths PROPORTIONAL to each syllable's frame count, so one frame = one fixed bar width across
    # all panels (a shared x-axis time scale); a longer syllable gets a wider panel, not wider bars.
    nfrs = [v - u for (_, u, v) in occs] or [1]
    # Compress the full history strip horizontally while keeping bars touching.
    # This shortens each frame bin physically without introducing discrete gaps.
    history_outer = gs[1].subgridspec(
        1, 3, width_ratios=[0.05, 0.90, 0.05], wspace=0)
    gsb = history_outer[1].subgridspec(1, M, wspace=0.26, width_ratios=nfrs)
    for kk, (s, u, v) in enumerate(occs):
        ax = fig.add_subplot(gsb[kk]); nfr = v - u
        c2 = cs[2, u:v]; c6 = cs[6, u:v]
        vals = {'op14': np.abs(cs[14, u:v]),
                'op2+': np.maximum(c2, 0), 'op2-': np.maximum(-c2, 0),
                'op6+': np.maximum(c6, 0), 'op6-': np.maximum(-c6, 0)}
        bottom = np.zeros(nfr)
        for ph in PH:
            ax.bar(np.arange(nfr), vals[ph], bottom=bottom, width=1.0, color=BASE[ph], linewidth=0)
            bottom += vals[ph]
        ax.set_xlim(-0.5, nfr - 0.5); ax.set_ylim(0, ymax)
        ax.set_xticks([]); ax.tick_params(labelsize=17, width=1.3, length=5)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        if kk == 0:
            ax.set_ylabel('recipe $|c|$', fontsize=19, fontweight='bold')
        else:
            ax.set_yticklabels([])
        ax.set_title(f'{workmap[s][1]}  ({nfr} fr)', fontsize=19, color=workmap[s][0],
                     fontweight='bold', pad=12)
        ax.set_xlabel('frame (time $\\to$)', fontsize=17, labelpad=2)
        # Map the discrete syllable interval onto the complete history panel:
        # onset -> left bin boundary, offset -> right bin boundary.  Two dashed
        # same-colour guides make the temporal extent explicit without implying
        # that only the interval midpoint matters.
        for strip_x, history_x in ((u / FPS, -0.5),
                                   (v / FPS, nfr - 0.5)):
            con = ConnectionPatch(
                xyA=(strip_x, 0), coordsA=axs.transData,
                xyB=(history_x, ymax), coordsB=ax.transData,
                color=workmap[s][0], lw=1.6, alpha=0.78,
                linestyle=(0, (4, 3)))
            fig.add_artist(con)
    handles = [Patch(color=BASE['op14'], label='op14'), Patch(color=BASE['op2+'], label='op2 $+$'),
               Patch(color=BASE['op2-'], label='op2 $-$'), Patch(color=BASE['op6+'], label='op6 $+$'),
               Patch(color=BASE['op6-'], label='op6 $-$')]
    fig.legend(handles=handles, ncol=5, fontsize=17, loc='lower center',
               bbox_to_anchor=(0.5, 0.008), frameon=False,
               handlelength=1.8, columnspacing=1.8)
    if title:
        axes[0].set_title(title, fontsize=18, fontweight='bold')
    out_pdf = OUT / outname
    fig.savefig(out_pdf, bbox_inches=CLIP_OUTPUT_BBOX, pad_inches=0)
    plt.close(fig)
    print('wrote', out_pdf)


# Operator-2 clip
make_clip('22_04_26_cage4_1_1', 38280, 210, WORK, 'fig5a_clip_timeseries_largefont.pdf', '')

print('Figure 1f large-font panel complete.')
