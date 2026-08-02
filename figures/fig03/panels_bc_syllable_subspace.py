"""Reproduce Figure 3b/c and the supplementary orthographic view."""

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa
import seaborn as sns
from math import gcd

from dlds_release import kpms_clip_video_frames as kcv
from dlds_release.empirical_wireframe import empirical_occupancy_wireframe, draw_empirical_wireframe
from dlds_release.paths import feature16_dir, moseq_results, out_dir, single_fit_dir

RUN = single_fit_dir()
FEAT = feature16_dir()
KPMS = moseq_results('single')
OUT = out_dir('fig03')
ALL = {'21_12_10_def6b_3': 'FEATURE16_kpmoseq_21_12_10_def6b_3.npy',
       '21_12_2_def6a_1': 'FEATURE16_kpmoseq_21_12_2_def6a_1.npy',
       '21_12_2_def6b_2': 'FEATURE16_kpmoseq_21_12_2_def6b_2.npy',
       '22_04_26_cage4_0': 'FEATURE16_kpmoseq_22_04_26_cage4_0.npy',
       '22_04_26_cage4_1_1': 'FEATURE16_kpmoseq_22_04_26_cage4_1_1.npy'}
OP = {'op14': 14, 'op2': 2, 'op6': 6}
FPS = 30.0
NOSE = kcv.USE_PARTS.index('nose')
SP4 = kcv.USE_PARTS.index('spine4')
VIEW = (64, -58)
SYL = 21                              # user-selected bounded within-syllable example
CLEAR_R, CLEAR_FADE = 0.07, 0.22
rng = np.random.default_rng(0)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def voxel_downsample(P, nb, cap, rng):
    mins = P.min(0)
    span = np.maximum(P.max(0) - mins, 1e-9)
    vox = np.minimum((((P - mins) / span) * np.array(nb)).astype(int), np.array(nb) - 1)
    key = (vox[:, 0] * nb[1] + vox[:, 1]) * nb[2] + vox[:, 2]
    order = rng.permutation(len(P))
    kss = key[order]
    srt = np.argsort(kss, kind='stable')
    ks = kss[srt]
    chg = np.r_[True, ks[1:] != ks[:-1]]
    start = np.maximum.accumulate(np.where(chg, np.arange(len(ks)), 0))
    within = np.arange(len(ks)) - start
    return order[srt[within < cap]]


def bout_progress(labels, syllable):
    """Return onset-to-offset coordinates for contiguous bouts of one syllable."""
    progress = np.full(len(labels), np.nan)
    frame = 0
    while frame < len(labels):
        if labels[frame] != syllable:
            frame += 1
            continue
        end = frame + 1
        while end < len(labels) and labels[end] == syllable:
            end += 1
        length = end - frame
        progress[frame:end] = 0.5 if length == 1 else np.linspace(0.0, 1.0, length)
        frame = end
    return progress


# Load the shared Figure 3 representation
Fs = np.load(f'{RUN}/Fs.npy')
Xrl, z_list, bout_progress_list, Ca_l, Xan_l, cf_list = [], [], [], [], [], []
gsum, gcnt = {}, {}


def normj(d):
    f = d.copy().astype(float)
    s = np.maximum(np.std(f, axis=1, keepdims=True), 1e-3)
    f /= s
    q = max(np.quantile(np.abs(f), 0.99), 1e-6)
    f /= q
    return f


for sid, fstem in ALL.items():
    cs = np.load(f'{RUN}/cs_{sid}.npy')
    xr = np.load(f'{FEAT}/{fstem}')[:, :cs.shape[1]]
    kp = kcv.load_clean_pixel_kp(sid)
    th = np.arctan2(kp[:, NOSE, 1] - kp[:, SP4, 1], kp[:, NOSE, 0] - kp[:, SP4, 0])
    om = np.r_[0, wrap(np.diff(th))] * FPS * 180 / np.pi
    with h5py.File(KPMS, 'r') as f:
        z0 = f[sid]['syllable'][:].astype(int)
    L = min(cs.shape[1], xr.shape[1], z0.shape[0], len(om))
    cs, xr, z = cs[:, :L], xr[:, :L], z0[:L]
    Ca_l.append(cs)
    Xan_l.append(normj(xr))
    Xrl.append(xr)
    cf_list.append(np.column_stack([cs[OP['op14']], cs[OP['op2']], cs[OP['op6']]]))
    z_list.append(z)
    bout_progress_list.append(bout_progress(z, SYL))
    for s in np.unique(z[z >= 0]):
        m = z == s
        gsum[s] = gsum.get(s, 0) + xr[:, m].sum(1)
        gcnt[s] = gcnt.get(s, 0) + int(m.sum())
Ca = np.concatenate(Ca_l, 1)
Xan = np.concatenate(Xan_l, 1)
mubN = Xan.mean(1)
Xrg = np.concatenate(Xrl, 1)
mubRg = Xrg.mean(1)
STDg = Xrg.std(1)
Cf_all = np.concatenate(cf_list, 0)
z_all = np.concatenate(z_list, 0)
bout_progress_all = np.concatenate(bout_progress_list, 0)


def d_pred(slot):
    A = Fs[slot]
    w, V = np.linalg.eig(A)
    o = np.argsort(-np.abs(w))
    V = V[:, o].real
    c = Ca[slot]
    ref = Xan[:, c > 0.05].mean(1) - mubN
    v0 = V[:, 0] * np.sign(V[:, 0] @ ref + 1e-12)
    v0 /= np.linalg.norm(v0)
    if (c < -0.05).sum():
        v1 = V[:, 1] * np.sign(V[:, 1] @ ref + 1e-12)
        v1 /= np.linalg.norm(v1)
        d = v0 + v1
    else:
        d = v0
    return d / np.linalg.norm(d)


ounit = {k: (d_pred(v) * STDg) / np.linalg.norm(d_pred(v) * STDg) for k, v in OP.items()}
gtot = sum(gcnt.values())
cls = sorted([s for s in gsum if 100 * gcnt[s] / gtot >= 0.5])
op_dirs = np.column_stack([ounit['op14'], ounit['op2'], ounit['op6']])
Uop = np.linalg.qr(op_dirs)[0]
shared_lim = float(np.ceil(max(float(np.percentile(np.abs(Cf_all[:, k]), 99.5)) * 1.10 for k in range(3)) * 10) / 10)
cframe_raw = (Uop.T @ (Xrg - mubRg[:, None])).T
ccenter_raw = np.array([(gsum[s] / gcnt[s] - mubRg) @ Uop for s in cls])
cscale_ref = max(float(np.percentile(np.abs(cframe_raw), 99.5)), float(np.abs(ccenter_raw).max()), 1e-9)
sc = 0.90 * shared_lim / cscale_ref
cframe, ccenter = cframe_raw * sc, ccenter_raw * sc

# Match the syllable colors used in the shared-space panel.
_order = sorted(cls, key=lambda s: -gcnt.get(s, 0))
_hu = sns.color_palette('husl', len(cls))
_stride = next(s for s in range(max(2, len(cls) // 3), len(cls)) if gcd(s, len(cls)) == 1)
syl_color = {s: tuple(_hu[(i * _stride) % len(cls)]) for i, s in enumerate(_order)}
syl_color[47] = matplotlib.colors.to_rgb('#a67c52')

op6lim = max(float(np.ceil(float(np.abs(ccenter[:, 2]).max()) / 0.05) * 0.05), 0.10)
zoom6 = shared_lim / op6lim
ticks_xy = np.round(np.arange(np.ceil(-shared_lim / 0.4) * 0.4, shared_lim + 0.01, 0.4), 1)
OPCOL = {'op14': '#8e44ad', 'op2': '#1f6fb2', 'op6': '#2e7d32'}

# ── syllable-21 display cloud ──
center = ccenter[cls.index(SYL)]
col_syl = syl_color[SYL]
BOUT_PURPLE = LinearSegmentedColormap.from_list(
    'syllable21_bout_progress', ['#f4df3f', col_syl, '#3b0f70']
)
inbox = ((np.abs(cframe[:, 0]) <= shared_lim) & (np.abs(cframe[:, 1]) <= shared_lim)
         & (np.abs(cframe[:, 2]) <= op6lim))
source = inbox & (z_all == SYL)
P_source = cframe[source]
progress_source = bout_progress_all[source]
keep = voxel_downsample(P_source, (31, 25, 14), 1, rng)
P = P_source[keep]
progress = progress_source[keep]
dnear = np.sqrt(((P[:, None, :2] - ccenter[None, :, :2]) ** 2).sum(-1)).min(1)
alpha = 0.72 * np.where(dnear < CLEAR_R, CLEAR_FADE, 1.0)
cols3d = BOUT_PURPLE(progress)
cols3d[:, 3] = alpha
wire_rings, wire_connectors = empirical_occupancy_wireframe(P)
lengths = []
for labels in z_list:
    is_syllable = labels == SYL
    starts = np.flatnonzero(is_syllable & np.r_[True, ~is_syllable[:-1]])
    ends = np.flatnonzero(is_syllable & np.r_[~is_syllable[1:], True])
    lengths.extend((ends - starts + 1).tolist())
print(f'syll {SYL}: {int(source.sum())} frames in box; '
      f'{len(P)} spatially uniform cloud pts; {len(lengths)} bouts; '
      f'median bout length {np.median(lengths):.0f} frames')


def fit_arrow(v, xl, yl, zl, frac=1.30):
    r = max(abs(v[0]) / xl, abs(v[1]) / yl, abs(v[2]) / zl, 1e-9)
    return v * (frac / r)


# Three-dimensional view
fig = plt.figure(figsize=(5.2, 4.7))
ax3d = fig.add_subplot(111, projection='3d')
fig.subplots_adjust(left=0.0, right=1.0, top=0.96, bottom=0.16)
ax3d.scatter(P[:, 0], P[:, 1], P[:, 2], c=cols3d,
             s=12.0,
             edgecolors='none', depthshade=False, zorder=2)
draw_empirical_wireframe(ax3d, wire_rings, wire_connectors, lw=1.7)
# the single syllable's star (same style as hero3v)
ax3d.scatter([center[0]], [center[1]], [center[2]], c='white', marker='*',
             s=300,
             edgecolors='none', depthshade=False, zorder=40)
ax3d.scatter([center[0]], [center[1]], [center[2]], c=[col_syl], marker='*',
             s=170,
             edgecolors='k', linewidths=0.9, depthshade=False, zorder=41)
# SOLID arrows = operator directions (identical to hero3v)
for lab, opn in [('op14', 'op14'), ('op2', 'op2'), ('op6', 'op6')]:
    v = fit_arrow(Uop.T @ ounit[opn], shared_lim, shared_lim, op6lim, frac=2.0 if opn == 'op6' else 1.5)
    ax3d.quiver(0, 0, 0, v[0], v[1], v[2], color=OPCOL[opn],
                 lw=3.4, arrow_length_ratio=0.12, zorder=42)
    text_frac = {'op14': 0.88, 'op2': 1.03, 'op6': 0.88}[opn]
    ax3d.text(v[0] * text_frac, v[1] * text_frac, v[2] * text_frac, lab, color=OPCOL[opn],
              fontsize=11,
              fontweight='bold', ha='center', va='center', zorder=43,
              path_effects=[pe.withStroke(linewidth=2.0, foreground='white')])
ax3d.set_xlim(-shared_lim, shared_lim)
ax3d.set_ylim(-shared_lim, shared_lim)
ax3d.set_zlim(-op6lim, op6lim)
ax3d.set_xticks(ticks_xy)
ax3d.set_yticks(ticks_xy)
ax3d.set_box_aspect((1, 1, 0.5))
ax3d.view_init(elev=VIEW[0], azim=VIEW[1])
ax3d.tick_params(labelsize=9, pad=0)
ax3d.set_xlabel('op14', fontsize=11, fontweight='bold', labelpad=2)
ax3d.set_ylabel('op2', fontsize=11, fontweight='bold', labelpad=2)
ax3d.set_zlabel(f'op6 ({zoom6:.0f}$\\times$)', fontsize=11, fontweight='bold', labelpad=1)
ax3d.set_title(f'syllable {SYL}: within-bout progression', fontsize=13, fontweight='bold', pad=4)

colorbar_axis = fig.add_axes([0.31, 0.045, 0.38, 0.035])
colorbar = fig.colorbar(
    matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap=BOUT_PURPLE),
    cax=colorbar_axis,
    orientation='horizontal',
)
colorbar.set_ticks([0, 0.5, 1])
colorbar.set_ticklabels(['onset', 'mid', 'offset'])
colorbar.ax.tick_params(labelsize=9, length=2.0, pad=0.8)
colorbar.set_label('within-bout time', fontsize=11, fontweight='bold', labelpad=1.2)

out3d = OUT / f'fig3b_syll{SYL}_3d.pdf'
fig.savefig(out3d, bbox_inches='tight', pad_inches=0.02)
plt.close(fig)
print('wrote', out3d)

# Two-dimensional op2 by op6 view. The tighter op6 scale makes structure along
# the weak-turn axis visible.
fig2, ax2 = plt.subplots(figsize=(5.4, 5.4))
ax2.axhline(0, color='0.9', lw=0.7, zorder=0)
ax2.axvline(0, color='0.9', lw=0.7, zorder=0)
ax2.scatter(P[:, 1], P[:, 2], c=progress, cmap=BOUT_PURPLE, vmin=0, vmax=1,
            s=7.2, alpha=0.72, edgecolors='none', zorder=2, rasterized=True)
ax2.scatter([center[1]], [center[2]], c='white', marker='*',
            s=320, edgecolors='none', zorder=40)
ax2.scatter([center[1]], [center[2]], c=[col_syl], marker='*',
            s=175, edgecolors='k',
            linewidths=0.8, zorder=41)
ax2.set_xlim(-shared_lim, shared_lim)
ax2.set_ylim(-op6lim, op6lim)
ax2.set_box_aspect(1)
ax2.set_xlabel('op2 (turn)', fontsize=13, color=OPCOL['op2'], fontweight='bold', labelpad=4)
ax2.set_ylabel(f'op6 ({zoom6:.0f}$\\times$)', fontsize=13,
               color=OPCOL['op6'], fontweight='bold', labelpad=4)
ax2.tick_params(labelsize=10, pad=3)
fig2.tight_layout(pad=1.08)
out2d = OUT / f'fig3c_syll{SYL}_op2_op6.pdf'
fig2.savefig(out2d, bbox_inches='tight', pad_inches=0.02)
plt.close(fig2)
print('wrote', out2d)


views = [
    (0, 1, shared_lim, shared_lim, 'op14', 'op2', 'op14 $\\times$ op2'),
    (0, 2, shared_lim, op6lim, 'op14', f'op6 ({zoom6:.0f}$\\times$)', 'op14 $\\times$ op6'),
    (1, 2, shared_lim, op6lim, 'op2', f'op6 ({zoom6:.0f}$\\times$)', 'op2 $\\times$ op6'),
]
fig_views, axes = plt.subplots(1, 3, figsize=(10.8, 3.9))
for axis, (i, j, limit_i, limit_j, x_label, y_label, title) in zip(axes, views):
    axis.axhline(0, color='0.88', lw=0.8, zorder=0)
    axis.axvline(0, color='0.88', lw=0.8, zorder=0)
    axis.scatter(P[:, i], P[:, j], c=progress, cmap=BOUT_PURPLE, vmin=0, vmax=1,
                 s=15, alpha=0.82, edgecolors='none', rasterized=True, zorder=2)
    axis.scatter([center[i]], [center[j]], c='white', marker='*', s=280,
                 edgecolors='none', zorder=4)
    axis.scatter([center[i]], [center[j]], c=[col_syl], marker='*', s=155,
                 edgecolors='k', linewidths=1.0, zorder=5)
    axis.set_xlim(-limit_i, limit_i)
    axis.set_ylim(-limit_j, limit_j)
    axis.set_box_aspect(1)
    axis.set_title(title, fontsize=13, fontweight='bold', pad=6)
    axis.set_xlabel(x_label, fontsize=12, fontweight='bold', labelpad=3)
    axis.set_ylabel(y_label, fontsize=12, fontweight='bold', labelpad=3)
    axis.tick_params(labelsize=10)

colorbar = fig_views.colorbar(
    matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap=BOUT_PURPLE),
    ax=axes, orientation='horizontal', fraction=0.055, pad=0.17, aspect=36,
)
colorbar.set_ticks([0, 0.5, 1])
colorbar.set_ticklabels(['onset', 'mid', 'offset'])
colorbar.ax.tick_params(labelsize=10, length=2.5, pad=1)
colorbar.set_label('within-bout time', fontsize=13, fontweight='bold', labelpad=3)
fig_views.suptitle(
    f'syllable {SYL}: three orthographic views of the same {len(P):,} sampled frames',
    fontsize=14, fontweight='bold', y=0.98,
)
fig_views.subplots_adjust(left=0.07, right=0.99, top=0.80, bottom=0.27, wspace=0.36)
out_views = OUT / f'fig3b_syll{SYL}_orthographic.pdf'
fig_views.savefig(out_views, bbox_inches='tight', pad_inches=0.02)
plt.close(fig_views)
print(f'wrote {out_views} using the same {len(P)} sampled frames as the 3D panel')
