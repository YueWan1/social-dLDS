"""Reproduce the published Figure 3a shared-space panel."""

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa
import seaborn as sns
from math import gcd

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
VIEW = (46, -58)
CLEAR_R, CLEAR_FADE = 0.07, 0.22   # mildly fade the 3D cloud near each star so stars peek through
FOCUS_SYL = 21
rng = np.random.default_rng(0)


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


# ── load (prep verbatim) ──
Fs = np.load(f'{RUN}/Fs.npy')
Xrl, z_list, Ca_l, Xan_l, cf_list = [], [], [], [], []
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
    with h5py.File(KPMS, 'r') as f:
        z0 = f[sid]['syllable'][:].astype(int)
    L = min(cs.shape[1], xr.shape[1], z0.shape[0])
    cs, xr, z = cs[:, :L], xr[:, :L], z0[:L]
    Ca_l.append(cs)
    Xan_l.append(normj(xr))
    Xrl.append(xr)
    cf_list.append(np.column_stack([cs[OP['op14']], cs[OP['op2']], cs[OP['op6']]]))
    z_list.append(z)
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
devg = np.array([gsum[s] / gcnt[s] - mubRg for s in cls])
op_dirs = np.column_stack([ounit['op14'], ounit['op2'], ounit['op6']])
Uop = np.linalg.qr(op_dirs)[0]
e_syll_in_op = float(((devg @ Uop) ** 2).sum() / (devg ** 2).sum())
shared_lim = float(np.ceil(max(float(np.percentile(np.abs(Cf_all[:, k]), 99.5)) * 1.10 for k in range(3)) * 10) / 10)
cframe_raw = (Uop.T @ (Xrg - mubRg[:, None])).T
ccenter_raw = np.array([(gsum[s] / gcnt[s] - mubRg) @ Uop for s in cls])
cscale_ref = max(float(np.percentile(np.abs(cframe_raw), 99.5)), float(np.abs(ccenter_raw).max()), 1e-9)
sc = 0.90 * shared_lim / cscale_ref
cframe, ccenter = cframe_raw * sc, ccenter_raw * sc

# husl palette (equal-luminance, harmonious), assigned by usage with a coprime hue stride so the
# most-used syllables get maximally separated hues
_order = sorted(cls, key=lambda s: -gcnt.get(s, 0))
_hu = sns.color_palette('husl', len(cls))
_stride = next(s for s in range(max(2, len(cls) // 3), len(cls)) if gcd(s, len(cls)) == 1)
syl_color = {s: tuple(_hu[(i * _stride) % len(cls)]) for i, s in enumerate(_order)}
# Syllable 47 was originally pink and visually competed with the lavender syllable 21.
# A muted warm brown keeps it categorical while allowing syllable 21 to sit above it.
syl_color[47] = matplotlib.colors.to_rgb('#a67c52')
clump = set(cls)
ccenter_col = np.array([syl_color[s] for s in cls])

op6lim = max(float(np.ceil(float(np.abs(ccenter[:, 2]).max()) / 0.05) * 0.05), 0.10)
inbox = (np.abs(cframe[:, 0]) <= shared_lim) & (np.abs(cframe[:, 1]) <= shared_lim) & (np.abs(cframe[:, 2]) <= op6lim)
Pin, zin = cframe[inbox], z_all[inbox]
# One spatially uniform sample is shared by the 3D view and all three projections.
# Two representatives per voxel retain the broad occupancy while opening enough
# space to make the centroid geometry legible in every view.
keep = voxel_downsample(Pin, (31, 25, 14), 2, rng)
P, zk = Pin[keep], zin[keep]

def render_order(points, syllables):
    """Draw background states first and syllable 21 last without fitting a boundary."""
    tier = np.array([3 if s == FOCUS_SYL else 1 if s == 47 else 2 if s in clump else 0
                     for s in syllables])
    order = np.argsort(tier, kind='stable')
    return points[order], syllables[order]

P, zk = render_order(P, zk)
rgb = np.array([syl_color[s] if s in clump else (0.80, 0.80, 0.80) for s in zk])
al_base = np.array([0.72 if s == FOCUS_SYL else 0.48 if s == 47 else
                    0.60 if s in clump else 0.12 for s in zk])
cols2d = np.column_stack([rgb, al_base])
# 3D cloud: mildly fade points near a centroid so its star remains readable.
dnear = np.sqrt(((P[:, None, :2] - ccenter[None, :, :2]) ** 2).sum(-1)).min(1)
cols3d = np.column_stack([rgb, al_base * np.where(dnear < CLEAR_R, CLEAR_FADE, 1.0)])
wire_rings, wire_connectors = empirical_occupancy_wireframe(P[zk == FOCUS_SYL])
zoom6 = shared_lim / op6lim
print(f'{len(cframe)} frames; {len(cls)} syllables; {len(P)} shared cloud pts; '
      f'{100*e_syll_in_op:.0f}% shared energy')

ticks_xy = np.round(np.arange(np.ceil(-shared_lim / 0.4) * 0.4, shared_lim + 0.01, 0.4), 1)
OPCOL = {'op14': '#8e44ad', 'op2': '#1f6fb2', 'op6': '#2e7d32'}


def fit_arrow(v, xl, yl, zl, frac=1.30):
    r = max(abs(v[0]) / xl, abs(v[1]) / yl, abs(v[2]) / zl, 1e-9)
    return v * (frac / r)


def stars(ax, C, threeD=False):
    if threeD:
        ax.scatter(C[:, 0], C[:, 1], C[:, 2], c='white', marker='*',
                   s=220, edgecolors='none',
                   depthshade=False, zorder=40)
        ax.scatter(C[:, 0], C[:, 1], C[:, 2], c=ccenter_col, marker='*',
                   s=120, edgecolors='k',
                   linewidths=0.9, depthshade=False, zorder=41)
    else:
        ax.scatter(C[:, 0], C[:, 1], c='white', marker='*',
                   s=135, edgecolors='none', zorder=40)
        ax.scatter(C[:, 0], C[:, 1], c=ccenter_col, marker='*',
                   s=75, edgecolors='k',
                   linewidths=0.8, zorder=41)


fig = plt.figure(figsize=(5.0, 4.2))
gs = fig.add_gridspec(
    2, 3, height_ratios=[2.15, 1.35], width_ratios=[1, 1, 1],
    hspace=0.16, wspace=0.28
)
point3d, point2d = 2.8, 3.0
arrow_lw, arrow_fs = 2.8, 7.2
axis_fs, tick_fs, title_fs = 7.0, 5.8, 6.8

# ── 3D hero ──
ax3d = fig.add_subplot(gs[0, :2], projection='3d')
focus = zk == FOCUS_SYL
ax3d.scatter(P[~focus, 0], P[~focus, 1], P[~focus, 2], c=cols3d[~focus], s=point3d,
             edgecolors='none', depthshade=False, zorder=2)
ax3d.scatter(P[focus, 0], P[focus, 1], P[focus, 2], c=cols3d[focus],
             s=3.3, edgecolors='none', depthshade=False, zorder=6)
draw_empirical_wireframe(ax3d, wire_rings, wire_connectors, lw=1.05)
stars(ax3d, ccenter, threeD=True)
# SOLID arrows = operator directions
for lab, opn in [('op14', 'op14'), ('op2', 'op2'), ('op6', 'op6')]:
    v = fit_arrow(Uop.T @ ounit[opn], shared_lim, shared_lim, op6lim,
                  frac=2.0 if opn == 'op6' else 1.5)
    ax3d.quiver(0, 0, 0, v[0], v[1], v[2], color=OPCOL[opn], lw=arrow_lw,
                 arrow_length_ratio=0.12, zorder=42)
    ax3d.text(v[0] * 1.07, v[1] * 1.07, v[2] * 1.07, lab, color=OPCOL[opn], fontsize=arrow_fs,
              fontweight='bold', ha='center', va='center', zorder=43,
              path_effects=[pe.withStroke(linewidth=2.6, foreground='white')])
ax3d.set_xlim(-shared_lim, shared_lim)
ax3d.set_ylim(-shared_lim, shared_lim)
ax3d.set_zlim(-op6lim, op6lim)
ax3d.set_xticks(ticks_xy)
ax3d.set_yticks(ticks_xy)
ax3d.set_zticks([-op6lim, 0, op6lim])
ax3d.set_box_aspect((1, 1, 0.68))
ax3d.view_init(elev=VIEW[0], azim=VIEW[1])
ax3d.tick_params(labelsize=tick_fs, pad=0)
ax3d.set_xlabel('op14', fontsize=axis_fs, fontweight='bold', labelpad=-3)
ax3d.set_ylabel('op2', fontsize=axis_fs, fontweight='bold', labelpad=-3)
ax3d.set_zlabel(f'op6 ({zoom6:.0f}$\\times$)', fontsize=axis_fs, fontweight='bold',
                labelpad=-2)

# ── syllable legend (top-right column) ──
axleg = fig.add_subplot(gs[0, 2])
axleg.axis('off')
star_handles = [Line2D([0], [0], marker='*', color='none', markerfacecolor=syl_color[s],
                       markeredgecolor='k', markeredgewidth=0.5,
                       markersize=8, label=str(int(s))) for s in cls]
axleg.legend(handles=star_handles, loc='lower center', ncol=2,
             fontsize=5.8, title='syllable $\\bigstar$',
             title_fontsize=7.0, handletextpad=0.05,
             columnspacing=0.35, labelspacing=0.20,
             framealpha=0.0, borderpad=0.4, bbox_to_anchor=(0.5, 0.0))

# ── three orthographic views below ──
views = [(0, 1, shared_lim, shared_lim, 'op14', 'op2', 'op14 $\\times$ op2'),
         (0, 2, shared_lim, op6lim, 'op14', f'op6 ({zoom6:.0f}$\\times$)', 'op14 $\\times$ op6'),
         (1, 2, shared_lim, op6lim, 'op2', f'op6 ({zoom6:.0f}$\\times$)', 'op2 $\\times$ op6')]
for col, (i, j, li, lj, xl, yl, ttl) in enumerate(views):
    ax = fig.add_subplot(gs[1, col])
    ax.axhline(0, color='0.9', lw=0.7, zorder=0)
    ax.axvline(0, color='0.9', lw=0.7, zorder=0)
    focus = zk == FOCUS_SYL
    ax.scatter(P[~focus, i], P[~focus, j], c=cols2d[~focus], s=point2d,
               edgecolors='none', zorder=2, rasterized=True)
    ax.scatter(P[focus, i], P[focus, j], c=cols2d[focus],
               s=2.8, edgecolors='none', zorder=3, rasterized=True)
    stars(ax, ccenter[:, [i, j]])
    ax.set_xlim(-li, li)
    ax.set_ylim(-lj, lj)
    ax.set_xlabel(xl, fontsize=6.2, labelpad=0)
    ax.set_ylabel(yl, fontsize=6.2, labelpad=0)
    ax.set_title(ttl, fontsize=title_fs, pad=1)
    ax.tick_params(labelsize=5.8, pad=1)
    ax.set_box_aspect(1)

outpath = OUT / 'fig3a_shared_space.pdf'
fig.savefig(outpath, bbox_inches='tight')
plt.close(fig)
print('wrote', outpath)
