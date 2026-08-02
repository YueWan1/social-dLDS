"""Generate the model and dictionary components for Figure 1c.

The dictionary contains 21 syllable dynamics matrices ordered by cosine
similarity. Each matrix is paired with its onset-aligned typical trajectory.
The current and next syllables are marked in blue and red. This script requires
the optional keypoint-MoSeq dependency. The learned AR atoms ship as a compact
derived artifact, so the original training checkpoint is not required.
"""
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.collections import LineCollection
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

from dlds_release import kpms_clip_video_frames as kcv
from dlds_release.lds_topchain import INFER_DY, R_INFER, draw_top_chain
from dlds_release.paths import DERIVED, moseq_results, out_dir, require
from keypoint_moseq.util import get_typical_trajectories

KPMS = moseq_results("single")
AB_PATH = require(DERIVED / "single_mouse" / "kpmoseq_single_Ab.npy")
OUT = out_dir("fig01")
SESS = ["21_12_10_def6b_3", "21_12_2_def6a_1", "21_12_2_def6b_2", "22_04_26_cage4_0", "22_04_26_cage4_1_1"]
USAGE_MIN = 0.5

NCOL = 7                                 # exact 3 x 7 bank for the 21 used syllables
META_SLOTS = 0                           # metadata is now a readable row above the bank

C_STATE, BLACK, GREY, GREYL = '#e8861a', '#111', '#566069', '#7e8893'
Z_COL, ACC, C_OBS = '#3b4252', '#c0392b', '#7a3aa0'          # one-hot / selected box / y_t purple

# onion-skin trajectory recipe (== fig_syllable_dict_21.py)
USE = kcv.USE_PARTS
TBONES = [(USE.index(a), USE.index(b)) for a, b in kcv.BONES]
TNOSE = USE.index("nose")
PRE_T, POST_T = 5, 15
NFR = PRE_T + POST_T
SHOW_FR = np.linspace(0, NFR - 1, 5).round().astype(int)
CMAP = plt.colormaps["viridis"]

# Usage and typical trajectories for the 21 retained syllables
counts = np.zeros(200, dtype=np.int64)
coordinates, results = {}, {}
with h5py.File(KPMS, "r") as f:
    for sid in SESS:
        kp = kcv.load_clean_pixel_kp(sid)
        z0 = f[sid]["syllable"][:].astype(int)
        cen = f[sid]["centroid"][:]; hea = f[sid]["heading"][:]
        L = min(len(kp), len(z0), len(cen), len(hea))
        coordinates[sid] = kp[:L]
        results[sid] = {"syllable": z0[:L], "centroid": cen[:L], "heading": hea[:L]}
        zz = z0[:L]
        counts += np.bincount(zz[zz >= 0], minlength=200)
usage = counts / counts.sum() * 100.0
qual = sorted([int(s) for s in np.where(usage >= USAGE_MIN)[0]], key=lambda s: -usage[s])
HELD, SWITCH = 42, 0     # syllables at the chosen boundary frame t=95079 (z42,z42 -> z0)

traj = get_typical_trajectories(coordinates, results, pre=PRE_T, post=POST_T, min_frequency=0.0,
                                min_duration=3, density_sample=True, sampling_options={"n_neighbors": 50})

# Extracted from model_snapshots/550/params/Ab in the published fit.
Ab = np.load(AB_PATH, allow_pickle=False)
if Ab.shape != (100, 4, 13):
    raise ValueError(
        f"expected 100 keypoint-MoSeq AR atoms with shape (4, 13), got {Ab.shape}"
    )
Alag1, bvec = Ab[:, :, :4], Ab[:, :, 12]


def atom(s):
    return np.concatenate([Alag1[s], bvec[s][:, None]], axis=1)        # (4,5): last col = bias


# Order atoms by cosine similarity.
A_flat = np.stack([atom(s).ravel() for s in qual])           # (21, 20)
Z = linkage(pdist(A_flat, metric='cosine'), method='average', optimal_ordering=True)
order = [qual[i] for i in leaves_list(Z)]                    # similarity order
K = len(order)
heldpos, switchpos = order.index(HELD), order.index(SWITCH)
print(f"{K} atoms; trajectories for {sum(s in traj for s in order)}/{K}; "
      f"held s{HELD}@{heldpos}, switch s{SWITCH}@{switchpos}")

vmax = np.percentile(np.abs(np.stack([atom(s) for s in order])), 85)

# Canvas for the model diagram and trajectory grid.
W, H = 9.6, 9.1
fig = plt.figure(figsize=(W, H), facecolor='none')
fig.patch.set_alpha(0.0)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.set_facecolor('none')
ax.set_aspect('equal'); ax.axis('off')
BASE_AX_ARTISTS = set(ax.get_children())


def circ(x, y, color, label, r, fs=11):
    ax.add_patch(Circle((x, y), r, facecolor='white', edgecolor=color, lw=2.2, zorder=4))
    ax.text(x, y, label, ha='center', va='center', fontsize=fs, color=color, fontweight='bold', zorder=5)


def sq(x, y, color, label, half, fs=10):
    ax.add_patch(FancyBboxPatch((x-half, y-half), 2*half, 2*half,
                 boxstyle='round,pad=0.02,rounding_size=0.05',
                 facecolor='white', edgecolor=color, lw=2.4, zorder=4))
    ax.text(x, y, label, ha='center', va='center', fontsize=fs, color=color, fontweight='bold', zorder=5)


def arrow(p0, p1, color=BLACK, lw=2.2, sh=6, ms=12, ls='-'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=ms,
                 color=color, lw=lw, shrinkA=sh, shrinkB=sh, zorder=2, linestyle=ls))


def thumb(cx, cy, w, h, mat, edge, lw, sep=False):
    axi = fig.add_axes([cx / W - (w/2) / W, cy / H - (h/2) / H, w / W, h / H])
    axi.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    if sep:
        axi.axvline(mat.shape[1] - 1.5, color='#555', lw=0.9, zorder=3)
    axi.set_xticks([]); axi.set_yticks([])
    for sp in axi.spines.values():
        sp.set_color(edge); sp.set_linewidth(lw)


def traj_inset(cx, cy, w, h, T, edgecolor=None):
    axi = fig.add_axes([cx / W - (w/2) / W, cy / H - (h/2) / H, w / W, h / H])
    axi.set_facecolor('none')
    Tc = T - T[PRE_T].mean(0)
    for fi in SHOW_FR:
        col = CMAP(fi / (NFR - 1)); a = 0.5 + 0.45 * fi / (NFR - 1)
        axi.add_collection(LineCollection([[Tc[fi, i], Tc[fi, j]] for i, j in TBONES],
                                          colors=[col], lw=1.2, alpha=a, zorder=int(fi)))
        axi.scatter(Tc[fi, :, 0], Tc[fi, :, 1], c=[col], s=2.6, lw=0, alpha=a, zorder=int(fi) + 50)
    axi.annotate('', xy=tuple(Tc[SHOW_FR[-1], TNOSE]), xytext=tuple(Tc[SHOW_FR[0], TNOSE]),
                 arrowprops=dict(arrowstyle='->', color='0.3', lw=0.8, alpha=0.75))
    axi.set_aspect('equal'); axi.set_xticks([]); axi.set_yticks([]); axi.invert_yaxis(); axi.margins(0.16)
    for sp in axi.spines.values():
        if edgecolor is not None:
            sp.set_color(edgecolor); sp.set_linewidth(1.8); sp.set_visible(True)
        else:
            sp.set_visible(False)


# Shared LDS chain.
CX_TOP, Y_X = 4.10, 6.90
boxx, _ = draw_top_chain(ax, cx=CX_TOP, ys=Y_X, mode='moseq', font_scale=2.00)

# Map z_t to a selected dynamics matrix.
# Put the selection rule immediately below the inferred z row.  The activated
# one-hot entry is centred under z_t so the short vertical arrow explains that
# z_t is a scalar index, not another dynamics matrix.
HELD_COL, SWITCH_COL = '#1f6fb2', '#c0392b'          # z_t (held) vs z_{t+1} (switch)
eqy = 5.29
ncell, cw, one_at = 7, 0.26, 3
ohx0 = 2.15
x_A = 0.92
x_eq = 1.45
x_times = 4.18
x_bank = 5.35

ax.text(x_A, eqy, r'$\mathbf{A}_{z_t}$', ha='center', va='center', fontsize=25,
        color=BLACK, fontweight='bold')
ax.text(x_eq, eqy, r'$=$', ha='center', va='center', fontsize=25,
        color=BLACK, fontweight='bold')
for j in range(ncell):
    on = (j == one_at)
    ax.add_patch(Rectangle((ohx0 + j * cw - cw * 0.42, eqy - 0.16), cw * 0.84, 0.32,
                 facecolor=(HELD_COL if on else '#eef0f2'),
                 edgecolor=(HELD_COL if on else '#c8ccd2'), lw=(1.4 if on else 0.5), zorder=4))
    ax.text(ohx0 + j * cw, eqy, ('1' if on else '0'), ha='center', va='center', fontsize=18,
            color=('white' if on else '#9aa0a8'), fontweight='bold', zorder=5)
ax.text(ohx0 + (ncell - 1) * cw / 2, eqy - 0.25, r'one-hot $\mathbf{e}_{z_t}$',
        ha='center', va='top', fontsize=18, color=HELD_COL, fontweight='bold')
ax.text(x_times, eqy, r'$\times$', ha='center', va='center', fontsize=24, color=BLACK)
ax.text(x_bank, eqy, r'$\{[\mathbf{A}_k\,|\,\mathbf{b}_k]\}_{k=1}^{K}$',
        ha='center', va='center', fontsize=20, color='#444', fontweight='bold')
arrow((boxx[1], Y_X - INFER_DY - R_INFER),
      (ohx0 + one_at * cw, eqy + 0.44), color=HELD_COL, lw=1.7, sh=2, ms=10)

# Compact panel selector.
# Solid blue is the current dynamics (syllable 42); dashed red is the next
# dynamics after the switch (syllable 0).  It sits beside the exact equation so
# both explanations remain above the dictionary.
ax.text(6.48, eqy, r'$\Longrightarrow$', ha='center', va='center',
        fontsize=19, color=GREYL)
selector_x = [6.74, 7.37, 8.00]
candidate_y, pivot_y = 5.45, 5.03
pivot_x = selector_x[1]
for sx in selector_x:
    ax.plot([pivot_x, sx], [pivot_y, candidate_y], color='#c8ccd2', lw=0.9, zorder=2)
ax.plot([pivot_x, selector_x[1]], [pivot_y, candidate_y], color=HELD_COL,
        lw=2.8, solid_capstyle='round', zorder=3)
ax.plot([pivot_x, selector_x[2]], [pivot_y, candidate_y], color=SWITCH_COL,
        lw=1.9, ls=(0, (4, 3)), solid_capstyle='round', zorder=3)

selector_edges = ['#9aa0a8', HELD_COL, SWITCH_COL]
selector_labels = [r'$\cdots\,\mathbf{A}_k\,\cdots$', r'$\mathbf{A}_{42}$', r'$\mathbf{A}_{0}$']
for sx, edge, lab in zip(selector_x, selector_edges, selector_labels):
    ax.add_patch(Circle((sx, candidate_y), 0.07, facecolor='white',
                        edgecolor=edge, lw=1.7, zorder=4))
    ax.text(sx, candidate_y + 0.16, lab, ha='center', va='bottom', fontsize=18,
            color=edge, fontweight='bold' if edge != '#9aa0a8' else 'normal')
ax.add_patch(Circle((pivot_x, pivot_y), 0.07, facecolor=BLACK,
                    edgecolor=BLACK, lw=0.8, zorder=5))
ax.text(pivot_x, pivot_y - 0.11, r'$z_t$ selector', ha='center', va='top',
        fontsize=18, color=Z_COL, fontweight='bold')

# Everything added to the main axes up to here belongs to the standalone
# structure/selector export.  Remember these artists so the two exports can be
# rendered independently instead of clipping text at a shared crop boundary.
TOP_ARTISTS = [artist for artist in ax.get_children()
               if artist not in BASE_AX_ARTISTS]

# Learned syllable dictionary.
gx0 = 1.35                                # col-0 centre
dx = 1.12
mry0, dry = 3.42, 1.20                    # matrix row-0 centre y; atom-row pitch
toff = 0.42                               # trajectory sits directly below its matrix
bw, bh = 0.62, 0.40                       # matrix thumb
tw, th = 0.64, 0.46                       # slightly enlarged trajectory inset
gxc = gx0 + (NCOL - 1) * dx / 2
dictionary_title = ax.text(
    gxc, 4.64,
    r'learned dynamics dictionary  ($K=21$ used syllables)',
    ha='center', va='center', fontsize=18, color=Z_COL, fontweight='bold')


def cell_xy(pos):
    return gx0 + (pos % NCOL) * dx, mry0 - (pos // NCOL) * dry


# Color the current and next syllables; use gray for the remaining atoms.
for i in range(K):
    slot = i + META_SLOTS
    r, c = slot // NCOL, slot % NCOL
    mxc, myc = gx0 + c * dx, mry0 - r * dry
    col = HELD_COL if i == heldpos else (SWITCH_COL if i == switchpos else None)
    thumb(mxc, myc, bw, bh, atom(order[i]), edge=(col or '#9aa0a8'), lw=(2.6 if col else 0.6), sep=True)
    # Print the keypoint-MoSeq syllable index beside each atom.
    ax.text(mxc + bw / 2 + 0.035, myc, str(order[i]), ha='left', va='center', fontsize=18,
            color=(col or GREY), fontweight=('bold' if col else 'normal'))
    if order[i] in traj:
        traj_inset(mxc, myc - toff, tw, th, traj[order[i]], edgecolor=col)

# Mark the current and next syllable indices.
held_x, held_y = cell_xy(heldpos + META_SLOTS)
sw_x, sw_y = cell_xy(switchpos + META_SLOTS)
ax.text(held_x, held_y - toff - th / 2 - 0.07, r'held ($z_{t-1}\!=\!z_t$)', ha='center', va='top',
        fontsize=18, color=HELD_COL, fontweight='bold')
ax.text(sw_x, sw_y - toff - th / 2 - 0.07, r'switch ($z_{t+1}$)', ha='center', va='top',
        fontsize=18, color=SWITCH_COL, fontweight='bold')

# Syllable usage on a log scale.
def usage_inset(cx, cy, w, h):
    axi = fig.add_axes([cx / W - (w / 2) / W, cy / H - (h / 2) / H, w / W, h / H])
    axi.set_facecolor('none')
    u = np.sort(usage[usage > 0])[::-1]
    ntot = u.size
    nq = int((u >= USAGE_MIN).sum())
    ranks = np.arange(1, ntot + 1)
    axi.bar(ranks[:nq], u[:nq], width=1.0, color=HELD_COL, lw=0, zorder=3)
    axi.bar(ranks[nq:], u[nq:], width=1.0, color='#c8ccd2', lw=0, zorder=2)
    axi.axhline(USAGE_MIN, color=GREYL, lw=0.8, ls=(0, (3, 2)), zorder=4)
    axi.set_yscale('log')
    axi.set_xlim(0.0, ntot + 1)
    axi.set_ylim(u.min() * 0.6, u.max() * 3.2)
    axi.set_xticks([]); axi.set_yticks([])
    axi.spines['top'].set_visible(False); axi.spines['right'].set_visible(False)
    for s in ('left', 'bottom'):
        axi.spines[s].set_color('#b8bec6'); axi.spines[s].set_linewidth(0.7)
    # Counts are stated once in the large metadata heading below; repeating
    # them inside this narrow inset would force sub-publication-size text.
    return axi


# Readable metadata row above the exact 3 x 7 dictionary.  Keeping these keys
# outside the bank avoids forcing them into sub-6-pt type at final placement.
usage_cx, time_x = 1.85, 5.35

usage_title = ax.text(
    usage_cx, 4.22,
    f'usage: {len(qual)}/{np.count_nonzero(usage)} $\\geq$ 0.5%',
    ha='center', va='center', fontsize=18,
    color=GREY, fontweight='bold')
usage_ax = usage_inset(usage_cx, 3.92, 2.10, 0.40)

# Onion-skin time legend occupies the final grid slot.
sm = plt.cm.ScalarMappable(cmap=CMAP,
                          norm=plt.Normalize(-PRE_T / 30.0 * 1000, POST_T / 30.0 * 1000))
sm.set_array([])
cbar_y, cbar_w = 3.91, 0.90
time_title = ax.text(time_x, 4.22, 'trajectory time', ha='center', va='center',
                     fontsize=18, color=GREY, fontweight='bold')
cax = fig.add_axes([(time_x - cbar_w / 2) / W, cbar_y / H, cbar_w / W, 0.065 / H])
cax.set_facecolor('none')
cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
cb.set_ticks([])
cb.outline.set_linewidth(0.5)
time_label = ax.text(time_x, cbar_y - 0.10, r'$0\ \rightarrow\ +0.5$ s',
                     ha='center', va='top', fontsize=18, color=GREY,
                     fontweight='bold')

# Export the structure/selection mechanism and learned dictionary as two
# independent assets for manual Illustrator assembly.  Hide the opposite
# layer before each save so both components can retain complete labels and
# comfortable margins even though their source regions overlap vertically.
BOTTOM_ARTISTS = [artist for artist in ax.get_children()
                  if artist not in BASE_AX_ARTISTS and artist not in TOP_ARTISTS]
BOTTOM_INSET_AXES = [inset_ax for inset_ax in fig.axes if inset_ax is not ax]

for artist in BOTTOM_ARTISTS:
    artist.set_visible(False)
for inset_ax in BOTTOM_INSET_AXES:
    inset_ax.set_visible(False)
out = OUT / 'fig_moseq_structure_top_largefont.pdf'
fig.savefig(out, dpi=300, bbox_inches=Bbox.from_extents(0, 4.45, W, H),
            pad_inches=0, transparent=True)
print('wrote', out)
for artist in BOTTOM_ARTISTS:
    artist.set_visible(True)
for inset_ax in BOTTOM_INSET_AXES:
    inset_ax.set_visible(True)

for artist in TOP_ARTISTS:
    artist.set_visible(False)

# The standalone dictionary does not need its redundant title.  Use that freed
# vertical space to lift both metadata labels and their plots, while leaving the
# 21 dictionary cells fixed for stable Illustrator alignment.
dictionary_title.set_visible(False)
META_EXPORT_DY = 0.32
metadata_text = (usage_title, time_title, time_label)
metadata_text_positions = [artist.get_position() for artist in metadata_text]
for artist, (xpos, ypos) in zip(metadata_text, metadata_text_positions):
    artist.set_position((xpos, ypos + META_EXPORT_DY))
metadata_axes = (usage_ax, cax)
metadata_axes_positions = [metadata_ax.get_position().frozen()
                           for metadata_ax in metadata_axes]
for metadata_ax, pos in zip(metadata_axes, metadata_axes_positions):
    metadata_ax.set_position([pos.x0, pos.y0 + META_EXPORT_DY / H,
                              pos.width, pos.height])
out = OUT / 'fig_moseq_dictionary_bottom_largefont.pdf'
fig.savefig(out, dpi=300, bbox_inches=Bbox.from_extents(0, 0, W, 4.90),
            pad_inches=0, transparent=True)
print('wrote', out)
dictionary_title.set_visible(True)
for artist, position in zip(metadata_text, metadata_text_positions):
    artist.set_position(position)
for metadata_ax, pos in zip(metadata_axes, metadata_axes_positions):
    metadata_ax.set_position(pos)
for artist in TOP_ARTISTS:
    artist.set_visible(True)
plt.close(fig)
