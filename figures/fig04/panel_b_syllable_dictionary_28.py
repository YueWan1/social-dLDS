"""Reproduce the compact dyadic keypoint-MoSeq dictionary in Figure 4b.

The panel shows onset-aligned typical trajectories for syllables used in at
least 0.5% of frames, ordered by pooled usage. It requires the ``moseq`` extra.
"""
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# The trajectory recipe is keypoint-MoSeq's own, so the panel is drawn with the
# same code the baseline was fitted with.  Needs the optional `moseq` extra.
from keypoint_moseq.util import get_typical_trajectories

from dlds_release.paths import moseq_results, out_dir, preprocessed_dir

KPMS = moseq_results("dyadic")
PRE_DIR = preprocessed_dir()
OUT = out_dir("fig04")

# resident MARS keypoints (config.yml use_bodyparts order) + visualization skeleton
NOSE, LE, RE, NECK, LH, RH, TAIL = range(7)
BONES = [(NECK, NOSE), (NECK, LE), (NECK, RE), (NECK, LH), (NECK, RH), (LH, TAIL), (RH, TAIL)]
PRE, POST, FPS = 5, 15, 30.0
USAGE_MIN = 0.5  # percent of frames -> 28 qualifying syllables


def session_ids():
    return sorted(p.name for p in PRE_DIR.glob("mouse*") if (p / "kp_clean.npy").exists())


# Pooled usage and per-session coordinates
counts = np.zeros(300, dtype=np.int64)
coordinates, results = {}, {}
with h5py.File(KPMS, "r") as f:
    keys = set(f.keys())
    for sid in session_ids():
        if sid not in keys:
            continue
        res = np.transpose(np.load(PRE_DIR / sid / "kp_clean.npy")[0], (2, 1, 0))  # (T,7,2) resident
        g = f[sid]
        z = g["syllable"][:].astype(int); cen = g["centroid"][:]; hea = g["heading"][:]
        L = min(res.shape[0], len(z), len(cen), len(hea))
        coordinates[sid] = res[:L]
        results[sid] = {"syllable": z[:L], "centroid": cen[:L], "heading": hea[:L]}
        zz = z[:L]
        counts += np.bincount(zz[zz >= 0], minlength=300)
tot = counts.sum()
usage = counts / tot * 100.0
qual = [int(s) for s in np.where(usage >= USAGE_MIN)[0]]
order = sorted(qual, key=lambda s: -usage[s])  # most-used first
print(f"{len(qual)} syllables with usage >= {USAGE_MIN}% over {len(coordinates)} sessions")
print("  " + ", ".join(f"s{s}({usage[s]:.1f}%)" for s in order))

# Onset-aligned keypoint-MoSeq typical trajectories
# min_frequency=0 because we already pre-select the 28 syllables by frame-usage (>=0.5%).
# get_typical_trajectories' own min_frequency is BOUT-frequency, which would wrongly drop
# few-but-long-bout syllables (e.g. s58: 0.53% of frames but only 101 long bouts); the
# density min_instances=50 floor is the trajectory-quality criterion we keep.
traj = get_typical_trajectories(coordinates, results, pre=PRE, post=POST, min_frequency=0.0,
                                min_duration=3, density_sample=True, sampling_options={"n_neighbors": 50})
syls = [s for s in order if s in traj]
missing = [s for s in order if s not in traj]
if missing:
    print(f"WARNING: no typical trajectory for {missing}")
print(f"{len(syls)} of {len(qual)} qualifying syllables have a typical trajectory")

# Four-by-seven onion-skin layout
nfr = PRE + POST
show_fr = np.linspace(0, nfr - 1, 7).round().astype(int)
cmap = plt.colormaps["viridis"]
ncol, nrow = 7, 4  # 4 x 7 = 28 panels, exactly the qualifying set
META_GREY = "#566069"
META_GREY_LIGHT = "#7e8893"
USAGE_BLUE = "#1f6fb2"
USAGE_TAIL = "#c8ccd2"

# Preserve usage ordering and label each thumbnail by syllable index.
# The extra canvas height accommodates the shared metadata row.
fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 2.92), facecolor="white")
axes = np.atleast_1d(axes).ravel()
for ax, s in zip(axes, syls):
    T = traj[s] - traj[s][PRE].mean(0)
    for fi in show_fr:
        col = cmap(fi / (nfr - 1)); alpha = 0.46 + 0.50 * fi / (nfr - 1)
        ax.add_collection(
            LineCollection(
                [[T[fi, i], T[fi, j]] for i, j in BONES],
                colors=[col], lw=1.05, alpha=alpha, zorder=fi,
            )
        )
        ax.scatter(
            T[fi, :, 0], T[fi, :, 1], c=[col], s=2.7,
            edgecolor="none", alpha=alpha, zorder=fi + 50,
        )
        if fi == PRE:
            ax.add_collection(
                LineCollection(
                    [[T[fi, i], T[fi, j]] for i, j in BONES],
                    colors=["k"], lw=0.55, alpha=0.78, zorder=300,
                )
            )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([]); ax.set_yticks([]); ax.invert_yaxis(); ax.margins(0.06)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.03, 0.96, f"s{s}", transform=ax.transAxes,
        ha="left", va="top", fontsize=6.2, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.35),
    )
for ax in axes[len(syls):]:
    ax.axis("off")

sm = plt.cm.ScalarMappable(
    cmap=cmap,
    norm=plt.Normalize(-PRE / FPS * 1000, POST / FPS * 1000),
)
# Match Figure 1c: usage and trajectory-time keys share a metadata row above
# the dictionary.  The grid has the same physical height and width as before;
# it is translated downward rather than compressed.
fig.subplots_adjust(left=0.012, right=0.988, top=0.719, bottom=0.025,
                    wspace=0.02, hspace=0.035)

# Usage and trajectory metadata.
# Left: the full 98-syllable usage spectrum on a log axis, with the 28
# substantive syllables in blue and the low-usage tail in grey.
usage_cx = 0.193
fig.text(
    usage_cx, 0.955,
    rf"usage: {len(syls)}/{np.count_nonzero(usage)} $\geq$ {USAGE_MIN:g}%",
    ha="center", va="center", fontsize=6.5,
    color=META_GREY, fontweight="bold",
)
uax = fig.add_axes([0.0835, 0.785, 0.219, 0.12])
u_sorted = np.sort(usage[usage > 0])[::-1]
n_tot = u_sorted.size
n_qual = int((u_sorted >= USAGE_MIN).sum())
ranks = np.arange(1, n_tot + 1)
uax.bar(ranks[:n_qual], u_sorted[:n_qual], width=1.0, color=USAGE_BLUE, lw=0, zorder=3)
uax.bar(ranks[n_qual:], u_sorted[n_qual:], width=1.0, color=USAGE_TAIL, lw=0, zorder=2)
uax.axhline(USAGE_MIN, color=META_GREY_LIGHT, lw=0.8, ls=(0, (3, 2)), zorder=4)
uax.set_yscale("log")
uax.set_xlim(0.0, n_tot + 1)
uax.set_ylim(u_sorted.min() * 0.6, u_sorted.max() * 3.2)
uax.set_xticks([]); uax.set_yticks([])
uax.spines["top"].set_visible(False); uax.spines["right"].set_visible(False)
for _sp in ("left", "bottom"):
    uax.spines[_sp].set_color("#b8bec6"); uax.spines[_sp].set_linewidth(0.7)

# Right: the same title / unlabelled viridis strip / time-range grammar used
# in Figure 1c.  The range retains the actual pre-onset window used here.
time_x = 0.557
fig.text(
    time_x, 0.955, "trajectory time",
    ha="center", va="center", fontsize=6.5,
    color=META_GREY, fontweight="bold",
)
cax = fig.add_axes([0.510, 0.844, 0.094, 0.010])
cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_ticks([])
cb.outline.set_linewidth(0.5)
fig.text(
    time_x, 0.805,
    rf"${-PRE / FPS:.2f}\ \rightarrow\ +{POST / FPS:.1f}\ \mathrm{{s}}$",
    ha="center", va="center", fontsize=6.5,
    color=META_GREY, fontweight="bold",
)
compact = OUT / "fig_syllable_dict_28_compact.pdf"
fig.savefig(compact, facecolor="white")
plt.close(fig)
print(f"wrote {compact}")
