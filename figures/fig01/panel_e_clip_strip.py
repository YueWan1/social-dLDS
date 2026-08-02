"""
Clip strip + ACTUAL syllable-instance trajectory (single worked syllable, syl 21).

Takes the same ~7 s single-mouse clip used in fig5_worked_example_clips.py
(session 22_04_26_cage4_1_1, frames 38280..38490). Draws ONLY the keypoint-MoSeq
syllable colour strip (numbers + WORK colours kept: 21 blue / 90 grey / 0 red,
rest grey), and pulls a leader line from the 36-frame syllable-21 segment down to
that segment's REAL trajectory (not the across-instance dict median): the actual
keypoints of these 36 frames, egocentric-to-onset via keypoint-MoSeq's
inverse_rigid_transform (the SAME alignment get_typical_trajectories uses), drawn
from ~PRE frames before onset through all 36 frames as a time-coloured onion-skin.

The review variant enlarges the downward-expanded syllable-21 trajectory while
keeping the same 7 s strip and the syllable-90 single-frame comparison.

Needs the `moseq` optional dependency group: the onset alignment reuses
keypoint-MoSeq's own inverse_rigid_transform rather than reimplementing it, so
this trajectory is in exactly the frame get_typical_trajectories uses.

Run:  python3 figures/fig01/panel_e_clip_strip.py
"""
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.transforms import Bbox

from dlds_release import kpms_clip_video_frames as kcv
from dlds_release.paths import moseq_results, out_dir
from keypoint_moseq.util import inverse_rigid_transform, filter_centroids_headings, np_io

KPMS = moseq_results("single")
OUT = out_dir("fig01")
SID = "22_04_26_cage4_1_1"
CLIP_A, CLIP_W = 38280, 210            # same clip as fig5_worked_example_clips.py (a)
WORK = {0: "#c0392b", 90: "#566573", 21: "#2471a3"}   # worked-syllable colours (kept)
TARGET = 21                            # the syllable instance we draw the trajectory for
PRE, FPS = 5, 30.0                     # PRE frames (~167 ms) before onset, like the dict
USE = kcv.USE_PARTS
BONES = [(USE.index(a), USE.index(b)) for a, b in kcv.BONES]
NOSE = USE.index("nose")
SYLL_LABEL_FS = 22.0
SYLL_LABEL_EFFECTS = [pe.withStroke(linewidth=2.0, foreground="white")]

# Load syllable labels, keypoints, centroid, and heading
with h5py.File(KPMS, "r") as f:
    z_full = f[SID]["syllable"][:].astype(int)
    cen_full = f[SID]["centroid"][:]
    hea_full = f[SID]["heading"][:]
kp_full = kcv.load_clean_pixel_kp(SID)
L = min(len(z_full), len(cen_full), len(hea_full), len(kp_full))
z_full, cen_full, hea_full, kp_full = z_full[:L], cen_full[:L], hea_full[:L], kp_full[:L]

z = z_full[CLIP_A:CLIP_A + CLIP_W]                       # the clip's syllable sequence

# Locate the target syllable's longest contiguous run
bounds = np.r_[0, np.where(np.diff(z) != 0)[0] + 1, CLIP_W]
runs = [(int(z[u]), u, v) for u, v in zip(bounds[:-1], bounds[1:])]
cand = [(u, v) for s, u, v in runs if s == TARGET]
u, v = max(cand, key=lambda uv: uv[1] - uv[0])           # the 36-frame occurrence
onset = CLIP_A + u                                       # global onset frame
POST = v - u                                             # = 36 frames of this instance
print(f"syllable {TARGET}: clip run u={u} v={v} ({POST} fr); global onset frame {onset}")

# Select one frame from the syllable-90 segment
TOKEN = 90
cand90 = [(u2, v2) for s, u2, v2 in runs if s == TOKEN]
u90, v90 = max(cand90, key=lambda uv: uv[1] - uv[0])
tframe = (u90 + v90) // 2                                 # the one frame t (clip index)
print(f"syllable {TOKEN}: clip run u={u90} v={v90} ({v90 - u90} fr); single frame t={tframe}")

# Align this trajectory to its onset in egocentric coordinates
cf, hf = filter_centroids_headings({SID: cen_full}, {SID: hea_full}, filter_size=9)
win = kp_full[onset - PRE:onset + POST]                  # (PRE+POST, k, 2)
ego = np.asarray(np_io(inverse_rigid_transform)(win, cf[SID][onset], hf[SID][onset]))
ego = ego - ego[PRE].mean(0)                             # centre on the onset pose
nfr = ego.shape[0]
show_fr = np.unique(np.linspace(0, nfr - 1, 13).round().astype(int))   # onion-skin layers

# Figure
# Keep the panel's panoramic aspect for the full-figure compositor, but use the
# whole canvas instead of reserving three empty coefficient-trace rows.  This
# gives the downward-expanded syllable substantially more height at final size.
FIGSIZE, DPI = (10.5, 5.8), 300
OUTPUT_BBOX = Bbox.from_bounds(0, 0, *FIGSIZE)
fig = plt.figure(figsize=FIGSIZE)
gs = fig.add_gridspec(2, 1, height_ratios=[0.72, 4.90], hspace=0.22,
                      left=0.075, right=0.885, top=0.86, bottom=0.025)
axs = fig.add_subplot(gs[0])
axt = fig.add_subplot(gs[1])                             # syl 21: full instance trajectory

# Lower the trajectory panel as a unit, leaving a clearer gap below the strip.
TRAJ_DY = -0.035
axt_pos = axt.get_position()
axt.set_position([axt_pos.x0, axt_pos.y0 + TRAJ_DY,
                  axt_pos.width, axt_pos.height])

# Syllable strip
sb = np.r_[0, np.where(np.diff(z) != 0)[0] + 1, CLIP_W]
for a, b in zip(sb[:-1], sb[1:]):
    s = int(z[a])
    color = WORK[s] if s in WORK else "0.9"
    axs.add_patch(Rectangle((a / FPS, 0), (b - a) / FPS, 1, color=color, lw=0))
    if s in WORK or b - a >= 20:
        axs.text((a + b) / (2 * FPS), 0.5, str(s), ha="center", va="center",
                 fontsize=SYLL_LABEL_FS, color="black",
                 fontweight="bold" if s in WORK else "normal", path_effects=SYLL_LABEL_EFFECTS)
axs.set_xlim(0, CLIP_W / FPS); axs.set_ylim(0, 1); axs.set_yticks([])
axs.set_xlabel(""); axs.tick_params(labelsize=20, width=1.6, length=6)
axs.set_ylabel("syl.", fontsize=22, fontweight="bold")

# Mark the selected syllable-90 frame
axs.add_patch(Rectangle((tframe / FPS, -0.45), 1 / FPS, 1.9, fill=False, ls=(0, (3, 2)),
                        ec=WORK[TOKEN], lw=1.6, clip_on=False, zorder=10))
ti_x = (tframe + 1) / FPS + 0.025
axs.text(ti_x, 1.03, "$t_i$", ha="left", va="bottom",
         fontsize=22, fontweight="bold", color=WORK[TOKEN], clip_on=False)

# Onion-skin trajectory for the 36-frame instance
cmap = plt.colormaps["viridis"]
for fi in show_fr:
    col = cmap(fi / (nfr - 1)); al = 0.35 + 0.6 * fi / (nfr - 1)
    axt.add_collection(LineCollection([[ego[fi, i], ego[fi, j]] for i, j in BONES],
                                      colors=[col], lw=5.0, alpha=al, zorder=fi))
    axt.scatter(ego[fi, :, 0], ego[fi, :, 1], c=[col], s=30, zorder=fi + 50,
                edgecolor="w", lw=0.6, alpha=al)
    if fi == PRE:                                        # onset pose in black
        axt.add_collection(LineCollection([[ego[fi, i], ego[fi, j]] for i, j in BONES],
                                          colors=["k"], lw=1.8, alpha=0.9, zorder=300))
axt.plot(ego[:, NOSE, 0], ego[:, NOSE, 1], color="0.45", lw=1.6, alpha=0.7, zorder=20)  # real nose path
axt.annotate("", xy=ego[show_fr[-1], NOSE], xytext=ego[show_fr[0], NOSE],
             arrowprops=dict(arrowstyle="->", color="0.3", lw=1.8, alpha=0.7))
axt.set_aspect("equal"); axt.set_xticks([]); axt.set_yticks([]); axt.invert_yaxis(); axt.margins(0.025)
for sp in axt.spines.values():
    sp.set_visible(False)
fig.text(0.37, 0.600,
         f"syllable {TARGET}: {POST} frames ({POST / FPS:.1f} s)",
         ha="center", va="bottom", fontsize=22,
         fontweight="bold", color=WORK[TARGET])

# Connect the syllable-21 segment to its trajectory
con = ConnectionPatch(xyA=((u + v) / 2 / FPS, 0), coordsA=axs.transData,
                      xyB=(0.17, 0.86), coordsB=axt.transAxes,
                      color=WORK[TARGET], lw=2.0, alpha=0.8)
fig.add_artist(con)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-PRE / FPS * 1000, POST / FPS * 1000))
cax = fig.add_axes([0.905, 0.035, 0.018, 0.590])
cb = fig.colorbar(sm, cax=cax)
cb.set_label("time (ms)", fontsize=20, fontweight="bold", labelpad=8)
cb.ax.tick_params(labelsize=20, width=1.4, length=5, pad=3)
cb.ax.yaxis.set_label_position("left")
cb.ax.yaxis.set_ticks_position("right")

out_pdf = OUT / "fig_clip_strip_syll21_traj_largefont.pdf"
fig.savefig(out_pdf, bbox_inches=OUTPUT_BBOX, pad_inches=0, transparent=True)
plt.close(fig)
print(f"wrote {out_pdf}")
