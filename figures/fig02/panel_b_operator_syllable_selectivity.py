"""Shared selectivity and trajectory calculations for Figure 2b."""
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

SYL_CMAP_NAME = "viridis"

# Syllables 0, 90, and 21 also appear in Figure 1e-f.
FIG1_EXAMPLES = {0, 90, 21}
from dlds_release import kpms_clip_video_frames as kcv
from dlds_release.paths import moseq_results, single_fit_dir

RUN = single_fit_dir()
KPMS = moseq_results("single")
SESS = ["21_12_10_def6b_3", "21_12_2_def6a_1", "21_12_2_def6b_2",
        "22_04_26_cage4_0", "22_04_26_cage4_1_1"]
USE = kcv.USE_PARTS
BONES = [(USE.index(a), USE.index(b)) for a, b in kcv.BONES]
NOSE, SP4 = USE.index("nose"), USE.index("spine4")
PRE, POST, FPS = 2, 10, 30.0     # ~400 ms = syllable median duration
THETA = 0.05
NSYL = 100
N_PERM = 500
MIN_FR = 30          # min frames of a syllable in a session to score it
MIN_SESS = 3
MIN_ONSETS = 8       # min onsets to draw an onion-skin
NTOP = 3             # enriched syllables shown per operator phase
ZTHR = 2.5           # min selectivity z to call a syllable enriched
RNG = np.random.default_rng(20260616)

# Sign distinguishes the two turn directions. The measured angular velocity
# supplies left/right because coefficient sign is not comparable across op2 and op6.
PHASES = [(14, +1, "op14", "forward baseline"),
          (2, +1, "op2$+$", "strong turn"),
          (2, -1, "op2$-$", "strong turn"),
          (6, +1, "op6$+$", "weak turn"),
          (6, -1, "op6$-$", "weak turn")]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def load():
    data = []
    with h5py.File(KPMS, "r") as f:
        for sid in SESS:
            kp = kcv.load_clean_pixel_kp(sid)                 # (T,8,2) px
            cs = np.load(RUN / f"cs_{sid}.npy")               # (15,T-1)
            z0 = f[sid]["syllable"][:].astype(int)
            L = min(len(kp), cs.shape[1], len(z0))
            kp, cs, z = kp[:L], cs[:, :L], z0[:L]
            th = np.arctan2(kp[:, NOSE, 1] - kp[:, SP4, 1], kp[:, NOSE, 0] - kp[:, SP4, 0])
            om = np.r_[0, wrap(np.diff(th))] * FPS * 180 / np.pi
            data.append((kp, cs, z, th, om))
    return data


def active_mask(cs, slot, sign):
    if slot == 14:                                            # baseline = both turns silent
        return (np.abs(cs[2]) < THETA) & (np.abs(cs[6]) < THETA)
    c = cs[slot]
    return (c > THETA) if sign > 0 else (c < -THETA)


def onion(data, mask_fn):
    """Median egocentric-to-onset pose trajectory + mean signed omega over a boolean mask."""
    wins, omegas = [], []
    for kp, cs, z, th, om in data:
        active = mask_fn(kp, cs, z)
        onsets = np.where(active[1:] & ~active[:-1])[0] + 1
        omegas.append(om[active])
        for t0 in onsets:
            if t0 - PRE < 0 or t0 + POST >= len(kp):
                continue
            c0 = kp[t0].mean(0)
            h0 = th[t0]
            R = np.array([[np.cos(h0), np.sin(h0)], [-np.sin(h0), np.cos(h0)]])
            wins.append((kp[t0 - PRE:t0 + POST + 1] - c0) @ R.T)
    if len(wins) < 1:
        return None, np.nan, 0
    traj = np.median(np.stack(wins), axis=0)
    mom = float(np.mean(np.concatenate(omegas))) if omegas else np.nan
    return traj, mom, len(wins)


def syll_omega_and_present(data):
    om_acc = {s: [] for s in range(NSYL)}
    present = np.zeros(NSYL, int)
    for kp, cs, z, th, om in data:
        cnt = np.bincount(z[z >= 0], minlength=NSYL)
        present += (cnt >= MIN_FR)
        for s in np.unique(z[z >= 0]):
            om_acc[s].append(om[z == s])
    omega = {s: float(np.mean(np.concatenate(v))) for s, v in om_acc.items() if v}
    return omega, present


def selectivity(data, slot, sign, cand):
    """median-over-session z of mean signed amplitude per syllable, circ-shift null."""
    per = {s: [] for s in cand}
    for kp, cs, z, th, om in data:
        a = np.maximum(sign * cs[slot], 0.0)
        L = len(a)
        cnt = np.bincount(z, minlength=NSYL)
        obs = np.bincount(z, weights=a, minlength=NSYL) / np.maximum(cnt, 1)
        null = np.zeros((N_PERM, NSYL))
        for j in range(N_PERM):
            sh = int(RNG.integers(1, L))
            null[j] = np.bincount(z, weights=np.roll(a, sh), minlength=NSYL) / np.maximum(cnt, 1)
        zsc = (obs - null.mean(0)) / (null.std(0, ddof=1) + 1e-12)
        for s in cand:
            if cnt[s] >= MIN_FR:
                per[s].append(zsc[s])
    return {s: (float(np.median(per[s])) if len(per[s]) >= MIN_SESS else np.nan) for s in cand}


def draw(ax, T, title, color, cmap=None):
    if T is None:
        ax.axis("off"); ax.set_title(title, fontsize=8.5, color=color); return
    T = T - T[PRE].mean(0)
    nfr = PRE + POST + 1
    if cmap is None:
        cmap = plt.colormaps[SYL_CMAP_NAME]
    show = np.unique(np.linspace(0, nfr - 1, 6).round().astype(int))
    for fi in show:
        col = cmap(fi / (nfr - 1)); al = 0.4 + 0.55 * fi / (nfr - 1)
        segs = [[T[fi, i], T[fi, j]] for i, j in BONES]
        ax.add_collection(LineCollection(segs, colors=[col], lw=2.3, alpha=al, zorder=fi))
        ax.scatter(T[fi, :, 0], T[fi, :, 1], c=[col], s=9, zorder=fi + 50, edgecolor="w", lw=0.3, alpha=al)
        if fi == PRE:
            ax.add_collection(LineCollection(segs, colors=["k"], lw=1.0, alpha=0.85, zorder=300))
    ax.annotate("", xy=T[show[-1], NOSE], xytext=T[show[0], NOSE],
                arrowprops=dict(arrowstyle="->", color="0.3", lw=1.0, alpha=0.7))
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.invert_yaxis(); ax.margins(0.16)
    ax.set_title(title, fontsize=8.5, color=color, fontweight="bold")
