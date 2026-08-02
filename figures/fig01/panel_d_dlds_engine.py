"""Generate the dLDS model diagram for Figure 1d.

The upper chain shows time-varying transition matrices and coefficients. The
lower row expands one transition matrix as a signed mixture of active
dictionary operators.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from dlds_release.lds_topchain import draw_top_chain, chain_halfwidth
from dlds_release.paths import out_dir, single_fit_dir

RUN = single_fit_dir()
OUT = out_dir('fig01')

SESS = [
    '21_12_10_def6b_3',
    '21_12_2_def6a_1',
    '21_12_2_def6b_2',
    '22_04_26_cage4_0',
    '22_04_26_cage4_1_1',
]
USAGE_MIN = 0.5

TERMS = [(2, '#1f6fb2'), (6, '#2e7d32'), (14, '#8e44ad')]      # (op index, colour) -- the 3 engaged ops
C_COEF, BLACK = '#2e8b3d', '#111'

Fs = np.load(RUN / 'Fs.npy')                                   # (15,16,16)
M = Fs.shape[0]
cs = np.load(RUN / 'cs_21_12_2_def6a_1.npy')
if cs.shape[0] == M:
    cs = cs.T
FRAME = 95079                                                  # shared with the MoSeq figure: a real syllable
cvec = cs[FRAME]                                               # boundary (z42,z42 -> z0); c ~ [+0.26,-0.24,+0.60]
F_blend = sum(cvec[idx] * Fs[idx] for idx, _ in TERMS)
vmax = np.percentile(np.abs(np.stack([F_blend] + [Fs[idx] for idx, _ in TERMS])), 85)

# Mirror the keypoint-MoSeq usage inset in panel 1c.  For dLDS, usage is the
# fraction of the pooled absolute coefficient mass carried by each operator,
# so the 15 bars sum to 100%, just as syllable frame usage sums to 100% in 1c.
op_mass = np.zeros(M, dtype=float)
for sid in SESS:
    session_cs = np.load(RUN / f'cs_{sid}.npy')
    if session_cs.shape[0] != M:
        session_cs = session_cs.T
    op_mass += np.abs(session_cs).sum(axis=1)
op_usage = op_mass / op_mass.sum() * 100.0
op_usage_sorted = np.sort(op_usage)[::-1]
n_used = int(np.count_nonzero(op_usage >= USAGE_MIN))
if n_used != 3:
    raise ValueError(f'Expected 3/15 operators at >= {USAGE_MIN}% usage; found {n_used}/{M}')

W, H = 10.6, 5.0
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.set_aspect('equal'); ax.axis('off')


def circ(x, y, color, label, r, fs):
    ax.add_patch(Circle((x, y), r, facecolor='white', edgecolor=color, lw=2.2, zorder=4))
    ax.text(x, y, label, ha='center', va='center', fontsize=fs, color=color, fontweight='bold', zorder=5)


def thumb(cx, cy, s, mat, edge, lw):
    axi = fig.add_axes([cx / W - (s/2) / W, cy / H - (s/2) / H, s / W, s / H])
    axi.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    axi.set_xticks([]); axi.set_yticks([])
    for sp in axi.spines.values():
        sp.set_color(edge); sp.set_linewidth(lw)


def usage_inset(cx, cy, w, h):
    """Draw the panel-1c-matched operator-usage distribution."""
    axi = fig.add_axes([cx / W - (w / 2) / W, cy / H - (h / 2) / H,
                        w / W, h / H])
    ranks = np.arange(1, M + 1)
    axi.bar(ranks[:n_used], op_usage_sorted[:n_used], width=1.0,
            color='#1f6fb2', lw=0, zorder=3)
    axi.bar(ranks[n_used:], op_usage_sorted[n_used:], width=1.0,
            color='#c8ccd2', lw=0, zorder=2)
    axi.axhline(USAGE_MIN, color='#7e8893', lw=0.8,
                ls=(0, (3, 2)), zorder=4)
    axi.set_yscale('log')
    axi.set_xlim(0.0, M + 1)
    axi.set_ylim(op_usage_sorted[-1] * 0.6, op_usage_sorted[0] * 3.2)
    axi.set_xticks([])
    axi.set_yticks([])
    axi.spines['top'].set_visible(False)
    axi.spines['right'].set_visible(False)
    for side in ('left', 'bottom'):
        axi.spines[side].set_color('#b8bec6')
        axi.spines[side].set_linewidth(0.7)
    return axi


# Shared LDS chain.
CX, YS = 3.75, 3.95
boxx, mty = draw_top_chain(ax, cx=CX, ys=YS, mode='dlds', font_scale=1.85)

# Same visual grammar as panel 1c: title + ranked usage bars, identical 0.5%
# threshold, log y-axis, blue substantive head, and grey low-usage tail.
USAGE_CX = 8.20
ax.text(USAGE_CX, 4.47,
        rf'usage: {n_used}/{M} $\geq$ {USAGE_MIN:.1f}%',
        ha='center', va='center', fontsize=17, color='#566069',
        fontweight='bold')
usage_inset(USAGE_CX, 4.08, 2.10, 0.40)

# Sparse mixture of the three engaged operators.
Y = 1.55
RC, TH = 0.38, 0.82
BOTTOM_DX = -1.73  # centre the full F_t construction beneath the top chain
lhs_x = 1.15 + BOTTOM_DX
thumb(lhs_x, Y, TH, F_blend, edge=BLACK, lw=3.4)
ax.text(lhs_x, Y - TH/2 - 0.27, r'$\mathbf{F}_t$', ha='center', va='center', fontsize=26,
        color=BLACK, fontweight='bold')
ax.plot([lhs_x, boxx[1]], [Y + TH/2 + 0.05, mty - 0.02], color='#9aa0a8', lw=1.1, ls=(0, (4, 3)), zorder=0)
ax.text(2.15 + BOTTOM_DX, Y, r'$=$', ha='center', va='center', fontsize=29, color=BLACK, fontweight='bold')

x = 2.85 + BOTTOM_DX
term_xs = []
for k, (idx, col) in enumerate(TERMS):
    term_xs.append(x)
    circ(x, Y, C_COEF, rf'$c_{{{idx},t}}$', r=RC, fs=22)
    ax.text(x + 0.62, Y, r'$\times$', ha='center', va='center', fontsize=25, color=BLACK)
    thumb(x + 1.40, Y, TH, Fs[idx], edge=col, lw=3.8)
    ax.text(x + 1.40, Y - TH/2 - 0.27, rf'$\mathbf{{f}}_{{{idx}}}$', ha='center', va='center',
            fontsize=24, color=col, fontweight='bold')
    if k < len(TERMS) - 1:
        ax.text(x + 2.18, Y, r'$+$', ha='center', va='center', fontsize=28, color=BLACK, fontweight='bold')
    x += 2.78

# Compact visual cue for simultaneous, signed mixing.  The explicit equation
# remains the load-bearing mathematical statement; the bracket/Sigma badge is
# only the intuitive counterpart of the lever used in the discrete MoSeq panel.
bracket_x0 = term_xs[0] - RC
bracket_x1 = term_xs[-1] + 1.40 + TH / 2
bracket_y = 2.08
mix_x = (bracket_x0 + bracket_x1) / 2
ax.plot([bracket_x0, bracket_x1], [bracket_y, bracket_y], color='#9aa0a8', lw=1.25, zorder=1)
ax.plot([bracket_x0, bracket_x0], [bracket_y, bracket_y - 0.12], color='#9aa0a8', lw=1.25, zorder=1)
ax.plot([bracket_x1, bracket_x1], [bracket_y, bracket_y - 0.12], color='#9aa0a8', lw=1.25, zorder=1)
ax.plot([mix_x, mix_x], [bracket_y, 2.17], color='#9aa0a8', lw=1.25, zorder=1)
MIX_R = 0.22
# Keep Sigma at the same font-size/radius ratio as the approved c_{i,t}
# coefficient circles below (22 pt at radius RC).
MIX_FS = 22 * MIX_R / RC
ax.add_patch(Circle((mix_x, 2.39), MIX_R, facecolor='white', edgecolor=BLACK, lw=1.8, zorder=4))
ax.text(mix_x, 2.39, r'$\Sigma$', ha='center', va='center', fontsize=MIX_FS,
        color=BLACK, fontweight='bold', zorder=5)
ax.text(mix_x, 2.58, 'signed weighted mixture',
        ha='center', va='bottom', fontsize=17, color=C_COEF,
        fontweight='bold', zorder=7)

out = OUT / 'fig_dlds_engine_products_v5_inferred_nodes_largefont.pdf'
fig.savefig(out, dpi=300, bbox_inches='tight', transparent=True)
print('wrote', out)
plt.close(fig)
