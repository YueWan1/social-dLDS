"""Shared latent-state chain for the paired dLDS and keypoint-MoSeq diagrams.

Both panels use the same geometry for three state transitions. In dLDS,
``c_t`` constructs ``F_t``; in keypoint-MoSeq, ``z_t`` selects ``A_z``.
"""
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch

# Geometry shared by both panels
R = 0.32            # state circle radius; sized for final-scale readable labels
SQ = 0.34           # operator box half-size
PITCH = 1.20        # centre-to-centre of consecutive chain elements (box,state,box,...)
RING = 0.085        # dLDS purple ring gap around x
OBS_DY = 1.05       # MoSeq: y node sits this far above its x
INFER_DY = 0.80     # inferred-variable row sits this far below the state/operator row
R_INFER = 0.27      # radius of c_t / z_t inferred-variable circles

# Colors shared by both panels
C_STATE = '#e8861a'   # state x (orange)
C_OBS = '#7a3aa0'     # observation y (purple)
BLACK = '#111'
GREY = '#566069'
GREYL = '#7e8893'
C_DLDS = '#2e8b3d'    # dLDS coefficient bracket (c)
C_MOSEQ = '#3b4252'   # MoSeq syllable bracket (z)


def _circ(ax, x, y, color, label, r, fs, ring=None):
    if ring is not None:
        ax.add_patch(Circle((x, y), r + RING, facecolor='white', edgecolor=ring, lw=2.2, zorder=3))
    ax.add_patch(Circle((x, y), r, facecolor='white', edgecolor=color, lw=2.2, zorder=4))
    ax.text(x, y, label, ha='center', va='center', fontsize=fs, color=color, fontweight='bold', zorder=5)


def _sq(ax, x, y, label, fs):
    ax.add_patch(FancyBboxPatch((x - SQ, y - SQ), 2 * SQ, 2 * SQ,
                 boxstyle='round,pad=0.02,rounding_size=0.05',
                 facecolor='white', edgecolor=BLACK, lw=2.4, zorder=4))
    ax.text(x, y, label, ha='center', va='center', fontsize=fs, color=BLACK, fontweight='bold', zorder=5)


def _arrow(ax, p0, p1, color=BLACK, lw=2.3, sh=6, ms=12):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=ms, color=color,
                 lw=lw, shrinkA=sh, shrinkB=sh, zorder=2))


def chain_halfwidth():
    """Half the horizontal extent of the 6-element chain (for centring / canvas sizing)."""
    return 2.5 * PITCH + R


def draw_top_chain(ax, cx, ys, mode, font_scale=1.0):
    """Draw a three-transition LDS chain centered at ``cx``.

    ``mode='dlds'`` places the observation ring on the state. ``mode='moseq'``
    draws a separate observation node and emission parameters.

    Returns the operator-box x positions and their common lower y coordinate.
    """
    def fs(value):
        return value * font_scale

    # 6 elements left->right: M_{t-1}, x_{t-1}, M_t, x_t, M_{t+1}, x_{t+1}
    xs = [cx + (i - 2.5) * PITCH for i in range(6)]
    B0, X0, B1, X1, B2, X2 = xs
    stt = ['t-1', 't', 't+1']
    statex = [X0, X1, X2]
    boxx = [B0, B1, B2]

    if mode == 'dlds':
        oplab = [r'$\mathbf{F}_{t-1}$', r'$\mathbf{F}_t$', r'$\mathbf{F}_{t+1}$']
        ring = C_OBS
        inferlab = [r'$\mathbf{c}_{t-1}$', r'$\mathbf{c}_t$', r'$\mathbf{c}_{t+1}$']
        infer_col = C_DLDS
        # Match every node to the two proportions approved in the standalone
        # dLDS panel: F_t inside its square and c_{i,t} inside the lower
        # coefficient circles.  The base sizes below are scaled by the caller's
        # font_scale; their radius-normalised sizes are therefore uniform.
        state_fs = [10.0, 10.0, 10.0]
        operator_fs = [11.0, 11.0, 11.0]
        inferred_fs = [8.45, 8.45, 8.45]
    else:
        oplab = [r'$\mathbf{A}_{z_{t-1}}$', r'$\mathbf{A}_{z_t}$', r'$\mathbf{A}_{z_{t+1}}$']
        ring = None
        inferlab = [r'$z_{t-1}$', r'$z_t$', r'$z_{t+1}$']
        infer_col = C_MOSEQ
        # Retain the established MoSeq typography; this revision only splits
        # that panel into independently exported upper and lower components.
        state_fs = [9.5, 10.5, 9.5]
        operator_fs = [8.5, 11.0, 8.5]
        # The longer neighbouring labels need more breathing room than z_t;
        # keeping one rigid size made z_{t-1} and z_{t+1} touch their circles.
        inferred_fs = [7.4, 8.6, 7.4]

    # state circles + operator boxes
    for k, sx in enumerate(statex):
        _circ(ax, sx, ys, C_STATE, rf'$\mathbf{{x}}_{{{stt[k]}}}$', R,
              fs(state_fs[k]), ring=ring)
    for k, bx in enumerate(boxx):
        _sq(ax, bx, ys, oplab[k], fs(operator_fs[k]))

    # transitions B0->X0->B1->X1->B2->X2
    _arrow(ax, (B0 + SQ, ys), (X0 - R, ys))
    _arrow(ax, (X0 + R, ys), (B1 - SQ, ys))
    _arrow(ax, (B1 + SQ, ys), (X1 - R, ys))
    _arrow(ax, (X1 + R, ys), (B2 - SQ, ys))
    _arrow(ax, (B2 + SQ, ys), (X2 - R, ys))

    # observation
    if mode == 'moseq':
        yy = ys + OBS_DY
        for k, sx in enumerate(statex):
            _arrow(ax, (sx, ys + R), (sx, yy - R), color=GREY, lw=1.8, sh=2, ms=11)
            # Purple y nodes have the same radius as the orange x nodes, so use
            # the corresponding x-label size rather than an unscaled 9.5 pt.
            _circ(ax, sx, yy, C_OBS, rf'$\mathbf{{y}}_{{{stt[k]}}}$', R,
                  fs(state_fs[k]))
        ax.text(X0 - R - 0.16, yy, 'keypoints', ha='right', va='center',
                fontsize=fs(10), color=GREY, fontweight='bold')
        for (px, py, lab) in [(X1 - 0.62, yy + 0.60, r'$s$'), (X1, yy + 0.78, r'$h_t$'),
                              (X1 + 0.62, yy + 0.60, r'$v_t$')]:
            ax.add_patch(Circle((px, py), 0.22, facecolor='white', edgecolor=GREYL, lw=2.0, zorder=4))
            ax.text(px, py, lab, ha='center', va='center', fontsize=fs(9.5), color=GREY, zorder=5)
            _arrow(ax, (px + (X1 - px) * 0.16, py - 0.22), (X1 + (px - X1) * 0.30, yy + R + 0.02),
                   color=GREYL, lw=1.6, sh=2, ms=10)

    # Inferred variables.  Circles distinguish these time-varying latent
    # quantities from the deterministic transition matrices drawn as boxes.
    yi = ys - INFER_DY
    for bx, lab, label_fs in zip(boxx, inferlab, inferred_fs):
        _circ(ax, bx, yi, infer_col, lab, R_INFER, fs(label_fs))
        _arrow(ax, (bx, yi + R_INFER), (bx, ys - SQ),
               color=infer_col, lw=1.8, sh=2, ms=10)
    for left, right in zip(boxx[:-1], boxx[1:]):
        _arrow(ax, (left + R_INFER, yi), (right - R_INFER, yi),
               color=infer_col, lw=1.8, sh=2, ms=10)

    return [B0, B1, B2], ys - SQ   # operator box centre-x's (M_{t-1}, M_t, M_{t+1}) + their bottom-y
