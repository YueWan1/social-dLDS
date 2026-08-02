"""Generate the 11 source panels placed in Figure 5.

Panel a shows the self-pose operators ``f_3``, ``f_2``, and both phases of
``f_6``. Panel b contains the ``f_4`` distance readout and the two ``f_11``
direction phases. Panel c combines pose and partner direction for ``f_9`` and
``f_15``. Positive and negative coefficient phases are red and blue.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch

from dlds_release.paths import (
    dyadic_cs_dir,
    dyadic_dictionary,
    feature27_dir,
    out_dir,
)

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.linewidth': 0.9, 'xtick.major.width': 0.9, 'ytick.major.width': 0.9,
})

RUN_DIR = dyadic_cs_dir()
FEAT_DIR = feature27_dir()
OUT = out_dir('fig05')

A_IDX = list(range(0, 14))
B_IDX = list(range(14, 25))
C_IDX = list(range(25, 27))
ACT_THR = 0.05
N_SCATTER = 30_000

PARTS = ['nose', 'l_ear', 'r_ear', 'neck', 'l_hip', 'r_hip', 'tail']
BONES = [('neck', 'nose'), ('neck', 'l_ear'), ('neck', 'r_ear'),
         ('neck', 'l_hip'), ('neck', 'r_hip'), ('l_hip', 'tail'), ('r_hip', 'tail')]
PART_COL = dict(nose='#e74c3c', l_ear='#2980b9', r_ear='#5dade2', neck='#f39c12',
                l_hip='#27ae60', r_hip='#58d68d', tail='#7d3c98')

POS_COL = '#d62728'          # +c   (red)   -- uniform across a/b/c
NEG_COL = '#1f77b4'          # -c   (blue)
FAM_COL = {'substrate': '#2e7d8f', 'geometry': '#c06a2b', 'social': '#8e4585'}


# Data
def normalize_features(d):
    f = d.copy().astype(float)
    f /= np.maximum(np.std(f, axis=1, keepdims=True), 1e-3)
    return f / max(np.quantile(np.abs(f), 0.99), 1e-6)


print('Loading sessions ...')
sessions = []
for sid in range(1, 71):
    cp, xp = RUN_DIR / f'cs_mouse{sid:03d}.npy', FEAT_DIR / f'FEATURE27_mouse{sid:03d}.npy'
    if not (cp.exists() and xp.exists()):
        continue
    cs = np.load(cp)
    xr = np.load(xp)
    if xr.shape[0] != 27:
        xr = xr.T
    x = normalize_features(xr)
    T = min(cs.shape[1], x.shape[1])
    sessions.append((cs[:, :T], x[:, :T]))
print(f'  {len(sessions)} sessions')
F_all = np.load(dyadic_dictionary())


def phase_means(slot):
    ps = np.zeros(14); ns = np.zeros(14); al = np.zeros(14)
    pc = nc = ac = 0.0
    for cs, x in sessions:
        c = cs[slot]
        xA = x[A_IDX]
        pm = (c > ACT_THR).astype(float); nm = (c < -ACT_THR).astype(float)
        ps += xA @ pm; ns += xA @ nm; al += xA.sum(1)
        pc += pm.sum(); nc += nm.sum(); ac += xA.shape[1]
    return (al / ac,
            ps / pc if pc > 0 else None,
            ns / nc if nc > 0 else None,
            int(pc), int(nc))


def reshape_for_plot(vec14):
    raw = vec14.reshape(7, 2)
    return np.column_stack([-raw[:, 1], raw[:, 0]])


# Drawing helpers
# Marker / arrow geometry copied from the PUBLISHED main figure
# (figure_result_panel1_rank2.py, the `figm` block: skeleton_ms=20, arrow_lw=0.90,
#  head 3.5-7.0, target_arrow_length=0.24).  Slightly enlarged because these
# sub-panels are drawn one operator at a time rather than in a 5x4 grid.
SKEL_MS = 22            # keypoint marker area; small enough that nose/neck separate
ARROW_LW = 1.0
HEAD_MIN, HEAD_MAX = 3.5, 7.5
ARROW_LEN = 0.26        # longest arrow per panel, in axis units


def draw_skeleton(ax, disp, lim, ms=SKEL_MS):
    pt = {p: disp[i] for i, p in enumerate(PARTS)}
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    for a, b in BONES:
        ax.plot([pt[a][0], pt[b][0]], [pt[a][1], pt[b][1]],
                color='#cdd2d5', lw=1.1, alpha=0.7, zorder=1)
    for i, p in enumerate(PARTS):
        ax.scatter(*disp[i], s=ms, color=PART_COL[p],
                   edgecolors='white', linewidths=0.55, zorder=5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def draw_arrows(ax, pose, field, color, scale, lw=ARROW_LW):
    mags = np.linalg.norm(field, axis=1) * scale
    mx = float(mags.max()) if mags.size else 1.0
    for i in range(7):
        x0, y0 = pose[i]
        dx, dy = field[i] * scale
        m = float(np.hypot(dx, dy))
        if m < 0.02:
            continue
        ux, uy = dx / m, dy / m
        sx, sy = x0 + 0.045 * ux, y0 + 0.045 * uy
        bdx, bdy = (x0 + dx) - sx, (y0 + dy) - sy
        head = float(np.clip(HEAD_MIN + (HEAD_MAX - HEAD_MIN)
                             * (np.hypot(bdx, bdy) / (mx + 1e-9)), HEAD_MIN, HEAD_MAX))
        ax.annotate('', xy=(sx + bdx, sy + bdy), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, alpha=0.92,
                                    mutation_scale=head, shrinkA=0, shrinkB=0), zorder=8)


def op_label(fig, text, family, sub=None):
    """Place the operator name and eigenvalue above the drawing."""
    fig.text(0.03, 0.985, text, fontsize=16, fontweight='bold',
             color=FAM_COL[family], va='top', ha='left')
    if sub:
        fig.text(0.03, 0.885, sub, fontsize=10.5, color='#444', va='top', ha='left')


def phase_tag(fig, text, color):
    """Coefficient phase + frame count, BELOW the axes."""
    fig.text(0.03, 0.012, text, fontsize=11, fontweight='bold',
             color=color, va='bottom', ha='left')


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f'  wrote {name}.pdf')


# Pose sub-panel factory
def pose_panel(name, pose14, field14, color, family, label, sublabel, tag,
               scale, mag=1.0, lim=None, figsize=(1.85, 2.30)):
    """`scale` is supplied by the caller so that operators of comparable amplitude
    share one arrow scale.  An operator drawn at a magnified scale carries an
    explicit `x N` note, so a short deformation is never mistaken for a long one."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.02, 0.085, 0.96, 0.75])
    pose = reshape_for_plot(pose14)
    field = reshape_for_plot(field14)
    end = pose + field * scale
    if lim is None:
        lim = 1.12 * float(max(np.abs(pose).max(), np.abs(end).max(), 0.6))
    draw_skeleton(ax, pose, lim=lim)
    draw_arrows(ax, pose, field, color, scale)
    if mag > 1.05:
        ax.text(0.985, 0.02, rf'arrows $\times${mag:.1f}', transform=ax.transAxes,
                fontsize=9, color='#666', ha='right', va='bottom', style='italic')
    op_label(fig, label, family, sublabel)
    phase_tag(fig, tag, color)
    save(fig, name)


# Panel a: self-pose substrate
print('\nPanel a  (self-pose substrate: f_3, f_2, f_6+/-)')
SUB = [('f_3', 2, 'single'), ('f_2', 1, 'single'), ('f_6', 5, 'bidir')]
rows_a = {}
for nm, slot, kind in SUB:
    mb, mp, mn, npos, nneg = phase_means(slot)
    d = (mp - mn) / 2.0 if (kind == 'bidir' and mn is not None) else mp - mb
    rows_a[nm] = dict(slot=slot, kind=kind, mu_bar=mb, mu_pos=mp, mu_neg=mn,
                      n_pos=npos, n_neg=nneg, d=d)
    print(f'  {nm}: |d|={np.linalg.norm(d):.4f}  n+={npos:,}  n-={nneg:,}')

SUB_META = {
    'f_3': r'$\lambda_1$ = +0.95    $\tau_e$ = 680 ms',
    'f_2': r'$\lambda_{1,2}$ = +0.90, -0.38    $\tau_e$ = 334 ms',
    'f_6': r'$\lambda_{1,2}$ = +0.82, -0.31    $\tau_e$ = 170 ms',
}


def group_scales(rows, reference_ops):
    """One arrow scale shared by `reference_ops` (the comparable, large-amplitude
    operators); every other operator keeps that scale unless it would be
    illegible, in which case it is magnified and the factor is returned."""
    peak = {nm: float(np.max(np.linalg.norm(reshape_for_plot(r['d']), axis=1)))
            for nm, r in rows.items()}
    ref = max(peak[nm] for nm in reference_ops)
    base = ARROW_LEN / (ref + 1e-9)
    out = {}
    for nm, pk in peak.items():
        if nm in reference_ops:
            out[nm] = (base, 1.0)
        else:
            mag = ref / (pk + 1e-9)
            out[nm] = (ARROW_LEN / (pk + 1e-9), mag)
    return out, peak, ref


SCALES_A, PEAK_A, REF_A = group_scales(rows_a, {'f_2', 'f_6'})
print(f'  f_2/f_6 share one scale (reference peak |d| = {REF_A:.4f});')
for nm in ('f_3', 'f_2', 'f_6'):
    print(f'    {nm:<5} peak |d| = {PEAK_A[nm]:.4f}   magnification = '
          f'x{SCALES_A[nm][1]:.2f}')

# common frame for the whole row so the mice are drawn at one body size
lim_a = 0.0
for nm, r in rows_a.items():
    sc = SCALES_A[nm][0]
    for mu, sgn in ((r['mu_pos'], +1), (r['mu_neg'], -1)):
        if mu is None or (sgn < 0 and r['kind'] != 'bidir'):
            continue
        p = reshape_for_plot(mu)
        e = p + reshape_for_plot(sgn * r['d']) * sc
        lim_a = max(lim_a, float(np.abs(p).max()), float(np.abs(e).max()))
lim_a *= 1.12

for nm in ('f_3', 'f_2', 'f_6'):
    r = rows_a[nm]
    sc, mag = SCALES_A[nm]
    tex = rf'$\mathbf{{f}}_{{{nm.split("_")[1]}}}$'
    if r['kind'] == 'bidir':
        pose_panel(f'a_{nm}_pos', r['mu_pos'], r['d'], POS_COL,
                   'substrate', tex, SUB_META[nm], f'+c   n={r["n_pos"]:,}',
                   scale=sc, mag=mag, lim=lim_a)
        pose_panel(f'a_{nm}_neg', r['mu_neg'], -r['d'], NEG_COL,
                   'substrate', tex, SUB_META[nm], f'-c   n={r["n_neg"]:,}',
                   scale=sc, mag=mag, lim=lim_a)
    else:
        pose_panel(f'a_{nm}', r['mu_pos'], r['d'], POS_COL,
                   'substrate', tex, SUB_META[nm],
                   f'+c   n={r["n_pos"]:,}  (single-signed)',
                   scale=sc, mag=mag, lim=lim_a)


# SVD and distance calculations for panels b and c
def compute_svd_op(slot):
    F_CB = F_all[slot][np.ix_(C_IDX, B_IDX)]
    U, sig, Vt = np.linalg.svd(F_CB, full_matrices=False)
    eta1 = float(sig[0] ** 2 / max((sig ** 2).sum(), 1e-12))
    u1, v1 = U[:, 0], Vt[0, :]
    pos_c, neg_c = [], []
    cpt = cnt = 0.0
    npos = nneg = 0
    for cs, x in sessions:
        c = cs[slot]
        xC = x[C_IDX]
        p_t = c * (v1 @ x[B_IDX])
        pm, nm_ = c > ACT_THR, c < -ACT_THR
        if pm.sum():
            pos_c.append(xC[:, pm]); cpt += float(p_t[pm].sum()); npos += int(pm.sum())
        if nm_.sum():
            neg_c.append(xC[:, nm_]); cnt += float(p_t[nm_].sum()); nneg += int(nm_.sum())
    if cpt / max(npos, 1) < 0:
        u1, v1 = -u1, -v1
        cpt, cnt = -cpt, -cnt
    mu_pos = np.concatenate(pos_c, 1).mean(1) if pos_c else np.zeros(2)
    mu_neg = np.concatenate(neg_c, 1).mean(1) if neg_c else np.zeros(2)
    return dict(eta1=eta1, u1=u1,
                theta_u=float(np.degrees(np.arctan2(u1[1], u1[0]))),
                mu_pos=mu_pos, mu_neg=mu_neg, n_pos=npos, n_neg=nneg,
                pred_pos=np.sign(cpt / max(npos, 1)) * u1,
                pred_neg=np.sign(cnt / max(nneg, 1)) * u1)


def draw_compass(ax, mu_pt, u_arrow, color, ang_label, u_label, lim=1.42,
                 ring=False, bare=False):
    """ring=False: star at the raw mean partner vector (published convention).
       ring=True : star on the unit circle at the mean BEARING, |mu| reported as R.
                   Same numbers, but the axis-vs-measurement comparison is legible."""
    def to_plot(v):
        return -v[1], +v[0]
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color='#cccccc', lw=0.9, ls='--', zorder=1)
    ax.axhline(0, color='#ececec', lw=0.6, zorder=0)
    ax.axvline(0, color='#ececec', lw=0.6, zorder=0)
    # orientation cues kept short so they stay inside the unit circle
    ax.annotate('', xy=(0, lim * 0.80), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#c8c8c8', lw=0.8))
    ax.text(-0.06, lim * 0.60, 'fwd', fontsize=6.5, color='#aaa',
            va='bottom', ha='right')
    ax.annotate('', xy=(-lim * 0.80, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#c8c8c8', lw=0.8))
    ax.text(-lim * 0.60, -0.10, 'left', fontsize=6.5, color='#aaa',
            ha='center', va='top')
    ax.scatter(0, 0, marker='o', s=24, color='#555', zorder=4)

    pt = np.array(to_plot(mu_pt))
    R = float(np.hypot(*pt))
    if ring:
        pt = pt / (R + 1e-12)
    mx, my = pt
    ax.scatter(mx, my, marker='*', s=210, color=color,
               edgecolors='black', linewidths=1.0, zorder=10)
    ux, uy = to_plot(u_arrow)
    n = np.hypot(ux, uy) + 1e-9
    ux, uy = ux / n, uy / n
    ex, ey = mx + 0.46 * ux, my + 0.46 * uy
    ax.annotate('', xy=(ex, ey), xytext=(mx, my),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=2.2,
                                mutation_scale=16, alpha=0.95, shrinkA=5, shrinkB=0), zorder=8)
    if not bare:
        px, py = -uy, ux
        ax.text((mx + ex) / 2 + 0.15 * px, (my + ey) / 2 + 0.15 * py, u_label,
                fontsize=9.5, fontweight='bold', color=color, ha='center', va='center',
                zorder=12, bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                                     edgecolor='none', alpha=0.85))

    # Put the angle read-out in whichever corner is farthest from the star, the
    # arrow tip and the two orientation cues, so nothing ever sits on top of it.
    if bare:
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        return R
    occupied = [(mx, my), (ex, ey), (0.0, lim * 0.60), (-lim * 0.60, 0.0)]
    corners = {'tr': (0.985, 0.985, 'right', 'top', (lim, lim)),
               'tl': (0.015, 0.985, 'left', 'top', (-lim, lim)),
               'br': (0.985, 0.030, 'right', 'bottom', (lim, -lim)),
               'bl': (0.015, 0.030, 'left', 'bottom', (-lim, -lim))}
    best = max(corners.values(),
               key=lambda c: min((c[4][0] - ox) ** 2 + (c[4][1] - oy) ** 2
                                 for ox, oy in occupied))
    ax.text(best[0], best[1], ang_label, transform=ax.transAxes, fontsize=8.6,
            color=color, fontweight='bold', ha=best[2], va=best[3], zorder=20,
            linespacing=1.4,
            bbox=dict(boxstyle='round,pad=0.16', facecolor='white',
                      edgecolor='none', alpha=0.92))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return R


def axial_err(a, b):
    d = abs((a - b + 180.0) % 360.0 - 180.0)
    return min(d, 180.0 - d)


def compass_panel(name, d, phase, family, label, ring=False, bare=False,
                  figsize=None):
    if figsize is None:
        figsize = (1.95, 2.52) if bare else (1.95, 2.30)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.02, 0.155, 0.96, 0.68] if bare else [0.02, 0.085, 0.96, 0.75])
    if phase == '+':
        mu, u, col, n = d['mu_pos'], d['pred_pos'], POS_COL, d['n_pos']
        ulab, tag = r'$+\mathbf{u}_1$', '+c'
    else:
        mu, u, col, n = d['mu_neg'], d['pred_neg'], NEG_COL, d['n_neg']
        ulab, tag = r'$-\mathbf{u}_1$', '-c'
    ang = float(np.degrees(np.arctan2(mu[1], mu[0])))
    err = axial_err(ang, d['theta_u'])
    R = draw_compass(ax, mu, u, col,
                     rf'$\angle\mu_C={ang:+.0f}\degree$' '\n'
                     rf'$\Delta={err:.0f}\degree$', ulab, ring=ring, bare=bare)
    op_label(fig, label, family,
             rf'$\eta_1$={d["eta1"]:.2f}   $\angle\mathbf{{u}}_1$={d["theta_u"]:+.0f}$\degree$')
    if bare:
        # arrow identity sits top-right; the two read-outs stack below the drawing
        # on their own lines, so nothing can collide inside the circle
        fig.text(0.97, 0.985, ulab, fontsize=11, fontweight='bold', color=col,
                 ha='right', va='top')
        fig.text(0.03, 0.075, f'{tag}   n={n:,}', fontsize=10, fontweight='bold',
                 color=col, ha='left', va='bottom')
        fig.text(0.03, 0.010,
                 rf'$\angle\mu_C={ang:+.0f}\degree$    $\Delta={err:.0f}\degree$',
                 fontsize=9.5, color=col, ha='left', va='bottom')
    # |mu_C| is not the circular concentration R, so it is not printed.
    if not bare:
        phase_tag(fig, f'{tag}   n={n:,}', col)
    save(fig, name)
    return dict(bearing=ang, axial_err=err, mu_norm=R, n=n)


# Panel b: partner geometry
print('\nPanel b  (partner geometry: f_4 distance, f_11 direction)')
d11 = compute_svd_op(10)
print(f'  f_11: eta1={d11["eta1"]:.3f} angU1={d11["theta_u"]:+.1f} '
      f'n+={d11["n_pos"]:,} n-={d11["n_neg"]:,}')
GEO_CHECK = {}
GEO_CHECK['f_11+'] = compass_panel(
    'b_f11_pos_bare', d11, '+', 'geometry', r'$\mathbf{f}_{11}$',
    ring=True, bare=True,
)
GEO_CHECK['f_11-'] = compass_panel(
    'b_f11_neg_bare', d11, '-', 'geometry', r'$\mathbf{f}_{11}$',
    ring=True, bare=True,
)

# f_4 distance scatter
F_BB = F_all[3][np.ix_(B_IDX, B_IDX)]
w4, V4 = np.linalg.eig(F_BB)
v1_f4 = V4[:, np.argsort(np.abs(w4))[::-1][0]].real
rs4, pc4, pp4 = [], [], []
for cs, x in sessions:
    c = cs[3]
    proj = v1_f4 @ x[B_IDX]
    if c.std() < 1e-9:
        continue
    rs4.append(float(np.corrcoef(c, proj)[0, 1]))
    pc4.append(c); pp4.append(proj)
rs4 = np.array(rs4)
if np.median(rs4) < 0:
    v1_f4 = -v1_f4; rs4 = -rs4; pp4 = [-p for p in pp4]
pc4 = np.concatenate(pc4); pp4 = np.concatenate(pp4)
r_pool4 = float(np.corrcoef(pc4, pp4)[0, 1])
print(f'  f_4: {len(rs4)} sessions, median r={np.median(rs4):+.3f}, r_pool={r_pool4:+.3f}')

rng = np.random.default_rng(0)
idx = rng.choice(len(pc4), size=min(N_SCATTER, len(pc4)), replace=False)
fig = plt.figure(figsize=(2.75, 2.30))
ax = fig.add_axes([0.20, 0.20, 0.76, 0.62])
ax.scatter(pp4[idx], pc4[idx], s=2.0, color='#5e6266', alpha=0.16,
           edgecolors='none', rasterized=True, zorder=2)
a4, b4 = np.polyfit(pp4[idx], pc4[idx], 1)
xs = np.linspace(pp4[idx].min(), pp4[idx].max(), 200)
ax.plot(xs, a4 * xs + b4, color=NEG_COL, lw=2.2, zorder=4)
ax.set_xlabel(r'distance-axis projection  $\mathbf{v}_1^\top\mathbf{x}^{\mathrm{DIST}}$', fontsize=8)
ax.set_ylabel(r'$c_4(t)$', fontsize=9)
ax.text(0.03, 0.96, f'$r_{{\\mathrm{{pool}}}}$ = {r_pool4:.2f}\n'
                    f'median $r$ = {np.median(rs4):.2f}',
        transform=ax.transAxes, fontsize=8.5, ha='left', va='top', color=NEG_COL,
        bbox=dict(boxstyle='round,pad=0.20', facecolor='white', edgecolor='#ddd', alpha=0.92))
op_label(fig, r'$\mathbf{f}_4$', 'geometry',
         r'inter-mouse distance   (single-signed, all $c_4<0$)')
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.grid(alpha=0.16)
save(fig, 'b_f4_scatter')


# Panel c: social primitives
print('\nPanel c  (social primitives: f_9, f_15 -- pose AND compass)')
SOC = [('f_9', 8, 'bidir'), ('f_15', 14, 'single')]
rows_c, svd_c = {}, {}
for nm, slot, kind in SOC:
    mb, mp, mn, npos, nneg = phase_means(slot)
    d = (mp - mn) / 2.0 if (kind == 'bidir' and mn is not None) else mp - mb
    rows_c[nm] = dict(slot=slot, kind=kind, mu_bar=mb, mu_pos=mp, mu_neg=mn,
                      n_pos=npos, n_neg=nneg, d=d)
    svd_c[nm] = compute_svd_op(slot)
    s = svd_c[nm]
    print(f'  {nm}: |d|={np.linalg.norm(d):.4f}  eta1={s["eta1"]:.3f} '
          f'angU1={s["theta_u"]:+.1f}  n+={s["n_pos"]:,} n-={s["n_neg"]:,}')

for nm, data in svd_c.items():
    for sign, phase in (('+', 'pos'), ('-', 'neg')):
        mean = data[f'mu_{phase}']
        bearing = float(np.degrees(np.arctan2(mean[1], mean[0])))
        GEO_CHECK[f'{nm}{sign}'] = {
            'bearing': bearing,
            'axial_err': axial_err(bearing, data['theta_u']),
            'mu_norm': float(np.linalg.norm(mean)),
            'n': data[f'n_{phase}'],
        }



# Combined self-pose deformation and partner direction
# One picture per coefficient phase.  The resident skeleton sits at the origin with
# its measured phase deformation; the partner is placed at the MEASURED mean bearing
# on a schematic radius (only the angle is data, the radius is not: distance is
# f_4's read-out, not f_9's or f_15's).  The matrix-predicted axis +/-u_1 is drawn
# from the same origin as a dashed ray, so the angle between the two rays IS the
# reported axial error.
# The ring sits just outside the resident's own body (max keypoint extent 0.923)
# rather than far out in empty space: the two animals are often close, and a wide
# ring both shrinks the pose and implies a separation the data do not show.
R_SCENE = 1.05
ARROW_LEN_C = 0.80       # panel c is scaled to itself, not to panel a
LIM_BARE = 1.42


def merged_scene(name, pose14, field14, scale, bearing_deg, u_axis_deg, color,
                 lim):
    fig = plt.figure(figsize=(2.35, 2.35))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    def ego_to_plot(deg):
        a = np.radians(deg)
        return np.array([-np.sin(a), np.cos(a)])   # same convention as the skeleton

    # Both rays start clear of the body: a ray drawn from the origin would cut
    # straight through the skeleton and make the pose unreadable.
    R0 = R_SCENE * 0.70
    up_d, mp_d = ego_to_plot(u_axis_deg), ego_to_plot(bearing_deg)
    # matrix prediction: dashed grey ray along the phase-oriented u_1 axis
    ax.annotate('', xy=tuple(up_d * R_SCENE), xytext=tuple(up_d * R0),
                arrowprops=dict(arrowstyle='-|>', color='#9aa0a6', lw=1.6,
                                ls='--', mutation_scale=12, shrinkA=0, shrinkB=0),
                zorder=6)
    # measurement: solid ray to the mean partner bearing, star at the end
    mp = mp_d * R_SCENE
    ax.annotate('', xy=tuple(mp), xytext=tuple(mp_d * R0),
                arrowprops=dict(arrowstyle='-', color=color, lw=1.6,
                                alpha=0.6, shrinkA=0, shrinkB=0), zorder=6)
    ax.scatter(*mp, marker='*', s=240, color=color,
               edgecolors='black', linewidths=1.0, zorder=12)

    pose = reshape_for_plot(pose14)
    field = reshape_for_plot(field14)
    draw_skeleton(ax, pose, lim=lim, ms=30)
    draw_arrows(ax, pose, field, color, scale, lw=1.5)

    save(fig, name)


print('\nPanel c merged scenes (bare)')
SC_ROWS = []
for nm, slot, kind in SOC:
    r = rows_c[nm]
    s = svd_c[nm]
    for ph, mu, n, bear in (('pos', r['mu_pos'], s['n_pos'],
                             float(np.degrees(np.arctan2(s['mu_pos'][1], s['mu_pos'][0])))),
                            ('neg', r['mu_neg'], s['n_neg'],
                             float(np.degrees(np.arctan2(s['mu_neg'][1], s['mu_neg'][0]))))):
        if mu is None:
            continue
        dev = mu - r['mu_bar']            # deviation measured IN THIS PHASE
        uax = s['theta_u'] if ph == 'pos' else s['theta_u'] + 180.0
        uax = (uax + 180.0) % 360.0 - 180.0
        SC_ROWS.append(dict(op=nm, ph=ph, mu=mu, dev=dev, n=n, bear=bear,
                            uax=uax, col=POS_COL if ph == 'pos' else NEG_COL))

peak_sc = max(float(np.max(np.linalg.norm(reshape_for_plot(s['dev']), axis=1)))
              for s in SC_ROWS)
SCALE_SC = ARROW_LEN_C / (peak_sc + 1e-9)
lim_sc = 0.0
for s in SC_ROWS:
    pp = reshape_for_plot(s['mu'])
    ee = pp + reshape_for_plot(s['dev']) * SCALE_SC
    lim_sc = max(lim_sc, float(np.abs(pp).max()), float(np.abs(ee).max()))
lim_sc = max(lim_sc * 1.10, R_SCENE * 1.16)

print(f'  shared scale across the four scenes (peak per-keypoint |dev| = {peak_sc:.4f}), '
      f'frame lim = {lim_sc:.2f}')
print(f'  {"scene":<12}{"n":>9}{"bearing":>10}{"axis u1":>10}{"axial err":>11}'
      f'{"peak |dev|":>12}{"rel":>7}')
for s in SC_ROWS:
    pk = float(np.max(np.linalg.norm(reshape_for_plot(s['dev']), axis=1)))
    err = axial_err(s['bear'], s['uax'])
    print(f'  {s["op"] + " " + s["ph"]:<12}{s["n"]:>9,}{s["bear"]:>+10.1f}'
          f'{s["uax"]:>+10.1f}{err:>11.1f}{pk:>12.4f}{pk / peak_sc:>7.2f}')
    merged_scene(f'c_scene_{s["op"]}_{s["ph"]}', s['mu'], s['dev'], SCALE_SC,
                 s['bear'], s['uax'], s['col'], LIM_BARE)


# Numerical summary used by the caption and Results
print('\n=== NUMBER CHECK (compare against MAIN_dyadic_figure2.pdf) ===')
print(f'{"op/phase":<10}{"bearing":>10}{"angU1":>9}{"axial err":>11}{"|muC|":>8}{"n":>10}')
for k in ('f_11+', 'f_11-', 'f_9+', 'f_9-', 'f_15+', 'f_15-'):
    if k not in GEO_CHECK:
        continue
    g = GEO_CHECK[k]
    op = k[:-1]
    tu = d11['theta_u'] if op == 'f_11' else svd_c[op]['theta_u']
    print(f'{k:<10}{g["bearing"]:>+10.1f}{tu:>+9.1f}{g["axial_err"]:>11.1f}'
          f'{g["mu_norm"]:>8.2f}{g["n"]:>10,}')
print(f'\nf_4  median r = {np.median(rs4):+.3f}   r_pool = {r_pool4:+.3f}   '
      f'sessions = {len(rs4)}')
print('\nDeformation amplitude |d_emp| (arrows are normalised PER PANEL, so these')
print('relative sizes are NOT visible in the drawings -- quote them in the caption):')
for lbl, rows in (('a', rows_a), ('c', rows_c)):
    for nm, r in rows.items():
        print(f'  panel {lbl}  {nm:<5} |d_emp| = {np.linalg.norm(r["d"]):.4f}')


print('\nAll published Figure 5 sub-panels are in:', OUT)
