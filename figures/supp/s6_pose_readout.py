"""Generate the resident-pose panels in Supplementary Figure S6.

Matrix-derived pose axes are compared with coefficient-conditioned mean poses.
The matched null recomputes phase orientation after each per-session circular
shift.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.linewidth': 0.9, 'xtick.major.width': 0.9, 'ytick.major.width': 0.9,
})

from dlds_release.paths import dyadic_cs_dir, dyadic_dictionary, feature27_dir, out_dir

CS_DIR = dyadic_cs_dir()
FEAT_DIR = feature27_dir()
OUT_DIR = out_dir('supp')
ANALYSIS_OUT = out_dir('analysis')

PARTS = ['nose', 'l_ear', 'r_ear', 'neck', 'l_hip', 'r_hip', 'tail']
A_IDX = list(range(14))
ACT_THR = 0.05
N_PERM = 1000
NECK_X_IDX = 2 * PARTS.index('neck')

BONES_H = [('neck', 'nose'), ('neck', 'l_ear'), ('neck', 'r_ear')]
BONES_B = [('neck', 'l_hip'), ('neck', 'r_hip'), ('l_hip', 'tail'), ('r_hip', 'tail')]
PART_COL = dict(nose='#e74c3c', l_ear='#2980b9', r_ear='#5dade2',
                neck='#f39c12', l_hip='#27ae60', r_hip='#58d68d', tail='#7d3c98')
SPECTRUM_COLOR = (0.8, 0.1, 0.1)
DOMINANT_COLOR = '#e67e22'
NULL_HIST_COLOR = '#d0d0d0'
OBS_COLOR = '#d62728'
NEG_COLOR = '#1f77b4'

# Operator name, zero-based slot, and pose model.
TARGETS = [('f_2', 1, 'rank2_single'),
           ('f_3', 2, 'rank1_single'),
           ('f_6', 5, 'rank2_bidir'),
           ('f_9', 8, 'rank2_bidir'),
           ('f_15', 14, 'rank1_single')]


def normalize_features(d):
    f = d.copy().astype(float)
    s = np.maximum(np.std(f, axis=1, keepdims=True), 1e-3); f /= s
    q = max(np.quantile(np.abs(f), 0.99), 1e-6); f /= q
    return f


def reshape_for_plot(vec14):
    raw = vec14.reshape(7, 2)
    disp = np.column_stack([-raw[:, 1], raw[:, 0]])  # plot_x=-ego_y, plot_y=+ego_x
    return disp, raw


def anchor_v(v):
    if np.iscomplexobj(v):
        v = v.real
    if v[NECK_X_IDX] < 0:
        v = -v
    return v / (np.linalg.norm(v) + 1e-12)


def draw_skeleton(ax, disp_xy, lim=1.7, ms=45):
    pt = {p: disp_xy[i] for i, p in enumerate(PARTS)}
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.axhline(0, color='#eee', lw=0.5, zorder=0)
    ax.axvline(0, color='#eee', lw=0.5, zorder=0)
    for a, b in BONES_H + BONES_B:
        ax.plot([pt[a][0], pt[b][0]], [pt[a][1], pt[b][1]],
                color='#cdd2d5', lw=1.1, alpha=0.7, zorder=1)
    for i, p in enumerate(PARTS):
        ax.scatter(*disp_xy[i], s=ms, color=PART_COL[p],
                   edgecolors='white', linewidths=0.8, zorder=5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def draw_arrows(ax, disp_pose, field_disp, color, scale,
                lw=2.6, dot_radius_data=0.06, head_min=4.0, head_max=16.0, arrow_min=0.04):
    mags = np.linalg.norm(field_disp, axis=1) * scale
    max_mag = float(mags.max()) if mags.size else 1.0
    for i in range(7):
        x0, y0 = disp_pose[i]
        dx, dy = field_disp[i] * scale
        mag = float(np.hypot(dx, dy))
        if mag < arrow_min:
            continue
        ux, uy = dx / mag, dy / mag
        sx, sy = x0 + dot_radius_data * ux, y0 + dot_radius_data * uy
        bdx, bdy = (x0 + dx) - sx, (y0 + dy) - sy
        head = head_min + (head_max - head_min) * (float(np.hypot(bdx, bdy)) / (max_mag + 1e-9))
        head = float(np.clip(head, head_min, head_max))
        ax.annotate('', xy=(sx + bdx, sy + bdy), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, alpha=0.9,
                                    mutation_scale=head, shrinkA=0, shrinkB=0), zorder=8)


def arrow_panel(ax, mu_pose, field, color, tag=None, tag_color=None,
                target_arrow_length=0.55, arrow_lw=2.6,
                head_min=4.0, head_max=16.0, skeleton_ms=45):
    """Draw the mean pose skeleton + the deformation arrow `field` (14-D) on it."""
    disp_pose, _ = reshape_for_plot(mu_pose)
    disp_f, _ = reshape_for_plot(field)
    max_f = float(np.max(np.linalg.norm(disp_f, axis=1)))
    scale = target_arrow_length / (max_f + 1e-6)
    arrow_end = disp_pose + disp_f * scale
    lim = 1.12 * float(max(np.abs(disp_pose).max(), np.abs(arrow_end).max(), 0.6))
    draw_skeleton(ax, disp_pose, lim=lim, ms=skeleton_ms)
    draw_arrows(ax, disp_pose, disp_f, color, scale, lw=arrow_lw,
                head_min=head_min, head_max=head_max)
    if tag:
        ax.text(0.03, 0.96, tag, transform=ax.transAxes, fontsize=10,
                color=tag_color or color, fontweight='bold', va='top', ha='left', zorder=20,
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none', alpha=0.85))


def na_panel(ax, text='n/a\n(single-signed)'):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha='center', va='center',
            fontsize=11, color='#aaa', style='italic')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def spectrum_panel(ax, w_sorted, hi_idx, op_name, marker_area_scale=1.0,
                   compact=False):
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color='#bbb', lw=0.8, ls='--', zorder=1)
    ax.axhline(0, color='#e9e9e9', lw=0.6, zorder=0)
    ax.axvline(0, color='#e9e9e9', lw=0.6, zorder=0)
    mask = np.ones(len(w_sorted), bool)
    for k in hi_idx:
        mask[k] = False
    other = w_sorted[mask]
    line_scale = np.sqrt(marker_area_scale)
    ax.scatter(other.real, other.imag, s=55 * marker_area_scale,
               color=SPECTRUM_COLOR, alpha=0.60,
               edgecolors='white', linewidths=0.8 * line_scale, zorder=3)
    for k in hi_idx:
        ax.scatter(w_sorted[k].real, w_sorted[k].imag, s=55 * marker_area_scale,
                   color=DOMINANT_COLOR,
                   edgecolors='black', linewidths=1.1 * line_scale, zorder=5)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect('equal')
    ax.set_xlabel(r'Re$(\lambda)$', fontsize=6.2 if compact else 8)
    ax.set_ylabel(r'Im$(\lambda)$', fontsize=6.2 if compact else 8)
    if compact:
        ax.tick_params(labelsize=5.8, pad=1.5)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.grid(alpha=0.2)
    ax.text(0.03, 0.95, rf'$\mathbf{{{op_name.replace("_", "_{") + "}"}}}$',
            transform=ax.transAxes, fontsize=8.2 if compact else 12,
            fontweight='bold', va='top', ha='left', zorder=20)
    # eigenvalue label(s) bottom-right
    parts = []
    for k in hi_idx:
        lk = w_sorted[k]
        if abs(lk.imag) > 1e-6:
            parts.append(rf'$\lambda_{{{k+1}}}\!=\!{lk.real:+.2f}{lk.imag:+.2f}i$')
        else:
            parts.append(rf'$\lambda_{{{k+1}}}\!=\!{lk.real:+.2f}$')
    ax.text(0.97, 0.04, '\n'.join(parts), transform=ax.transAxes,
            fontsize=6.2 if compact else 9, ha='right',
            va='bottom', color=DOMINANT_COLOR, fontweight='bold', zorder=20, linespacing=1.3,
            bbox=dict(boxstyle='round,pad=0.20', facecolor='white', edgecolor='none', alpha=0.85))


def null_panel(ax, null, obs, p, xlabel, stat_label, val_fmt='{:.2f}'):
    ax.hist(null, bins=35, color=NULL_HIST_COLOR, edgecolor='white', alpha=0.9, zorder=2)
    ax.axvline(obs, color=OBS_COLOR, lw=2.2, zorder=4)
    mu, sd = float(null.mean()), float(null.std(ddof=1))
    ax.axvline(mu, color='#555', lw=1.1, ls=':', zorder=3)
    ax.axvspan(mu - sd, mu + sd, color='#888', alpha=0.18, zorder=1)
    ax.set_xlabel(xlabel, fontsize=7.5); ax.set_ylabel('null count', fontsize=8)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.grid(axis='y', alpha=0.25)
    ptxt = '$p\\!<\\!0.001$' if p < 1e-3 else rf'$p\!=\!{p:.3f}$'
    col = OBS_COLOR if p < 0.05 else '#666'
    ax.text(0.97, 0.95, (stat_label + r'$\!=\!$' + val_fmt.format(obs)) + '\n' + ptxt,
            transform=ax.transAxes, fontsize=10, ha='right', va='top', color=col,
            fontweight='bold', zorder=20, linespacing=1.3,
            bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor=col,
                      alpha=0.92, linewidth=0.8))


# Load aligned coefficients and features.
print('Loading sessions ...')
sessions = []
for sid in range(1, 71):
    cp = CS_DIR / f'cs_mouse{sid:03d}.npy'
    xp = FEAT_DIR / f'FEATURE27_mouse{sid:03d}.npy'
    if not (cp.exists() and xp.exists()):
        continue
    cs = np.load(cp); xr = np.load(xp)
    if xr.shape[0] != 27:
        xr = xr.T
    x = normalize_features(xr)
    T = min(cs.shape[1], x.shape[1])
    sessions.append((cs[:, :T], x[A_IDX, :T]))
print(f'  {len(sessions)} sessions')
F_all = np.load(dyadic_dictionary())


def phase_means(slot, shift_rng=None):
    """Pooled mu_bar, mu_+, mu_- (+counts) in normalised A-block coords."""
    ps = np.zeros(len(A_IDX)); ns = np.zeros(len(A_IDX)); alls = np.zeros(len(A_IDX))
    pc = nc = ac = 0.0
    for cs, xA in sessions:
        c = cs[slot]
        if shift_rng is not None:
            c = np.roll(c, int(shift_rng.integers(1, len(c))))
        pmf = (c > ACT_THR).astype(float); nmf = (c < -ACT_THR).astype(float)
        ps += xA @ pmf; ns += xA @ nmf; alls += xA.sum(1)
        pc += pmf.sum(); nc += nmf.sum(); ac += xA.shape[1]
    mu_bar = alls / max(ac, 1.0)
    mu_pos = ps / pc if pc > 0 else None
    mu_neg = ns / nc if nc > 0 else None
    return mu_bar, mu_pos, mu_neg, pc, nc


def deformation(mu_bar, mu_pos, mu_neg, bidir):
    if bidir and mu_neg is not None:
        return (mu_pos - mu_neg) / 2.0
    return mu_pos - mu_bar


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def predicted_direction(v1, v2, rank2, mu_bar, mu_pos):
    """Orient the matrix eigendirection using one coefficient-phase grouping."""
    positive_displacement = mu_pos - mu_bar
    s1 = float(np.sign(v1 @ positive_displacement)) or 1.0
    if rank2:
        s2 = float(np.sign(v2 @ positive_displacement)) or 1.0
        direction = unit(s1 * v1 + s2 * v2)
    else:
        s2 = None
        direction = unit(s1 * v1)
    return direction, s1, s2


def compute_row(name, slot, kind):
    F_AA = F_all[slot][np.ix_(A_IDX, A_IDX)]
    w, V = np.linalg.eig(F_AA)
    order = np.argsort(np.abs(w))[::-1]
    w, V = w[order], V[:, order]
    bidir = (kind == 'rank2_bidir')
    rank2 = kind.startswith('rank2')
    mu_bar, mu_pos, mu_neg, n_pos, n_neg = phase_means(slot)
    res = dict(name=name, slot=slot, kind=kind, w=w, mu_bar=mu_bar, mu_pos=mu_pos,
               mu_neg=mu_neg, n_pos=int(n_pos), n_neg=int(n_neg))

    # Rank-2 predictions use the equal-weight eigenplane bisector. Rank-1
    # predictions use the dominant eigenvector. Phase means orient both axes.
    v1 = unit(V[:, 0].real); v2 = unit(V[:, 1].real)
    d_pred, s1, s2 = predicted_direction(v1, v2, rank2, mu_bar, mu_pos)
    d_raw = deformation(mu_bar, mu_pos, mu_neg, bidir)
    obs = abs(float(d_pred @ unit(d_raw)))
    rng = np.random.default_rng(20260615 + slot)
    null = np.empty(N_PERM)
    for j in range(N_PERM):
        mb, mp, mn, pc, nc = phase_means(slot, rng)
        shifted_pred, _, _ = predicted_direction(v1, v2, rank2, mb, mp)
        shifted_emp = deformation(mb, mp, mn, bidir)
        null[j] = abs(float(shifted_pred @ unit(shifted_emp)))
    z = float((obs - null.mean()) / (null.std(ddof=1) + 1e-12))
    exceedances = int(np.count_nonzero(null >= obs))
    p = float((exceedances + 1) / (N_PERM + 1))
    res.update(d_raw=d_raw, d_pred=d_pred, v1_pred=s1 * v1,
               v2_pred=(s2 * v2 if rank2 else None),
               obs=obs, null=null, p=p, z=z)

    if rank2:
        res['hi_idx'] = [0, 1]          # highlight the 2-D eigenplane (lambda_1, lambda_2)
        print(f'  {name}: rank-2 {kind}  arrows=empirical d  d_pred=v1+v2  '
              f'|cos|={obs:.2f} null={null.mean():.2f} z={z:+.1f} p={p:.4f}  '
              f'(n+={int(n_pos)}, n-={int(n_neg)})')
    else:
        res['v'] = anchor_v(V[:, 0])    # rank-1 arrows = the dominant eigenvector
        res['alpha_p'] = float(res['v'] @ (mu_pos - mu_bar))
        res['hi_idx'] = [0]
        print(f'  {name}: rank-1 {kind}  arrows=v_1  d_pred=v1  '
              f'|cos|={obs:.2f} null={null.mean():.2f} z={z:+.1f} p={p:.4f} '
              f'-> {"PASS" if p < 0.05 else "n.s."}')
    return res


# Compute the five operator rows.
print('\nComputing rows ...')
rows = [compute_row(n, s, k) for n, s, k in TARGETS]

np.savez_compressed(
    ANALYSIS_OUT / 'dyadic_pose_alignment_nulls.npz',
    operator=np.asarray([row['name'] for row in rows]),
    observed=np.asarray([row['obs'] for row in rows]),
    null=np.stack([row['null'] for row in rows]),
    z=np.asarray([row['z'] for row in rows]),
    p_mc=np.asarray([row['p'] for row in rows]),
    n_positive=np.asarray([row['n_pos'] for row in rows]),
    n_negative=np.asarray([row['n_neg'] for row in rows]),
)

def matrix_direction_panel(ax, r):
    """Show the matrix-only predicted direction in its leading eigenspace."""
    ax.axhline(0, color='#eeeeee', lw=0.55)
    ax.axvline(0, color='#eeeeee', lw=0.55)
    if r['kind'].startswith('rank2'):
        e1 = unit(r['v1_pred'])
        e2 = r['v2_pred'] - (r['v2_pred'] @ e1) * e1
        e2 = unit(e2)

        def proj(v):
            return np.array([v @ e1, v @ e2])

        v1p, v2p, dp = proj(r['v1_pred']), proj(r['v2_pred']), proj(r['d_pred'])
        for vv, lab in ((v1p, r'$v_1$'), (v2p, r'$v_2$')):
            ax.annotate('', xy=vv, xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', lw=0.85, color='#888888',
                                        mutation_scale=6.5))
            ax.annotate(lab, xy=vv, xytext=(-3 if lab == r'$v_1$' else 2, 3),
                        textcoords='offset points', fontsize=6.3, color='#666666',
                        ha='right' if lab == r'$v_1$' else 'left', va='bottom')
        ax.annotate('', xy=dp, xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', lw=1.35, color='crimson',
                                    mutation_scale=7.5))
        ax.annotate('', xy=-dp, xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', lw=0.70, color='crimson',
                                    alpha=0.34, mutation_scale=6.5))
    else:
        ax.annotate('', xy=(1, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', lw=1.35, color='crimson',
                                    mutation_scale=7.5))
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25); ax.set_aspect('equal')
    ax.set_xlabel('eigenplane axis 1', fontsize=5.9, labelpad=1.0)
    ax.set_ylabel('axis 2', fontsize=5.9, labelpad=1.0)
    ax.tick_params(labelsize=5.4, pad=1.0)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


print('\nComposing refined main-text resident-pose panel ...')
figm = plt.figure(figsize=(6.9, 5.70))
gsm = figm.add_gridspec(
    5, 4, left=0.055, right=0.995, bottom=0.060, top=0.855,
    width_ratios=[1.00, 1.00, 1.08, 1.08], wspace=0.055, hspace=0.26)
axm = np.array([[figm.add_subplot(gsm[i, j]) for j in range(4)] for i in range(5)])

for i, r in enumerate(rows):
    spectrum_panel(axm[i, 0], r['w'], r['hi_idx'], r['name'],
                   marker_area_scale=0.24, compact=True)
    matrix_direction_panel(axm[i, 1], r)
    if i < len(rows) - 1:
        axm[i, 0].set_xlabel('')
        axm[i, 1].set_xlabel('')

    arrow_panel(axm[i, 2], r['mu_pos'], r['d_pred'], OBS_COLOR,
                tag=r'$+d_{\rm pred}$', target_arrow_length=0.24,
                arrow_lw=0.90, head_min=3.5, head_max=7.0,
                skeleton_ms=20)
    if r['kind'] == 'rank2_bidir':
        arrow_panel(axm[i, 3], r['mu_neg'], -r['d_pred'], NEG_COLOR,
                    tag=r'$-d_{\rm pred}$', tag_color=NEG_COLOR,
                    target_arrow_length=0.24, arrow_lw=0.90,
                    head_min=3.5, head_max=7.0, skeleton_ms=20)
    else:
        na_panel(axm[i, 3])

    # Compact the annotations generated by the shared helpers at final size.
    for j in (2, 3):
        for txt in axm[i, j].texts:
            if txt.get_fontsize() >= 10:
                txt.set_fontsize(6.4)

col_heads = ['eigenvalue spectrum', r'matrix direction $d_{\rm pred}$',
             r'empirical pose $+c$', r'empirical pose $-c$']
col_colors = ['#444444', 'crimson', OBS_COLOR, NEG_COLOR]
for j, (head, color) in enumerate(zip(col_heads, col_colors)):
    axm[0, j].annotate(head, xy=(0.5, 1.25), xycoords='axes fraction',
                       ha='center', fontsize=7.2, fontweight='bold', color=color)

p0, p1 = axm[0, 0].get_position(), axm[0, 1].get_position()
p2, p3 = axm[0, 2].get_position(), axm[0, 3].get_position()
xdiv = (p1.x1 + p2.x0) / 2
figm.add_artist(plt.Line2D([xdiv, xdiv], [0.03, 0.885], color='#b8b8b8', lw=0.80,
                           ls=(0, (4, 3)), transform=figm.transFigure))
figm.text((p0.x0 + p1.x1) / 2, 0.960, 'PREDICTION FROM OPERATOR MATRIX',
          ha='center', va='center', fontsize=8.7, fontweight='bold', color='crimson')
figm.text((p2.x0 + p3.x1) / 2, 0.960, 'EMPIRICAL VALIDATION',
          ha='center', va='center', fontsize=8.7, fontweight='bold', color='#2e7d32')

main_base = OUT_DIR / 'dyadic_fig2a_matrix_pose_refined'
figm.savefig(f'{main_base}.pdf', facecolor='white')
plt.close(figm)
print(f'  saved {main_base}.pdf')


# Circular-shift null panels.
print('\nComposing supplementary resident-pose null tests ...')
fign, axn = plt.subplots(1, 5, figsize=(6.9, 1.72))
plt.subplots_adjust(left=0.055, right=0.995, bottom=0.25, top=0.75, wspace=0.34)
for j, r in enumerate(rows):
    null_panel(axn[j], r['null'], r['obs'], r['p'],
               r'$|\cos(d_{\rm pred},d_{\rm emp})|$', r'$|\cos|$')
    axn[j].set_title(rf'$\mathbf{{{r["name"].replace("_", "_{") + "}"}}}$',
                     fontsize=8.0, fontweight='bold', pad=2)
    axn[j].set_ylabel('null count' if j == 0 else '', fontsize=6.1)
    axn[j].set_xlabel(r'$|\cos(d_{\rm pred},d_{\rm emp})|$', fontsize=5.6, labelpad=1.5)
    axn[j].tick_params(labelsize=5.4, pad=1.2)
    for txt in axn[j].texts:
        txt.set_fontsize(6.2)
fign.suptitle('Resident-pose alignment against 1,000 per-session circular shifts',
              fontsize=8.5, fontweight='bold', y=0.96)
null_base = OUT_DIR / 'dyadic_supp_pose_alignment_nulls'
fign.savefig(f'{null_base}.pdf', facecolor='white')
plt.close(fign)
print(f'  saved {null_base}.pdf')
