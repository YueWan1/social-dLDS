"""Recompute the DIST and DIRC operator screens in Supplementary Figure S4."""
import numpy as np

from dlds_release.paths import dyadic_cs_dir, dyadic_dictionary, feature27_dir, out_dir

RUN_DIR = dyadic_cs_dir()
FEAT_DIR = feature27_dir()
OUT_DIR = out_dir('analysis')

A_IDX = list(range(0, 14))
B_IDX = list(range(14, 25))
C_IDX = list(range(25, 27))

SLOT = {'f4': 3, 'f7': 6, 'f10': 9, 'f11': 10}

N_PERM = 1000
ENGAGE_THR = 0.05
MIN_ENGAGED = 5
STD_EPS = 1e-9


def normalize_features(d):
    f = d.copy().astype(float)
    s = np.maximum(np.std(f, axis=1, keepdims=True), 1e-3)
    f /= s
    q = max(np.quantile(np.abs(f), 0.99), 1e-6)
    f /= q
    return f


print('Loading sessions ...')
sessions = []
for sid in range(1, 71):
    cp = RUN_DIR / f'cs_mouse{sid:03d}.npy'
    xp = FEAT_DIR / f'FEATURE27_mouse{sid:03d}.npy'
    if not (cp.exists() and xp.exists()):
        continue
    cs = np.load(cp)
    xr = np.load(xp)
    if xr.shape[0] != 27:
        xr = xr.T
    x = normalize_features(xr)
    T = min(cs.shape[1], x.shape[1])
    sessions.append(dict(sid=sid, T=T, cs=cs[:, :T], x=x[:, :T]))
print(f'  {len(sessions)} sessions loaded')
session_ids = np.array([s['sid'] for s in sessions])

F_all = np.load(dyadic_dictionary())
print(f'  F_universal shape = {F_all.shape}')


def distance_readout(op):
    slot = SLOT[op]
    F_BB = F_all[slot][np.ix_(B_IDX, B_IDX)]
    w, V = np.linalg.eig(F_BB)
    order = np.argsort(np.abs(w))[::-1]
    v1 = V[:, order[0]].real
    lam1 = w[order[0]]

    rs, ids, pooled_c, pooled_p = [], [], [], []
    n_const = 0
    for s in sessions:
        c = s['cs'][slot]
        xB = s['x'][B_IDX]
        if c.std() < STD_EPS:
            n_const += 1
            continue
        proj = v1 @ xB
        rs.append(float(np.corrcoef(c, proj)[0, 1]))
        ids.append(s['sid'])
        pooled_c.append(c)
        pooled_p.append(proj)

    rs = np.array(rs)
    ids = np.array(ids)
    flipped = False
    if len(rs) and np.median(rs) < 0:
        v1 = -v1
        rs = -rs
        pooled_p = [-p for p in pooled_p]
        flipped = True
    pc = np.concatenate(pooled_c)
    pp = np.concatenate(pooled_p)
    r_pool = float(np.corrcoef(pc, pp)[0, 1])

    return dict(v1=v1, lam1_re=float(np.real(lam1)), lam1_im=float(np.imag(lam1)),
                rs=rs, ids=ids, r_pool=r_pool, n_sess=len(rs),
                n_const=n_const, n_frames=int(len(pc)),
                n_gt05=int((rs > 0.5).sum()), flipped=flipped)


def _dir_rs(slot, u1, v1, offsets=None):
    rs = []
    for i, s in enumerate(sessions):
        c = s['cs'][slot]
        if offsets is not None:
            c = np.roll(c, offsets[i])
        y = u1 @ s['x'][C_IDX]
        p = c * (v1 @ s['x'][B_IDX])
        if y.std() < 1e-9 or p.std() < 1e-9:
            continue
        rs.append(float(np.corrcoef(y, p)[0, 1]))
    return np.array(rs)


def _dir_ids(slot, u1, v1):
    ids = []
    for s in sessions:
        c = s['cs'][slot]
        y = u1 @ s['x'][C_IDX]
        p = c * (v1 @ s['x'][B_IDX])
        if y.std() < 1e-9 or p.std() < 1e-9:
            continue
        ids.append(s['sid'])
    return np.array(ids)


def direction_readout(op, seed=20260601):
    slot = SLOT[op]
    F_CB = F_all[slot][np.ix_(C_IDX, B_IDX)]
    U, sig, Vt = np.linalg.svd(F_CB, full_matrices=False)
    u1 = U[:, 0].copy()
    v1 = Vt[0, :].copy()
    eta1 = float(sig[0] ** 2 / max((sig ** 2).sum(), 1e-12))
    theta_u = float(np.degrees(np.arctan2(u1[1], u1[0])))

    rs = _dir_rs(slot, u1, v1)
    flipped = False
    if len(rs) and np.median(rs) < 0:
        u1, v1 = -u1, -v1
        theta_u = theta_u + 180 if theta_u <= 0 else theta_u - 180
        rs = _dir_rs(slot, u1, v1)
        flipped = True
    ids = _dir_ids(slot, u1, v1)

    rng = np.random.default_rng(seed)
    null_med = np.zeros(N_PERM)
    for p in range(N_PERM):
        offs = [int(rng.integers(1, s['T'])) for s in sessions]
        null_med[p] = float(np.median(_dir_rs(slot, u1, v1, offsets=offs)))
    nm, nsd = float(null_med.mean()), float(null_med.std(ddof=1))
    med = float(np.median(rs))

    ys, ps = [], []
    for s in sessions:
        ys.append(u1 @ s['x'][C_IDX])
        ps.append(s['cs'][slot] * (v1 @ s['x'][B_IDX]))
    r_pool = float(np.corrcoef(np.concatenate(ys), np.concatenate(ps))[0, 1])

    return dict(u1=u1, v1=v1, sigma1=float(sig[0]), sigma2=float(sig[1]),
                eta1=eta1, theta_u=theta_u, rs=rs, ids=ids, median=med,
                r_pool=r_pool, n_sess=len(rs), n_gt05=int((rs > 0.5).sum()),
                null_med=null_med, null_mean=nm, null_sd=nsd,
                z=float((med - nm) / (nsd + 1e-12)), flipped=flipped)


def _R_from_shift(slot, offsets=None):
    xs, ys = [], []
    for i, s in enumerate(sessions):
        c = s['cs'][slot]
        if offsets is not None:
            c = np.roll(c, offsets[i])
        m = np.abs(c) > ENGAGE_THR
        if m.sum() < MIN_ENGAGED:
            continue
        xC = s['x'][C_IDX][:, m]
        xs.append(xC[0])
        ys.append(xC[1])
    if not xs:
        return np.nan, np.nan, 0
    ang = np.arctan2(np.concatenate(ys), np.concatenate(xs))
    n = len(ang)
    C = np.cos(ang).sum()
    S = np.sin(ang).sum()
    return float(np.sqrt(C ** 2 + S ** 2) / n), float(np.degrees(np.arctan2(S, C))), n


def concentration(op, seed=42):
    slot = SLOT[op]
    R, mu, n = _R_from_shift(slot)
    rng = np.random.default_rng(seed)
    null = np.zeros(N_PERM)
    for p in range(N_PERM):
        offs = [int(rng.integers(1, s['T'])) for s in sessions]
        null[p] = _R_from_shift(slot, offsets=offs)[0]
    null = null[np.isfinite(null)]
    nm, nsd = float(null.mean()), float(null.std(ddof=1))
    sess_R, sess_ids, sess_mu = [], [], []
    for s in sessions:
        m = np.abs(s['cs'][slot]) > ENGAGE_THR
        if m.sum() < MIN_ENGAGED:
            continue
        xC = s['x'][C_IDX][:, m]
        a = np.arctan2(xC[1], xC[0])
        sess_R.append(float(np.hypot(np.cos(a).mean(), np.sin(a).mean())))
        sess_mu.append(float(np.degrees(np.arctan2(np.sin(a).mean(),
                                                   np.cos(a).mean()))))
        sess_ids.append(s['sid'])
    return dict(R=R, mu_deg=mu, n_frames=n, null=null, null_mean=nm,
                null_sd=nsd, z=float((R - nm) / (nsd + 1e-12)),
                sess_R=np.array(sess_R), sess_ids=np.array(sess_ids),
                sess_mu_deg=np.array(sess_mu))


def u1_angle_deg(op):
    """Return the leading C-from-B axis after fixing its joint SVD sign."""
    slot = SLOT[op]
    F_CB = F_all[slot][np.ix_(C_IDX, B_IDX)]
    U, sig, Vt = np.linalg.svd(F_CB, full_matrices=False)
    u1 = U[:, 0].copy()
    v1 = Vt[0, :].copy()
    raw = float(np.degrees(np.arctan2(u1[1], u1[0])))
    rs = _dir_rs(slot, u1, v1)
    flipped = bool(len(rs) and np.median(rs) < 0)
    if flipped:
        u1, v1 = -u1, -v1
    return float(np.degrees(np.arctan2(u1[1], u1[0]))), raw, flipped, u1


def wrap180(d):
    """Wrap a degree difference into (-180, 180]."""
    return float(np.degrees(np.angle(np.exp(1j * np.radians(d)))))


out = {'session_ids': session_ids}
L = []


def say(s=''):
    print(s)
    L.append(s)


say('Operator screening read-outs')
say(f'run dir : {RUN_DIR}')
say(f'sessions: {len(sessions)}')
say('')

say('(1) DIST read-out  [V1 correlation]')
say('    v_1 = leading eigenvector of the operator\'s DIST<-DIST block')
say(f'{"op":>5} {"n_sess":>7} {"n_const":>8} {"median r":>10} {"mean r":>8} '
    f'{"min r":>8} {"max r":>8} {"r_pool":>9} {"#r>0.5":>8} {"lam1":>9}')
for op in ['f4', 'f7']:
    d = distance_readout(op)
    out[f'{op}_r_per_session'] = d['rs']
    out[f'{op}_r_session_ids'] = d['ids']
    out[f'{op}_r_pool'] = np.array(d['r_pool'])
    out[f'{op}_v1_BB'] = d['v1']
    out[f'{op}_n_gt05'] = np.array(d['n_gt05'])
    say(f'{op:>5} {d["n_sess"]:>7d} {d["n_const"]:>8d} {np.median(d["rs"]):>+10.6f} '
        f'{d["rs"].mean():>+8.4f} {d["rs"].min():>+8.4f} {d["rs"].max():>+8.4f} '
        f'{d["r_pool"]:>+9.6f} {d["n_gt05"]:>4d}/{d["n_sess"]:<3d} {d["lam1_re"]:>+9.4f}')
say('')

say('(2) DIRC read-out  [V2 frame-level, DIRC<-DIST]')
say('    u_1,v_1 = the operator\'s DIRC<-DIST leading singular vectors')
say(f'{"op":>5} {"sig1":>7} {"eta1":>7} {"angU1":>8} {"n_sess":>7} {"median r":>10} '
    f'{"mean r":>8} {"min r":>8} {"#r>0.5":>8} {"r_pool":>9} {"null mu":>9} '
    f'{"null sd":>8} {"z":>9}')
for op in ['f11', 'f10']:
    d = direction_readout(op)
    out[f'{op}_dir_r_per_session'] = d['rs']
    out[f'{op}_dir_r_session_ids'] = d['ids']
    out[f'{op}_dir_median_r'] = np.array(d['median'])
    out[f'{op}_dir_r_pool'] = np.array(d['r_pool'])
    out[f'{op}_dir_null_medians'] = d['null_med']
    out[f'{op}_dir_z'] = np.array(d['z'])
    out[f'{op}_u1_CB'] = d['u1']
    out[f'{op}_v1_CB'] = d['v1']
    say(f'{op:>5} {d["sigma1"]:>7.4f} {d["eta1"]:>7.4f} {d["theta_u"]:>+8.1f} '
        f'{d["n_sess"]:>7d} {d["median"]:>+10.6f} {d["rs"].mean():>+8.4f} '
        f'{d["rs"].min():>+8.4f} {d["n_gt05"]:>4d}/{d["n_sess"]:<3d} '
        f'{d["r_pool"]:>+9.6f} {d["null_mean"]:>+9.5f} {d["null_sd"]:>8.5f} '
        f'{d["z"]:>+9.4f}')
say('')

say('(3) Azimuthal concentration R')
say(f'{"op":>5} {"R":>9} {"mu_deg":>9} {"n_frames":>10} {"n_sess":>7} '
    f'{"null mu":>9} {"null sd":>9} {"z":>9}')
for op in ['f11', 'f10', 'f4', 'f7']:
    d = concentration(op)
    out[f'{op}_R'] = np.array(d['R'])
    out[f'{op}_null_R'] = d['null']
    out[f'{op}_R_z'] = np.array(d['z'])
    out[f'{op}_R_per_session'] = d['sess_R']
    out[f'{op}_R_session_ids'] = d['sess_ids']
    out[f'{op}_mu_bearing_deg'] = np.array(d['mu_deg'])
    out[f'{op}_mu_bearing_rad'] = np.array(np.radians(d['mu_deg']))
    out[f'{op}_mu_per_session_deg'] = d['sess_mu_deg']
    say(f'{op:>5} {d["R"]:>9.6f} {d["mu_deg"]:>+9.3f} {d["n_frames"]:>10d} '
        f'{len(d["sess_R"]):>7d} {d["null_mean"]:>9.6f} {d["null_sd"]:>9.6f} '
        f'{d["z"]:>+9.4f}')
say('')

say('(4) Matrix-predicted axis vs empirical bearing')
say('    angle_u1 = atan2(u_1[1], u_1[0]) after joint (u_1,v_1) sign anchoring')
say('    sign anchoring of the V2^frame statistic (median per-session r > 0).')
say('    signed error = wrap180(angle_u1 - mu)   in (-180, 180]')
say('    axial error  = min(|signed|, 180 - |signed|)   in [0, 90]')
say(f'{"op":>5} {"angU1_anch":>11} {"angU1_raw":>10} {"flip":>5} {"mu_deg":>9} '
    f'{"signed_err":>11} {"axial_err":>10}')
for op in ['f11', 'f10', 'f4', 'f7']:
    ang, raw, flipped, u1 = u1_angle_deg(op)
    mu = float(out[f'{op}_mu_bearing_deg'])
    signed = wrap180(ang - mu)
    axial = min(abs(signed), 180.0 - abs(signed))
    out[f'{op}_u1_angle_deg'] = np.array(ang)
    out[f'{op}_u1_angle_raw_deg'] = np.array(raw)
    out[f'{op}_u1_sign_flipped'] = np.array(flipped)
    out[f'{op}_u1_CB_anchored'] = u1
    out[f'{op}_axis_error_signed_deg'] = np.array(signed)
    out[f'{op}_axis_error_deg'] = np.array(axial)
    say(f'{op:>5} {ang:>+11.3f} {raw:>+10.3f} {str(flipped):>5} {mu:>+9.3f} '
        f'{signed:>+11.3f} {axial:>10.3f}')
say('')
say('    Axis convention: report min(|signed|, 180-|signed|). The SVD axis is')
say('    defined only up to a joint sign flip, so its axial error is invariant.')
say('')

npz_path = OUT_DIR / 'readout_reproducibility.npz'
txt_path = OUT_DIR / 'readout_reproducibility.txt'
np.savez_compressed(npz_path, **out)
say(f'Saved arrays -> {npz_path}')
txt_path.write_text('\n'.join(L) + '\n')
print(f'Saved summary -> {txt_path}')
