"""Nonparametric 3D occupancy wireframe for the Figure 3 syllable-21 cloud.

The wireframe is built from smoothed empirical density contours in five
quantile slices along the cloud's first principal direction.  It deliberately
does not fit an ellipsoid or other parametric boundary.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter


EDGE_COLOR = '#3f006e'


def _largest_contour(x, y, density, level):
    """Extract the largest closed contour without retaining a helper figure."""
    helper, ax = plt.subplots(figsize=(1, 1))
    contour = ax.contour(x, y, density.T, levels=[level])
    segments = contour.allsegs[0]
    plt.close(helper)
    if not segments:
        return None

    def area(segment):
        if len(segment) < 3:
            return 0.0
        xx, yy = segment[:, 0], segment[:, 1]
        return 0.5 * abs(np.dot(xx, np.roll(yy, 1)) - np.dot(yy, np.roll(xx, 1)))

    return max(segments, key=area)


def empirical_occupancy_wireframe(P, coverage=0.88, n_slices=3, bins=40, sigma=1.80):
    """Return transverse density contours and longitudinal connectors.

    Parameters
    ----------
    P : (n, 3) array
        The exact displayed syllable points.  Using this array keeps the
        wireframe tied to the same empirical sample shown in panels a--c.
    coverage : float
        Approximate within-slice highest-density mass enclosed by each contour.
    """
    center = P.mean(0)
    _, _, vt = np.linalg.svd(P - center, full_matrices=False)
    basis = vt.T
    q = (P - center) @ basis

    q1_lo, q1_hi = np.quantile(q[:, 1], [0.01, 0.99])
    q2_lo, q2_hi = np.quantile(q[:, 2], [0.01, 0.99])
    pad1 = max((q1_hi - q1_lo) * 0.06, 1e-4)
    pad2 = max((q2_hi - q2_lo) * 0.06, 1e-4)
    e1 = np.linspace(q1_lo - pad1, q1_hi + pad1, bins + 1)
    e2 = np.linspace(q2_lo - pad2, q2_hi + pad2, bins + 1)
    x = 0.5 * (e1[:-1] + e1[1:])
    y = 0.5 * (e2[:-1] + e2[1:])

    slice_edges = np.quantile(q[:, 0], np.linspace(0.05, 0.95, n_slices + 1))
    rings_q = []
    for lo, hi in zip(slice_edges[:-1], slice_edges[1:]):
        take = (q[:, 0] >= lo) & (q[:, 0] <= hi)
        hist, _, _ = np.histogram2d(q[take, 1], q[take, 2], bins=(e1, e2))
        density = gaussian_filter(hist.astype(float), sigma=sigma, mode='nearest')
        values = np.sort(density.ravel())[::-1]
        cumulative = np.cumsum(values)
        if cumulative[-1] <= 0:
            continue
        level = values[np.searchsorted(cumulative, coverage * cumulative[-1])]
        segment = _largest_contour(x, y, density, level)
        if segment is None or len(segment) < 8:
            continue
        q0 = float(np.median(q[take, 0]))
        rings_q.append(np.column_stack([np.full(len(segment), q0), segment]))

    rings = [center + ring @ basis.T for ring in rings_q]

    connectors = []
    for angle in np.linspace(0, 2 * np.pi, 3, endpoint=False):
        selected = []
        for ring in rings_q:
            yz = ring[:, 1:]
            rel = yz - yz.mean(0, keepdims=True)
            theta = np.arctan2(rel[:, 1], rel[:, 0])
            distance = np.abs(np.angle(np.exp(1j * (theta - angle))))
            selected.append(ring[np.argmin(distance)])
        if len(selected) >= 2:
            connectors.append(center + np.asarray(selected) @ basis.T)

    return rings, connectors


def draw_empirical_wireframe(ax, rings, connectors, lw=1.05):
    """Draw a restrained dashed 3D wireframe with a narrow white halo."""
    halo = [pe.Stroke(linewidth=lw + 0.55, foreground='white', alpha=0.72), pe.Normal()]
    for ring in rings:
        line, = ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=EDGE_COLOR,
                        lw=lw, ls=(0, (3.0, 2.2)), alpha=0.96, zorder=8)
        line.set_path_effects(halo)
    for connector in connectors:
        line, = ax.plot(connector[:, 0], connector[:, 1], connector[:, 2], color=EDGE_COLOR,
                        lw=0.76 * lw, ls=(0, (2.8, 2.8)), alpha=0.82, zorder=7)
        line.set_path_effects(halo)
