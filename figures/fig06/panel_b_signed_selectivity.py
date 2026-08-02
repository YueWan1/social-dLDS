"""Figure 6b: signed behavior selectivity of resident-pose operators.

Each cell reports the signed-amplitude z score against 1,000 per-session
circular shifts. Values are read from the released analysis table.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'font.family': 'sans-serif', 'font.size': 10,
                     'axes.linewidth': 0.8})

from dlds_release.paths import DERIVED, out_dir

OUT = out_dir('fig06') / 'fig6b_signed_selectivity'
VALUES = DERIVED / 'dyadic' / 'signed_selectivity_ztable.npz'

OPS = ['f_2', 'f_3', 'f_6', 'f_9', 'f_15']
BEH = ['attack', 'invest.', 'mount']

with np.load(VALUES) as released:
    Z = np.array(
        [
            [
                float(released[f'z_{operator}_signed_{behavior}'])
                for behavior in ('attack', 'invest', 'mount')
            ]
            for operator in OPS
        ]
    )

APEX = [(3, 2), (4, 0)]  # (f9, mount), (f15, attack)


def star(z):
    a = abs(z)
    return '***' if a > 10 else '**' if a > 5 else '*' if a > 3 else ''


VMAX = 10.0  # cap so the +24.4 investigation outlier does not wash out the +-5 cells
norm = TwoSlopeNorm(vmin=-VMAX, vcenter=0.0, vmax=VMAX)
cmap = plt.get_cmap('RdBu_r')

fig, ax = plt.subplots(figsize=(3.7, 3.5))
im = ax.imshow(np.clip(Z, -VMAX, VMAX), cmap=cmap, norm=norm, aspect='auto')

ax.set_xticks(range(len(BEH)))
ax.set_xticklabels(BEH, fontsize=10)
ax.set_yticks(range(len(OPS)))
ax.set_yticklabels([rf'$\mathbf{{{o.replace("_", "_{") + "}"}}}$' for o in OPS],
                   fontsize=11.5)

for i in range(len(OPS)):
    for j in range(len(BEH)):
        z = Z[i, j]
        s = star(z)
        txt = f'{z:+.1f}' + (f'\n{s}' if s else '')
        shade = abs(np.clip(z, -VMAX, VMAX))
        ax.text(j, i, txt, ha='center', va='center', fontsize=8.6,
                color='white' if shade > VMAX * 0.62 else '#1a1a1a',
                fontweight='bold' if s else 'normal', linespacing=0.82)

for (i, j) in APEX:
    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                           edgecolor='black', lw=2.4, zorder=5))

ax.set_xticks(np.arange(-.5, len(BEH), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(OPS), 1), minor=True)
ax.grid(which='minor', color='white', lw=1.2)
ax.tick_params(which='minor', length=0)
ax.tick_params(which='major', length=0)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.08, extend='max')
cbar.set_label(r'behavior selectivity $z$', fontsize=9)
cbar.ax.tick_params(labelsize=8)
# make the sign meaningful: red end = positive coefficient phase, blue = negative
cbar.ax.text(0.5, 1.045, r'$c_m\!>\!0$ phase', transform=cbar.ax.transAxes,
             ha='center', va='bottom', fontsize=7.4, color='#b2182b')
cbar.ax.text(0.5, -0.055, r'$c_m\!<\!0$ phase', transform=cbar.ax.transAxes,
             ha='center', va='top', fontsize=7.4, color='#2166ac')

ax.set_title(r'sign-resolved behavior selectivity', fontsize=9.5, pad=7)

fig.tight_layout()
fig.savefig(str(OUT) + '.pdf', bbox_inches='tight')
print(f'Saved {OUT}.pdf')
print('Z (rows f2,f3,f6,f9,f15 x cols atk,inv,mnt):')
print(Z)
