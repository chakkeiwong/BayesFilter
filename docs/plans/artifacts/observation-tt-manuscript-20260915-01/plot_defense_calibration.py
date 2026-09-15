"""Deterministic diagnostic illustration of A05's declared safety bounds.

This reporting script does not select parameters or enter a runtime path.
All plotted values follow the displayed mixture formula, not fitted data.
"""
from pathlib import Path
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parent
n = 512
stress = math.sqrt(2.220446049250313e-16)
mass = [10 ** (-8 + i * (math.log10(.05) + 8) / 600) for i in range(601)]
healthy = [n*a/((1-a)*.25+a) for a in mass]
unprotected = [n*(1-a)*stress/((1-a)*stress+a) for a in mass]
lower = (n-1)*stress/(1+(n-1)*stress)
upper = .25/(n-1+.25)
plt.rcParams.update({'font.family':'serif', 'mathtext.fontset':'cm', 'font.size':10})
fig, ax = plt.subplots(figsize=(7.1, 3.65), layout='constrained')
ax.axvspan(lower, upper, color='#dfeddc', label='Both bounds satisfied')
ax.loglog(mass, healthy, color='#215c92', label='Changed healthy draws')
ax.loglog(mass, unprotected, color='#303030', linestyle='--', label='Unprotected stress draws')
ax.axhline(1, color='#9a3826', linewidth=1, linestyle=':', label='One-draw allowance')
ax.axvline(1e-5, color='#527943', linewidth=.9, linestyle='-.')
ax.annotate(r'Selected $a=10^{-5}$', xy=(1e-5, .004), xytext=(2e-7, .0006),
            arrowprops={'arrowstyle':'->', 'color':'#527943'}, fontsize=9)
ax.set(xlabel=r'Joint Gaussian fraction $a$', ylabel='Expected draws out of 512',
       xlim=(1e-8, .05), ylim=(1e-5, 800))
ax.legend(loc='upper right', framealpha=.95, fontsize=8.7)
ax.spines[['top','right']].set_visible(False)
fig.savefig(out/'defense-calibration.pdf')
fig.savefig(out/'defense-calibration.png', dpi=170)
print(out/'defense-calibration.pdf')
