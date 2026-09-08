"""C2 row-law mechanism evidence (2026-08-24): raw eta rows vs Christoffel vs half-mixture.

Empirical design-Gram conditioning at ell=13 for Monte Carlo least squares
under the Gaussian reference. NumPy diagnostic (backend rule). Feeds the
C2 derivation note Engine Findings section.
"""
import numpy as np, math
rng = np.random.default_rng(0)
ELL = 13
def hermite(u):
    V = np.zeros((u.size, ELL)); V[:, 0] = 1.0
    if ELL > 1: V[:, 1] = u
    for k in range(1, ELL - 1):
        V[:, k + 1] = (u * V[:, k] - math.sqrt(k) * V[:, k - 1]) / math.sqrt(k + 1)
    return V
for N in (2048, 14336):
    u = rng.standard_normal(N)
    V = hermite(u)
    G = V.T @ V / N
    print(f"raw eta rows      N={N:6d}: cond={np.linalg.cond(G):.2e} maxdev={np.max(np.abs(G-np.eye(ELL))):.2e}")
# Christoffel: q(u) = eta(u) * mean_k He~_k(u)^2 ; sample via k~U, u~He_k^2 eta (grid inverse-CDF)
grid = np.linspace(-14, 14, 40001)
eta = np.exp(-grid**2 / 2) / math.sqrt(2 * math.pi)
Vg = hermite(grid)
q = eta * (Vg**2).mean(axis=1)
cdf = np.cumsum(q); cdf /= cdf[-1]
for N in (2048, 14336):
    u = np.interp(rng.uniform(size=N), cdf, grid)
    V = hermite(u)
    w = eta_u = np.exp(-u**2/2)/math.sqrt(2*math.pi)
    qu = eta_u * (V**2).mean(axis=1)
    w = eta_u / qu
    G = (V * w[:, None]).T @ V / N
    print(f"christoffel rows  N={N:6d}: cond={np.linalg.cond(G):.2e} maxdev={np.max(np.abs(G-np.eye(ELL))):.2e}")

# defensive half-mixture: q = eta*(0.5 + 0.5*cbar), w = 1/(0.5+0.5*cbar) <= 2
q2 = eta * (0.5 + 0.5 * (Vg**2).mean(axis=1))
cdf2 = np.cumsum(q2); cdf2 /= cdf2[-1]
for N in (2048, 14336):
    u = np.interp(rng.uniform(size=N), cdf2, grid)
    V = hermite(u)
    w = 1.0 / (0.5 + 0.5 * (V**2).mean(axis=1))
    G = (V * w[:, None]).T @ V / N
    wn = w / w.sum()
    print(f"half-mixture rows N={N:6d}: cond={np.linalg.cond(G):.2e} "
          f"maxdev={np.max(np.abs(G-np.eye(ELL))):.2e} ess={1.0/np.sum(wn**2):.0f}")
