"""Diagnostic checks for the reference-domain selection derivation note (2026-08-23).

NumPy-only closed-form/reference checks (diagnostic code per repo backend rule).
Feeds: docs/plans/bayesfilter-reference-domain-selection-derivation-note-2026-08-23.md

Checks:
  (A) In-box Gaussian mass (2*Phi(kappa)-1)^n at the containment-capped kappa*  [O2 bound]
  (B) Normalized probabilists' Hermite magnitude |He_k(z)|/sqrt(k!)             [O5]
  (C) Hermite mass-matrix orthonormality under N(0,1) via Gauss-Hermite        [R1 for C2]
  (D) Legendre coefficient decay of sqrt(F1) under AlgebraicMapping(1)          [O4 for C3]
      vs sqrt(truncated Gaussian) comparator (analytic => geometric)            [O4 for C1]
"""
import math
import numpy as np
from numpy.polynomial import legendre as L
from numpy.polynomial import hermite_e as HE

def Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

print("== (A) in-box mass (2*Phi(k)-1)^n ==")
for kappa in (2.0, 2.2, 3.0):
    per = 2 * Phi(kappa) - 1
    row = ", ".join(f"n={n}: in={per**n:.3f} out={1-per**n:.3f}" for n in (2, 4, 8))
    print(f"kappa*={kappa}: per-axis={per:.4f} | {row}")

print("== (B) |He_k(z)|/sqrt(k!) ==")
for k in (8, 12, 16):
    hek = HE.HermiteE.basis(k)
    for z in (3.0, 4.0):
        print(f"k={k} z={z}: {abs(hek(z))/math.sqrt(math.factorial(k)):.3e}")

print("== (C) Hermite orthonormality max |B - I|, degrees 0..12 ==")
nodes_h, w_h = HE.hermegauss(60)
w_h = w_h / math.sqrt(2 * math.pi)  # normalize weight e^{-z^2/2} -> N(0,1)
V = np.stack([HE.HermiteE.basis(k)(nodes_h) / math.sqrt(math.factorial(k))
              for k in range(13)])
B = (V * w_h) @ V.T
print(f"max|B - I| = {np.abs(B - np.eye(13)).max():.2e}")

print("== (D) Legendre decay: sqrt(F1) under algebraic map vs truncated-Gaussian sqrt ==")
nodes, weights = np.polynomial.legendre.leggauss(4000)

def legendre_tail_profile(vals, kmax=80):
    c = np.array([(2 * k + 1) / 2 * np.sum(weights * vals * L.Legendre.basis(k)(nodes))
                  for k in range(kmax + 1)])
    norms = c**2 * 2 / (2 * np.arange(kmax + 1) + 1)
    tot = norms.sum()
    tail = np.sqrt(np.maximum(np.cumsum(norms[::-1])[::-1], 0.0) / tot)
    return c, tail

for s in (0.5, 1.0, 2.0):
    with np.errstate(over="ignore", under="ignore"):
        x = s * nodes / np.sqrt(1 - nodes**2)
        eta = np.exp(-x**2 / 2) / math.sqrt(2 * math.pi)
        jac = s * (1 - nodes**2) ** (-1.5)
        F1 = eta * jac
    mass = np.sum(weights * F1)
    vals = np.sqrt(F1)
    c, tail = legendre_tail_profile(vals)
    deg = {tol: int(np.argmax(tail < tol)) if (tail < tol).any() else -1
           for tol in (1e-2, 1e-3, 1e-4)}
    print(f"s={s}: mass check ∫F1 dz = {mass:.6f} (want 1)")
    print(f"  |c_k| at k=0,4,8,16,32,64: "
          + ", ".join(f"{abs(c[k]):.1e}" for k in (0, 4, 8, 16, 32, 64)))
    print(f"  degree for rel-L2 tail < 1e-2/1e-3/1e-4: "
          f"{deg[1e-2]}/{deg[1e-3]}/{deg[1e-4]}")

# comparator: sqrt of a kappa*=2 truncated Gaussian on the box (entire function)
vals = np.exp(-(2.0 * nodes) ** 2 / 4)
c, tail = legendre_tail_profile(vals)
deg = {tol: int(np.argmax(tail < tol)) if (tail < tol).any() else -1
       for tol in (1e-2, 1e-3, 1e-4)}
print("truncated-Gaussian sqrt (C1 comparator, kappa=2):")
print("  |c_k| at k=0,4,8,16,32,64: "
      + ", ".join(f"{abs(c[k]):.1e}" for k in (0, 4, 8, 16, 32, 64)))
print(f"  degree for rel-L2 tail < 1e-2/1e-3/1e-4: {deg[1e-2]}/{deg[1e-3]}/{deg[1e-4]}")
