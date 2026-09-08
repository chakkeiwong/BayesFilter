"""Diagnostic spot-checks for Sections 13, 14, and 5.1 of the ZLB survey.

Companion to derivation_check_ukf_section_20260819.py; same diagnostic scope
statement applies (NumPy/SciPy reference checks, not runtime inference code).

Checked closed forms (equation numbers of zlb_discontinuous_hmc_survey.md,
19 August 2026 second-block numbering):

  1. (82): a Nelson-Siegel forward curve crosses a level at most twice
     (randomized sign-change count over dense horizon grids).
  2. (84)--(87): exact censored-measurement cell decomposition -- the cell
     evidence sum equals a Monte Carlo estimate of the marginal observation
     density for a piecewise-affine max-measurement in two dimensions.
  3. (88)--(89) and (91): softplus-max gap bounds and the curvature bound.
  4. (94)--(96): the lagged-rule ZLB example -- both branches solve the LCP
     exactly on the multiplicity region, and neither solves it below the
     threshold pi_0 = -r/A^2.

Every check asserts a tolerance and prints one PASS line.
"""

import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(20260819)


def check_crossing_lemma():
    lam_grid = (0.65, 0.45)
    worst = 0
    s = np.linspace(1e-4, 30.0, 200_001)
    for _ in range(4000):
        lam = lam_grid[rng.integers(0, 2)]
        L, S, C = rng.normal(0.0, 0.05, size=3)
        ell = rng.normal(0.0, 0.02)
        g = (L - ell) + np.exp(-lam * s) * (S + C * lam * s)
        signs = np.sign(g)
        signs = signs[signs != 0]
        crossings = int(np.sum(signs[1:] != signs[:-1]))
        worst = max(worst, crossings)
        assert crossings <= 2, (lam, L, S, C, ell, crossings)
    print(f"PASS crossing lemma (82): max sign changes over 4000 random "
          f"NS curves = {worst} (bound 2)")


def check_cell_evidence():
    # Two-dimensional state, measurement y = sum_k w_k max(ell, a_k' x) + u.
    m = np.array([0.1, -0.2])
    P = np.array([[0.5, 0.1], [0.1, 0.3]])
    A = np.array([[1.0, 0.4], [1.0, -0.8], [0.6, 1.0]])   # three node loadings
    w = np.array([0.5, 0.3, 0.2])
    ell, Rv = 0.0, 0.05**2

    def h(x):
        return w @ np.maximum(ell, A @ x)

    y0 = 0.12
    # Exact cell sum (84)--(87): enumerate all 2^3 sign patterns.
    total = 0.0
    for pattern in range(8):
        slack = np.array([(pattern >> k) & 1 for k in range(3)], dtype=bool)
        # On the cell, h(x) = (w*slack) @ A x + ell * sum(w[~slack])
        Hrow = (w * slack) @ A
        d = ell * w[~slack].sum()
        S = Hrow @ P @ Hrow + Rv
        K = P @ Hrow / S
        m_plus = m + K * (y0 - Hrow @ m - d)
        P_plus = P - np.outer(K, K) * S
        # Cell probability under N(m_plus, P_plus) by dense MC (2-d, exact enough)
        Z = rng.multivariate_normal(m_plus, P_plus, size=400_000)
        gvals = Z @ A.T - ell
        inside = np.all((gvals > 0) == slack[None, :], axis=1).mean()
        total += norm.pdf(y0, Hrow @ m + d, np.sqrt(S)) * inside
    # Direct Monte Carlo of p(y0)
    X = rng.multivariate_normal(m, P, size=2_000_000)
    direct = norm.pdf(y0, np.array([h(x) for x in X[:200_000]]),
                      np.sqrt(Rv)).mean()  # subsample for speed
    rel = abs(total - direct) / direct
    assert rel < 2e-2, (total, direct, rel)
    print(f"PASS cell evidence (84)--(87): sum of cell terms {total:.5f} vs "
          f"MC {direct:.5f} (rel err {rel:.1%})")


def check_softplus_bounds():
    alpha, ell = 1.5e-3, 0.0
    u = np.linspace(-0.05, 0.05, 200_001)
    gap = (ell + alpha * np.logaddexp(0.0, (u - ell) / alpha)
           - np.maximum(ell, u))
    # Strict positivity holds mathematically; in float64 the gap underflows and
    # the alpha*((u-ell)/alpha) round trip can leave residuals of order 1e-17,
    # so assert nonnegativity up to that rounding and strict positivity on the
    # resolvable range.
    assert np.all(gap >= -1e-15)
    near = np.abs(u - ell) <= 20 * alpha
    assert np.all(gap[near] > 0.0)
    assert np.max(gap) <= alpha * np.log(2.0) + 1e-15
    assert abs(np.max(gap) - alpha * np.log(2.0)) < 1e-9  # attained at kink
    far = np.abs(u - ell) >= 5 * alpha
    assert np.max(gap[far]) <= alpha * np.log(1 + np.exp(-5)) + 1e-15
    # curvature bound (91)
    z = (u - ell) / alpha
    sig = 1.0 / (1.0 + np.exp(-z))
    curv = sig * (1 - sig) / alpha
    assert np.max(curv) <= 0.25 / alpha + 1e-9
    print(f"PASS softplus bounds (88)--(89),(91): max gap "
          f"{np.max(gap):.3e} = alpha*log2 = {alpha*np.log(2):.3e}; "
          f"max curvature {np.max(curv):.1f} <= 1/(4 alpha) = {0.25/alpha:.1f}")


def check_holden_example():
    r = 0.02
    for psi in (0.2, 0.5, 0.8):
        phi = 2.0
        A = 1.0 - np.sqrt(1.0 - psi)
        assert abs(A * (phi - A) - psi) < 1e-14   # identity used in (96)
        thresh = -r / A**2

        def lcp_residual(pi1, pi0):
            return max(-r - A * pi1, (phi - A) * pi1 - psi * pi0)

        for pi0 in np.linspace(thresh + 1e-6, thresh + 0.5, 200):
            f = psi * pi0 / (phi - A)          # fundamental branch
            b = -r / A                          # bound branch
            assert abs(lcp_residual(f, pi0)) < 1e-12
            assert abs(lcp_residual(b, pi0)) < 1e-12
            assert abs(f - b) > 0 or abs(pi0 - thresh) < 1e-9
        for pi0 in np.linspace(thresh - 0.5, thresh - 1e-6, 200):
            grid = np.linspace(-5.0, 5.0, 20_001)
            res = np.array([lcp_residual(p, pi0) for p in grid])
            assert np.min(np.abs(res)) > 1e-7   # no solution on the grid
    print("PASS lagged-rule example (94)--(96): both branches solve the LCP "
          "exactly above pi_0 = -r/A^2; no grid solution below it "
          "(psi in {0.2, 0.5, 0.8})")


if __name__ == "__main__":
    check_crossing_lemma()
    check_cell_evidence()
    check_softplus_bounds()
    check_holden_example()
    print("ALL CHECKS PASSED")
