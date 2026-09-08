"""Diagnostic spot-check for Section 4.3 closed forms of the ZLB survey.

Scope: this is an explicitly diagnostic, NumPy/SciPy-based reference check in
the sense of the repository backend rule. It verifies the survey's project
derivations numerically; it is not runtime inference code, not a TensorFlow
route, and not a proof.

Checked closed forms (equation numbers of zlb_discontinuous_hmc_survey.md,
19 August 2026 numbering):

  1. (22), (27), (28): affine-boundary branch probabilities and exact
     truncated-Gaussian branch moments, against Monte Carlo conditioning.
  2. (30)--(35): censored policy-rate predictive density, its normalization,
     and the posterior binding probability, against quadrature and Monte Carlo.
  3. (47)--(48): moment-preserving mixture collapse, against direct sampling.
  4. (49)--(51): IMM mode prediction and mixing moments, against enumeration
     on a two-mode toy chain.

Every check asserts a tolerance consistent with its Monte Carlo error and
prints one PASS line, so the log is a complete record.
"""

import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

rng = np.random.default_rng(20260819)


def check_truncated_branch_moments():
    d = 3
    A = rng.normal(size=(d, d))
    P = A @ A.T + d * np.eye(d)
    m = rng.normal(size=d)
    a = rng.normal(size=d)
    c = a @ m + 0.4 * np.sqrt(a @ P @ a)  # boundary inside the bulk

    sig = np.sqrt(a @ P @ a)
    gam = (c - a @ m) / sig
    alpha_b = norm.cdf(gam)
    lam_b = norm.pdf(gam) / norm.cdf(gam)
    lam_n = norm.pdf(gam) / (1.0 - norm.cdf(gam))
    Pa = P @ a
    m_b = m - Pa / sig * lam_b
    P_b = P - np.outer(Pa, Pa) / sig**2 * lam_b * (lam_b + gam)
    m_n = m + Pa / sig * lam_n
    P_n = P - np.outer(Pa, Pa) / sig**2 * lam_n * (lam_n - gam)

    N = 4_000_000
    Z = rng.multivariate_normal(m, P, size=N)
    bind = Z @ a - c <= 0.0
    alpha_mc = bind.mean()
    Zb, Zn = Z[bind], Z[~bind]
    tol_mean, tol_cov, tol_p = 5e-3 * np.sqrt(np.trace(P)), 5e-2, 5e-4
    assert abs(alpha_mc - alpha_b) < tol_p, (alpha_mc, alpha_b)
    assert np.max(np.abs(Zb.mean(0) - m_b)) < tol_mean
    assert np.max(np.abs(Zn.mean(0) - m_n)) < tol_mean
    assert np.max(np.abs(np.cov(Zb.T) - P_b)) < tol_cov
    assert np.max(np.abs(np.cov(Zn.T) - P_n)) < tol_cov
    print(f"PASS truncated branch moments (22)/(27)/(28): "
          f"alpha_b={alpha_b:.4f} vs MC {alpha_mc:.4f}; "
          f"max mean err {max(np.max(np.abs(Zb.mean(0)-m_b)), np.max(np.abs(Zn.mean(0)-m_n))):.2e}; "
          f"max cov err {max(np.max(np.abs(np.cov(Zb.T)-P_b)), np.max(np.abs(np.cov(Zn.T)-P_n))):.2e}")


def check_censored_likelihood():
    mu, sig, ell, V = 0.6, 1.3, 0.0, 0.35**2

    def p_y(y):
        alpha_b = norm.cdf((ell - mu) / sig)
        st2 = sig**2 * V / (sig**2 + V)
        mt = (V * mu + sig**2 * y) / (sig**2 + V)
        return (alpha_b * norm.pdf(y, ell, np.sqrt(V))
                + norm.pdf(y, mu, np.sqrt(V + sig**2))
                * (1.0 - norm.cdf((ell - mt) / np.sqrt(st2))))

    # (34) must equal the direct integral over the censored latent rate.
    for y in (-0.5, 0.0, 0.3, 1.5):
        direct = (norm.cdf((ell - mu) / sig) * norm.pdf(y, ell, np.sqrt(V))
                  + quad(lambda i: norm.pdf(y, i, np.sqrt(V))
                         * norm.pdf(i, mu, sig), ell, np.inf)[0])
        assert abs(p_y(y) - direct) < 1e-10, (y, p_y(y), direct)
    total, err = quad(p_y, -np.inf, np.inf)
    assert abs(total - 1.0) < 1e-8, total

    # (35) posterior binding probability against Monte Carlo.
    N = 4_000_000
    i_star = rng.normal(mu, sig, size=N)
    i_obs = np.maximum(ell, i_star)
    y_sim = i_obs + rng.normal(0.0, np.sqrt(V), size=N)
    y0, h = 0.05, 0.01
    sel = np.abs(y_sim - y0) < h
    pb_mc = (i_star[sel] <= ell).mean()
    pb = norm.cdf((ell - mu) / sig) * norm.pdf(y0, ell, np.sqrt(V)) / p_y(y0)
    assert abs(pb - pb_mc) < 2e-2, (pb, pb_mc)
    print(f"PASS censored likelihood (30)--(35): integral of (34) = {total:.10f}; "
          f"Pr(b|y={y0}) = {pb:.4f} vs MC {pb_mc:.4f} (window +/-{h})")


def check_mixture_collapse():
    w = np.array([0.35, 0.65])
    ms = [np.array([0.0, 1.0]), np.array([2.0, -1.0])]
    Ps = [np.array([[1.0, 0.3], [0.3, 0.8]]), np.array([[0.5, -0.1], [-0.1, 1.2]])]
    mbar = sum(wi * mi for wi, mi in zip(w, ms))
    Pbar = sum(wi * (Pi + np.outer(mi - mbar, mi - mbar))
               for wi, mi, Pi in zip(w, ms, Ps))
    N = 2_000_000
    comp = rng.random(N) < w[1]
    X = np.where(comp[:, None],
                 rng.multivariate_normal(ms[1], Ps[1], size=N),
                 rng.multivariate_normal(ms[0], Ps[0], size=N))
    assert np.max(np.abs(X.mean(0) - mbar)) < 3e-3
    assert np.max(np.abs(np.cov(X.T) - Pbar)) < 1e-2
    print(f"PASS mixture collapse (47)--(48): mean err {np.max(np.abs(X.mean(0)-mbar)):.2e}, "
          f"cov err {np.max(np.abs(np.cov(X.T)-Pbar)):.2e}")


def check_imm_mixing():
    Pi = np.array([[0.9, 0.1], [0.3, 0.7]])
    mu_prev = np.array([0.4, 0.6])
    ms = [np.array([0.0]), np.array([3.0])]
    Ps = [np.array([[1.0]]), np.array([[2.0]])]
    c = Pi.T @ mu_prev                     # (49) mode prediction
    mix = Pi * mu_prev[:, None] / c[None, :]  # (49) mixing weights mu_{s|r}
    assert np.allclose(mix.sum(0), 1.0)
    for r in range(2):
        m_mix = sum(mix[s, r] * ms[s] for s in range(2))     # (50)
        P_mix = sum(mix[s, r] * (Ps[s] + np.outer(ms[s] - m_mix, ms[s] - m_mix))
                    for s in range(2))                        # (51)
        # enumeration: same object computed from the joint (s, r) table
        joint = np.array([Pi[s, r] * mu_prev[s] for s in range(2)])
        m_enum = (joint / joint.sum()) @ np.array([ms[0][0], ms[1][0]])
        second = (joint / joint.sum()) @ np.array(
            [Ps[0][0, 0] + ms[0][0]**2, Ps[1][0, 0] + ms[1][0]**2])
        P_enum = second - m_enum**2
        assert abs(m_mix[0] - m_enum) < 1e-12
        assert abs(P_mix[0, 0] - P_enum) < 1e-12
    assert abs(c.sum() - 1.0) < 1e-12
    print(f"PASS IMM mixing (49)--(51): c = {np.round(c, 6)}, "
          f"mixing columns normalize, moments match enumeration to 1e-12")


if __name__ == "__main__":
    check_truncated_branch_moments()
    check_censored_likelihood()
    check_mixture_collapse()
    check_imm_mixing()
    print("ALL CHECKS PASSED")
