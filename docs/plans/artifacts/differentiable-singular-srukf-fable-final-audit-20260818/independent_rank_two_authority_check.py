"""Independent Fable audit check: rank-two NON-diagonal singular authority.

Harder than the repository test: the observation support has rank 2 inside a
3-dimensional ambient observation space (their rank-two test uses a full-rank
2-dim observation space). Dense Moore-Penrose authority + centered FD.
Diagnostic-only audit artifact; not repository runtime code.
"""
import numpy as np
import tensorflow as tf

from bayesfilter.linear.rectangular_factor_tf import (
    batched_fixed_support_qr_update,
    batched_fixed_support_qr_likelihood,
)

np.set_printoptions(precision=10)
failures = []

def check(name, err, tol):
    ok = err <= tol
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {err:.3e} (tol {tol:.1e})")
    if not ok:
        failures.append(name)

# ---- fixture: ny=3 ambient, observation stack rank 2, non-diagonal mixing
M = np.array([[1.0, 0.4], [-0.3, 1.2], [0.5, -0.7]])   # 3x2 mixing (col space = support)
C = np.array([[0.9, -0.2, 0.4, 0.1], [0.3, 0.8, -0.5, 0.6]])  # 2x4 coefficients
Y0 = M @ C                                              # 3x4, rank 2
X0 = np.array([[1.0, -0.2, 0.3, 0.7],
               [0.1, 0.9, -0.4, 0.5],
               [0.6, 0.2, 0.8, -0.1]])                  # 3x4 state stack
alpha0 = np.array([0.3, -0.5])
e0 = M @ alpha0                                          # innovation ON support

DY = 0.1 * (np.array([[0.2, -0.1], [0.4, 0.3], [-0.2, 0.1]]) @ C
            + M @ np.array([[0.1, 0.3, -0.2, 0.0], [0.2, -0.1, 0.4, 0.1]]))
DX = 0.1 * np.array([[0.3, 0.1, -0.2, 0.4],
                     [-0.1, 0.2, 0.3, -0.3],
                     [0.2, -0.4, 0.1, 0.2]])
# innovation derivative stays inside the moving support: d(M(t) a(t)) with
# M(t) = M + t*DM, a(t) = alpha + t*da  ->  de = DM@alpha + M@da
DM = 0.1 * np.array([[0.2, -0.1], [0.4, 0.3], [-0.2, 0.1]])
da = np.array([0.1, 0.2])
DE = DM @ alpha0 + M @ da

def path(t):
    # keep Y(t) rank-2 with a moving support: Y(t) = (M + t*DM) @ (C + t*DC)
    DC = np.array([[0.1, 0.3, -0.2, 0.0], [0.2, -0.1, 0.4, 0.1]])
    Yt = (M + t * DM) @ (C + t * DC)
    Xt = X0 + t * DX
    et = (M + t * DM) @ (alpha0 + t * da)
    return Yt, Xt, et

# analytic DY consistent with the path: d/dt[(M+tDM)(C+tDC)]|0 = DM@C + M@DC
DC = np.array([[0.1, 0.3, -0.2, 0.0], [0.2, -0.1, 0.4, 0.1]])
DY = DM @ C + M @ DC

obs_perm = tf.constant([0, 1, 2], tf.int32)     # rows 0,1 of Y span... check pivots
cond_perm = tf.constant([0, 1, 2], tf.int32)
OBS_RANK, COND_RANK = 2, 3                       # state residual X(I-VV') in 3-dim: rank 2 or 3?
# X(I-VV') has rank min(3, 4-2)=2 in exact arithmetic? I-VV' has rank K-r = 2, so X(I-VV') rank <= 2.
COND_RANK = 2
# choose conditional permutation by preflight: any rows of X(I-VV') with full rank 2 chart

def run(t, with_deriv):
    Yt, Xt, et = path(t)
    args = [
        tf.constant(Yt[None], tf.float64),
        tf.constant(Xt[None], tf.float64),
        tf.constant(et[None], tf.float64),
        obs_perm, OBS_RANK, cond_perm, COND_RANK,
    ]
    if with_deriv:
        args += [
            tf.constant(DY[None, None], tf.float64),
            tf.constant(DX[None, None], tf.float64),
            tf.constant(DE[None, None], tf.float64),
        ]
    return batched_fixed_support_qr_update(*args)

res = run(0.0, True)
lik, score, inc, d_inc, Gf, dGf, diag = res
assert bool(diag["chart_valid"][0]), "observation chart invalid"
assert bool(diag["conditional_chart_valid"][0]), "conditional chart invalid"
assert bool(diag["on_support"][0]), "innovation not on support"

# ---- dense Moore-Penrose authority
S = Y0 @ Y0.T                       # rank-2 3x3
Pxy = X0 @ Y0.T
Sp = np.linalg.pinv(S, rcond=1e-12)
K_auth = Pxy @ Sp
Pf_auth = X0 @ X0.T - K_auth @ S @ K_auth.T
check("gain vs dense pseudo-inverse authority", np.abs(diag["gain"].numpy()[0] - K_auth).max(), 1e-11)
check("mean increment vs K e", np.abs(inc.numpy()[0] - K_auth @ e0).max(), 1e-11)
check("posterior factor Gf Gf' vs dense P_f", np.abs((Gf.numpy()[0] @ Gf.numpy()[0].T) - Pf_auth).max(), 1e-11)

# ---- support likelihood vs independent eigen authority
lam, V = np.linalg.eigh(S)
pos = lam > 1e-12 * lam.max()
r = int(pos.sum()); assert r == 2
logdet_plus = np.log(lam[pos]).sum()
quad = e0 @ (V[:, pos] @ np.diag(1.0 / lam[pos]) @ V[:, pos].T) @ e0
lik_auth = -0.5 * (r * np.log(2 * np.pi) + logdet_plus + quad)
check("support likelihood vs eigen authority", abs(lik.numpy()[0] - lik_auth), 1e-11)

# ---- renormalized epsilon limit (independent)
for eps_reg in (1e-4, 1e-6, 1e-8):
    Pe = S + eps_reg * np.eye(3)
    amb = -0.5 * (3 * np.log(2 * np.pi) + np.linalg.slogdet(Pe)[1] + e0 @ np.linalg.solve(Pe, e0))
    ren = amb + 0.5 * (3 - r) * np.log(2 * np.pi * eps_reg)
    check(f"renormalized ambient limit (eps={eps_reg:g})", abs(ren - lik_auth), 30 * eps_reg)

# ---- centered finite differences for likelihood, increment, factor
for h in (1e-6, 5e-7):
    p = run(+h, False); m = run(-h, False)
    fd_lik = (p[0].numpy() - m[0].numpy()) / (2 * h)
    fd_inc = (p[2].numpy() - m[2].numpy()) / (2 * h)
    fd_G = (p[4].numpy() - m[4].numpy()) / (2 * h)
    check(f"score vs FD (h={h:g})", abs(score.numpy()[0, 0] - fd_lik[0]), 5e-7)
    check(f"d_increment vs FD (h={h:g})", np.abs(d_inc.numpy()[0, 0] - fd_inc[0]).max(), 5e-7)
    check(f"d_factor vs FD (h={h:g})", np.abs(dGf.numpy()[0, 0] - fd_G[0]).max(), 5e-7)

# ---- off-support rejection on this singular geometry
e_off = e0 + 1e-3 * np.linalg.svd(M, full_matrices=True)[0][:, 2]  # add null-direction
v_off, s_off, d_off = batched_fixed_support_qr_likelihood(
    tf.constant(e_off[None], tf.float64),
    tf.constant((M @ np.linalg.cholesky(np.eye(2)))[None] @ tf.eye(2, dtype=tf.float64).numpy(), tf.float64) * 0
    + tf.constant((Y0 @ np.linalg.pinv(C) @ np.eye(2))[None], tf.float64),  # a rank-2 factor of S? use G from update instead
)
print("off-support value:", v_off.numpy()[0], "on_support:", bool(d_off["on_support"][0]))
assert np.isneginf(v_off.numpy()[0]) and not bool(d_off["on_support"][0])

print(f"\nSUMMARY: {'ALL PASS' if not failures else 'FAILURES: ' + str(failures)}")
raise SystemExit(1 if failures else 0)
