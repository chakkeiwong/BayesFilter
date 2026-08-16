"""Focused re-audit diagnostic checks (FS-1..FS-5 numerics).

Diagnostic-only NumPy script under the audit artifact root, per handoff
section 3. It is independent reference evidence and is not repository
runtime code. dtype float64 throughout. Deterministic fixtures (no RNG).
"""
import numpy as np

np.set_printoptions(precision=12)
EPS = np.finfo(np.float64).eps
report = []

def rec(name, value, tol):
    ok = value <= tol
    report.append((name, value, tol, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {value:.3e} (tol {tol:.1e})")
    return ok

def solve_lower(L, B):
    return np.linalg.solve(L, B)  # L lower, exact small-scale

def solve_upper(U, B):
    return np.linalg.solve(U, B)

# ---------------------------------------------------------------- check 1
print("== Check 1: FS-1 gain solve orientation (value + directional derivative)")
Sy0 = np.array([[2.0, 0.0], [1.0, 1.0]])
Pxy0 = np.array([[1.0, 2.0], [3.0, 4.0]])
DSy = np.array([[0.3, 0.0], [-0.2, 0.4]])   # lower-triangular direction
DPxy = np.array([[0.1, -0.5], [0.2, 0.7]])

def gain_path(theta):
    Sy = Sy0 + theta * DSy
    Pxy = Pxy0 + theta * DPxy
    Ut = solve_lower(Sy, Pxy.T)          # U' = Sy^{-1} Pxy'   (revised 3.5)
    U = Ut.T
    Kt = solve_upper(Sy.T, Ut)           # K' = Sy^{-T} U'
    K = Kt.T
    return Sy, Pxy, U, K

Sy, Pxy, U, K = gain_path(0.0)
Py = Sy @ Sy.T
K_oracle = Pxy @ np.linalg.inv(Py)
rec("revised K vs dense P_xy P_y^{-1} oracle", np.abs(K - K_oracle).max(), 1e-13)

# old (pre-revision) wrong orientation, for the record
Syi = np.linalg.inv(Sy)
K_old = Pxy @ Syi @ Syi
print(f"      old-plan K error vs oracle (must be large): {np.abs(K_old - K_oracle).max():.3e}")
assert np.abs(K_old - K_oracle).max() > 1.0  # counterexample still refutes old form

# analytic derivatives per revised 3.5
dUt = solve_lower(Sy, DPxy.T - DSy @ U.T)   # dU' = Sy^{-1}(dPxy' - dSy U')
dU = dUt.T
dK = (dU - K @ DSy) @ np.linalg.inv(Sy)     # dK = (dU - K dSy) Sy^{-1}

for h in (1e-6, 5e-7):
    _, _, Up, Kp = gain_path(+h)
    _, _, Um, Km = gain_path(-h)
    rec(f"dU analytic vs centered FD (h={h:g})", np.abs((Up - Um) / (2 * h) - dU).max(), 1e-8)
    rec(f"dK analytic vs centered FD (h={h:g})", np.abs((Kp - Km) / (2 * h) - dK).max(), 1e-8)

# derivative-solve residual: Sy dU' + dSy U' = dPxy'
rec("derivative solve residual Sy dU' + dSy U' - dPxy'",
    np.abs(Sy @ dUt + DSy @ U.T - DPxy.T).max(), 1e-13)

# ---------------------------------------------------------------- check 2
print("== Check 2: FS-2 rank-one downdate recurrence (value forms + derivative)")
L0 = np.array([[2.0, 0.0], [1.0, 1.5]])
x0 = np.array([0.5, 0.3])
DL = np.array([[0.2, 0.0], [-0.1, 0.3]])
Dx = np.array([0.05, -0.04])

def downdate_value(L, x, form):
    L = L.copy(); x = x.copy(); n = L.shape[0]
    for k in range(n):
        m = L[k, k] ** 2 - x[k] ** 2
        assert m > 0.0, "margin must be positive"
        r = np.sqrt(m); c = r / L[k, k]; s = x[k] / L[k, k]
        L[k, k] = r
        a_old = L[k + 1:, k].copy(); u_old = x[k + 1:].copy()
        a_new = (a_old - s * u_old) / c
        if form == "normative":      # u_new = c u_old - s a_new  (revised plan)
            u_new = c * u_old - s * a_new
        elif form == "all_old":      # documented equivalent
            u_new = (u_old - s * a_old) / c
        elif form == "old_wrong":    # pre-revision text (refuted)
            u_new = c * u_old - s * a_old
        L[k + 1:, k] = a_new; x[k + 1:] = u_new
    return L

def downdate_with_derivative(L, x, dL, dx):
    """Revised-plan scalar-pivot downdate with first derivatives."""
    L = L.copy(); x = x.copy(); dL = dL.copy(); dx = dx.copy()
    n = L.shape[0]
    for k in range(n):
        Lkk, xk, dLkk, dxk = L[k, k], x[k], dL[k, k], dx[k]
        r = np.sqrt(Lkk ** 2 - xk ** 2)
        c = r / Lkk; s = xk / Lkk
        dr = (Lkk * dLkk - xk * dxk) / r                      # old pivot values
        dc = (dr * Lkk - r * dLkk) / Lkk ** 2
        ds = (dxk * Lkk - xk * dLkk) / Lkk ** 2
        L[k, k] = r; dL[k, k] = dr
        a = L[k + 1:, k].copy(); u = x[k + 1:].copy()
        da = dL[k + 1:, k].copy(); du = dx[k + 1:].copy()
        a_new = (a - s * u) / c
        da_new = ((da - ds * u - s * du) * c - (a - s * u) * dc) / c ** 2   # first
        u_new = c * u - s * a_new
        du_new = dc * u + c * du - ds * a_new - s * da_new                  # then
        L[k + 1:, k] = a_new; x[k + 1:] = u_new
        dL[k + 1:, k] = da_new; dx[k + 1:] = du_new
    return L, dL

Pf = L0 @ L0.T - np.outer(x0, x0)
for form in ("normative", "all_old"):
    Ld = downdate_value(L0, x0, form)
    rec(f"reconstruction L_new L_new' = LL' - xx' ({form})",
        np.abs(Ld @ Ld.T - Pf).max(), 1e-14)
Ld_n = downdate_value(L0, x0, "normative")
Ld_a = downdate_value(L0, x0, "all_old")
rec("normative vs all-old form agreement", np.abs(Ld_n - Ld_a).max(), 1e-14)
Ld_w = downdate_value(L0, x0, "old_wrong")
print(f"      pre-revision rule reconstruction error (must stay refuted): "
      f"{np.abs(Ld_w @ Ld_w.T - Pf).max():.3e}")
assert np.abs(Ld_w @ Ld_w.T - Pf).max() > 1e-4

Sf, dSf = downdate_with_derivative(L0, x0, DL, Dx)
for h in (1e-6, 5e-7):
    Lp = downdate_value(L0 + h * DL, x0 + h * Dx, "normative")
    Lm = downdate_value(L0 - h * DL, x0 - h * Dx, "normative")
    rec(f"dS_f analytic vs centered FD (h={h:g})", np.abs((Lp - Lm) / (2 * h) - dSf).max(), 1e-8)
dPf = DL @ L0.T + L0 @ DL.T - np.outer(Dx, x0) - np.outer(x0, Dx)
rec("dS_f S_f' + S_f dS_f' = dP_f", np.abs(dSf @ Sf.T + Sf @ dSf.T - dPf).max(), 1e-13)

# ---------------------------------------------------------------- check 3
print("== Check 3: FS-5 zero-extended block comparator identity + derivative")
nx, ny, N = 2, 2, 5                       # N sigma columns, ny noise columns
Ax0 = np.array([[0.9, -0.4, 0.3, 0.1, -0.2],
                [0.2, 0.6, -0.5, 0.4, 0.1]])
Ay_res0 = np.array([[0.5, -0.3, 0.2, -0.1, 0.4],
                    [-0.2, 0.4, 0.3, 0.2, -0.3]])
Sr0 = np.array([[0.8, 0.0], [0.3, 0.6]])
DAx = 0.1 * np.array([[0.2, -0.1, 0.4, 0.0, 0.3],
                      [-0.3, 0.2, 0.1, -0.2, 0.0]])
DAy_res = 0.1 * np.array([[0.1, 0.3, -0.2, 0.4, -0.1],
                          [0.2, -0.4, 0.0, 0.1, 0.3]])
DSr = 0.1 * np.array([[0.5, 0.0], [-0.2, 0.4]])

def comparator_path(theta):
    Ax = Ax0 + theta * DAx
    Ay = np.hstack([Ay_res0 + theta * DAy_res, Sr0 + theta * DSr])
    Axp = np.hstack([Ax, np.zeros((nx, ny))])
    Pm = Axp @ Axp.T; Pxy = Axp @ Ay.T; Py = Ay @ Ay.T
    K = Pxy @ np.linalg.inv(Py)
    Af = Axp - K @ Ay
    return Ax, Ay, Axp, Pm, Pxy, Py, K, Af

Ax, Ay, Axp, Pm, Pxy, Py, K, Af = comparator_path(0.0)
Pf_blk = Pm - K @ Py @ K.T
rec("comparator identity A~_f A~_f' = P^- - K P_y K'",
    np.abs(Af @ Af.T - Pf_blk).max(), 1e-13)

dAxp = np.hstack([DAx, np.zeros((nx, ny))])
dAy = np.hstack([DAy_res, DSr])
dPxy = dAxp @ Ay.T + Axp @ dAy.T
dPy = dAy @ Ay.T + Ay @ dAy.T
dK = (dPxy - K @ dPy) @ np.linalg.inv(Py)
dAf = dAxp - dK @ Ay - K @ dAy            # revised-plan comparator derivative
for h in (1e-6, 5e-7):
    *_, Afp = comparator_path(+h)
    *_, Afm = comparator_path(-h)
    rec(f"dA~_f analytic vs centered FD (h={h:g})", np.abs((Afp - Afm) / (2 * h) - dAf).max(), 1e-8)
dPf_ind = (dAxp @ Axp.T + Axp @ dAxp.T) - dK @ Py @ K.T - K @ dPy @ K.T - K @ Py @ dK.T
rec("d(A~_f A~_f') = dP_f (independent assembly)",
    np.abs(dAf @ Af.T + Af @ dAf.T - dPf_ind).max(), 1e-13)

# ---------------------------------------------------------------- check 4
print("== Check 4: FS-4 partial-covariance feasibility after each column")
Sm = np.array([[2.0, 0.0, 0.0], [0.5, 1.8, 0.0], [-0.3, 0.4, 1.5]])
V = np.array([[0.6, -0.2], [0.3, 0.5], [-0.4, 0.3]])
Pf3 = Sm @ Sm.T - V @ V.T
lam = np.linalg.eigvalsh(Pf3)
print(f"      fixture lambda_min(P_f) = {lam.min():.6f} (SPD target)")
assert lam.min() > 0

def downdate_columns(L, V, order):
    L = L.copy(); mins = []
    partials = []
    for j in order:
        x = V[:, j].copy(); n = L.shape[0]
        for k in range(n):
            m = L[k, k] ** 2 - x[k] ** 2
            mins.append(m)
            assert m > 0
            r = np.sqrt(m); c = r / L[k, k]; s = x[k] / L[k, k]
            L[k, k] = r
            a = L[k + 1:, k].copy(); u = x[k + 1:].copy()
            a_new = (a - s * u) / c
            L[k + 1:, k] = a_new
            x[k + 1:] = c * u - s * a_new
        partials.append(L.copy())
    return L, partials, min(mins)

for order in ([0, 1], [1, 0]):
    L_final, partials, min_margin = downdate_columns(Sm, V, order)
    errs = []
    for step, Lk in enumerate(partials, start=1):
        remaining = [j for j in order[step:]]
        target = Pf3 + sum(np.outer(V[:, j], V[:, j]) for j in remaining) if remaining else Pf3
        errs.append(np.abs(Lk @ Lk.T - target).max())
    rec(f"P^(k) = P_f + sum_(j>k) v_j v_j' along order {order}", max(errs), 1e-13)
    print(f"      order {order}: min scalar margin {min_margin:.6f} > 0 "
          f"(exact-SPD feasibility, both orders)")

n_fail = sum(1 for *_, ok in report if not ok)
print(f"\nSUMMARY: {len(report) - n_fail}/{len(report)} checks passed, {n_fail} failed")
raise SystemExit(1 if n_fail else 0)
