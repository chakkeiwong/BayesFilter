"""Deterministic analytic reference: initial epsilon at Gaussian acceptance .7.

Question: for stationary N(0,I4), identity mass, standard leapfrog and the
existing q20 L grid, what is the smallest positive stable epsilon with mean
acceptance .7? No q20 target, framework, GPU or stochastic sampling is used.

Pre-run skeptical audit: the ideal Gaussian is the stated comparator, not a
verified description of the learned posterior. Smallest-root selection is an
explicit proposal convention, not an efficiency optimum. Multiple roots are
isolated between exact return points rather than assuming globally monotone
acceptance. Each earlier lobe must have a maximum below the target equation.
Stop on a missing root or failed direct-matrix arithmetic check; never change
the campaign config. The 1e-12 check is a float64 arithmetic tolerance only.

The exact acceptance derivation is recorded in the q20 mathematical audit M6.
Root isolation uses g(epsilon)=epsilon^3*|U_(L-1)(1-epsilon^2/2)|. Its positive
zeros are r_k=2*sin(k*pi/(2*L)), k=1,...,L-1. Between zeros, log(g) has derivative
3/epsilon+sum_k[1/(epsilon-r_k)+1/(epsilon+r_k)], with strictly negative
derivative. Each interior lobe therefore has exactly one maximum. The final
lobe increases to epsilon=2. Bisection stops at floating-point resolution.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def boundary(left, right, belongs_left):
    while True:
        middle = left + (right - left) / 2
        if middle == left or middle == right:
            return middle
        if belongs_left(middle):
            left = middle
        else:
            right = middle


def chebyshev_u(order, x):
    previous, current = 0.0, 1.0
    for _ in range(order):
        previous, current = current, 2 * x * current - previous
    return current


def equation_value(epsilon, length):
    return epsilon**3 * abs(chebyshev_u(length - 1, 1 - epsilon**2 / 2))


def smallest_root(length, target):
    zeros = [2 * math.sin(k * math.pi / (2 * length))
             for k in range(1, length)]
    endpoints = [0.0, *zeros, 2.0]
    excluded = []
    for lobe, (left, right) in enumerate(zip(endpoints, endpoints[1:])):
        if right == 2.0:
            peak = right
        else:
            def positive_log_slope(epsilon):
                return (3 / epsilon + sum(
                    1 / (epsilon - zero) + 1 / (epsilon + zero)
                    for zero in zeros)) > 0
            peak = boundary(left, right, positive_log_slope)
        height = equation_value(peak, length)
        if height < target:
            excluded.append({"lobe_index": lobe, "peak_epsilon": peak,
                             "peak_equation_value": height})
            continue
        root = boundary(left, peak,
                        lambda epsilon: equation_value(epsilon, length) < target)
        return root, lobe, excluded
    raise ArithmeticError(f"No stable root for L={length}")


def direct_matrix_acceptance(epsilon, length):
    a = d = 1 - epsilon**2 / 2
    b = epsilon
    c = -epsilon * (1 - epsilon**2 / 4)
    u, v, w, z = 1.0, 0.0, 0.0, 1.0
    for _ in range(length):
        u, v, w, z = a*u + b*w, a*v + b*z, c*u + d*w, c*v + d*z
    gram00, gram01, gram11 = u*u + w*w, u*v + w*z, v*v + z*z
    rho = (gram00 + gram11 + math.sqrt(
        (gram00 - gram11)**2 + 4 * gram01**2)) / 2
    x = 1 / (1 + rho)
    return 6*x*x - 4*x*x*x, u*z - v*w


def main():
    started = time.perf_counter()
    alpha = 0.7
    lengths = (3, 5, 9, 13, 18, 25)
    x = boundary(0.0, 0.5, lambda x: 6*x*x - 4*x*x*x < alpha)
    rho = 1 / x - 1
    b_target = (rho - 1)**2 / rho
    target = 4 * math.sqrt(b_target)
    rows = []
    for length in lengths:
        epsilon, lobe, excluded = smallest_root(length, target)
        acceptance, determinant = direct_matrix_acceptance(epsilon, length)
        residual = equation_value(epsilon, length) - target
        if not (0 < epsilon < 2 and abs(residual) < 1e-12
                and abs(acceptance - alpha) < 1e-12
                and abs(determinant - 1) < 1e-12):
            raise ArithmeticError(f"Gaussian formula validation failed for L={length}")
        rows.append({"L": length, "epsilon": epsilon,
                     "expected_acceptance_direct_matrix": acceptance,
                     "equation_residual": residual,
                     "matrix_determinant": determinant,
                     "root_lobe_index": lobe,
                     "excluded_earlier_lobes": excluded})
    here = Path(__file__).resolve()
    result = {
        "role": "conditional_analytic_gaussian_reference_only",
        "assumptions": ["target N(0,I4)", "stationary position starts",
                        "independent N(0,I4) momenta", "identity mass",
                        "standard leapfrog with endpoint Metropolis correction"],
        "root_selection": "smallest positive root in 0<epsilon<2",
        "root_selection_provenance": "explicit proposal convention, not optimum",
        "acceptance_target": alpha, "equation_rhs": target, "rows": rows,
        "checks": "earlier-lobe exclusion and independent direct matrix evaluation",
        "arithmetic_check_tolerance": 1e-12,
        "not_concluded": ["actual q20 acceptance", "actual q20 whitening",
                          "optimal kernel", "posterior convergence"],
        "command": f"python3 {here.relative_to(Path.cwd())}",
        "python": sys.version,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
        "framework_imports": False, "gpu_used": False,
        "random_seeds": "N/A: deterministic analytic calculation",
        "wall_seconds": time.perf_counter() - started,
    }
    with here.with_name("result.json").open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"equation_rhs": target, "rows": [
        {key: row[key] for key in ("L", "epsilon", "expected_acceptance_direct_matrix",
                                  "equation_residual", "root_lobe_index")}
        for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
