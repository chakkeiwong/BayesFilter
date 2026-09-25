"""Independent exact mathematical reference; no production computation or GPU."""
import json
from pathlib import Path

import sympy as sp

t, y1, y2 = sp.symbols("t y1 y2", real=True)
a = sp.Matrix([1, t])
y = sp.Matrix([y1, y2])
sigma = sp.eye(2) + a * a.T
log_likelihood = -sp.log(sigma.det()) / 2 - (y.T * sigma.inv() * y)[0] / 2
score_from_covariance = sp.diff(log_likelihood, t)
posterior_precision = 1 + (a.T * a)[0]
posterior_mean = (a.T * y)[0] / posterior_precision
posterior_variance = 1 / posterior_precision
score_from_disturbances = y2 * posterior_mean - t * (
    posterior_variance + posterior_mean**2
)
assert sp.simplify(score_from_covariance - score_from_disturbances) == 0
assert sp.simplify(sigma.det() - (2 + t**2)) == 0

# This derivative calculation is independent of MathDevMCP's text parser.
# Its result coincides with the displayed moving-line score for every real t.
displayed_score = (
    -t / (2 + t**2)
    + y2 * (y1 + t * y2) / (2 + t**2)
    - t * (y1 + t * y2)**2 / (2 + t**2)**2
)
assert sp.simplify(score_from_covariance - displayed_score) == 0

beta, cov, va = sp.symbols("beta cov va", real=True)
vb = sp.symbols("vb", positive=True)
variance = va - 2 * beta * cov + beta**2 * vb
optimum = sp.solve(sp.diff(variance, beta), beta)
assert optimum == [cov / vb]
assert sp.diff(variance, beta, 2) == 2 * vb
assert sp.simplify(variance.subs(beta, optimum[0]) - (va - cov**2 / vb)) == 0
assert sp.Rational(1, 2) * (2 + sp.Rational(2, 3)) == sp.Rational(4, 3)

report = {
    "status": "PASS",
    "role": "independent exact symbolic reference, not an implementation test",
    "backend": f"SymPy {sp.__version__}",
    "hardware": "CPU only; no TensorFlow/JAX/PyTorch/CUDA imported",
    "checks": {
        "gaussian_covariance_score_equals_conditional_disturbance_score": True,
        "gaussian_covariance_score_equals_manuscript_formula": True,
        "positive_observation_covariance_determinant": str(sigma.det()),
        "control_coefficient_is_unique_quadratic_minimum_when_vb_positive": True,
        "ratio_bias_exact_counterexample": "4/3 != 1",
    },
    "limits": [
        "No automatic proof of dominated differentiation hypotheses.",
        "No finite-particle variance, DSGE performance, or HMC validation.",
    ],
}
Path(__file__).with_name("symbolic-verification.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
