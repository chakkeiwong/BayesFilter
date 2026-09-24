"""Exact-arithmetic reference checks for the written derivations; no GPU use.

This checks algebra and document links, not particle-filter scientific quality.
"""

from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
repo = root.parents[3]
report = repo / "docs/plans/younis-model-score-analysis-2026-09-11.md"

# Positive unbiased normalizer, biased derivative of its logarithm.
theta, radius = F(2), F(1, 2)
mean_score = (1 / (theta + radius) + 1 / (theta - radius)) / 2
assert mean_score == theta / (theta * theta - radius * radius)
assert mean_score - 1 / theta == F(1, 30)

# Single-Gaussian IWSG, centered IWSG, and pathwise estimators for phi(z)=2z+1.
mu, bandwidth, slope, offset = F(3), F(1, 4), F(2), F(1)
normal_moments = {0: F(1), 1: F(0), 2: F(1), 3: F(0), 4: F(3)}

def polynomial_moments(coefficients):
    mean = sum(c * normal_moments[k] for k, c in coefficients.items())
    second = sum(a * b * normal_moments[i + j]
                 for i, a in coefficients.items() for j, b in coefficients.items())
    return mean, second - mean * mean

raw = polynomial_moments({1: (slope * mu + offset) / bandwidth, 2: slope})
centered = polynomial_moments({2: slope})
pathwise = polynomial_moments({0: slope})
assert raw == (F(2), F(792))
assert centered == (F(2), F(8))
assert pathwise == (F(2), F(0))

# Complete two-state, two-observation model. The initial law AND transitions
# depend on theta. Enumerating the joint measure is independent of the
# backward-conditional recursion and catches omission of the initial score.
theta = F(3, 5)
initial = [1 - theta, theta]
initial_score = [-1 / (1 - theta), 1 / theta]
observation = [[F(1, 5), F(4, 5)], [F(7, 10), F(3, 10)]]

def transition(previous, current):
    return theta if previous == current else 1 - theta

def transition_score(previous, current):
    return 1 / theta if previous == current else -1 / (1 - theta)

normalizer = F(0)
derivative = F(0)
for x0, x1, x2 in product(range(2), repeat=3):
    weight = (initial[x0] * transition(x0, x1) * observation[0][x1]
              * transition(x1, x2) * observation[1][x2])
    score = initial_score[x0] + transition_score(x0, x1) + transition_score(x1, x2)
    normalizer += weight
    derivative += weight * score
weights, additive = initial, initial_score
for likelihood in observation:
    predictive = [sum(weights[j] * transition(j, i) for j in range(2)) for i in range(2)]
    next_additive = [sum(weights[j] * transition(j, i) / predictive[i]
                         * (additive[j] + transition_score(j, i)) for j in range(2))
                     for i in range(2)]
    raw_weights = [predictive[i] * likelihood[i] for i in range(2)]
    weights = [weight / sum(raw_weights) for weight in raw_weights]
    additive = next_additive
recursive_score = sum(weights[i] * additive[i] for i in range(2))
assert recursive_score == derivative / normalizer

missing = []
for label, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", report.read_text()):
    if target.startswith(("http://", "https://", "#")):
        continue
    if not (report.parent / target.split("#", 1)[0]).resolve().exists():
        missing.append({"label": label, "target": target})
assert not missing, missing

summary = {
    "purpose": "exact-arithmetic independent reference and local-link checks",
    "backend": "Python standard library fractions; no ML frameworks imported",
    "gpu_status": "not used or probed",
    "normalizer_example": {"expected_estimated_score": str(mean_score),
                           "true_score": "1/2", "bias": "1/30"},
    "gaussian_gradient_example": {"IWSG_mean_variance": list(map(str, raw)),
                                  "centered_mean_variance": list(map(str, centered)),
                                  "pathwise_mean_variance": list(map(str, pathwise))},
    "two_state_fisher_check": {"joint_derivative_over_normalizer": str(derivative / normalizer),
                               "recursive_score": str(recursive_score), "exactly_equal": True},
    "missing_local_links": missing,
    "nonclaims": ["No particle-filter implementation was tested.",
                  "No empirical performance or convergence evidence is created."],
}
(root / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary))
