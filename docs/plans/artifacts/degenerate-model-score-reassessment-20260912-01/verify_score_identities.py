"""Independent exact-arithmetic diagnostics; no algorithmic runtime or ranking."""

from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path


root = Path(__file__).resolve().parent


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def mv(a, x):
    return [dot(row, x) for row in a]


# Moving-support model X=(U,theta U), U~N(0,1), Y=X+N(0,I).
# The conditional expectation of the innovation path score equals the exact
# Gaussian likelihood score, including nonzero theta.
moving_support = []
for theta in [F(0), F(1, 3), F(-2, 3)]:
    y = [F(7, 10), F(-2, 5)]
    c = [[F(2), theta], [theta, 1 + theta**2]]
    dc = [[F(0), F(1)], [F(1), 2 * theta]]
    determinant = c[0][0] * c[1][1] - theta**2
    inv = [[c[1][1] / determinant, -theta / determinant],
           [-theta / determinant, c[0][0] / determinant]]
    v = mv(inv, y)
    tr = sum(inv[i][j] * dc[j][i] for i in range(2) for j in range(2))
    gaussian_score = (dot(v, mv(dc, v)) - tr) / 2
    post_var = 1 / (2 + theta**2)
    post_mean = (y[0] + theta * y[1]) * post_var
    innovation_score = y[1] * post_mean - theta * (post_var + post_mean**2)
    assert innovation_score == gaussian_score
    if theta == 0:
        assert innovation_score == y[0] * y[1] / 2
    moving_support.append({"theta": str(theta), "score": str(gaussian_score),
                           "innovation_identity": "exact"})

# Initial-law sensitivity in stationary AR(1), with a single observation.
# Freezing the initial variance loses 25% of the variance derivative at a=1/2.
a, sigma2, r, y = F(1, 2), F(1), F(1), F(2)
initial_var = sigma2 / (1 - a**2)
true_dvar = 2 * a * sigma2 / (1 - a**2)**2
frozen_initial_dvar = 2 * a * initial_var
score_factor = (y**2 / (initial_var + r)**2 - 1 / (initial_var + r)) / 2
assert frozen_initial_dvar / true_dvar == F(3, 4)
initial_law = {"a": str(a), "correct_score": str(true_dvar * score_factor),
               "frozen_initial_score": str(frozen_initial_dvar * score_factor),
               "fraction_retained": "3/4"}

# Exactly enumerable two-time observation model, parameter-independent ±1
# innovations. A deterministic lag retains u0 in each state.
# g1=P(Y1=1|u0)=1/2+theta*u0/4;
# g2=P(Y2=1|u0,u1)=1/2+(theta*u0+u1)/8.
# N=2, multinomial resampling after the first observation. Enumerate every
# innovation and ancestor choice; there is no random numerical error.
theta = F(1, 2)
g1 = lambda u: F(1, 2) + theta * u / 4
d1 = lambda u: F(u, 4)
g2 = lambda u, v: F(1, 2) + (theta * u + v) / 8
d2 = lambda u, v: F(u, 8)
z = sum(g1(u) * g2(u, v) for u, v in product([-1, 1], repeat=2)) / 4
dz = sum(d1(u) * g2(u, v) + g1(u) * d2(u, v)
         for u, v in product([-1, 1], repeat=2)) / 4
expect = {key: F(0) for key in
          ["mass", "z", "d_genealogical", "d_resampling_lr",
           "score_genealogical", "score_frozen_ancestry"]}
rows = []
for incoming in product([-1, 1], repeat=2):
    w1 = [g1(u) for u in incoming]
    c1 = sum(w1) / 2
    dc1 = sum(d1(u) for u in incoming) / 2
    for ancestors in product([0, 1], repeat=2):
        prob_a = product_prob = w1[ancestors[0]] * w1[ancestors[1]] / (2 * c1)**2
        for innovations in product([-1, 1], repeat=2):
            prob = prob_a / 16
            parents = [incoming[a] for a in ancestors]
            w2 = [g2(u, v) for u, v in zip(parents, innovations)]
            c2 = sum(w2) / 2
            dc2 = sum(d2(u, v) for u, v in zip(parents, innovations)) / 2
            weights = [w / (2 * c2) for w in w2]
            h = [d1(u) / g1(u) + d2(u, v) / g2(u, v)
                 for u, v in zip(parents, innovations)]
            score = dot(weights, h)
            frozen_score = dc1 / c1 + dc2 / c2
            ancestry_score = sum(d1(u) / g1(u) - dc1 / c1 for u in parents)
            zhat = c1 * c2
            values = {
                "mass": F(1), "z": zhat, "d_genealogical": zhat * score,
                "d_resampling_lr": zhat * (frozen_score + ancestry_score),
                "score_genealogical": score, "score_frozen_ancestry": frozen_score,
            }
            for key, value in values.items():
                expect[key] += prob * value
            rows.append({"incoming": incoming, "ancestors": ancestors,
                         "innovations": innovations, "probability": str(prob),
                         **{key: str(value) for key, value in values.items()}})
assert expect["mass"] == 1
assert expect["z"] == z
assert expect["d_genealogical"] == dz
assert expect["d_resampling_lr"] == dz
assert expect["d_genealogical"] / expect["z"] == dz / z
assert expect["score_genealogical"] != dz / z  # finite-N ratio bias
assert expect["score_frozen_ancestry"] != dz / z

result = {
    "status": "PASS",
    "role": "exact mathematical references; not a stochastic method comparison",
    "backend": "Python standard library Fraction; no numerical framework or GPU",
    "moving_support": moving_support,
    "initial_law": initial_law,
    "enumerated_particle_filter": {
        "N": 2, "T": 2, "outcomes": len(rows), "theta": str(theta),
        "true_z": str(z), "true_dz": str(dz), "true_score": str(dz / z),
        "expectations": {key: str(value) for key, value in expect.items()},
        "ratio_of_expectations_is_exact_score": True,
        "single_run_normalized_score_is_biased": True,
        "no_claim": "No long-horizon stability, DSGE performance, or production implementation verified.",
    },
}
(root / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
(root / "enumerated-particle-filter.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(result, indent=2))
