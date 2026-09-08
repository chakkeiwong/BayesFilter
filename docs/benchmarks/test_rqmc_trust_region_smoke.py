#!/usr/bin/env python3
"""RQMC-on-corrected-trust-region wiring smoke check (CPU, seconds).

Question: can the RQMC initial-cloud entry point reach main's corrected
dual-cap trust-region route, and does the route actually engage?

This is a wiring and finiteness check only. It is NOT evidence about
initialization quality, variance reduction, or any RQMC-vs-IID comparison;
those need the tuned campaign with replication.

CPU-only by construction: GPU devices are intentionally hidden below.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.highdim.cubature_genut_candidate import cubature_design
from bayesfilter.highdim.ledh_pfpf_genut_initial_rqmc_tf import (
    finite_value_standard_score_initial_rqmc,
)
from bayesfilter.highdim.ledh_pfpf_genut_model_callbacks_tf import (
    diagonal_lgssm_callbacks,
)
from bayesfilter.highdim.sqmc_tf import randomized_halton_gaussian

COUNT = 24
HORIZON = 2
DIM = 3
PROCESS_STEPS = HORIZON - 1  # transition_before_first_observation = False

callbacks = diagonal_lgssm_callbacks()
theta = tf.constant([0.9, 0.85, 0.8, 0.15, 0.2], tf.float32)
observations = tf.random.stateless_normal([HORIZON, DIM], [2026, 831])

# The RQMC lever under test: the initial cloud is a randomized Halton
# point set rather than an IID normal draw. Process noise is left IID so
# this check isolates the initial-cloud path.
initial_noise = randomized_halton_gaussian(
    num_particles=COUNT, dimension=DIM, seed=2026, salt=831
)
process_noise = tf.random.stateless_normal([PROCESS_STEPS, COUNT, DIM], [2026, 832])
design = cubature_design(dim=DIM, num_particles=COUNT)

shared = dict(
    ancestry_policy="existing_one_to_one",
    process_ancestor_uniforms=tf.zeros([PROCESS_STEPS, COUNT], tf.float32),
    reset_policy="contract_e",
    dual_cap_enabled=True,
    dual_cap_diagonal_steps=4,
    dual_cap_diagonal_strength=0.2,
    dual_cap_pairwise_steps=4,
    dual_cap_pairwise_strength=0.02,
    dual_cap_pairwise_particle_rms_cap=2.0,
    dual_cap_coordinate_cap=0.98,
    dual_cap_coordinate_cap_power=8,
    epsilon=2.0,
    sinkhorn_steps=8,
    balance_steps=8,
    ridge=1.0e-5,
)
inputs = (callbacks, theta, observations, initial_noise, process_noise, design)

# reset_route_id: 2 = trust-region, 1 = dual-cap only, 0 = plain reset.
without = finite_value_standard_score_initial_rqmc(
    *inputs, trust_region_enabled=False, **shared
)
with_tr = finite_value_standard_score_initial_rqmc(
    *inputs,
    trust_region_enabled=True,
    trust_region_lm_damping=1.0e-2,
    trust_region_lm_scale_floor=1.0e-4,
    trust_region_radius=0.5,
    **shared,
)

for label, (value, score, diagnostics), route, solver in (
    ("dual-cap only ", without, 1, 0),
    ("trust-region  ", with_tr, 2, 1),
):
    observed_route = int(diagnostics["reset_route_id"].numpy())
    observed_solver = int(diagnostics["trust_region_solver_id"].numpy())
    valid = bool(diagnostics["program_valid"].numpy())
    print(
        f"{label}: value={float(value):+.6f} "
        f"score[0]={float(score[0]):+.6f} "
        f"route_id={observed_route} solver_id={observed_solver} "
        f"program_valid={valid}"
    )
    assert valid, f"{label.strip()}: program_valid is False"
    assert observed_route == route, (
        f"{label.strip()}: expected reset_route_id={route}, got {observed_route}"
    )
    assert observed_solver == solver, (
        f"{label.strip()}: expected trust_region_solver_id={solver}, "
        f"got {observed_solver}"
    )
    assert bool(tf.math.is_finite(value).numpy()), f"{label.strip()}: value not finite"
    assert bool(tf.reduce_all(tf.math.is_finite(score)).numpy()), (
        f"{label.strip()}: score not finite"
    )

value_gap = float(tf.abs(with_tr[0] - without[0]).numpy())
print(f"\n|value(trust-region) - value(dual-cap only)| = {value_gap:.3e}")
assert value_gap > 0.0, (
    "trust-region produced a bitwise-identical value: the mechanism did not "
    "engage, so the route flag is not reaching the correction"
)
print(
    "PASS: RQMC initial-cloud path reaches the corrected trust-region route "
    "(route_id 1 -> 2, solver 0 -> 1), both arms finite and program-valid, "
    "and the mechanism changes the computed value."
)
print(
    "\nNot established here: initialization quality, variance reduction, or "
    "any RQMC-vs-IID ranking. Wiring and finiteness only."
)
