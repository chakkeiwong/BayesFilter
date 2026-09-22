"""P7 rebind gates: canonical NeuTra target on the frozen Austria scope.

The claim-critical binding: `make_canonical_neutra_target("austria_sir")`
replaces the bootstrap-lane factory for NeuTra use. Gates: finite batched
value/score on the frozen data, fresh target signature (comparability
sever is INTENDED), fused-lane theta-row sensitivity (the per-point theta
bridge actually routes), and G-1 registration.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_neutra_targets_tf import (
    make_canonical_neutra_target,
)

DTYPE = tf.float64


def test_austria_canonical_neutra_target_value_score_finite():
    target = make_canonical_neutra_target(
        "austria_sir", particle_count=126, substeps=8
    )
    theta = tf.constant([[0.0, 0.0, 0.0], [0.1, -0.1, 0.0]], DTYPE)
    directions = tf.constant(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], DTYPE
    )
    value, score, diagnostics = target.batch_value_score(theta, directions)
    assert value.shape == (2,)
    assert bool(tf.reduce_all(diagnostics["program_valid"]).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(value)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(score)).numpy())
    # theta-row sensitivity: distinct rows -> distinct values
    assert abs(float(value[0].numpy()) - float(value[1].numpy())) > 1e-8


def test_austria_canonical_signature_is_fresh_and_stable():
    target_a = make_canonical_neutra_target(
        "austria_sir", particle_count=126, substeps=8
    )
    target_b = make_canonical_neutra_target(
        "austria_sir", particle_count=126, substeps=8
    )
    signature = target_a.target_signature()
    assert signature == target_b.target_signature(), "signature not stable"
    # Fresh lineage: must NOT equal the invalidated bootstrap-lane
    # signature for the same scope (recorded historical constant).
    historical_bootstrap_signature = (
        "4845e7322685e19650024e5886e47d89c8b9c4b70c5d36a639c9b1218d39b5c3"
    )
    assert signature != historical_bootstrap_signature
    assert target_a.algorithm_id == (
        "ledh_canonical_pfpf_ot_contract_e_dual_cap_trust_region_analytical_v2"
    )
    assert target_a.score_kwargs["reset_policy"] == "contract_e"
    assert target_a.score_kwargs["correction_steps"] == 4
    assert target_a.score_kwargs["pairwise_steps"] == 4
    assert target_a.score_kwargs["coordinate_cap"] == 0.98
    assert target_a.reset_design.shape == (126, 18)


def test_austria_canonical_score_direction_routes():
    target = make_canonical_neutra_target(
        "austria_sir", particle_count=126, substeps=8
    )
    theta = tf.constant([[0.0, 0.0, 0.0]], DTYPE)
    score_d0 = target.batch_value_score(
        theta, tf.constant([[1.0, 0.0, 0.0]], DTYPE)
    )[1]
    score_d1 = target.batch_value_score(
        theta, tf.constant([[0.0, 1.0, 0.0]], DTYPE)
    )[1]
    assert abs(
        float(score_d0[0].numpy()) - float(score_d1[0].numpy())
    ) > 1e-8, "direction routing inert"


def test_predator_prey_and_lgssm_bridges_value_score_finite():
    for name, p_count in (("predator_prey", 6), ("lgssm", 5)):
        target = make_canonical_neutra_target(
            name, particle_count=126, substeps=6
        )
        if name == "predator_prey":
            theta = tf.constant([[0.8, 90.0, 25.0, 0.5, 0.4, 0.3]], DTYPE)
        else:
            theta = tf.constant([[0.72, 0.55, 0.35, 0.35, 0.45]], DTYPE)
        directions = tf.constant([[1.0] + [0.0] * (p_count - 1)], DTYPE)
        value, score, diagnostics = target.batch_value_score(
            theta, directions
        )
        assert bool(diagnostics["program_valid"][0].numpy()), name
        assert bool(tf.math.is_finite(value[0]).numpy()), name
        assert bool(tf.math.is_finite(score[0]).numpy()), name


def test_ksc_bridge_value_score_finite_smoke():
    """KSC frozen scope is T=1000; the bridge gate runs a reduced particle
    count as a finite/valid smoke (claim-scale runs belong to campaigns)."""

    target = make_canonical_neutra_target(
        "ksc", particle_count=48, substeps=3
    )
    assert int(target.observations.shape[0]) == 1000
    theta = tf.constant([[0.5, 0.1]], DTYPE)
    directions = tf.constant([[1.0, 0.0]], DTYPE)
    value, score, diagnostics = target.batch_value_score(theta, directions)
    assert bool(diagnostics["program_valid"][0].numpy())
    assert bool(tf.math.is_finite(value[0]).numpy())
    assert bool(tf.math.is_finite(score[0]).numpy())
