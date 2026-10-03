"""Independent diagnosis of graph/XLA strict-incumbent reporting differences."""

import dataclasses
from decimal import Decimal, localcontext

import tensorflow as tf

from bayesfilter.inference.joint_center import JointCenterStagedConfig
from bayesfilter.inference.joint_center_staged_tf import (
    StagedJointCenterProgram,
    run_staged_program,
)
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_quadratic_batches import _equal_records

D = tf.float64


def test_staged_graph_xla_incumbent_attribution(request):
    original = FrozenCheckpoint("3582b4ac", "staged_rounding_original")
    prior = original.load("bayesfilter.inference.joint_center")
    precision = tf.linalg.diag(tf.cast(tf.range(1), D) + 1.3) + .07
    mode = .14 + tf.cast(tf.range(1), D) * .03

    def callback(point):
        delta = point - mode
        score = -tf.linalg.matvec(precision, delta)
        return .5 * tf.reduce_sum(delta * score), score

    inputs = (.6 - tf.cast(tf.range(1), D) * .14,
        .8 + tf.cast(tf.range(1), D) * .09)
    changed = tuple(value + .11 for value in inputs)
    records = []
    for jit in (False, True):
        config = JointCenterStagedConfig(checkpoint_iterations=1, total_iterations=10,
            gradient_tolerance=1e-8, max_objective_evaluations=120, jit_compile=jit)
        owner = StagedJointCenterProgram(callback, 1, config)
        for operands in (inputs, changed):
            expected = prior.locate_joint_center_staged(callback, operands[0], scale=operands[1],
                config=prior.JointCenterStagedConfig(**dataclasses.asdict(config)),
                checkpoint_validator=lambda _: True)
            actual = run_staged_program(owner, *operands, lambda _: True)
            optimizer, state, checkpoint = owner.checkpoint(*operands)
            initial, scale, z = (float(operands[0][0]), float(operands[1][0]), float(state["z"][0]))
            with localcontext() as context:
                context.prec = 100
                fused = float(Decimal.from_float(initial) +
                    Decimal.from_float(scale) * Decimal.from_float(z))
            separate = initial + scale * z
            records.append({"jit_compile": jit, "initial": initial, "scale": scale,
                "actual": clean(dataclasses.asdict(actual)), "original": clean(dataclasses.asdict(expected)),
                "optimizer": clean(optimizer._asdict()), "state": clean(state), "checkpoint": clean(checkpoint),
                "independent_affine": {"fused": fused, "separate": separate,
                    "fused_delta": fused - float(mode[0]), "separate_delta": separate - float(mode[0])}})
    report = {"original_source_sha256": original.hashes(), "records": records,
        "comparison": "Full original records within each declared execution setting; XLA admission authority unchanged.",
        "nonclaims": ["Does not waive failed original-XLA versus candidate-graph comparisons03331/03334.",
            "No timing conclusion; Decimal is independent diagnostic arithmetic only."]}
    save(request, "staged-center-rounding.json", report)
    for row in records:
        _equal_records(row["actual"], row["original"])
    graph, xla = records[0], records[2]
    assert graph["state"]["value"] == graph["checkpoint"]["endpoint_objective"] == 0.
    assert xla["state"]["value"] < xla["checkpoint"]["endpoint_objective"] == 0.
    assert graph["actual"]["best_evaluated_source"] == "optimizer_callback"
    assert xla["actual"]["best_evaluated_source"] == "checkpoint_replay"
    assert graph["independent_affine"]["separate_delta"] == 0.
    assert xla["independent_affine"]["fused_delta"] != 0.
    assert xla["independent_affine"]["separate_delta"] == 0.
