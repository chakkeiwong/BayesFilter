"""Evidence-integrity and optimizer-resumption regressions; CPU only."""

import copy
import json
import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

from bayesfilter.score_study.neural_moment_evidence import (
    PARTICLE_ARMS, HEURISTIC_ARMS, analyze_holdout, validate_holdout,
)
from bayesfilter.score_study.neural_kdm_study import NeuralKDMStudy


def records():
    rows = []
    datasets = [101, 102, 103, 104]
    for index, dataset in enumerate(datasets):
        common = dict(dataset=dataset, valid=True, observation_regime="low" if index < 2 else "high",
                      compartment_regime="nearer_boundary" if index == 0 else "interior")
        for arm in PARTICLE_ARMS:
            loss = .5 if arm == "trained" else 1.
            rows.extend(dict(common, arm=arm, loss=loss, blocks=[loss, loss], replicate=replicate,
                             particle_seed=283030000 + index * 4 + replicate) for replicate in range(4))
        rows.extend(dict(common, arm=arm, loss=1., blocks=[1., 1.], replicate=None, particle_seed=None)
                    for arm in HEURISTIC_ARMS)
    return rows, datasets


def test_complete_pairing_and_primary_decision():
    rows, datasets = records()
    result = analyze_holdout(rows, datasets)
    assert result["integrity"]["rows"] == 92
    assert result["primary_pass"]
    assert result["decision"] == "optional_diagnostic_candidate"
    assert not result["model_score_accuracy_evaluated"]
    assert not result["default_ready"]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "seed", "invalid", "nan", "blocks",
                                    "regime", "boolean_replicate"])
def test_bad_pairing_or_invalid_evidence_rejected(defect):
    rows, datasets = records()
    target = rows[0]
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows.append(copy.copy(target))
    elif defect == "seed":
        target["particle_seed"] += 1
    elif defect == "invalid":
        target["valid"] = False
    elif defect == "nan":
        target["loss"] = float("nan")
    elif defect == "blocks":
        target["blocks"] = [1., 1.]
    elif defect == "regime":
        target["observation_regime"] = "high"
    else:
        target["replicate"] = False
    with pytest.raises(ValueError):
        validate_holdout(rows, datasets)


def test_heuristic_tail_failure_vetoes_favorable_primary_comparison():
    rows, datasets = records()
    for row in rows:
        if row["dataset"] == 101 and row["arm"] == "observation_only":
            row.update(loss=.1, blocks=[.1, .1])
    result = analyze_holdout(rows, datasets)
    assert result["primary_pass"]
    assert not result["heuristic_dominance"]["passed"]
    assert result["decision"] == "candidate_not_promoted"
    assert any(row["regime"] == "nearer_boundary" for row in result["heuristic_dominance"]["failures"])


def fake_study(tmp_path):
    study = object.__new__(NeuralKDMStudy)
    study.arguments = SimpleNamespace(components=2, width=16, beta=.5, learning_rate=.001, seed=263044001)
    study.network = SimpleNamespace(parameters=[tf.Variable([.3, -.7], dtype=tf.float64)])
    study.optimizer = tf.keras.optimizers.Adam(.001, epsilon=1e-8)
    study.optimizer.build(study.network.parameters)
    study.source = {"source_hashes": {"dependency.py": "test-hash"}}
    study.completed_updates = 0
    study.output = tmp_path
    return study


def update(study):
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(tf.square(study.network.parameters[0] - .2))
    study.optimizer.apply_gradients(zip(tape.gradient(loss, study.network.parameters), study.network.parameters))
    study.completed_updates += 1


def test_checkpoint_restores_adam_and_exact_next_update(tmp_path):
    original = fake_study(tmp_path)
    update(original)
    path = original.checkpoint(dict(valid=True, loss=1.))
    restored = fake_study(tmp_path)
    restored.restore(type(tmp_path)(path))
    update(original)
    update(restored)
    assert restored.completed_updates == original.completed_updates == 2
    for left, right in zip(original.network.parameters + original.optimizer.variables,
                           restored.network.parameters + restored.optimizer.variables):
        tf.debugging.assert_equal(tf.convert_to_tensor(left), tf.convert_to_tensor(right))


def test_checkpoint_source_and_data_index_are_checked(tmp_path):
    study = fake_study(tmp_path)
    update(study)
    path = type(tmp_path)(study.checkpoint(dict(valid=True, loss=1.)))
    checkpoint = json.loads(path.read_text())
    checkpoint["updates"] += 1
    path.write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match="iteration"):
        fake_study(tmp_path).restore(path)
    checkpoint["updates"] -= 1
    checkpoint["source_hashes"] = {}
    path.write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match="source"):
        fake_study(tmp_path).restore(path)


def test_heuristic_blocks_preserve_existing_losses():
    from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
    from bayesfilter.score_study.physical_proposal_diagnostic_campaign import PhysicalCampaign
    spec = NonlinearSQMCSpec("sir_d18")
    theta = spec.default_theta(tf.float64)
    mean = spec.initial_mean(tf.float64)
    context = SimpleNamespace(spec=spec, theta=theta,
        calibration=(mean, tf.ones([18], tf.float64), tf.ones([189], tf.float64)))
    model = spec.model(theta, tf.zeros_like(theta))[0]
    observation = model.observation_fn(mean[None])[0]
    observations = tf.broadcast_to(observation, [5, 9])
    old_losses, old_valid = PhysicalCampaign.make_heuristics(context)(observations, mean)
    losses, valid, blocks = PhysicalCampaign.make_heuristics(context, return_blocks=True)(observations, mean)
    tf.debugging.assert_equal(losses, old_losses)
    tf.debugging.assert_equal(valid, old_valid)
    tf.debugging.assert_near(losses, tf.reduce_mean(blocks, axis=1), atol=1e-12, rtol=1e-12)
