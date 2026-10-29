"""CPU-only recovery mechanics; tiny fixtures do not evaluate SIR quality."""

import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
import tensorflow as tf

from bayesfilter.score_study.neural_kdm_diagnostic_campaign import NeuralKDMCampaign
from bayesfilter.score_study.neural_kdm_study import NeuralKDMStudy
from scripts.continue_sir_neural_kdm import (
    RECOVERY_SCHEMA, RecoverableNeuralKDMStudy, continuation_projection, continuation_rungs,
)


@pytest.mark.parametrize("starting,target,expected", [
    (128, 256, [192, 256]), (192, 256, [256]), (256, 512, [384, 512]),
    (362, 512, [384, 512]), (384, 512, [512]), (511, 512, [512]),
])
def test_continuation_rungs_preserve_completed_and_pending_checkpoints(starting, target, expected):
    assert continuation_rungs(starting, target) == expected


@pytest.mark.parametrize("starting,target", [(127, 256), (256, 256), (512, 512), (256, 1024)])
def test_continuation_rungs_reject_unsupported_ranges(starting, target):
    with pytest.raises(ValueError, match="Unsupported continuation range"):
        continuation_rungs(starting, target)


def fixture_study(path, updates=8):
    path.mkdir()
    study = object.__new__(RecoverableNeuralKDMStudy)
    study.arguments = SimpleNamespace(components=2, width=16, beta=.9,
        learning_rate=.001, seed=263044001, updates=updates, budget_seconds=60.)
    study.network = SimpleNamespace(parameters=[tf.Variable([.1 * (index + 1)], dtype=tf.float64)
                                                for index in range(6)])
    study.optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
    study.optimizer.build(study.network.parameters)
    study.completed_updates = 0
    study.last_recovery_update = -1
    study.latest_recovery = None
    study.source = dict(source_hashes={"fixture.py": "fixed"})
    study.output = path
    study.started = time.monotonic()
    study.seen_seeds = []
    study.validation = lambda: dict(valid=True, loss=1., datasets=32, replicates=2)

    def make_batch(dataset_seed, particle_seed):
        study.seen_seeds.append((int(dataset_seed), int(particle_seed)))
        return dataset_seed, particle_seed

    def update(dataset_seed, particle_seed):
        displacement = tf.cast(dataset_seed - 263041000, tf.float64) * .01
        gradients = [2. * parameter + displacement for parameter in study.network.parameters]
        study.optimizer.apply_gradients(zip(gradients, study.network.parameters))
        return dict(loss=tf.add_n([tf.reduce_sum(tf.square(parameter))
                                 for parameter in study.network.parameters]),
                    valid=tf.constant(True), applied=tf.constant(True))

    study.make_batch = make_batch
    study.update = update
    return study


def test_actual_fit_and_trainer_are_inherited(monkeypatch):
    assert RecoverableNeuralKDMStudy.fit is NeuralKDMStudy.fit
    assert RecoverableNeuralKDMStudy._make_batch is NeuralKDMStudy._make_batch
    update = object()
    gradient = SimpleNamespace(input_signature=[tf.TensorSpec([], tf.float64)])
    optimizer = object()

    def initialize(study, arguments):
        study.update, study.row_gradient, study.optimizer = update, gradient, optimizer

    monkeypatch.setattr(NeuralKDMCampaign, "__init__", initialize)
    study = RecoverableNeuralKDMStudy(SimpleNamespace(resume=None))
    assert study.update is update
    assert study.row_gradient is gradient
    assert study.optimizer is optimizer


def test_interrupted_resume_matches_all_adam_and_parameter_tensors(tmp_path, monkeypatch):
    uninterrupted = fixture_study(tmp_path / "uninterrupted")
    uninterrupted.fit()
    interrupted = fixture_study(tmp_path / "interrupted")
    original_check = NeuralKDMCampaign.check_budget

    def interrupt(study):
        if study is interrupted and study.completed_updates == 3:
            raise TimeoutError("fixture budget")
        original_check(study)

    monkeypatch.setattr(NeuralKDMCampaign, "check_budget", interrupt)
    with pytest.raises(TimeoutError):
        interrupted.fit()
    recovery = Path(interrupted.latest_recovery)
    payload = json.loads(recovery.read_text())
    assert payload["updates"] == 3
    assert payload["selectable"] is False
    assert payload["validation"]["valid"] is False
    resumed = fixture_study(tmp_path / "resumed")
    resumed.restore(recovery)
    resumed.fit()
    assert interrupted.seen_seeds + resumed.seen_seeds == uninterrupted.seen_seeds
    assert resumed.completed_updates == uninterrupted.completed_updates == 8
    for actual, expected in zip(resumed.network.parameters + list(resumed.optimizer.variables),
                                uninterrupted.network.parameters + list(uninterrupted.optimizer.variables)):
        tf.debugging.assert_equal(actual, expected)


@pytest.mark.parametrize("mutation", ["source", "configuration", "iteration", "seed",
                                      "selectable", "schema", "shape", "nonfinite", "unmarked"])
def test_corrupt_recovery_rejects(tmp_path, mutation):
    original = fixture_study(tmp_path / "original")
    path = Path(original.save_recovery())
    payload = json.loads(path.read_text())
    if mutation == "source":
        payload["source_hashes"] = {"fixture.py": "changed"}
    elif mutation == "configuration":
        payload["configuration"]["beta"] = .5
    elif mutation == "iteration":
        payload["updates"] = 1
        payload["next_training_seed"] += 1
        payload["next_particle_seed"] += 1
    elif mutation == "seed":
        payload["next_training_seed"] += 1
    elif mutation == "selectable":
        payload["selectable"] = True
    elif mutation == "schema":
        payload["recovery_schema"] = "unknown"
    elif mutation == "shape":
        payload["parameters"][0] = [1., 2.]
    elif mutation == "nonfinite":
        payload["parameters"][0] = [float("inf")]
    elif mutation == "unmarked":
        del payload["recovery_schema"]
    path.write_text(json.dumps(payload))
    resumed = fixture_study(tmp_path / "resumed")
    with pytest.raises(ValueError):
        resumed.restore(path)


def test_save_rejects_inconsistent_or_nonfinite_state(tmp_path):
    study = fixture_study(tmp_path / "study")
    study.completed_updates = 1
    with pytest.raises(ValueError, match="inconsistent"):
        study.save_recovery()
    study.completed_updates = 0
    study.network.parameters[0].assign([float("inf")])
    with pytest.raises(ValueError, match="nonfinite"):
        study.save_recovery()


def test_recovery_is_distinct_from_validated_checkpoint(tmp_path):
    study = fixture_study(tmp_path / "study")
    study.fit()
    recovery = Path(study.save_recovery())
    assert recovery.name == "recovery-0008.json"
    assert json.loads(recovery.read_text())["recovery_schema"] == RECOVERY_SCHEMA
    assert json.loads((study.output / "checkpoint-0008.json").read_text())["validation"]["valid"]
    assert not list(study.output.glob("*.tmp"))


def test_projection_uses_recent_upper_tail_not_old_average():
    assert continuation_projection([1.] * 100 + [28.] * 20, 60., 64) == 2435.
    assert continuation_projection([28.] * 20, 60., 384) > 4891.197895057


@pytest.mark.parametrize("times,validation,updates", [([], 0., 64), ([float("nan")], 0., 64),
    ([0.], 0., 64), ([1.], -1., 64), ([1.], 0., 0)])
def test_projection_rejects_bad_evidence(times, validation, updates):
    with pytest.raises(ValueError):
        continuation_projection(times, validation, updates)
