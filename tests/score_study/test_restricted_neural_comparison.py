"""CPU-only integrity checks for fresh restricted SIR comparators."""

import copy
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

from bayesfilter.score_study.neural_moment_evidence import HEURISTIC_ARMS, PARTICLE_ARMS
from scripts.refit_sir_moment_baselines import analyze_restricted_comparisons
from scripts import refit_sir_moment_baselines as baseline_module


def comparison_rows():
    neural, restricted = [], []
    datasets = [101, 102, 103, 104]
    for index, dataset in enumerate(datasets):
        common = dict(dataset=dataset, valid=True,
            observation_regime="low" if index < 2 else "high",
            compartment_regime="nearer_boundary" if index == 0 else "interior")
        for arm in PARTICLE_ARMS:
            neural.extend(dict(common, arm=arm, loss=.5, blocks=[.4, .6], replicate=replicate,
                particle_seed=283030000 + 4 * index + replicate) for replicate in range(4))
        neural.extend(dict(common, arm=arm, loss=1., blocks=[1., 1.], replicate=None,
                           particle_seed=None) for arm in HEURISTIC_ARMS)
        for arm in ("csmc_unadapted", "csmc_adapted", "rbf_unadapted", "rbf_adapted"):
            restricted.extend(dict(common, arm=arm, loss=1., blocks=[.8, 1.2], replicate=replicate,
                particle_seed=283030000 + 4 * index + replicate) for replicate in range(4))
    return neural, restricted, datasets


def test_cluster_comparison_preserves_blocks_and_regimes():
    result = analyze_restricted_comparisons(*comparison_rows())
    assert result["rows"] == 64
    assert not result["missing_regimes"]
    pooled = [row for row in result["comparisons"] if row["regime"] == "all"]
    assert len(pooled) == 4
    assert all(row["datasets"] == 4 and row["mean_difference"] == -.5 for row in pooled)
    assert all(row["paired_cluster_interval95"] == [-.5, -.5] for row in pooled)
    assert all(row["blocks"] == pytest.approx([.8, 1.2], rel=1e-14) for row in result["summaries"])
    assert all(not row["multiplicity_adjusted"] for row in result["comparisons"])


@pytest.mark.parametrize("defect", ["missing", "duplicate", "seed", "regime", "invalid", "arm", "blocks"])
def test_incomplete_or_mismatched_comparisons_reject(defect):
    neural, rows, datasets = comparison_rows()
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows.append(copy.copy(rows[0]))
    elif defect == "seed":
        rows[0]["particle_seed"] += 1
    elif defect == "regime":
        rows[0]["observation_regime"] = "high"
    elif defect == "invalid":
        rows[0]["valid"] = False
    elif defect == "arm":
        rows[0]["arm"] = "unknown"
    else:
        rows[0]["blocks"] = [.8, 1.]
    with pytest.raises(ValueError):
        analyze_restricted_comparisons(neural, rows, datasets)


def test_missing_regime_remains_visible():
    neural, rows, datasets = comparison_rows()
    for row in neural + rows:
        row["compartment_regime"] = "interior"
    result = analyze_restricted_comparisons(neural, rows, datasets)
    assert result["missing_regimes"] == ["nearer_boundary"]
    assert result["status"] == "incomplete_required_regimes"


def test_remaining_cost_counts_full_search_continuation_and_holdout():
    result = baseline_module.remaining_work_projection(
        dict(csmc=[100.] + [1.] * 63, rbf=[200.] + [2.] * 63), [300.] + [3.] * 31)
    assert result["evaluation_seconds"] == dict(csmc=2112., rbf=4224.)
    assert result["fitting_seconds"] == 336.
    assert result["projected_seconds"] == pytest.approx(1.2 * (2112. + 4224. + 336.))


def test_projection_requires_measured_warm_execution_for_both_families():
    with pytest.raises(ValueError, match="warm evaluations"):
        baseline_module.remaining_work_projection(dict(csmc=[1., 1.], rbf=[1.]), [1., 1.])


def test_continuation_uses_fresh_batches_with_paired_objective_probes(monkeypatch, tmp_path):
    study = baseline_module.RestrictedStudy.__new__(baseline_module.RestrictedStudy)
    study.output, study.batch_cache = tmp_path, {}
    batches, evaluations = [], []

    def make_batch(dataset_seed, particle_seed):
        batches.append((int(dataset_seed), int(particle_seed)))
        return (tf.range(4, dtype=tf.float64),) * 10

    def evaluate(family, capacity, eta, inputs, fit_seed):
        evaluations.append((fit_seed, float(inputs[-1])))
        return dict(valid=True, loss=tf.constant(1., tf.float64))

    def paired_optimizer(objective, initial, *, updates, checkpoint, **unused):
        checkpoint(0, initial)
        for update in range(updates):
            objective(initial, update)
            objective(initial + .1, update)
        checkpoint(updates, initial)
        return initial, 0

    study.make_batch, study.evaluate_message = make_batch, evaluate
    study.validate_adapter = lambda *arguments: 1.
    monkeypatch.setattr(baseline_module, "random_direction_descent", paired_optimizer)
    candidates = study.fit_adapter("csmc", "pooled", 263048001, .05,
        start=8, stop=10, initial=tf.zeros([1, 3], tf.float64))
    assert batches == [(263041008, 283010008), (263041009, 283010009)]
    assert evaluations == [(284040016, 0.), (284040017, 1.)] * 2 + [(284040018, 0.), (284040019, 1.)] * 2
    assert [candidate["updates"] for candidate in candidates] == [8, 10]
