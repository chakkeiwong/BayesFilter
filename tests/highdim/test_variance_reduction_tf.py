from __future__ import annotations

import tensorflow as tf
import json
import importlib.util
from pathlib import Path
import pytest

from bayesfilter.independent_score import variance_reduction_tf as variance
from bayesfilter.independent_score import anchored_orthogonal_ratio_score_tf as anchored


def test_crossed_summary_recovers_known_paired_variance_reduction() -> None:
    bundles, paths, cells = 10, 24, 3
    path_effect = tf.random.stateless_normal([paths, cells], [101, 1], dtype=tf.float64)
    bundle_noise = tf.random.stateless_normal([bundles, paths, cells], [101, 2], dtype=tf.float64)
    scales = tf.constant([1.0, 0.7, 0.6, 0.3], tf.float64)
    outputs = path_effect[None, None, :, :] + scales[:, None, None, None] * bundle_noise[None, :, :, :]
    fixed_noise = tf.random.stateless_normal([bundles, cells], [101, 3], dtype=tf.float64)
    fixed = scales[:, None, None] * fixed_noise[None, :, :]
    summary = variance.summarize_crossed_outputs(outputs, fixed, bootstrap_replicates=200)
    combined_audit = summary["audit_variance_ratio"][3]
    combined_fixed = summary["fixed_variance_ratio"][3]
    assert combined_audit["upper_95"] < 1.0
    assert combined_fixed["upper_95"] < 1.0
    assert summary["audit_effect_ratio"][-1]["effect"] == "combined"
    assert summary["audit_effect_ratio"][-1]["upper_95"] < 1.0
    assert variance.classify_combined_arm(summary)["status"] == "variance_reduction_supported"


def test_gaussian_accuracy_guard_can_veto_low_variance_biased_arm() -> None:
    bundles, paths, cells = 10, 24, 2
    exact = tf.random.stateless_normal([paths, cells], [202, 1], dtype=tf.float64)
    noise = tf.random.stateless_normal([bundles, paths, cells], [202, 2], dtype=tf.float64)
    arms = [exact[None, :, :] + noise]
    arms.append(exact[None, :, :] + 0.8 * noise)
    arms.append(exact[None, :, :] + 0.7 * noise)
    arms.append(exact[None, :, :] + 5.0 + 0.1 * noise)
    outputs = tf.stack(arms)
    exact_fixed = tf.zeros([cells], tf.float64)
    fixed_noise = tf.random.stateless_normal([bundles, cells], [202, 3], dtype=tf.float64)
    fixed = tf.stack([fixed_noise, 0.8 * fixed_noise, 0.7 * fixed_noise, 5.0 + 0.1 * fixed_noise])
    summary = variance.summarize_crossed_outputs(
        outputs,
        fixed,
        exact_scores=exact,
        exact_fixed_score=exact_fixed,
        bootstrap_replicates=200,
    )
    decision = variance.classify_combined_arm(summary)
    assert decision["accuracy_harmed"] is True
    assert decision["status"] == "accuracy_harmed"
    assert "exact_fixed_mse_ratio" in summary
    assert "exact_mse_effect_ratio" in summary
    assert "exact_bias_by_cell" in summary
    assert "exact_rmse_by_cell" in summary


def test_bundle_variance_is_separate_from_path_variance() -> None:
    base = tf.reshape(tf.range(6.0, dtype=tf.float64), [1, 1, 6, 1])
    bundle_offsets = tf.reshape(tf.constant([0.0, 1.0], tf.float64), [1, 2, 1, 1])
    outputs = tf.tile(base + bundle_offsets, [4, 1, 1, 1])
    fixed = tf.tile(tf.reshape(tf.constant([0.0, 1.0], tf.float64), [1, 2, 1]), [4, 1, 1])
    summary = variance.summarize_crossed_outputs(outputs, fixed, bootstrap_replicates=100)
    assert summary["bundle_variance_by_cell"][0, 0].numpy() == 0.5
    assert summary["natural_path_variance_by_cell"][0].numpy() > 0.5


def test_aggregator_refuses_incomplete_campaign(tmp_path: Path) -> None:
    path = Path(__file__).resolve().parents[2] / "docs/benchmarks/aggregate_classifier_score_variance_20260815.py"
    spec = importlib.util.spec_from_file_location("variance_aggregator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="missing bundle"):
        module.load_campaign(tmp_path)


def test_aggregator_adds_repository_root_before_local_imports() -> None:
    path = Path(__file__).resolve().parents[2] / "docs/benchmarks/aggregate_classifier_score_variance_20260815.py"
    source = path.read_text(encoding="utf-8")
    assert 'ROOT = Path(__file__).resolve().parents[2]' in source
    assert 'sys.path.insert(0, str(ROOT))' in source
    assert '"device_policy": "cpu_only"' in source
    assert 'parser.add_argument("--output", type=Path)' in source


def test_aggregator_accepts_one_complete_crossed_campaign(tmp_path: Path) -> None:
    path = Path(__file__).resolve().parents[2] / "docs/benchmarks/aggregate_classifier_score_variance_20260815.py"
    spec = importlib.util.spec_from_file_location("variance_aggregator_complete", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for bundle in range(10):
        folder = tmp_path / f"bundle_{bundle:02d}"
        folder.mkdir()
        outputs = tf.random.stateless_normal([4, 8, 2], [bundle + 1, 7], dtype=tf.float64).numpy().tolist()
        fixed = tf.random.stateless_normal([4, 2], [bundle + 1, 9], dtype=tf.float64).numpy().tolist()
        records = {}
        for delta in ("0.005",):
            records[delta] = {
                "minus_noise_sha256": "minus-small",
                "plus_noise_sha256": "plus-small",
                "minus_prefix_sha256": "minus-small",
                "plus_prefix_sha256": "plus-small",
                "noise_identical": False,
            }
        independent_small = records
        independent_large = {
            "0.005": {
                "minus_noise_sha256": "minus-large",
                "plus_noise_sha256": "plus-large",
                "minus_prefix_sha256": "minus-small",
                "plus_prefix_sha256": "plus-small",
                "noise_identical": False,
            }
        }
        crn_small = {
            "0.005": {
                "minus_noise_sha256": "minus-small",
                "plus_noise_sha256": "minus-small",
                "minus_prefix_sha256": "minus-small",
                "plus_prefix_sha256": "minus-small",
                "noise_identical": True,
            }
        }
        crn_large = {
            "0.005": {
                "minus_noise_sha256": "minus-large",
                "plus_noise_sha256": "minus-large",
                "minus_prefix_sha256": "minus-small",
                "plus_prefix_sha256": "minus-small",
                "noise_identical": True,
            }
        }
        pair_by_arm = {
            "independent_n2048": independent_small,
            "crn_n2048": crn_small,
            "independent_n8192": independent_large,
            "crn_n8192": crn_large,
        }
        shared_hashes = {"j0": {"validation": "v", "calibration": "c", "test": "t"}}
        arm_rows = {}
        for arm in variance.ARM_NAMES:
            cells = {"T20_j0": {"finite": True, "temperature": 1.0, "optimizer_complete": True, "pair_hashes": pair_by_arm[arm]}}
            arm_rows[arm] = {"cells": cells, "shared_split_hashes": shared_hashes}
        payload = {
            "status": "COMPLETED",
            "kind": "sir",
            "bundle": bundle,
            "completed_arms": list(variance.ARM_NAMES),
            "arm_rows": arm_rows,
            "audit_outputs": outputs,
            "fixed_outputs": fixed,
            "audit_path_sha256": "shared-audit",
            "fixed_path_sha256": "shared-fixed",
        }
        (folder / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    result = module.aggregate(tmp_path, bootstrap_replicates=100)
    assert result["status"] == "COMPLETED"
    assert result["all_hard_valid"] is True


def test_bundle_runner_serializes_resource_variables() -> None:
    path = Path(__file__).resolve().parents[2] / "docs/benchmarks/run_classifier_score_variance_bundle_20260815.py"
    source = path.read_text(encoding="utf-8")
    assert "tf.is_tensor(value) or isinstance(value, tf.Variable)" in source
    assert "plus_noise = minus_noise if crn" in source
    assert 'if name == "capacity"' in source
    assert 'if name == "full_cell"' in source
    assert "paired_training_datasets(" in source
    assert 'cell filtering is not allowed for the full campaign' in source


def test_cached_training_banks_exactly_match_original_per_arm_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BAYESFILTER_CPU_ONLY_SMOKE", "true")
    path = Path(__file__).resolve().parents[2] / "docs/benchmarks/run_classifier_score_variance_bundle_20260815.py"
    spec = importlib.util.spec_from_file_location("variance_bundle_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def simulator(parameters: tf.Tensor, noise: tf.Tensor) -> tf.Tensor:
        shift = tf.reshape(tf.reduce_sum(parameters), [1, 1, 1])
        return noise + shift

    cfg = {"n_small": 4, "n_large": 8}
    cached = module.paired_training_datasets(
        "gaussian",
        simulator,
        coordinate=1,
        bundle=2,
        cfg=cfg,
        arm_names=variance.ARM_NAMES,
    )
    for arm in variance.ARM_NAMES:
        count = cfg["n_large"] if arm.endswith("n8192") else cfg["n_small"]
        original = module.conditional_dataset(
            "gaussian",
            simulator,
            coordinate=1,
            bundle=2,
            role=10,
            count=count,
            max_count=cfg["n_large"],
            crn=arm.startswith("crn"),
        )
        for cached_tensor, original_tensor in zip(cached[arm][:4], original[:4]):
            tf.debugging.assert_equal(cached_tensor, original_tensor)
        assert cached[arm][4] == original[4]


def test_pair_rows_validate_and_preserve_whole_minibatch_pairs() -> None:
    pair_ids = tf.constant([7, 9, 7, 8, 9, 8], tf.int64)
    labels = tf.constant([0, 1, 1, 0, 0, 1], tf.float32)
    deltas = tf.constant([0.01, 0.02, 0.01, 0.03, 0.02, 0.03], tf.float32)
    rows = anchored._training_pair_rows(pair_ids, labels, deltas)
    permutation = tf.reshape(tf.gather(rows, tf.constant([2, 0, 1])), [-1])
    for start in range(0, 6, 2):
        batch_ids = tf.gather(pair_ids, permutation[start:start + 2])
        tf.debugging.assert_equal(batch_ids[0], batch_ids[1])
    with pytest.raises(ValueError, match="one row per class"):
        anchored._training_pair_rows(pair_ids, tf.zeros([6]), deltas)
