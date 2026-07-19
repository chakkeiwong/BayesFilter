from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "docs/benchmarks/run_ssl_lstm_a4_hmc_repair_02_2026_07_14.py"


@pytest.fixture(scope="module")
def repair() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ssl_lstm_a4_hmc_repair_02", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _samples() -> tf.Tensor:
    draw = tf.reshape(tf.cast(tf.range(64), tf.float64), (64, 1, 1))
    chain = tf.reshape(tf.cast(tf.range(4), tf.float64), (1, 4, 1))
    parameter = tf.reshape(tf.cast(tf.range(4), tf.float64), (1, 1, 4))
    return 0.01 * draw + chain + 0.1 * parameter


def _trace(accepted_per_chain: tuple[int, int, int, int]) -> dict[str, tf.Tensor]:
    accepted = tf.stack(
        [
            tf.range(64) < count
            for count in accepted_per_chain
        ],
        axis=1,
    )
    return {
        "is_accepted": accepted,
        "step_size": tf.fill([64], tf.constant(0.3, tf.float64)),
        "log_accept_ratio": tf.zeros((64, 4), tf.float64),
        "target_log_prob": tf.zeros((64, 4), tf.float64),
    }


def test_frozen_adaptation_contract(repair: ModuleType) -> None:
    config = repair.build_adaptation_config()
    assert config.step_size == 0.3925
    assert config.num_leapfrog_steps == 4
    assert config.num_burnin_steps == 320
    assert config.num_results == 64
    assert config.use_xla is True
    assert config.tuning_policy.label == "fixed_mass_dual_averaging"
    assert config.tuning_policy.num_adaptation_steps == 256
    assert config.tuning_policy.target_accept_prob == pytest.approx(0.70)


def test_real_tfp_adaptation_freezes_scalar_step_after_warmup(
    repair: ModuleType,
) -> None:
    from bayesfilter.inference import ValueScoreCapability
    from bayesfilter.inference.hmc import (
        FullChainHMCConfig,
        build_reusable_full_chain_tfp_hmc_runner,
    )
    from bayesfilter.inference.hmc_tuning import HMCTuningPolicy

    class GaussianAdapter:
        def adapter_signature(self) -> str:
            return "repair-02-dual-averaging-test-gaussian-v1"

        def value_score_capability(self) -> ValueScoreCapability:
            return ValueScoreCapability(
                value_score_authority="graph_native",
                xla_hmc_ready=True,
                runtime_backend="tensorflow",
                evidence_path="tests/test_ssl_lstm_a4_hmc_repair_02.py",
                target_scope="repair_02_dual_averaging_test_gaussian",
                nonclaims=("tiny CPU-hidden adaptation semantics fixture",),
                full_chain_xla_diagnostic_ready=True,
            )

        def log_prob_and_grad(self, value: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            tensor = tf.convert_to_tensor(value, tf.float64)
            return -0.5 * tf.reduce_sum(tf.square(tensor), axis=-1), -tensor

    policy = HMCTuningPolicy.fixed_mass_dual_averaging(
        num_adaptation_steps=16,
        target_accept_prob=0.70,
        source="tests/test_ssl_lstm_a4_hmc_repair_02.py",
    )
    config = FullChainHMCConfig(
        num_results=8,
        num_burnin_steps=20,
        step_size=0.2,
        num_leapfrog_steps=1,
        seed=(20260714, 1690),
        use_xla=False,
        trace_policy="standard",
        tuning_policy=policy,
        target_scope="repair_02_dual_averaging_test_gaussian",
    )
    initial = tf.constant([[0.0, 0.0], [0.5, -0.5]], tf.float64)
    result = build_reusable_full_chain_tfp_hmc_runner(
        GaussianAdapter(), initial, config
    ).run()
    step_trace = tf.convert_to_tensor(result.trace["step_size"], tf.float64)
    assert tuple(step_trace.shape) == (8,)
    tf.debugging.assert_near(step_trace, tf.fill([8], step_trace[-1]), atol=1e-12)
    assert bool(tf.math.is_finite(step_trace[-1]).numpy())
    assert result.metadata["adaptation_policy"] == "dual_averaging_step_size"


def test_prior_receipts_are_hash_bound_and_budget_complete(repair: ModuleType) -> None:
    consumed = repair.validate_prior_receipts()
    assert consumed == pytest.approx(1556.7344745269511)
    assert repair.base.GPU_BUDGET_SECONDS - consumed == pytest.approx(
        27243.26552547305
    )


def test_prior_receipt_hash_drift_fails_closed(
    repair: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _digest = repair.PRIOR_RECEIPTS[0]
    monkeypatch.setattr(repair, "PRIOR_RECEIPTS", ((path, "0" * 64),))
    with pytest.raises(repair.RepairError, match="SHA-256 drift"):
        repair.validate_prior_receipts()


def test_adaptation_classification_selects_target_neighborhood(
    repair: ModuleType,
) -> None:
    result = repair.classify_adaptation(_samples(), _trace((45, 44, 46, 43)))
    assert result["status"] == "SELECTED"
    assert result["selected"] is True
    assert result["hard_vetoes"] == []
    assert result["final_step_size"] == pytest.approx(0.3)
    assert result["post_warmup_step_spread"] == pytest.approx(0.0)


def test_adaptation_classification_rejects_over_acceptance_without_hard_veto(
    repair: ModuleType,
) -> None:
    result = repair.classify_adaptation(_samples(), _trace((60, 58, 61, 59)))
    assert result["status"] == "NOT_SELECTED"
    assert result["selected"] is False
    assert result["acceptance_passed"] is False
    assert result["hard_vetoes"] == []


def test_adaptation_uses_aggregate_target_band_and_broad_chain_safety(
    repair: ModuleType,
) -> None:
    result = repair.classify_adaptation(_samples(), _trace((23, 51, 52, 53)))
    assert result["acceptance_rate_by_chain"][0] == pytest.approx(23 / 64)
    assert result["acceptance_rate"] == pytest.approx(179 / 256)
    assert result["status"] == "SELECTED"


def test_adaptation_classification_vetoes_post_warmup_step_changes(
    repair: ModuleType,
) -> None:
    trace = _trace((45, 44, 46, 43))
    trace["step_size"] = tf.linspace(
        tf.constant(0.30, tf.float64), tf.constant(0.31, tf.float64), 64
    )
    result = repair.classify_adaptation(_samples(), trace)
    assert result["status"] == "HARD_VETO"
    assert "step_changed_after_warmup" in result["hard_vetoes"]


def test_private_tensor_round_trip_and_hash_veto(
    repair: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(repair, "ROOT", tmp_path)
    row = repair._write_tensor(Path("private/value.tftensor"), _samples())
    tf.debugging.assert_equal(repair._read_tensor(row), _samples())
    path = tmp_path / row["path"]
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(repair.RepairError, match="hash mismatch"):
        repair._read_tensor(row)


def test_no_overwrite_guard_checks_public_and_private(
    repair: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(repair, "ROOT", tmp_path)
    output = Path("result.json")
    private = (Path("private/state.tftensor"),)
    repair._require_fresh(output, private)
    (tmp_path / output).write_text("occupied", encoding="ascii")
    with pytest.raises(repair.RepairError, match="refusing overwrite"):
        repair._require_fresh(output, private)


def test_private_adaptation_paths_are_distinct_from_prior_repairs(
    repair: ModuleType,
) -> None:
    paths = repair._adaptation_private_paths()
    assert len(set(paths.values())) == len(paths)
    assert all("repair-02" in path.as_posix() for path in paths.values())
    assert all("repair-01" not in path.as_posix() for path in paths.values())
    assert repair.SEGMENT_OUTPUT != repair.ADAPTATION_OUTPUT
