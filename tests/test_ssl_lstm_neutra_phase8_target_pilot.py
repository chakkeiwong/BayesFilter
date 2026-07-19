from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "docs/benchmarks/run_ssl_lstm_neutra_phase8_target_pilot_2026_07_17.py"
)


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("phase8_target_pilot", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_entry_receipts_and_split_contract_are_exact(harness: ModuleType) -> None:
    entry = harness.validate_entry_receipts()
    assert entry["phase7"]["both_charts_admitted"] is True
    assert entry["canary"]["status"] == "PASSED"
    assert entry["chunk_canary"]["contract"]["draw_chunk_size"] == 16
    assert entry["chunk_prefix"]["scope"]["target_pilot_retried"] is False
    assert entry["distance_canary"]["contract"]["path_shape"] == [8, 64, 2, 10]
    assert harness.PILOT_DRAWS_PER_CHAIN == 64
    assert harness.CONFIRMATION_DRAWS_PER_CHAIN == 448
    assert harness.PILOT_DRAWS_PER_CHAIN + harness.CONFIRMATION_DRAWS_PER_CHAIN == 512
    assert harness.PILOT_DRAWS_PER_CHAIN % harness.BLOCK_LENGTH == 0
    assert harness.CONFIRMATION_DRAWS_PER_CHAIN % harness.BLOCK_LENGTH == 0
    assert harness.CONFIRMATION_DRAWS_PER_CHAIN // harness.BLOCK_LENGTH == 28
    assert harness.FORECAST_DRAW_CHUNK_SIZE == 16
    assert 4 * harness.PILOT_DRAWS_PER_CHAIN % harness.FORECAST_DRAW_CHUNK_SIZE == 0


def test_archive_reader_parses_only_frozen_prefix(harness: ModuleType) -> None:
    phase7 = harness.validate_entry_receipts()["phase7"]
    samples, audit = harness.read_frozen_pilot_prefix(phase7)
    assert set(samples) == {"fresh-g", "fresh-h"}
    for chart in samples:
        assert tuple(samples[chart].shape) == (4, 64, 4)
        assert bool(tf.reduce_all(tf.math.is_finite(samples[chart])))
        assert audit[chart][0]["sample_values_parsed"] is True
        assert audit[chart][0]["parsed_draw_indices"] == [0, 255]
        assert audit[chart][0]["selected_for_pilot_draw_indices"] == [0, 63]
        assert audit[chart][0]["sample_tensor_deserialization_scope"] == (
            "full_256_draw_tensor_required_by_tftensor_format"
        )
        assert audit[chart][1]["sample_values_parsed"] is False
        assert audit[chart][1]["parsed_draw_indices"] is None
        assert audit[chart][1]["selected_for_pilot_draw_indices"] is None
        assert audit[chart][1]["sample_tensor_deserialization_scope"] == "none_hash_only"


def test_scale_policy_is_pooled_finite_and_fail_closed(harness: ModuleType) -> None:
    base = tf.reshape(tf.range(8 * 4 * 2 * 10, dtype=tf.float64), [8, 4, 2, 10])
    center, scale, floor, active = harness._pilot_scale(base / 17.0)
    assert tuple(center.shape) == tuple(scale.shape) == tuple(floor.shape) == (10,)
    assert bool(tf.reduce_all(scale > floor))
    assert not bool(tf.reduce_any(active))
    with pytest.raises(harness.Phase8PilotError, match="degenerate"):
        harness._pilot_scale(tf.ones((8, 4, 2, 10), tf.float64))


def test_seed_and_bandwidth_domains_are_prospective(harness: ModuleType) -> None:
    assert harness.PILOT_SEED == (12001, 12002)
    assert harness.ARM_IDS == {"fresh-g": 1, "fresh-h": 2}
    assert harness.BANDWIDTH_FACTORS == (0.25, 0.5, 1.0, 2.0, 4.0)
    config = harness.SSLLSTMForecastConfig()
    g = harness.make_ssl_lstm_innovation_bank(
        config, 4, tf.constant(harness.PILOT_SEED, tf.int32), "independent_arm", 1
    )
    h = harness.make_ssl_lstm_innovation_bank(
        config, 4, tf.constant(harness.PILOT_SEED, tf.int32), "independent_arm", 2
    )
    assert g.content_signature != h.content_signature
    assert set(g.tensor_hashes().values()).isdisjoint(h.tensor_hashes().values())


def test_public_receipt_contract_forbids_arm_comparison(harness: ModuleType) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"segment0_full_tensor_deserialized": True' in source
    assert '"segment0_confirmation_suffix_selected_or_evaluated": False' in source
    assert '"segment1_tensor_deserialized": False' in source
    assert '"confirmation_values_used_in_any_computation": False' in source
    assert '"confirmation_forecast_bank_opened": False' in source
    assert '"arm_specific_predictive_summaries_emitted": False' in source
    assert '"g_h_predictive_difference_computed": False' in source
    assert "segment-001_retained_samples.tftensor" not in source
    assert "PILOT_DRAWS_PER_CHAIN = 64" in source
    assert "ssl_lstm_terminal_covariance_audit_compiled_program" in source
    assert '"terminal_covariance_audit": _trace_count' in source
    assert "draw_chunk_size=FORECAST_DRAW_CHUNK_SIZE" in source
    assert '"forecast_draw_chunk_size": FORECAST_DRAW_CHUNK_SIZE' in source
