from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/benchmarks/run_pp_ukf_posterior_validation_20260730.py"
SPEC = importlib.util.spec_from_file_location("pp_ukf_posterior_validation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def test_reference_archive_identity_hash_shape_and_summary() -> None:
    result = json.loads((ROOT / driver.REFERENCE_RESULT).read_text())
    hashes = json.loads((ROOT / driver.REFERENCE_HASHES).read_text())

    reference = driver._reference_samples(result, hashes)
    summary = driver._summary(reference["samples"])

    assert tuple(reference["samples"].shape) == (4000, 4, 6)
    assert reference["archive_sha256"] == (
        "caba62d43f033cc7c40d2ebc603120a50a507f31b09932a9ff8ecf8d298b70de"
    )
    assert summary["mean"].numpy().tolist() == pytest.approx(
        result["posterior_summary"]["mean"], abs=1.0e-12
    )
    assert summary["sd"].numpy().tolist() == pytest.approx(
        result["posterior_summary"]["sd"], abs=1.0e-12
    )


def test_tensorflow_block_bootstrap_is_deterministic_and_shaped(monkeypatch) -> None:
    monkeypatch.setattr(driver, "BOOTSTRAP_REPLICATES", 40)
    monkeypatch.setattr(driver, "BOOTSTRAP_BATCH_SIZE", 10)
    draws = tf.random.stateless_normal(
        [81, 4, 6], seed=(20260730, 4001), dtype=tf.float64
    )

    first = driver._block_bootstrap_summary(draws, (20260730, 4002))
    second = driver._block_bootstrap_summary(draws, (20260730, 4002))

    assert tuple(first["mean"].shape) == (40, 6)
    assert tuple(first["sd"].shape) == (40, 6)
    assert tuple(first["quantiles"].shape) == (40, 3, 6)
    tf.debugging.assert_near(first["mean"], second["mean"], atol=1.0e-14, rtol=1.0e-14)
    tf.debugging.assert_all_finite(first["quantiles"], "bootstrap quantiles")


def test_current_candidates_are_hash_verified_and_unique() -> None:
    public = json.loads((ROOT / driver.PUBLIC_RESULT).read_text())

    candidates = driver._candidate_samples(public)

    assert len(candidates) == 10
    assert len({row["candidate_id"] for row in candidates}) == 10
    assert tuple(row["L"] for row in candidates) == (5, 9, 12, 13, 14, 17, 18, 19, 24, 25)
    assert all(tuple(row["samples"].shape[1:]) == (4, 6) for row in candidates)


def test_reference_rejects_mathematical_target_or_scope_drift() -> None:
    result = json.loads((ROOT / driver.REFERENCE_RESULT).read_text())
    hashes = json.loads((ROOT / driver.REFERENCE_HASHES).read_text())
    result["target_identity"]["mathematical_target_signature"] = "wrong"
    with pytest.raises(ValueError, match="mathematical target"):
        driver._reference_samples(result, hashes)

    result = json.loads((ROOT / driver.REFERENCE_RESULT).read_text())
    result["target_identity"]["batch_execution_surface"]["target_scope"] = "wrong"
    with pytest.raises(ValueError, match="target scope"):
        driver._reference_samples(result, hashes)


def test_compatibility_classification_separates_inconclusive_from_disagreement(
    monkeypatch,
) -> None:
    monkeypatch.setattr(driver, "BOOTSTRAP_REPLICATES", 40)
    monkeypatch.setattr(driver, "BOOTSTRAP_BATCH_SIZE", 10)
    reference_samples = tf.random.stateless_normal(
        [81, 4, 6], seed=(20260730, 4010), dtype=tf.float64
    )
    reference = {"samples": reference_samples}
    summary = driver._summary(reference_samples)
    bootstrap = driver._block_bootstrap_summary(reference_samples, (20260730, 4011))
    candidate = {
        "candidate_index": 0,
        "candidate_id": "test",
        "L": 5,
        "step_size": 0.5,
        "draws_per_chain": 81,
        "archive_path": "test.tensor",
        "archive_sha256": "a" * 64,
        "samples": reference_samples,
    }

    result = driver._compatibility(candidate, reference, summary, bootstrap)

    assert result["material_disagreement_supported"] is False
    assert result["decision"] in {
        "POSTERIOR_EQUIVALENCE_ESTABLISHED",
        "POSTERIOR_EQUIVALENCE_INCONCLUSIVE",
    }
    assert all(
        check["status"] in {
            "equivalence_established",
            "inconclusive",
            "material_disagreement_supported",
        }
        for row in result["parameter_checks"]
        for check in row["checks"].values()
    )
