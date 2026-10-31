"""Independent diagnostic checks for dataset pairing and promotion vetoes."""

import copy
import hashlib

import numpy as np
import pytest

from bayesfilter.score_study.moment_training_analysis_diagnostic import (
    METHODS, analyze_records, interrupted_summary, paired_interval, source_hash_checks,
)


def test_archived_source_requires_exact_original_hash(tmp_path):
    current = tmp_path / "current.py"
    archived = tmp_path / "executed.py"
    current.write_text("updated reporting")
    archived.write_text("original runtime")
    expected = {str(current): hashlib.sha256(archived.read_bytes()).hexdigest()}
    assert not source_hash_checks(expected)[str(current)]
    mapping = {str(current): str(archived)}
    assert source_hash_checks(expected, mapping)[str(current)]
    archived.write_text("changed archive")
    assert not source_hash_checks(expected, mapping)[str(current)]
    with pytest.raises(ValueError, match="absent from the manifest"):
        source_hash_checks(expected, {"unknown": str(archived)})


def records():
    rows = []
    for seed in range(1, 5):
        for replicate in range(2):
            for method in METHODS:
                loss = seed + replicate / 2 - (.5 if method.endswith("trained") else 0.)
                rows.append(dict(dataset_seed=seed, particle_seed=274000000 + 2 * seed + replicate,
                    method=method, regime="low" if seed <= 2 else "high", valid=True,
                    loss=loss, blocks=[loss, loss], raw_blocks=[loss * 2, loss * 3],
                    score_loss=[1., 2., 3.]))
    return rows


def test_pairing_preserves_exact_shift_and_dataset_unit():
    result = analyze_records(records(), range(1, 5), draws=100)
    contrast = result["contrasts"]["csmc_trained - unguided"]
    np.testing.assert_allclose(contrast["interval"], [-.5, -.5])
    assert contrast["datasets"] == 4
    assert result["datasets"][0]["methods"]["unguided"]["loss"] == 1.25
    assert result["decisions"]["csmc_trained"]["scoped_promotion"]


def test_conditional_heuristic_failure_vetoes_favorable_pooled_mean():
    rows = records()
    for row in rows:
        if row["method"] == "csmc_trained":
            row["loss"] += 1. if row["regime"] == "low" else -2.
    result = analyze_records(rows, range(1, 5), draws=100)
    assert result["summary"]["csmc_trained"]["means"]["loss"] < result["summary"]["unguided"]["means"]["loss"]
    assert not result["decisions"]["csmc_trained"]["heuristic_pass"]
    assert not result["decisions"]["csmc_trained"]["scoped_promotion"]


def test_invalid_rows_are_not_silently_discarded():
    rows = records()
    invalid = next(row for row in rows if row["method"] == "csmc_trained")
    invalid["valid"] = False
    result = analyze_records(rows, range(1, 5), draws=100)
    assert result["summary"]["csmc_trained"]["means"] is None
    assert result["contrasts"]["csmc_trained - unguided"] is None
    assert not result["decisions"]["csmc_trained"]["validity_pass"]


def test_incomplete_or_duplicated_grid_fails_closed():
    rows = records()
    for bad in (rows[:-1], rows + [copy.copy(rows[0])]):
        with pytest.raises(ValueError, match="grid"):
            analyze_records(bad, range(1, 5), draws=100)
    with pytest.raises(ValueError, match="independent datasets"):
        paired_interval([1.], draws=100)


def test_interrupted_study_never_ranks_surviving_datasets():
    rows = records()[:13]
    rows[-1]["valid"] = False
    result = interrupted_summary(rows, range(1, 5), {"error": "invalid baseline"})
    assert result["complete_datasets"] == 1
    assert result["recorded_rows"] == 13
    assert result["missing_rows"] == 35
    assert result["contrasts"] is None
    assert not result["ranking_supported"]
    assert not result["default_ready"]
    assert len(result["invalid_keys"]) == 1
    with pytest.raises(ValueError, match="keys"):
        interrupted_summary(rows + [rows[0]], range(1, 5), {})
