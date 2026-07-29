from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import subprocess
import sys

import pytest
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/benchmarks/run_pp_ukf_true_hmc_validation_20260722.py"
SPEC = importlib.util.spec_from_file_location("pp_ukf_true_hmc_validation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def test_campaign_budget_is_twenty_four_hours():
    assert driver.CAMPAIGN_CAP_SECONDS == 24.0 * 60.0 * 60.0
    assert driver.MAX_RESULTS == 10_000


def test_campaign_budget_check_includes_prior_charge_before_each_chunk():
    assert driver._campaign_budget_exhausted(
        prior_elapsed_seconds=42_403.504540,
        current_elapsed_seconds=43_996.495459,
    ) is False
    assert driver._campaign_budget_exhausted(
        prior_elapsed_seconds=42_403.504540,
        current_elapsed_seconds=43_996.495460,
    ) is True
    assert driver._campaign_budget_exhausted(
        prior_elapsed_seconds=42_746.868394,
        current_elapsed_seconds=42_753.131606,
        reserve_seconds=900.0,
    ) is True


def test_run_request_rejects_ambiguous_or_invalid_budget_inputs(tmp_path):
    progress = ROOT / "docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-09/progress.json"
    with pytest.raises(ValueError, match="exclusive"):
        driver._validate_run_request(
            candidate_indices=(1,),
            replacement_indices=(1, 2, 5),
            prior_elapsed_seconds=0.0,
            resume_progress=progress,
        )
    for value in (-1.0, math.nan, math.inf, driver.CAMPAIGN_CAP_SECONDS):
        with pytest.raises(ValueError, match="prior_elapsed_seconds"):
            driver._validate_run_request(
                candidate_indices=None,
                replacement_indices=None,
                prior_elapsed_seconds=value,
                resume_progress=None,
            )
    with pytest.raises(FileNotFoundError):
        driver._validate_run_request(
            candidate_indices=None,
            replacement_indices=(1, 2, 5),
            prior_elapsed_seconds=0.0,
            resume_progress=tmp_path / "missing.json",
        )


def test_candidate_manifest_is_frozen_and_partition_is_fresh():
    manifest = driver._candidate_manifest()
    assert tuple(item["num_leapfrog_steps"] for item in manifest["candidates"]) == (
        5,
        9,
        12,
        13,
        14,
        17,
        18,
        19,
        24,
        25,
    )
    partition = driver._fresh_partition(tf, 6)
    assert partition["tuning_draws_reused"] is False
    assert partition["tuning_seed_disjoint"] is True
    assert len(partition["partition_signature"]) == 64


def test_bound_adapter_forwards_target_status_telemetry():
    base = driver._load_operational_module()
    adapter = base.build_pp_ukf_bound_adapter()
    status = adapter.target_status_telemetry(tf.zeros((2, 6), tf.float64))
    assert set(status) == {"status_code", "valid_pre_regularized_score"}
    assert tuple(status["status_code"].shape) == (2,)
    assert tuple(status["valid_pre_regularized_score"].shape) == (2,)


def test_progress_payload_preserves_resumed_rows_at_terminal_checkpoint():
    prior_rows = [{"candidate": {"candidate_id": "prior"}}]
    rows = [{"candidate": {"candidate_id": "new"}}]

    payload = driver._progress_payload(
        prior_rows=prior_rows,
        rows=rows,
        planned_candidate_count=10,
        elapsed_seconds=12.5,
        terminal=True,
    )

    assert payload["completed_candidate_count"] == 2
    assert payload["planned_candidate_count"] == 10
    assert payload["candidate_rows"] == tuple(prior_rows + rows)
    assert payload["elapsed_seconds"] == 12.5
    assert payload["terminal"] is True


def test_progress_payload_does_not_mutate_resumed_rows():
    prior_rows = [{"candidate": {"candidate_id": "prior"}}]
    rows: list[dict[str, object]] = []

    driver._progress_payload(
        prior_rows=prior_rows,
        rows=rows,
        planned_candidate_count=10,
        elapsed_seconds=0.0,
        terminal=False,
    )

    assert prior_rows == [{"candidate": {"candidate_id": "prior"}}]
    assert rows == []


def test_progress_counts_inflight_replacement_as_incomplete():
    rows = [
        {"candidate": {"candidate_id": "terminal"}},
        {
            "candidate": {"candidate_id": "continuing"},
            "terminal_candidate_result": False,
        },
    ]

    payload = driver._progress_payload(
        prior_rows=rows,
        rows=[],
        planned_candidate_count=10,
        elapsed_seconds=1.0,
        terminal=False,
    )

    assert payload["completed_candidate_count"] == 1
    assert len(payload["candidate_rows"]) == 2


def test_replacement_merge_preserves_all_ten_rows_and_replaces_only_selected():
    prior = [
        {"candidate": {"candidate_id": f"c{index}"}, "passed": index % 2 == 0}
        for index in range(10)
    ]
    replacement = [
        {"candidate": {"candidate_id": "c1"}, "passed": True},
        {"candidate": {"candidate_id": "c5"}, "passed": True},
    ]

    merged = driver._merge_replacement_rows(prior, replacement)

    assert len(merged) == 10
    assert [row["candidate"]["candidate_id"] for row in merged] == [f"c{i}" for i in range(10)]
    assert merged[1]["passed"] is True
    assert merged[5]["passed"] is True
    assert merged[0] is prior[0]


def test_attempt09_replacement_contract_validates_real_prefixes():
    progress = ROOT / "docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-09/progress.json"

    payload, prefixes = driver._validate_replacement_contract(progress, (1, 2, 5))

    assert len(payload["candidate_rows"]) == 10
    assert tuple(prefixes) == (1, 2, 5)
    assert all(item["prefix_results_per_chain"] == 3000 for item in prefixes.values())
    assert all(item["next_retained_chunk_index"] == 6 for item in prefixes.values())


def test_real_prefix_cumulative_tensors_equal_archived_chunks():
    progress = ROOT / "docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-09/progress.json"
    _, prefixes = driver._validate_replacement_contract(progress, (1, 2, 5))

    for prefix in prefixes.values():
        latent, raw = driver._load_verified_continuation_tensors(prefix, tf)
        assert tuple(latent.shape) == (3000, 4, 6)
        assert tuple(raw.shape) == (3000, 4, 6)


def test_continuation_row_preserves_prefix_evidence_and_marks_inflight():
    progress = ROOT / "docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-09/progress.json"
    _, prefixes = driver._validate_replacement_contract(progress, (1, 2, 5))
    prefix = prefixes[1]

    class Config:
        @staticmethod
        def payload(*, chain_count):
            return {"chain_count": chain_count, "retained_max_results": 10_000}

    row = driver._continuation_row(
        candidate=prefix["source_row"]["candidate"],
        config=Config(),
        prefix=prefix,
        continuation={
            "passed": False,
            "decision": "INCOMPLETE_SEQUENTIAL_FIXED_NEUTRA_HMC_KERNEL",
            "completion_status": "continuation_in_progress",
            "retained_passed": False,
            "retained_cap_hit": False,
            "retained_results_per_chain": 3500,
            "retained_checks": ({"chunk_index": 6},),
            "hard_vetoes": (),
            "elapsed_seconds": 2.0,
        },
        new_archive=({"stage": "retained", "chunk_index": 6},),
    )

    source = prefix["source_row"]
    assert row["terminal_candidate_result"] is False
    assert row["warmup_checks"] == tuple(source["warmup_checks"])
    assert len(row["retained_checks"]) == len(source["retained_checks"]) + 1
    assert len(row["archive"]) == len(source["archive"]) + 1
    assert row["archive"][-1]["continuation_archive_origin"] == "current_attempt"


def test_pp_ukf_route_uses_public_shared_hmc_controllers_only():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "run_retained_neutra_hmc_continuation" in source
    assert "run_sequential_neutra_hmc" in source
    assert "_build_batched_hmc_program" not in source
    assert "_summarize_batched_hmc_output" not in source
    assert "ContinuationChunkConfig" not in source


def test_cli_persists_structured_failure_after_output_root_creation(tmp_path):
    output = tmp_path / "attempt"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-root",
            str(output),
            "--candidate-index",
            "0",
        ],
        cwd=ROOT,
        env={"CUDA_VISIBLE_DEVICES": "-1"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    failure = output / "failure.json"
    assert failure.is_file()
    payload = __import__("json").loads(failure.read_text(encoding="ascii"))
    assert payload["status"] == "harness_exception"
    assert payload["exception_type"]
    assert "traceback" in payload
