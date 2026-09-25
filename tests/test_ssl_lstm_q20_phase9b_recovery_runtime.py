"""CPU reference tests for Phase 0 durable controller recovery and health."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import tensorflow as tf

from bayesfilter.inference import neutra_hmc
from bayesfilter.inference.hmc_convergence import RankNormalizedHMCThresholds, rank_normalized_hmc_diagnostics
from bayesfilter.runtime.durable_tensor_checkpoint import CheckpointError, DurableTensorCheckpoint


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("phase9b_recovery_tests", ROOT / "docs/benchmarks/run_ssl_lstm_q20_phase9b_recovery_runtime_2026_09_09.py")
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class GaussianAdapter:
    @staticmethod
    def adapter_signature():
        return "cpu_gaussian_checkpoint_reference"

    @staticmethod
    def log_prob_and_grad(values):
        return -0.5 * tf.reduce_sum(tf.square(values), axis=-1), -values

    @staticmethod
    def target_status_telemetry(values):
        shape = tf.shape(values)[:-1]
        return {"status_code": tf.zeros(shape, tf.int32),
                "valid_pre_regularized_score": tf.ones(shape, tf.bool)}


def test_actual_shared_controller_checkpoint_replay_matches_uninterrupted(tmp_path):
    config = neutra_hmc.SequentialNeuTraHMCConfig(
        step_size=0.4, num_leapfrog_steps=2, warmup_seed=(20260909, 1), retained_seed=(20260909, 2),
        warmup_chunk_results=4, warmup_min_results=12, warmup_check_window_results=4,
        warmup_max_results=12, retained_chunk_results=4, retained_min_results=4, retained_max_results=4,
    )
    initial = tf.zeros((4, 2), tf.float64)

    def execute(root, event_callback=None):
        allowed = iter((True, True, False))
        with DurableTensorCheckpoint(root, {"identity": "gaussian"}, event_callback=event_callback) as store:
            result = neutra_hmc.run_sequential_neutra_hmc(
                adapter=GaussianAdapter(), initial_state=initial, parameter_names=("first", "second"),
                config=config, budget_check=lambda count: next(allowed), checkpoint_store=store,
            )
            return result, list(store.records)

    reference, _ = execute(tmp_path / "reference")

    def stop_before_second_commit(event, payload):
        if event == "tensors-written" and payload["key"] == "chunk-000001":
            raise InterruptedError("synthetic partial write")

    with pytest.raises(InterruptedError):
        execute(tmp_path / "resumed", stop_before_second_commit)
    resumed, records = execute(tmp_path / "resumed")
    assert records[0]["replayed"] is True
    assert records[1]["replayed"] is False
    assert bool(tf.reduce_all(reference["private_warmup_z"] == resumed["private_warmup_z"]).numpy())
    assert resumed["warmup_results_per_chain"] == 8
    assert resumed["retained_results_per_chain"] == 0
    assert resumed["hard_vetoes"] == ("campaign_resource_cap",)


def _health_fixture():
    samples = tf.reshape(tf.range(32, dtype=tf.float64), (2, 4, 4)) / 100.0
    trace = {"is_accepted": tf.ones((2, 4), tf.bool), "log_accept_ratio": tf.zeros((2, 4), tf.float64),
             "target_log_prob": tf.ones((2, 4), tf.float64),
             "target_status": {"status_code": tf.zeros((2, 4), tf.int32),
                               "valid_pre_regularized_score": tf.ones((2, 4), tf.bool)}}
    config = neutra_hmc.BatchedHMCConfig(num_results=2, num_burnin_steps=0, step_size=0.05,
                                        num_leapfrog_steps=3, seed=(1, 2), jit_compile=True)
    return samples, trace, tf.zeros((4, 4), tf.float64), config


@pytest.mark.parametrize("case", ("state", "target", "acceptance", "status", "score", "movement", "floor", "eigen"))
def test_complete_health_rejects_invalid_traces(case):
    samples, trace, initial, config = _health_fixture()
    if case == "state":
        samples = tf.fill(samples.shape, tf.constant(float("nan"), tf.float64))
    elif case == "target":
        trace["target_log_prob"] = tf.fill((2, 4), tf.constant(float("nan"), tf.float64))
    elif case == "acceptance":
        trace["log_accept_ratio"] = tf.fill((2, 4), tf.constant(float("inf"), tf.float64))
    elif case == "status":
        trace["target_status"]["status_code"] = tf.ones((2, 4), tf.int32)
    elif case == "score":
        trace["target_status"]["valid_pre_regularized_score"] = tf.zeros((2, 4), tf.bool)
    elif case == "movement":
        samples = tf.zeros_like(samples)
    elif case == "floor":
        trace["target_status"]["floor_count_value"] = tf.ones((2, 4), tf.int32)
    elif case == "eigen":
        trace["target_status"]["min_innovation_eigenvalue"] = -tf.ones((2, 4), tf.float64)
    assert runner.full_health(tf, samples, trace, initial, config)["passed"] is False


def test_health_requires_complete_status_and_finite_extremes_are_not_divergences():
    samples, trace, initial, config = _health_fixture()
    trace["log_accept_ratio"] = tf.fill((2, 4), tf.constant(-1e6, tf.float64))
    receipt = runner.full_health(tf, samples, trace, initial, config)
    assert receipt["passed"] is True
    assert receipt["energy"]["native_divergences"] is None
    del trace["target_status"]
    with pytest.raises(CheckpointError, match="missing"):
        runner.full_health(tf, samples, trace, initial, config)


def _typed_tuning_fixture(diagnostic_value=float("nan")):
    from bayesfilter.inference.fixed_transport_hmc_tuning_tf import (
        FixedTransportHMCCandidateResult, FixedTransportHMCKernelTuningConfig,
        FixedTransportHMCKernelTuningResult,
    )

    config = FixedTransportHMCKernelTuningConfig(
        initial_step_size=0.1, step_size_candidates=(0.05, 0.1), leapfrog_grid=(2, 4),
        target_scope="cpu-serializer-test",
    )
    viable = FixedTransportHMCCandidateResult(
        candidate_index=0, num_leapfrog_steps=2, ladder_result={"selected_step_size": 0.1},
        verification_config_payload={}, verification_diagnostics={"tail_ess": 16.0},
        final_status="passed", diagnostic_role="cpu_serializer_fixture_not_admission",
    )
    rejected = replace(viable, candidate_index=1, num_leapfrog_steps=4,
                       verification_diagnostics={"tail_ess": diagnostic_value}, final_status="failed",
                       hard_vetoes=("nonfinite_convergence_diagnostic",))
    return FixedTransportHMCKernelTuningResult(
        config=config, transformed_adapter_signature="transformed", base_adapter_signature="base",
        fixed_transport_manifest_hash="transport", target_dimension=1,
        identity_z_mass_artifact_payload={}, identity_z_mass_artifact_signature="mass",
        candidates=(viable, rejected), selected_candidate_index=0, final_status="passed",
        final_kernel_payload={"step_size": 0.1}, tuning_scope_payload={}, route_record_payload={},
        coordinate_payload={}, source_dependency_closure={},
        candidate_selection_payload={"diagnostics": {"min_tail_ess": diagnostic_value},
                                     "replications": [(diagnostic_value, {"upper_tail_ess": diagnostic_value})]},
        hard_vetoes=("candidate_1_selection_replication_1_verification_selection_efficiency_nonfinite",),
    )


@pytest.mark.parametrize("marker", ("nan", "inf", "-inf"))
def test_typed_tuning_checkpoint_round_trips_nonfinite_diagnostic_values(tmp_path, marker):
    result = _typed_tuning_fixture(float(marker))
    encoded = runner.typed_tuning_payload(result)
    assert encoded["fields"]["candidates"]["tuple_items"][1]["verification_diagnostics"]["tail_ess"] == {
        "__nonfinite__": marker
    }
    with DurableTensorCheckpoint(tmp_path / "typed", {"identity": "typed-tuning"}) as store:
        store.run("tuning", {}, lambda: encoded)
    with DurableTensorCheckpoint(tmp_path / "typed", {"identity": "typed-tuning"}) as store:
        stored = store.run("tuning", {}, lambda: pytest.fail("committed result must replay"))
        assert store.records[0]["replayed"] is True
    restored = runner.restore_tuning(stored)
    assert repr(restored.candidates[1].verification_diagnostics["tail_ess"]) == marker
    assert repr(restored.candidate_selection_payload["diagnostics"]["min_tail_ess"]) == marker
    assert isinstance(restored.candidate_selection_payload["replications"][0], tuple)
    assert restored.candidates[1].hard_vetoes == result.candidates[1].hard_vetoes
    assert restored.hard_vetoes == result.hard_vetoes
    assert restored.selected_candidate_index == 0
    assert restored.selected_candidate.passed is True
    assert restored.candidates[1].passed is False
    assert encoded["payload_hash"] == result.artifact_hash
    assert runner.typed_tuning_payload(restored)["payload_hash"] == encoded["payload_hash"]


def test_legacy_finite_tuning_checkpoint_retains_public_hash(tmp_path):
    result = _typed_tuning_fixture(16.0)
    legacy = {"fields": runner._encode_tuning_value(asdict(result)),
              "payload_hash": runner.payload_hash(result.payload())}
    with DurableTensorCheckpoint(tmp_path, {"identity": "legacy"}) as store:
        store.run("tuning", {}, lambda: legacy)
    with DurableTensorCheckpoint(tmp_path, {"identity": "legacy"}) as store:
        stored, _ = store.load("tuning", {})
    restored = runner.restore_tuning(stored)
    assert restored == result
    assert restored.artifact_hash == legacy["payload_hash"]
    assert runner.typed_tuning_payload(restored)["payload_hash"] == legacy["payload_hash"]


def test_typed_tuning_rejects_changed_veto_and_invalid_markers():
    payload = runner.typed_tuning_payload(_typed_tuning_fixture())
    payload["fields"]["hard_vetoes"] = {"tuple_items": []}
    with pytest.raises(CheckpointError, match="differs"):
        runner.restore_tuning(payload)
    payload = runner.typed_tuning_payload(_typed_tuning_fixture())
    payload["fields"]["candidate_selection_payload"]["diagnostics"]["min_tail_ess"] = {"__nonfinite__": "0"}
    with pytest.raises(CheckpointError, match="unknown nonfinite"):
        runner.restore_tuning(payload)
    with pytest.raises(CheckpointError, match="schema"):
        runner.restore_tuning({"schema": "unsupported"})


def test_degenerate_tail_ess_remains_a_hard_veto():
    values = tf.zeros((16, 4, 1), tf.float64)
    flat = tf.reshape(values, [-1])
    flat = tf.tensor_scatter_nd_update(flat, [[0], [1]], [1.0, 1.0])
    draws = tf.reshape(flat, (16, 4, 1))
    diagnostics = rank_normalized_hmc_diagnostics(
        draws, parameter_names=("observation_bias.0",),
        thresholds=RankNormalizedHMCThresholds(rhat_max=3.0, bulk_ess_min=1.0, tail_ess_min=1.0),
    )
    row = diagnostics["parameter_diagnostics"][0]
    assert diagnostics["input_all_finite"] is True
    assert diagnostics["diagnostics_all_finite"] is False
    assert diagnostics["hard_vetoes"] == ("nonfinite_convergence_diagnostic",)
    assert math.isnan(row["upper_tail_ess"])
    assert diagnostics["passed"] is False


def test_forecast_uses_eight_hour_cap_and_rejects_short_fixture():
    result = {"setup": {"stages": [{"compute_seconds": 300, "serialization_seconds": 1}], "bridge_seconds": 1},
              "elapsed_seconds": 2850, "controller": {"chunk_results": 500, "controller_elapsed_seconds": 2800,
              "chunks": [{"compute_seconds": 1360}, {"compute_seconds": 1390}]}}
    forecast = runner.forecast(result)
    assert forecast["arm_cap_seconds"] == 28800
    assert forecast["fits_arm_cap"] is True
    result["controller"]["chunk_results"] = 8
    with pytest.raises(CheckpointError, match="500"):
        runner.forecast(result)


def test_partial_round_trip_movement_counts_as_movement():
    samples, trace, initial, config = _health_fixture()
    samples = tf.concat((samples[:1], initial[None]), axis=0)
    result = neutra_hmc._summarize_batched_hmc_output(initial_state=initial, samples=samples, trace=trace,
                    config=config, chain_count=4, elapsed_seconds=0.0)
    assert result["diagnostics"]["all_states_moved"] is True


@pytest.mark.parametrize("prepared,observed,expected", ((False, False, 0.0), (True, False, 100.0), (True, True, 12.5)))
def test_lost_parent_reconciliation_preserves_budget_and_is_idempotent(tmp_path, prepared, observed, expected):
    from bayesfilter.runtime.campaign_budget_ledger import CampaignBudgetLedger

    output = tmp_path / "launches" / "attempt"
    output.mkdir(parents=True)
    ledger = CampaignBudgetLedger.create(tmp_path / "budget.json", campaign_id="test", total_budget_seconds=200,
                     source_hash="source", plan_hash="plan", claim_boundary="cpu_accounting_test")
    binding = {"source_hash": "source", "plan_hash": "plan"}
    ledger.start_attempt(attempt_id="attempt", output_root=output, seed_namespace={}, **binding)
    ledger.reserve_chunk(attempt_id="attempt", arm="factor", chunk_index=0, reserve_seconds=100,
                         output_root=output, seed_namespace={}, **binding)
    if prepared:
        (output / "factor-job.json").write_text("{}")
    if observed:
        supervision = output / "factor-supervision"
        supervision.mkdir()
        (supervision / "parallel_worker_result.json").write_text(json.dumps({"elapsed_seconds": 12.5}))
    runner.reconcile_interrupted_supervisor(tmp_path, ledger)
    runner.reconcile_interrupted_supervisor(tmp_path, ledger)
    assert ledger.read()["consumed_seconds"] == expected
    assert ledger.read()["reserved_seconds"] == 0


@pytest.mark.parametrize("migrated", (False, True))
def test_retry_reuses_successful_sibling_and_selects_pending_gpu(tmp_path, monkeypatch, migrated):
    from bayesfilter.runtime import display_gpu_policy, parallel_tuning
    from bayesfilter.runtime.campaign_budget_ledger import CampaignBudgetLedger

    prior = tmp_path / "launches" / "resume-old"
    prior.mkdir(parents=True)
    manifest = prior / "factor-result.json"
    manifest.write_text(json.dumps({"job": {"mode": "resume", "campaign_root": str(tmp_path)}}))
    factor = {"status": "completed", "required_artifact": str(manifest),
              "required_artifact_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "task": {"task_id": "factor", "gpu_uuid": "GPU-factor", "output_dir": str(prior / "factor"), "command": []},
              "elapsed_seconds": 10.0}
    (prior / "summary.json").write_text(json.dumps({"results": {"factor": factor}}))
    (tmp_path / "strict-gpu.json").write_text(json.dumps({"uuid": "GPU-strict"}))
    (tmp_path / "strict-interruption-stream.json").write_text(json.dumps({"branch": "interrupted-old"}))
    start = {"sources": {}, "plan_hash": "plan"}
    execution_sources = {"wrapper.py": "repair"} if migrated else {}
    migration = {"from_sources": {}, "to_sources": execution_sources} if migrated else None
    monkeypatch.setattr(runner, "verify_binding", lambda start, campaign=None: None)
    monkeypatch.setattr(runner, "execution_source_binding", lambda start, campaign: (execution_sources, migration))
    ledger = CampaignBudgetLedger.create(tmp_path / "budget.json", campaign_id="test", total_budget_seconds=1000,
                     initial_consumed_seconds=10, source_hash=runner.payload_hash({}), plan_hash="plan", claim_boundary="cpu_test")
    monkeypatch.setattr(runner, "_validate_worker", lambda *args: None)
    monkeypatch.setattr(display_gpu_policy, "probe_inventory", lambda: {})

    def select(snapshot, count, **kwargs):
        assert count == 2
        return {"selected": [{"uuid": "GPU-factor"}, {"uuid": "GPU-strict"}]}

    def run_wave(tasks, **kwargs):
        assert len(tasks) == 1
        assert tasks[0].task_id == "strict"
        assert tasks[0].gpu_uuid == "GPU-strict"
        job = json.loads(Path(tasks[0].command[-1]).read_bytes())
        assert job["sources"] == execution_sources
        assert job["setup_sources"] == job["stream_sources"] == job["binding_sources"] == start["sources"]
        assert job.get("source_migration") == migration
        return {"status": "completed", "results": [{"status": "completed", "returncode": 0,
                 "elapsed_seconds": 20.0, "task": tasks[0].payload()}]}

    monkeypatch.setattr(display_gpu_policy, "select_gpus", select)
    monkeypatch.setattr(parallel_tuning, "run_parallel_tuning_wave", run_wave)
    result = runner._wave(tmp_path, "resume", 400, ledger, start)
    assert result["passed"]
    assert ledger.read()["consumed_seconds"] == 30
    assert result["results"]["factor"] == factor


@pytest.fixture
def tiny_p1(tmp_path, monkeypatch):
    config_factory = neutra_hmc.SequentialNeuTraHMCConfig

    def small_config(**kwargs):
        kwargs.update(warmup_chunk_results=8, warmup_min_results=32, warmup_check_window_results=16,
                      warmup_max_results=32, retained_chunk_results=8, retained_min_results=16,
                      retained_max_results=16)
        return config_factory(**kwargs)

    monkeypatch.setattr(neutra_hmc, "SequentialNeuTraHMCConfig", small_config)
    monkeypatch.setattr(neutra_hmc, "rank_normalized_split_rhat_summary",
                        lambda *args, **kwargs: {"passed": True, "test_role": "forced_stage_transition_not_convergence_evidence"})
    chart = SimpleNamespace(forward_batch=lambda values: values)
    handoff = SimpleNamespace(step_size=0.4, num_leapfrog_steps=2, handoff_hash="cpu-test",
                              transformed_adapter=GaussianAdapter())
    profile = SimpleNamespace(initial_state_bank=[[0.0] * 4] * 4)

    def execute(branch):
        job = {"campaign_root": str(tmp_path), "mode": "p1", "arm": "factor", "stream_branch": branch,
               "sources": {}, "gpu_uuid": "intentionally_hidden_cpu_reference", "cap_seconds": 3600,
               "next_chunk_reserve_seconds": 1, "output_dir": str(tmp_path / f"output-{branch}")}
        return runner._run_controller(tf, job, chart, handoff, profile)

    return execute


@pytest.mark.parametrize("interrupt_index", (2, 3, 4))
def test_actual_p1_replays_third_fourth_and_first_retained_chunks(tiny_p1, monkeypatch, interrupt_index):
    reference = tiny_p1("reference")
    original_event = DurableTensorCheckpoint._event

    def interrupt(store, attempt, name, payload):
        original_event(store, attempt, name, payload)
        if store.root.name == "chunks" and name == "tensors-written" and payload["key"] == f"chunk-{interrupt_index:06d}":
            raise InterruptedError("CPU interruption before publication")

    with monkeypatch.context() as patch:
        patch.setattr(DurableTensorCheckpoint, "_event", interrupt)
        with pytest.raises(InterruptedError):
            tiny_p1("resumed")
    resumed = tiny_p1("resumed")
    assert resumed["sample_sha256"] == reference["sample_sha256"]
    assert resumed["retained_sample_sha256"] == reference["retained_sample_sha256"]
    assert [row["replayed"] for row in resumed["chunks"]] == [index < interrupt_index for index in range(6)]
    assert [row["checkpoint_index"] for row in resumed["health"]] == list(range(6))
    assert [row["stage_chunk_index"] for row in resumed["health"]] == [0, 1, 2, 3, 0, 1]
    assert [row["stage"] for row in resumed["health"]] == ["warmup"] * 4 + ["retained"] * 2
    assert resumed["health"][4]["seed"] != resumed["health"][0]["seed"]
    for index in range(6):
        bundles = [json.loads((Path(row["stream_root"]) / "chunks" / "committed" / f"chunk-{index:06d}" / "bundle.json").read_bytes())
                   for row in (reference, resumed)]
        assert bundles[0]["tree"] == bundles[1]["tree"]
        assert bundles[0]["inputs_hash"] == bundles[1]["inputs_hash"]
    assert runner.p1_outcome(resumed) == "bounded_sequential_screen_passed_not_posterior_admission"


def test_p1_graceful_partial_resume_versions_cumulative_archives(tiny_p1, monkeypatch):
    original_run = neutra_hmc.run_sequential_neutra_hmc

    def stop_after_two(**kwargs):
        checks = iter((True, True, False))
        kwargs["budget_check"] = lambda count: next(checks)
        return original_run(**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(neutra_hmc, "run_sequential_neutra_hmc", stop_after_two)
        partial = tiny_p1("partial")
    assert runner.p1_outcome(partial) == "resource_cap_partial_not_completed"
    complete = tiny_p1("partial")
    archives = Path(complete["stream_root"]) / "archives" / "committed"
    assert (archives / "warmup-cumulative-16").is_dir()
    assert (archives / "warmup-cumulative-32").is_dir()
    assert [row["replayed"] for row in complete["chunks"]] == [True, True, False, False, False, False]


def test_p1_warmup_failure_is_a_terminal_candidate_result(tiny_p1, monkeypatch):
    monkeypatch.setattr(neutra_hmc, "rank_normalized_split_rhat_summary", lambda *args, **kwargs: {"passed": False})
    result = tiny_p1("failed-warmup")
    assert runner.p1_outcome(result) == "candidate_warmup_screen_failed"
    assert len(result["health"]) == 4
    assert result["sequential_result"]["retained_results_per_chain"] == 0


@pytest.fixture
def prepared_campaign(tmp_path, monkeypatch):
    from bayesfilter.runtime.campaign_budget_ledger import CampaignBudgetLedger

    monkeypatch.setattr(runner, "source_hashes", lambda: {"code.py": "final-source"})
    plan = tmp_path / "plan.md"
    plan.write_text("CPU initialization fixture")
    monkeypatch.setattr(runner, "PLAN", plan)
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    ledger = CampaignBudgetLedger.create(campaign / "campaign_budget_ledger.json", campaign_id=campaign.name,
                    total_budget_seconds=86400, source_hash="prepared-source", plan_hash="prepared-plan",
                    claim_boundary="phase9b_p0_recovery_runtime_8h_amendment_only")
    return campaign, ledger


def test_initialization_adopts_unused_ledger_without_new_allocation(prepared_campaign):
    campaign, ledger = prepared_campaign
    original = ledger.path.read_bytes()
    start, adopted = runner.initialize_campaign(campaign, resume=False)
    assert (campaign / "prepared-budget-ledger.json").read_bytes() == original
    assert adopted.path == ledger.path
    assert adopted.read()["total_budget_seconds"] == 86400
    assert adopted.read()["consumed_seconds"] == 0
    assert adopted.read()["attempts"] == []
    assert start["tuning_repair"] == 1
    before = ledger.path.read_bytes()
    runner.initialize_campaign(campaign, resume=True)
    assert ledger.path.read_bytes() == before


@pytest.fixture
def migrated_campaign(prepared_campaign, monkeypatch):
    campaign, ledger = prepared_campaign
    script_path = str(runner.SCRIPT.relative_to(ROOT))
    original_sources = {script_path: "original", "numerical.py": "unchanged",
                        runner.NUMERICAL_CORE_PATH: "original-core",
                        runner.ACCOUNTING_REPAIR_PATH: "original-ledger"}
    monkeypatch.setattr(runner, "source_hashes", lambda: original_sources)
    start, ledger = runner.initialize_campaign(campaign, resume=False)
    binding = {"source_hash": runner.payload_hash(original_sources), "plan_hash": start["plan_hash"]}
    ledger.start_attempt(attempt_id="prior", output_root=campaign / "launches/prior", seed_namespace={}, **binding)
    ledger.reserve_chunk(attempt_id="prior", arm="factor", chunk_index=0, reserve_seconds=60,
                         output_root=campaign / "launches/prior", seed_namespace={}, **binding)
    ledger.settle_arm(attempt_id="prior", arm="factor", measured_seconds=12, status="completed")
    ledger.finish_attempt(attempt_id="prior", status="completed")
    current = {**original_sources, script_path: "repaired"}
    monkeypatch.setattr(runner, "source_hashes", lambda: current)
    migration = {"schema": runner.SOURCE_MIGRATION_SCHEMA, "migration_id": runner.SOURCE_MIGRATION_ID,
                 "from_sources": original_sources, "to_sources": current, "changed_paths": [script_path],
                 "plan_hash": start["plan_hash"], "claim_boundary": runner.CLAIM_BOUNDARY,
                 "scientific_contract_unchanged": True, "serializer_only": True}
    runner.durable_json(campaign / runner.SOURCE_MIGRATION_FILENAME, migration)
    return campaign, ledger, start, current, migration


def test_source_migration_preserves_start_and_spent_ledger(migrated_campaign):
    campaign, ledger, start, current, migration = migrated_campaign
    original_start = (campaign / "campaign-start.json").read_bytes()
    original_ledger = ledger.path.read_bytes()
    for _attempt in range(2):
        restored_start, restored_ledger = runner.initialize_campaign(campaign, resume=True)
        assert restored_start == start
        assert restored_ledger.read()["consumed_seconds"] == 12
        assert runner.execution_source_binding(restored_start, campaign) == (current, migration)
    assert (campaign / "campaign-start.json").read_bytes() == original_start
    assert ledger.path.read_bytes() == original_ledger


@pytest.fixture
def numerical_campaign(migrated_campaign, monkeypatch):
    campaign, ledger, start, sources, migration = migrated_campaign
    current = {**sources, runner.NUMERICAL_CORE_PATH: "refined-eigen-core"}
    numerical = {**migration, "migration_id": runner.NUMERICAL_MIGRATION_ID,
                 "to_sources": current, "serializer_only": False,
                 "changed_paths": sorted([str(runner.SCRIPT.relative_to(ROOT)), runner.NUMERICAL_CORE_PATH]),
                 "execution_namespace": "numerical-repairs/eigh-refinement-r1"}
    monkeypatch.setattr(runner, "source_hashes", lambda: current)
    runner.durable_json(campaign / runner.SOURCE_MIGRATION_FILENAME, numerical)
    return campaign, ledger, start, current, numerical


@pytest.fixture
def accounting_campaign(numerical_campaign, monkeypatch):
    campaign, ledger, start, setup_sources, migration = numerical_campaign
    script_path = str(runner.SCRIPT.relative_to(ROOT))
    current = {**setup_sources, script_path: "accounting-wrapper",
               runner.ACCOUNTING_REPAIR_PATH: "repaired-ledger"}
    accounting = {**migration, "migration_id": runner.ACCOUNTING_MIGRATION_ID,
                  "to_sources": current, "reusable_setup_sources": setup_sources,
                  "changed_paths": sorted([*migration["changed_paths"], runner.ACCOUNTING_REPAIR_PATH])}
    monkeypatch.setattr(runner, "source_hashes", lambda: current)
    runner.durable_json(campaign / runner.SOURCE_MIGRATION_FILENAME, accounting)
    return campaign, ledger, start, current, accounting


def test_accounting_migration_restores_committed_setup_without_recomputing(accounting_campaign):
    campaign, ledger, start, current, migration = accounting_campaign
    before = ledger.path.read_bytes()
    execution = campaign / migration["execution_namespace"]
    setup_root = execution / "setup" / "factor"
    identity = {"sources": migration["reusable_setup_sources"], "profile": "fixed", "target": "fixed"}
    runner.initialize_campaign(campaign, resume=True)
    _, effective = runner.prepare_execution_namespace(campaign, start, current, migration)
    with DurableTensorCheckpoint(setup_root, identity) as store:
        store.run("chart", {}, lambda: tf.constant([1.0, 2.0], tf.float64))
    restored_identity = {**identity, "sources": effective["setup_sources"]}
    with DurableTensorCheckpoint(setup_root, restored_identity) as store:
        restored = store.run("chart", {}, lambda: pytest.fail("committed setup must be reused"))
        assert restored.numpy().tolist() == [1.0, 2.0]
        assert store.records[0]["replayed"] is True
    assert effective["stream_sources"] == current
    assert effective["sources"] == start["sources"]
    assert ledger.path.read_bytes() == before


@pytest.mark.parametrize("case", ("missing", "changed_core", "changed_target"))
def test_accounting_migration_rejects_unreviewed_setup_reuse(accounting_campaign, case):
    campaign, ledger, _, _, migration = accounting_campaign
    before = ledger.path.read_bytes()
    if case == "missing":
        migration.pop("reusable_setup_sources")
    elif case == "changed_core":
        migration["reusable_setup_sources"][runner.NUMERICAL_CORE_PATH] = "unreviewed-core"
    else:
        migration["reusable_setup_sources"]["numerical.py"] = "changed-target"
    runner.durable_json(campaign / runner.SOURCE_MIGRATION_FILENAME, migration)
    with pytest.raises(CheckpointError, match="setup"):
        runner.initialize_campaign(campaign, resume=True)
    assert ledger.path.read_bytes() == before


@pytest.mark.parametrize("prior_sources", ("execution", "setup"))
def test_accounting_retry_reuses_completed_sibling(accounting_campaign, monkeypatch, prior_sources):
    from bayesfilter.runtime import display_gpu_policy, parallel_tuning

    campaign, ledger, start, current, migration = accounting_campaign
    execution, effective = runner.prepare_execution_namespace(campaign, start, current, migration)
    setup_sources = migration["reusable_setup_sources"]
    prior = execution / "launches" / "resume-prior"
    manifest = prior / "factor-result.json"
    sources = current if prior_sources == "execution" else setup_sources
    runner.durable_json(manifest, {"job": {"mode": "resume", "campaign_root": str(execution),
                        "sources": sources, "setup_sources": setup_sources, "stream_sources": sources}})
    factor = {"status": "completed", "required_artifact": str(manifest),
              "required_artifact_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "task": {"gpu_uuid": "GPU-factor", "output_dir": str(prior / "factor"), "command": []}}
    runner.durable_json(prior / "summary.json", {"results": {"factor": factor}})
    runner.durable_json(execution / "strict-gpu.json", {"uuid": "GPU-strict"})
    runner.durable_json(execution / "strict-interruption-stream.json", {"branch": "interrupted-prior"})
    monkeypatch.setattr(runner, "_validate_worker", lambda *args: None)
    monkeypatch.setattr(display_gpu_policy, "probe_inventory", lambda: {})
    monkeypatch.setattr(display_gpu_policy, "select_gpus", lambda *args, **kwargs: {
        "selected": [{"uuid": "GPU-factor"}, {"uuid": "GPU-strict"}]})

    def run_wave(tasks, **kwargs):
        assert len(tasks) == 1
        task = tasks[0]
        assert task.task_id == "strict"
        job = json.loads(Path(task.command[-1]).read_bytes())
        assert job["sources"] == job["stream_sources"] == current
        assert job["setup_sources"] == setup_sources
        return {"status": "completed", "results": [{"status": "completed", "returncode": 0,
                "elapsed_seconds": 20.0, "task": task.payload()}]}

    monkeypatch.setattr(parallel_tuning, "run_parallel_tuning_wave", run_wave)
    result = runner._wave(execution, "resume", 400, ledger, effective)
    assert result["passed"]
    assert result["results"]["factor"] == factor
    assert ledger.read()["consumed_seconds"] == 32


def test_numerical_namespace_preserves_budget_and_refreshes_all_seed_families(numerical_campaign):
    campaign, ledger, start, current, migration = numerical_campaign
    before = ledger.path.read_bytes()
    original_start = (campaign / "campaign-start.json").read_bytes()
    restored, same_ledger = runner.initialize_campaign(campaign, resume=True)
    execution, effective = runner.prepare_execution_namespace(campaign, restored, current, migration)
    assert same_ledger.path == ledger.path
    assert effective["sources"] == start["sources"]
    assert effective["setup_sources"] == effective["stream_sources"] == current
    assert runner.execution_source_binding(effective, execution) == (current, migration)
    runner.verify_binding(effective, execution)
    parallel = runner.load_parallel()
    source = parallel._load_source()
    for arm in runner.ARMS:
        old_profile = parallel._profile(source, campaign, arm, runner.ARM_CAP_SECONDS)
        new_profile = parallel._profile(source, execution, arm, runner.ARM_CAP_SECONDS)
        for field in ("initialization_roots", "preflight_roots", "training_roots", "tuning_roots"):
            assert set(getattr(old_profile, field)).isdisjoint(getattr(new_profile, field))
        seeds = [runner.stream_seeds(root, arm, family) for root in (campaign, execution)
                 for family in ("canary", "runtime", "p1")]
        assert len(set(seeds)) == 6
    assert runner.prepare_execution_namespace(campaign, restored, current, migration) == (execution, effective)
    assert ledger.path.read_bytes() == before
    assert (campaign / "campaign-start.json").read_bytes() == original_start
    with pytest.raises(CheckpointError, match="original budget campaign"):
        runner.initialize_campaign(execution, resume=False)
    assert not (execution / "campaign_budget_ledger.json").exists()


@pytest.mark.parametrize("case", ("serializer_only", "missing_namespace", "parent_namespace", "old_namespace",
                                 "extra_source", "changed_core", "changed_plan"))
def test_numerical_migration_rejects_stale_or_unbound_scope(numerical_campaign, monkeypatch, case):
    campaign, ledger, start, current, migration = numerical_campaign
    before = ledger.path.read_bytes()
    if case == "serializer_only":
        migration["serializer_only"] = True
    elif case == "missing_namespace":
        migration.pop("execution_namespace")
    elif case == "parent_namespace":
        migration["execution_namespace"] = "numerical-repairs/.."
    elif case == "old_namespace":
        migration["execution_namespace"] = "streams/runtime"
    elif case == "extra_source":
        current["numerical.py"] = "undeclared-change"
        migration["changed_paths"] = sorted([*migration["changed_paths"], "numerical.py"])
    elif case == "changed_core":
        monkeypatch.setattr(runner, "source_hashes", lambda: {**current, runner.NUMERICAL_CORE_PATH: "unbound"})
    else:
        runner.PLAN.write_text("changed plan")
    runner.durable_json(campaign / runner.SOURCE_MIGRATION_FILENAME, migration)
    with pytest.raises(CheckpointError):
        runner.initialize_campaign(campaign, resume=True)
    assert ledger.path.read_bytes() == before


def test_numerical_namespace_rejects_existing_unbound_evidence(numerical_campaign):
    campaign, ledger, start, current, migration = numerical_campaign
    execution = campaign / migration["execution_namespace"]
    runner.durable_json(execution / "reference-complete.json", {"stale": True})
    with pytest.raises(CheckpointError, match="unbound prior evidence"):
        runner.prepare_execution_namespace(campaign, start, current, migration)
    assert ledger.read()["consumed_seconds"] == 12


def test_numerical_namespace_rejects_changed_execution_identity(numerical_campaign):
    campaign, ledger, start, current, migration = numerical_campaign
    runner.prepare_execution_namespace(campaign, start, current, migration)
    with pytest.raises(CheckpointError, match="binding changed"):
        runner.prepare_execution_namespace(campaign, start, {**current, "numerical.py": "changed"}, migration)
    assert ledger.read()["consumed_seconds"] == 12


def test_numerical_namespace_recovers_interrupted_initial_record(numerical_campaign, monkeypatch):
    campaign, ledger, start, current, migration = numerical_campaign
    original = runner.durable_json
    interrupted = []

    def write(path, payload):
        if path.name == "execution-start.json" and not interrupted:
            partial = path.with_name("execution-start.json.partial-interrupted-fixture")
            partial.write_bytes(b'{"incomplete":')
            interrupted.append(partial)
            raise InterruptedError("initial namespace record interrupted")
        original(path, payload)

    monkeypatch.setattr(runner, "durable_json", write)
    before = ledger.path.read_bytes()
    with pytest.raises(InterruptedError):
        runner.prepare_execution_namespace(campaign, start, current, migration)
    execution, effective = runner.prepare_execution_namespace(campaign, start, current, migration)
    assert (execution / "execution-start.json").is_file()
    assert interrupted[0].read_bytes() == b'{"incomplete":'
    runner.verify_binding(effective, execution)
    assert ledger.path.read_bytes() == before


def test_numerical_coordinator_ignores_old_markers_and_resumes_new_scope(numerical_campaign, monkeypatch):
    campaign, ledger, start, current, migration = numerical_campaign
    before = ledger.path.read_bytes()
    modes = []
    for mode in ("reference", "interrupt", "resume", "runtime", "p1"):
        runner.durable_json(campaign / f"{mode}-complete.json", {"stale": True})
    runner.durable_json(campaign / "phase0-readiness.json", {"stale": True})

    def wave(root, mode, cap, passed_ledger, effective, **kwargs):
        assert root == campaign / migration["execution_namespace"]
        assert passed_ledger.path == ledger.path
        assert effective["sources"] == start["sources"]
        assert effective["setup_sources"] == effective["stream_sources"] == current
        runner.verify_binding(effective, root)
        modes.append(mode)
        result = {"mode": mode, "results": {arm: {"arm": arm} for arm in runner.ARMS}}
        runner.durable_json(root / f"{mode}-complete.json", result)
        return result

    monkeypatch.setattr(runner, "_wave", wave)
    monkeypatch.setattr(runner, "_read_results", lambda wave: {arm: {"controller": {}} for arm in runner.ARMS})
    monkeypatch.setattr(runner, "compare_canary", lambda *args: {"passed": True, "exact_tensor_and_trace_equality": [True, True]})
    monkeypatch.setattr(runner, "forecast", lambda result: {
        "fits_arm_cap": True, "with_one_steady_chunk_reserve_and_grace_seconds": 5000,
        "first_chunk_seconds": 500, "steady_chunk_seconds": 500, "observed_overhead_per_chunk_seconds": 1})
    monkeypatch.setattr(runner, "p1_outcome", lambda controller: "candidate_warmup_screen_failed")
    assert runner.coordinator(campaign, resume=True, through_p1=True) == 0
    assert modes == ["reference", "interrupt", "resume", "runtime", "p1"]
    execution = campaign / migration["execution_namespace"]
    readiness = json.loads((execution / "phase0-readiness.json").read_bytes())
    assert readiness["execution_sources"] == current
    assert readiness["source_migration"] == migration
    modes.clear()
    assert runner.coordinator(campaign, resume=True, through_p1=True) == 0
    assert modes == []
    assert ledger.path.read_bytes() == before
    assert json.loads((campaign / "phase0-readiness.json").read_bytes()) == {"stale": True}


def test_numerical_wave_launches_both_arms_with_current_setup_and_original_budget(numerical_campaign, monkeypatch):
    from bayesfilter.runtime import display_gpu_policy, parallel_tuning

    campaign, ledger, start, current, migration = numerical_campaign
    execution, effective = runner.prepare_execution_namespace(campaign, start, current, migration)
    monkeypatch.setattr(display_gpu_policy, "probe_inventory", lambda: {})
    monkeypatch.setattr(display_gpu_policy, "select_gpus", lambda *args, **kwargs: {
        "selected": [{"uuid": "GPU-factor"}, {"uuid": "GPU-strict"}]})

    def run_wave(tasks, **kwargs):
        assert len(tasks) == 2
        results = []
        for task in tasks:
            job = json.loads(Path(task.command[-1]).read_bytes())
            assert job["campaign_root"] == str(execution)
            assert job["sources"] == job["setup_sources"] == job["stream_sources"] == current
            assert job["binding_sources"] == start["sources"]
            assert job["source_migration"] == migration
            results.append({"status": "completed", "returncode": 0,
                            "elapsed_seconds": 20.0, "task": task.payload()})
        return {"status": "completed", "results": results}

    monkeypatch.setattr(parallel_tuning, "run_parallel_tuning_wave", run_wave)
    result = runner._wave(execution, "reference", 3600, ledger, effective)
    assert result["passed"]
    assert ledger.read()["consumed_seconds"] == 52
    assert ledger.read()["source_hash"] == runner.payload_hash(start["sources"])


@pytest.mark.parametrize("case", ("missing", "malformed", "to_sources", "from_sources", "changed_paths",
                                 "plan_hash", "claim_boundary", "serializer_only", "scientific_contract_unchanged",
                                 "numerical_change", "plan_change"))
def test_source_migration_rejects_unbound_changes_without_spending(migrated_campaign, monkeypatch, case):
    campaign, ledger, start, current, migration = migrated_campaign
    path = campaign / runner.SOURCE_MIGRATION_FILENAME
    before = ledger.path.read_bytes()
    if case == "missing":
        path.unlink()
    elif case == "malformed":
        path.write_text("{")
    elif case == "plan_change":
        runner.PLAN.write_text("changed plan")
    elif case == "numerical_change":
        changed = {**current, "numerical.py": "different-target"}
        monkeypatch.setattr(runner, "source_hashes", lambda: changed)
        migration["to_sources"] = changed
        migration["changed_paths"] = sorted(changed)
        runner.durable_json(path, migration)
    else:
        runner.durable_json(path, {**migration, case: False})
    with pytest.raises(CheckpointError):
        runner.initialize_campaign(campaign, resume=True)
    assert ledger.path.read_bytes() == before
    assert json.loads((campaign / "campaign-start.json").read_bytes()) == start


@pytest.mark.parametrize("failure_file", ("campaign-initialization.json", "campaign_budget_ledger.json", "campaign-start.json"))
def test_initialization_recovers_after_each_atomic_write(prepared_campaign, monkeypatch, failure_file):
    campaign, ledger = prepared_campaign
    original = runner.durable_json

    def interrupted(path, payload):
        original(path, payload)
        if path.name == failure_file:
            raise InterruptedError("synthetic initialization interruption")

    with monkeypatch.context() as patch:
        patch.setattr(runner, "durable_json", interrupted)
        with pytest.raises(InterruptedError):
            runner.initialize_campaign(campaign, resume=False)
    runner.initialize_campaign(campaign, resume=True)
    assert ledger.read()["total_budget_seconds"] == 86400
    assert ledger.read()["consumed_seconds"] == ledger.read()["reserved_seconds"] == 0
    assert sum(row["event"] == "unused_prepared_ledger_bound_for_execution" for row in ledger.read()["events"]) == 1


@pytest.mark.parametrize("field,value", (("consumed_seconds", 1), ("total_budget_seconds", 172800),
                                       ("campaign_id", "different-campaign"), ("schema", "invalid")))
def test_initialization_rejects_used_wrong_or_malformed_ledger_without_mutation(prepared_campaign, field, value):
    campaign, ledger = prepared_campaign
    payload = dict(ledger.read())
    payload[field] = value
    runner.durable_json(ledger.path, payload)
    before = ledger.path.read_bytes()
    with pytest.raises(RuntimeError):
        runner.initialize_campaign(campaign, resume=False)
    assert ledger.path.read_bytes() == before
    assert not (campaign / "campaign-start.json").exists()
    assert not (campaign / "prepared-budget-ledger.json").exists()


def test_p1_allocations_require_canary_caps_and_aggregate_budget():
    canaries = {arm: {"passed": True, "exact_tensor_and_trace_equality": [True, True]} for arm in runner.ARMS}
    forecasts = {arm: {"fits_arm_cap": True, "with_one_steady_chunk_reserve_and_grace_seconds": 14001,
                       "first_chunk_seconds": 2100, "steady_chunk_seconds": 2000,
                       "observed_overhead_per_chunk_seconds": 10} for arm in runner.ARMS}
    allocations = runner.p1_allocations(canaries, forecasts, 86400)
    assert allocations["strict"]["cap_seconds"] == 14400
    with pytest.raises(CheckpointError, match="remaining"):
        runner.p1_allocations(canaries, forecasts, 28000)
    forecasts["strict"]["with_one_steady_chunk_reserve_and_grace_seconds"] = 28801
    with pytest.raises(CheckpointError, match="per-arm"):
        runner.p1_allocations(canaries, forecasts, 86400)
    canaries["factor"]["passed"] = False
    with pytest.raises(CheckpointError, match="recovery"):
        runner.p1_allocations(canaries, forecasts, 86400)


def test_diagnostic_and_p1_seed_families_are_disjoint(tmp_path):
    seeds = [runner.stream_seeds(tmp_path, arm, family) for arm in runner.ARMS for family in ("canary", "runtime", "p1")]
    assert len(set(seeds)) == 6


def test_postrun_verification_checks_nested_tensors(tmp_path):
    with DurableTensorCheckpoint(tmp_path / "store", {}) as store:
        store.run("nested", {}, lambda: {"first": (tf.constant([1]), {"second": tf.constant([2])})})
    assert runner.verify_campaign_bundles(tmp_path) == {"verified_bundles": 1, "verified_tensors": 2}
    tensor = next(tmp_path.glob("**/tensor-0001.bin"))
    tensor.write_bytes(b"corrupt")
    with pytest.raises(CheckpointError, match="tensor checksum"):
        runner.verify_campaign_bundles(tmp_path)


def test_coordinator_runs_canary_runtime_then_measured_parallel_p1(prepared_campaign, monkeypatch):
    campaign, ledger = prepared_campaign
    modes = []

    def wave(root, mode, cap, ledger, start, **kwargs):
        modes.append(mode)
        if mode == "p1":
            assert cap == {"factor": 14400, "strict": 18000}
            assert set(kwargs["allocations"]) == set(runner.ARMS)
            assert (root / "phase0-readiness.json").is_file()
        result = {"mode": mode, "results": {arm: {"arm": arm} for arm in runner.ARMS}}
        runner.durable_json(root / f"{mode}-complete.json", result)
        return result

    def forecast(result):
        return {"fits_arm_cap": True, "with_one_steady_chunk_reserve_and_grace_seconds":
                14000 if result["arm"] == "factor" else 17000,
                "first_chunk_seconds": 2000, "steady_chunk_seconds": 2000,
                "observed_overhead_per_chunk_seconds": 10}

    monkeypatch.setattr(runner, "_wave", wave)
    monkeypatch.setattr(runner, "_read_results", lambda wave: {arm: {"arm": arm, "controller": {}} for arm in runner.ARMS})
    monkeypatch.setattr(runner, "compare_canary", lambda *args: {"passed": True, "exact_tensor_and_trace_equality": [True, True]})
    monkeypatch.setattr(runner, "forecast", forecast)
    monkeypatch.setattr(runner, "p1_outcome", lambda row: "candidate_warmup_screen_failed")
    assert runner.coordinator(campaign, through_p1=True) == 0
    assert modes == ["reference", "interrupt", "resume", "runtime", "p1"]
    result = json.loads((campaign / "p1-result.json").read_bytes())
    assert result["p1_completed"] is True
    assert result["posterior_admitted"] is False
    assert result["p2_launched"] is False
    modes.clear()
    assert runner.coordinator(campaign, resume=True, through_p1=True) == 0
    assert modes == []


@pytest.mark.parametrize("cap", (float("nan"), float("inf"), 30, 28801))
def test_wave_rejects_invalid_per_arm_allocations_before_spending(prepared_campaign, monkeypatch, cap):
    campaign, ledger = prepared_campaign
    start, ledger = runner.initialize_campaign(campaign, resume=False)
    with pytest.raises(CheckpointError, match="caps"):
        runner._wave(campaign, "runtime", {"factor": 100, "strict": cap}, ledger, start)
    assert ledger.read()["attempts"] == []
