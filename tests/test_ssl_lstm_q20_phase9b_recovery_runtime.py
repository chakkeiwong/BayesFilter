"""CPU reference tests for Phase 0 durable controller recovery and health."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import tensorflow as tf

from bayesfilter.inference import neutra_hmc
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


def test_retry_reuses_successful_sibling_and_selects_pending_gpu(tmp_path, monkeypatch):
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
    monkeypatch.setattr(runner, "verify_binding", lambda start: None)
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
