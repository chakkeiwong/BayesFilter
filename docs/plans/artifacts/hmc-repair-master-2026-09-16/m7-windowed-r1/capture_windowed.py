"""Before/after windowed-stage reference checks; no posterior or GPU claim."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
from bayesfilter.inference import hmc_kernel_tuning as stage
from tests.test_hmc_kernel_tuning_windowed_mass import (
    _ToyGaussianAdapter, _MismatchedAdapter, _geometry, _bootstrap, _stage_config,
    _runtime_shaped_result, _fake_result, _operational_inputs, _operational_budget,
)
from bayesfilter.hmc_route_contract import OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID


def serial(value):
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def scripted_cases():
    records = {}
    for name, cfg, runner in (
        ("passed", {}, lambda: _runtime_shaped_result()),
        ("identity", {"mass_policy": "fixed_identity"}, lambda: _runtime_shaped_result()),
        ("nonfinite_draw", {}, lambda: _runtime_shaped_result(finite_samples=False)),
        ("nonfinite_ratio", {}, lambda: _runtime_shaped_result(finite_log_accept=False)),
        ("nonfinite_target", {}, lambda: _runtime_shaped_result(finite_target_log_prob=False)),
        ("fixture_runtime", {}, lambda: _fake_result()),
        ("default_acceptance", {}, lambda: _runtime_shaped_result(acceptance_trace=[True]*12)),
        ("missing_acceptance", {}, lambda: replace(_runtime_shaped_result(), trace={})),
        ("timeout", {"public_timeout_budget_s": 1., "public_timeout_started_perf_counter_s": 1.},
         lambda: _runtime_shaped_result()),
    ):
        calls, progress = [], []
        def run(adapter, state, config):
            calls.append({"adapter_signature": adapter.adapter_signature(),
                          "state": serial(state), "config": config.signature_payload()})
            return runner()
        with patch.object(stage.time, "perf_counter", return_value=100.):
            result = stage.run_hmc_windowed_mass_stage(
                adapter=_ToyGaussianAdapter(), geometry=_geometry(), bootstrap=_bootstrap(),
                config=_stage_config(**cfg), run_full_chain=run,
                _progress_callback=lambda event, payload: progress.append([event, payload]),
            )
        records[name] = {"payload": result.payload(), "artifact_hash": result.artifact_hash,
                         "calls": calls, "progress": progress}
    def failed(*args, **kwargs):
        raise RuntimeError("fixed diagnostic failure")
    with patch.object(stage.time, "perf_counter", return_value=100.):
        failure = stage.run_hmc_windowed_mass_stage(adapter=_ToyGaussianAdapter(),
            geometry=_geometry(), bootstrap=_bootstrap(), config=_stage_config(), run_full_chain=failed)
    records["runner_error"] = {"payload": failure.payload(), "artifact_hash": failure.artifact_hash}
    for name, call in (
        ("adapter_mismatch", lambda: stage.run_hmc_windowed_mass_stage(
            adapter=_MismatchedAdapter(), geometry=_geometry(), bootstrap=_bootstrap(), config=_stage_config())),
        ("invalid_metric_policy", lambda: _stage_config(metric_update_requirement="unknown")),
        ("incompatible_metric_policy", lambda: _stage_config(metric_update_requirement="require_operational_update")),
        ("invalid_timeout", lambda: _stage_config(public_timeout_budget_s=-1.)),
    ):
        try:
            call()
        except Exception as exc:
            records[name] = {"exception": type(exc).__name__, "message": str(exc)}
        else:
            raise AssertionError(f"{name} failed to reject")
    return serial(records)


def real_reference():
    adapter, geometry, bootstrap = _operational_inputs()
    result = stage.run_hmc_windowed_mass_stage(adapter=adapter, geometry=geometry, bootstrap=bootstrap,
        config=_stage_config(algorithm_id=OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
                             chain_execution_mode="tf_function"),
        _attempt_budget_policy=_operational_budget())
    assert result.passed, result.final_status
    operational = result.operational_warmup_result
    handoff = stage.build_operational_fixed_mass_hmc_adapter(
        adapter=adapter, geometry=geometry, windowed_stage=result,
        target_scope="kernel_windowed_mass_toy_gaussian")
    # Timings and hashes containing timing values are not numerical parity targets.
    windows = [{name: getattr(window, name) for name in (
        "adaptation_canonical_states", "consumed_step_size_trace", "log_accept_ratio",
        "is_accepted", "target_log_prob", "coordinate_signature_used", "metric_signature_used",
    )} for window in operational.windows]
    probes = [[0., 0.], [.2, -.4]]
    values, scores = handoff["final_adapter"].log_prob_and_grad(probes)
    return serial({"status": result.final_status, "seed_report": result.seed_report,
        "config": result.config.payload(), "geometry_hash": geometry.artifact_hash,
        "adapted_mass": handoff["adapted_mass_artifact"].to_payload(include_arrays=True),
        "adapted_mass_signature": handoff["adapted_mass_artifact_signature"],
        "final_adapter_signature": handoff["final_adapter_signature"],
        "initial_position": handoff["initial_position"], "lineage": handoff["start_lineage"],
        "final_theta": operational.final_kernel_state.canonical_theta,
        "final_epsilon": operational.final_kernel_state.epsilon,
        "transform_signature": operational.final_kernel_state.transform.signature,
        "private_start_bank": operational.private_start_bank_theta,
        "windows": windows, "probe_values": values, "probe_scores": scores})


parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
parser.add_argument("--compare", type=Path)
parser.add_argument("--real", action="store_true")
args = parser.parse_args()
payload = real_reference() if args.real else scripted_cases()
args.output.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
if args.compare:
    assert payload == json.loads(args.compare.read_text()), "windowed preparation changed"
print(json.dumps({"kind": "real_reference" if args.real else "scripted",
                  "exact_parity": bool(args.compare), "record_count": len(payload)}))
