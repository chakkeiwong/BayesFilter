"""Complete-cost and budget regressions; no framework or GPU initialization."""
import json

import pytest

from bayesfilter.inference import surrogate_hmc_program as program
from scripts.qualify_surrogate_hmc_nonlinear_endpoint import complete_forecast


def timings():
    return {name: [dict(compilation_included=i == 0, cpu_worker_seconds=float(i+1),
                       wall_seconds=float(i+2), healthy=True) for i in range(5)]
            for name in ("L1", "L2")}


REFERENCE_TIMINGS = {f"reference_fine_{i}": dict(cpu_worker_seconds=.01, wall_seconds=.02)
                     for i in range(1, 15)}


def test_forecast_covers_slowest_complete_L2_and_tuning():
    costs = timings()
    result = complete_forecast(costs, REFERENCE_TIMINGS)
    assert result["transitions_per_four_chain_batch"] == 3778
    assert result["trajectory_candidates"] == 2
    assert result["required"]["cpu_worker_seconds"] == 1.5*(3778*5+330+80)
    assert result["retained_looks"] == [1000, 2000]
    assert result["authorizes_inference"] is False
    costs["L2"][1]["cpu_worker_seconds"] = 9.
    assert complete_forecast(costs, REFERENCE_TIMINGS)["required"]["cpu_worker_seconds"] == 1.5*(3778*9+330+80)


@pytest.mark.parametrize("kind", ["missing", "short", "health", "nonfinite"])
def test_partial_or_invalid_cost_cannot_authorize_forecast(kind):
    costs = timings()
    if kind == "missing":
        del costs["L2"]
    elif kind == "short":
        costs["L2"].pop()
    elif kind == "health":
        costs["L2"][2]["healthy"] = False
    else:
        costs["L2"][2]["cpu_worker_seconds"] = float("nan")
    with pytest.raises(ValueError):
        complete_forecast(costs, REFERENCE_TIMINGS)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_retry_preserves_failed_attempt_and_spends_same_stage_allocation(tmp_path):
    out = tmp_path/"docs/plans/artifacts/surrogate-hmc-all-models-test-m10-endpoint-01"
    write(out/"accounting.json", dict(charged_this_run=dict(cpu_worker_seconds=110., gpu_seconds=120.)))
    write(out/"run-manifest.json", dict(command=["python", "qualify_surrogate_hmc_nonlinear_endpoint.py"]))
    write(out/"report.json", dict(status="EXECUTION_FAILED", qualified=False))
    state = program.m10_endpoint_state(tmp_path)
    assert state["qualified"] is None
    assert state["cpu_worker_seconds"] == 250.
    assert state["wall_seconds"] == 360.
    assert state["latest"]["status"] == "EXECUTION_FAILED"


def test_success_flag_without_required_checks_fails_closed(tmp_path):
    out = tmp_path/"docs/plans/artifacts/surrogate-hmc-all-models-test-m10-endpoint-01"
    write(out/"accounting.json", dict(charged_this_run=dict(cpu_worker_seconds=1., gpu_seconds=2.)))
    write(out/"run-manifest.json", dict(command=["python", "qualify_surrogate_hmc_nonlinear_endpoint.py"], exit_code=0, inputs={}))
    write(out/"report.json", dict(status="QUALIFIED_INFERENCE_ENDPOINT", qualified=True,
        checks=dict(one=True), retained_inference=False))
    with pytest.raises(ValueError, match="missing M10 endpoint evidence"):
        program.m10_endpoint_state(tmp_path)
