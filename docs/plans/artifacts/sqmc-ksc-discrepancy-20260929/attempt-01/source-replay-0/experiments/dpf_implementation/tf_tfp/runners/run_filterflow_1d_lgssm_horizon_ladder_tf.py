"""Controlled 1D LGSSM horizon ladder against executable filterflow."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import subprocess
import textwrap
import time
from dataclasses import dataclass
from typing import Any

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_lgssm_step_gradient_comparison_tf as step,
)
from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    REPO_ROOT,
    environment_manifest,
    load_json,
    stable_digest,
    utc_now,
    write_json,
    write_text,
)


DTYPE = tf.float64
PLAN_PATH = "docs/plans/bayesfilter-dpf-1d-lgssm-horizon-ladder-plan-2026-06-01.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-1d-lgssm-horizon-ladder-result-2026-06-01.md"
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_1d_lgssm_horizon_ladder_2026-06-01.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-1d-lgssm-horizon-ladder-2026-06-01.md"
FILTERFLOW_ENV_PYTHON = REPO_ROOT / ".localenv" / "filterflow-py311" / "bin" / "python"
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    observations: tuple[float, ...]
    transition_noises: tuple[tuple[float, ...], ...]

    @property
    def horizon(self) -> int:
        return len(self.observations)


SCENARIOS = (
    Scenario(
        scenario_id="T2_anchor",
        observations=(0.05, -0.1),
        transition_noises=((0.0, 0.1, -0.2, 0.3), (0.2, -0.1, 0.0, -0.3)),
    ),
    Scenario(
        scenario_id="T4_extension",
        observations=(0.05, -0.1, 0.08, -0.04),
        transition_noises=(
            (0.0, 0.1, -0.2, 0.3),
            (0.2, -0.1, 0.0, -0.3),
            (-0.1, 0.0, 0.25, -0.15),
            (0.15, -0.25, 0.05, 0.1),
        ),
    ),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(load_json(JSON_PATH))
        return 0
    start = time.perf_counter()
    payload = _run()
    payload["run_manifest"]["wall_time_seconds"] = time.perf_counter() - start
    payload["reproducibility_digest"] = _digest_payload(payload)
    write_json(JSON_PATH, payload)
    write_text(REPORT_PATH, _markdown(payload))
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict[str, Any]:
    rows = [_run_scenario(scenario) for scenario in SCENARIOS]
    all_implementation_agreement = all(row["implementation_agreement"] for row in rows)
    all_forward_match = all(row["forward_match"] for row in rows)
    decision = "one_d_lgssm_horizon_ladder_forward_match"
    if all_implementation_agreement and not all_forward_match:
        decision = "one_d_lgssm_horizon_ladder_agreement_residual_veto"
    if not all_forward_match:
        decision = "one_d_lgssm_horizon_ladder_forward_mismatch"
    if all_implementation_agreement and not all_forward_match:
        decision = "one_d_lgssm_horizon_ladder_agreement_residual_veto"
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "question": "Do BayesFilter and executable filterflow remain aligned as the controlled 1D LGSSM grows from T=2 to T=4?",
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "reference_policy": {
            "canonical_executable_reference": "current local patched .localsource/filterflow",
            "filterflow_source_mutation": "runner does not mutate filterflow source",
            "fixed_target_sinkhorn": "not used",
        },
        "filterflow_checkout": step._filterflow_checkout_manifest(),
        "tolerances": step.TOLERANCES,
        "rows": rows,
        "summary": {
            "scenario_count": len(rows),
            "all_implementation_agreement": all_implementation_agreement,
            "all_forward_match": all_forward_match,
            "gradient_promotion": "not_concluded",
            "scope": "fixed scalar-state T2/T4 forward ledger plus AD-vs-FD diagnostics",
        },
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_filterflow_1d_lgssm_horizon_ladder_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": step._non_implications(),
    }


def _run_scenario(scenario: Scenario) -> dict[str, Any]:
    bayesfilter = _bayesfilter_reference(scenario)
    filterflow = _run_filterflow_subprocess(scenario)
    comparison = _compare_runs(bayesfilter, filterflow)
    return {
        "scenario_id": scenario.scenario_id,
        "horizon": scenario.horizon,
        "observations": list(scenario.observations),
        "transition_noises": [list(row) for row in scenario.transition_noises],
        "bayesfilter": bayesfilter,
        "filterflow": filterflow,
        "comparison": comparison,
        "implementation_agreement": _implementation_agreement(comparison),
        "forward_match": _forward_match(comparison),
    }


def _bayesfilter_reference(scenario: Scenario) -> dict[str, Any]:
    theta = tf.Variable(step.THETA0, dtype=DTYPE)
    with tf.GradientTape() as tape:
        scalar_value, ledger = _run_bayesfilter_scalar(theta, scenario)
    gradient = tape.gradient(scalar_value, theta)
    plus, _ = _run_bayesfilter_scalar(tf.constant(step.THETA0 + step.FINITE_DIFF_STEP, DTYPE), scenario)
    minus, _ = _run_bayesfilter_scalar(tf.constant(step.THETA0 - step.FINITE_DIFF_STEP, DTYPE), scenario)
    finite_difference = (plus - minus) / tf.constant(2.0 * step.FINITE_DIFF_STEP, DTYPE)
    return {
        "status": "executed",
        "scalar": step._float(scalar_value),
        "gradient_tape": step._maybe_float(gradient),
        "finite_difference_gradient": step._float(finite_difference),
        "gradient_delta_vs_finite_difference": step._maybe_float(
            gradient - finite_difference if gradient is not None else None
        ),
        "finite_scalar": step._finite(scalar_value),
        "finite_gradient": gradient is not None and step._finite(gradient),
        "ledger": ledger,
    }


def _run_bayesfilter_scalar(theta: tf.Tensor, scenario: Scenario) -> tuple[tf.Tensor, list[dict[str, Any]]]:
    original_horizon = step.HORIZON
    original_observations = step.OBSERVATIONS
    original_noises = step.TRANSITION_NOISES
    try:
        step.HORIZON = scenario.horizon
        step.OBSERVATIONS = list(scenario.observations)
        step.TRANSITION_NOISES = [list(row) for row in scenario.transition_noises]
        return step._run_bayesfilter_scalar(theta)
    finally:
        step.HORIZON = original_horizon
        step.OBSERVATIONS = original_observations
        step.TRANSITION_NOISES = original_noises


def _run_filterflow_subprocess(scenario: Scenario) -> dict[str, Any]:
    if not FILTERFLOW_ENV_PYTHON.exists():
        return {"status": "blocked", "blocker": f"missing filterflow env python: {FILTERFLOW_ENV_PYTHON}"}
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = str(FILTERFLOW_PATH)
    env["MPLCONFIGDIR"] = str(REPO_ROOT / ".cache" / "filterflow-mpl")
    completed = subprocess.run(
        [str(FILTERFLOW_ENV_PYTHON), "-c", _filterflow_script(scenario)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    if completed.returncode != 0:
        return {
            "status": "blocked",
            "blocker": "filterflow subprocess failed",
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    stdout = completed.stdout
    start = stdout.rfind("FILTERFLOW_1D_LADDER_JSON_BEGIN")
    end = stdout.rfind("FILTERFLOW_1D_LADDER_JSON_END")
    if start < 0 or end < 0 or end <= start:
        return {"status": "blocked", "blocker": "filterflow JSON sentinels missing"}
    return json.loads(stdout[start + len("FILTERFLOW_1D_LADDER_JSON_BEGIN"):end].strip())


def _filterflow_script(scenario: Scenario) -> str:
    return textwrap.dedent(
        f"""
        import json
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")

        import experiments.dpf_implementation.tf_tfp.runners.run_filterflow_1d_lgssm_step_gradient_comparison_tf as step

        step.HORIZON = {scenario.horizon}
        step.OBSERVATIONS = {list(scenario.observations)!r}
        step.TRANSITION_NOISES = {[list(row) for row in scenario.transition_noises]!r}

        payload = step._run_filterflow_subprocess()
        print("FILTERFLOW_1D_LADDER_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("FILTERFLOW_1D_LADDER_JSON_END")
        """
    )


def _compare_runs(bayesfilter: dict[str, Any], filterflow: dict[str, Any]) -> dict[str, Any]:
    if filterflow.get("status") != "executed":
        return {"status": "blocked_filterflow_reference", "blocker": filterflow.get("blocker", "unknown")}
    max_deltas = {key: 0.0 for key in (
        "predicted_particles",
        "observation_log_likelihoods",
        "normalized_log_weights",
        "transport_cost_matrix",
        "transport_matrix",
        "post_transport_particles",
        "per_step_log_normalizer",
    )}
    absolute_residuals = {
        "bayesfilter_max_row_residual": 0.0,
        "bayesfilter_max_column_residual": 0.0,
        "filterflow_max_row_residual": 0.0,
        "filterflow_max_column_residual": 0.0,
    }
    bf_flags = []
    ff_flags = []
    field_map = {
        "predicted_particles": "predicted_particles",
        "observation_log_likelihoods": "observation_log_likelihoods",
        "normalized_log_weights": "post_update_log_weights",
        "transport_cost_matrix": "transport_cost_matrix",
        "transport_matrix": "transport_matrix",
        "post_transport_particles": "post_transport_particles",
        "per_step_log_normalizer": "per_step_log_normalizer",
    }
    for bf_step, ff_step in zip(bayesfilter["ledger"], filterflow["ledger"], strict=True):
        bf_flags.extend(bf_step["resampling_flags"])
        ff_flags.extend(ff_step["resampling_flags"])
        for label, key in field_map.items():
            max_deltas[label] = max(max_deltas[label], step._max_abs_nested(bf_step[key], ff_step[key]))
        absolute_residuals["bayesfilter_max_row_residual"] = max(
            absolute_residuals["bayesfilter_max_row_residual"], abs(float(bf_step["row_residual"]))
        )
        absolute_residuals["bayesfilter_max_column_residual"] = max(
            absolute_residuals["bayesfilter_max_column_residual"], abs(float(bf_step["column_residual"]))
        )
        absolute_residuals["filterflow_max_row_residual"] = max(
            absolute_residuals["filterflow_max_row_residual"], abs(float(ff_step["row_residual"]))
        )
        absolute_residuals["filterflow_max_column_residual"] = max(
            absolute_residuals["filterflow_max_column_residual"], abs(float(ff_step["column_residual"]))
        )
    scalar_delta = abs(float(bayesfilter["scalar"]) - float(filterflow["scalar"]))
    gradient_delta = abs(float(bayesfilter["gradient_tape"]) - float(filterflow["gradient_tape"]))
    fd_delta = abs(float(bayesfilter["finite_difference_gradient"]) - float(filterflow["finite_difference_gradient"]))
    residuals_ok = (
        absolute_residuals["bayesfilter_max_row_residual"] <= step.TOLERANCES["row_residual"]
        and absolute_residuals["filterflow_max_row_residual"] <= step.TOLERANCES["row_residual"]
        and absolute_residuals["bayesfilter_max_column_residual"] <= step.TOLERANCES["column_residual"]
        and absolute_residuals["filterflow_max_column_residual"] <= step.TOLERANCES["column_residual"]
    )
    ledger_ok = all(max_deltas[key] <= step.TOLERANCES[key] for key in max_deltas)
    return {
        "status": "compared",
        "bayesfilter_flags": bf_flags,
        "filterflow_flags": ff_flags,
        "trigger_match": bf_flags == ff_flags,
        "max_deltas": max_deltas,
        "absolute_residuals": absolute_residuals,
        "absolute_residuals_within_tolerance": residuals_ok,
        "ledger_within_tolerance": ledger_ok,
        "scalar_delta": scalar_delta,
        "scalar_within_tolerance": scalar_delta <= step.TOLERANCES["total_scalar"],
        "gradient_delta": gradient_delta,
        "finite_difference_delta": fd_delta,
        "gradient_promotion": "not_concluded",
    }


def _forward_match(comparison: dict[str, Any]) -> bool:
    if comparison.get("status") != "compared":
        return False
    return bool(
        comparison["trigger_match"]
        and comparison["ledger_within_tolerance"]
        and comparison["scalar_within_tolerance"]
        and comparison["absolute_residuals_within_tolerance"]
    )


def _implementation_agreement(comparison: dict[str, Any]) -> bool:
    if comparison.get("status") != "compared":
        return False
    return bool(
        comparison["trigger_match"]
        and comparison["ledger_within_tolerance"]
        and comparison["scalar_within_tolerance"]
    )


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload["decision"] not in {
        "one_d_lgssm_horizon_ladder_forward_match",
        "one_d_lgssm_horizon_ladder_forward_mismatch",
        "one_d_lgssm_horizon_ladder_agreement_residual_veto",
    }:
        raise ValueError(f"unexpected decision: {payload['decision']}")
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise ValueError("parent CPU-only pre-import invariant failed")
    if payload["run_manifest"]["gpu_devices_visible"] != []:
        raise ValueError("parent GPU devices visible")
    if len(payload["rows"]) != 2:
        raise ValueError("expected exactly two horizon scenarios")
    for row in payload["rows"]:
        if row["filterflow"].get("status") != "executed":
            raise ValueError(f"filterflow did not execute: {row['scenario_id']}")
        if row["filterflow"]["cpu_only_manifest"]["pre_import_cuda_visible_devices"] != "-1":
            raise ValueError("filterflow CPU-only pre-import invariant failed")
        if row["filterflow"]["cpu_only_manifest"]["gpu_devices_visible"] != []:
            raise ValueError("filterflow GPU devices visible")
    all_agreement = all(row["implementation_agreement"] for row in payload["rows"])
    all_forward_match = all(row["forward_match"] for row in payload["rows"])
    if all_forward_match:
        expected = "one_d_lgssm_horizon_ladder_forward_match"
    elif all_agreement:
        expected = "one_d_lgssm_horizon_ladder_agreement_residual_veto"
    else:
        expected = "one_d_lgssm_horizon_ladder_forward_mismatch"
    if payload["decision"] != expected:
        raise ValueError(f"decision {payload['decision']} inconsistent with rows; expected {expected}")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# 1D LGSSM Horizon Ladder",
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        "| Scenario | T | Flags | Scalar delta | Ledger match | Residuals pass | Max row residual | BF AD | FF AD | BF FD | FF FD |",
        "| --- | ---: | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        comparison = row["comparison"]
        max_row_residual = max(
            comparison["absolute_residuals"]["bayesfilter_max_row_residual"],
            comparison["absolute_residuals"]["filterflow_max_row_residual"],
        )
        lines.append(
            f"| `{row['scenario_id']}` | `{row['horizon']}` | `{comparison['bayesfilter_flags']}` | "
            f"`{comparison['scalar_delta']}` | `{comparison['ledger_within_tolerance']}` | "
            f"`{comparison['absolute_residuals_within_tolerance']}` | `{max_row_residual}` | "
            f"`{row['bayesfilter']['gradient_tape']}` | `{row['filterflow']['gradient_tape']}` | "
            f"`{row['bayesfilter']['finite_difference_gradient']}` | "
            f"`{row['filterflow']['finite_difference_gradient']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a fixed scalar-state forward-ledger ladder. Gradient diagnostics",
            "remain non-promotional.",
            "",
            "## Non-Implications",
            "",
            *[f"- {item}" for item in payload["non_implications"]],
            "",
        ]
    )
    return "\n".join(lines)


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("run_manifest", None)
    clone.pop("created_at_utc", None)
    clone.pop("reproducibility_digest", None)
    return stable_digest(clone)


if __name__ == "__main__":
    raise SystemExit(main())
