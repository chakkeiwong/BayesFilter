"""T4 LGSSM residual ladder for the BayesFilter/FilterFlow comparator."""

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
    run_filterflow_1d_lgssm_horizon_ladder_tf as horizon,
)
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


PLAN_PATH = "docs/plans/bayesfilter-dpf-1d-lgssm-t4-residual-solver-ladder-plan-2026-06-06.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-1d-lgssm-t4-residual-solver-ladder-result-2026-06-06.md"
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_1d_lgssm_t4_residual_solver_ladder_2026-06-06.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-1d-lgssm-t4-residual-solver-ladder-2026-06-06.md"
FILTERFLOW_ENV_PYTHON = REPO_ROOT / ".localenv" / "filterflow-py311" / "bin" / "python"
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"
T4_SCENARIO = next(scenario for scenario in horizon.SCENARIOS if scenario.scenario_id == "T4_extension")


@dataclass(frozen=True)
class SolverConfig:
    config_id: str
    convergence_threshold: float
    max_iterations: int


CONFIGS = (
    SolverConfig("baseline_1e-6_iter200", 1e-6, 200),
    SolverConfig("threshold_1e-7_iter200", 1e-7, 200),
    SolverConfig("threshold_1e-6_iter500", 1e-6, 500),
    SolverConfig("threshold_1e-8_iter500", 1e-8, 500),
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
    markdown = _markdown(payload)
    write_json(JSON_PATH, payload)
    write_text(REPORT_PATH, markdown)
    write_text(REPO_ROOT / RESULT_PATH, markdown)
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict[str, Any]:
    rows = [_run_config(config) for config in CONFIGS]
    all_contract_match = all(row["filterflow_contract_match"] for row in rows)
    any_residual_pass = any(row["comparison"]["absolute_residuals_within_tolerance"] for row in rows)
    if not all_contract_match:
        decision = "one_d_lgssm_t4_residual_solver_ladder_contract_mismatch"
    elif any_residual_pass:
        decision = "one_d_lgssm_t4_residual_solver_ladder_residual_resolved"
    else:
        decision = "one_d_lgssm_t4_residual_solver_ladder_residual_persists"
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "question": (
            "Does tightening convergence_threshold or raising max_iterations clear "
            "the shared T4 1D LGSSM row-residual veto while preserving BayesFilter/FilterFlow parity?"
        ),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "reference_policy": {
            "canonical_executable_reference": "current local patched .localsource/filterflow",
            "filterflow_source_mutation": "runner does not mutate filterflow source",
            "fixed_fixture": "T4_extension from the 1D LGSSM horizon ladder",
        },
        "filterflow_checkout": step._filterflow_checkout_manifest(),
        "tolerances": step.TOLERANCES,
        "configs": [config.__dict__ for config in CONFIGS],
        "rows": rows,
        "summary": {
            "all_contract_match": all_contract_match,
            "any_residual_pass": any_residual_pass,
            "min_max_row_residual": min(row["max_row_residual"] for row in rows),
            "best_config_id": min(rows, key=lambda row: row["max_row_residual"])["config_id"],
            "finite_difference_role": "explanatory_only",
        },
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_1d_lgssm_t4_residual_solver_ladder_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": step._non_implications(),
    }


def _run_config(config: SolverConfig) -> dict[str, Any]:
    bayesfilter = _bayesfilter_reference(config)
    filterflow = _run_filterflow_subprocess(config)
    comparison = horizon._compare_runs(bayesfilter, filterflow)
    gradient_relative_delta = step._relative_delta(
        bayesfilter["gradient_tape"],
        filterflow.get("gradient_tape") if filterflow.get("status") == "executed" else None,
    )
    gradient_within_tolerance = (
        comparison.get("status") == "compared"
        and (
            comparison["gradient_delta"] <= step.TOLERANCES["gradient_abs"]
            or gradient_relative_delta <= step.TOLERANCES["gradient_rel"]
        )
    )
    implementation_agreement = horizon._implementation_agreement(comparison)
    max_row_residual = _max_row_residual(comparison)
    return {
        "config_id": config.config_id,
        "convergence_threshold": config.convergence_threshold,
        "max_iterations": config.max_iterations,
        "scenario_id": T4_SCENARIO.scenario_id,
        "horizon": T4_SCENARIO.horizon,
        "bayesfilter": bayesfilter,
        "filterflow": filterflow,
        "comparison": comparison,
        "implementation_agreement": implementation_agreement,
        "gradient_relative_delta": gradient_relative_delta,
        "gradient_within_tolerance": gradient_within_tolerance,
        "filterflow_contract_match": implementation_agreement and gradient_within_tolerance,
        "max_row_residual": max_row_residual,
        "bayesfilter_max_iterations_used": _bayesfilter_max_iterations_used(bayesfilter),
    }


def _bayesfilter_reference(config: SolverConfig) -> dict[str, Any]:
    originals = _patch_step_globals(config)
    try:
        return horizon._bayesfilter_reference(T4_SCENARIO)
    finally:
        _restore_step_globals(originals)


def _patch_step_globals(config: SolverConfig) -> dict[str, Any]:
    originals = {
        "convergence_threshold": step.CONVERGENCE_THRESHOLD,
        "max_iterations": step.MAX_ITERATIONS,
    }
    step.CONVERGENCE_THRESHOLD = config.convergence_threshold
    step.MAX_ITERATIONS = config.max_iterations
    return originals


def _restore_step_globals(originals: dict[str, Any]) -> None:
    step.CONVERGENCE_THRESHOLD = originals["convergence_threshold"]
    step.MAX_ITERATIONS = originals["max_iterations"]


def _run_filterflow_subprocess(config: SolverConfig) -> dict[str, Any]:
    if not FILTERFLOW_ENV_PYTHON.exists():
        return {"status": "blocked", "blocker": f"missing filterflow env python: {FILTERFLOW_ENV_PYTHON}"}
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = str(FILTERFLOW_PATH)
    env["MPLCONFIGDIR"] = str(REPO_ROOT / ".cache" / "filterflow-mpl")
    completed = subprocess.run(
        [str(FILTERFLOW_ENV_PYTHON), "-c", _filterflow_script(config)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    if completed.returncode != 0:
        return {
            "status": "blocked",
            "blocker": "filterflow subprocess failed",
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    stdout = completed.stdout
    start = stdout.rfind("FILTERFLOW_T4_RESIDUAL_JSON_BEGIN")
    end = stdout.rfind("FILTERFLOW_T4_RESIDUAL_JSON_END")
    if start < 0 or end < 0 or end <= start:
        return {"status": "blocked", "blocker": "filterflow JSON sentinels missing"}
    return json.loads(stdout[start + len("FILTERFLOW_T4_RESIDUAL_JSON_BEGIN"):end].strip())


def _filterflow_script(config: SolverConfig) -> str:
    return textwrap.dedent(
        f"""
        import json
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")

        import experiments.dpf_implementation.tf_tfp.runners.run_filterflow_1d_lgssm_step_gradient_comparison_tf as step

        step.HORIZON = {T4_SCENARIO.horizon}
        step.OBSERVATIONS = {list(T4_SCENARIO.observations)!r}
        step.TRANSITION_NOISES = {[list(row) for row in T4_SCENARIO.transition_noises]!r}
        step.CONVERGENCE_THRESHOLD = {config.convergence_threshold!r}
        step.MAX_ITERATIONS = {config.max_iterations}

        payload = step._run_filterflow_subprocess()
        print("FILTERFLOW_T4_RESIDUAL_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("FILTERFLOW_T4_RESIDUAL_JSON_END")
        """
    )


def _max_row_residual(comparison: dict[str, Any]) -> float:
    if comparison.get("status") != "compared":
        return float("inf")
    residuals = comparison["absolute_residuals"]
    return max(
        residuals["bayesfilter_max_row_residual"],
        residuals["filterflow_max_row_residual"],
    )


def _bayesfilter_max_iterations_used(bayesfilter: dict[str, Any]) -> float:
    return max(float(row["iterations_used"]) for row in bayesfilter["ledger"])


def _validate_payload(payload: dict[str, Any]) -> None:
    allowed = {
        "one_d_lgssm_t4_residual_solver_ladder_contract_mismatch",
        "one_d_lgssm_t4_residual_solver_ladder_residual_persists",
        "one_d_lgssm_t4_residual_solver_ladder_residual_resolved",
    }
    if payload["decision"] not in allowed:
        raise ValueError(f"unexpected decision: {payload['decision']}")
    manifest = payload["run_manifest"]
    if manifest["pre_import_cuda_visible_devices"] != "-1":
        raise ValueError("parent CPU-only pre-import invariant failed")
    if manifest["gpu_devices_visible"] != []:
        raise ValueError("parent GPU devices visible")
    if len(payload["rows"]) != len(CONFIGS):
        raise ValueError("unexpected config row count")
    all_contract_match = all(row["filterflow_contract_match"] for row in payload["rows"])
    any_residual_pass = any(row["comparison"]["absolute_residuals_within_tolerance"] for row in payload["rows"])
    for row in payload["rows"]:
        if row["filterflow"].get("status") != "executed":
            raise ValueError(f"filterflow did not execute for {row['config_id']}")
        cpu = row["filterflow"]["cpu_only_manifest"]
        if cpu["pre_import_cuda_visible_devices"] != "-1":
            raise ValueError("filterflow CPU-only pre-import invariant failed")
        if cpu["gpu_devices_visible"] != []:
            raise ValueError("filterflow GPU devices visible")
    if payload["summary"]["all_contract_match"] != all_contract_match:
        raise ValueError("summary all_contract_match mismatch")
    if payload["summary"]["any_residual_pass"] != any_residual_pass:
        raise ValueError("summary any_residual_pass mismatch")
    if payload["decision"].endswith("contract_mismatch") and all_contract_match:
        raise ValueError("contract mismatch decision but all configs matched")
    if payload["decision"].endswith("residual_persists") and (not all_contract_match or any_residual_pass):
        raise ValueError("residual persists decision inconsistent with rows")
    if payload["decision"].endswith("residual_resolved") and (not all_contract_match or not any_residual_pass):
        raise ValueError("residual resolved decision inconsistent with rows")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# 1D LGSSM T4 Residual Solver Ladder",
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        "## Evidence Contract",
        "",
        "Comparator: current local patched FilterFlow executable. Primary",
        "criterion: BayesFilter-vs-FilterFlow scalar, ledger, trigger, and AD",
        "gradient agreement. Residual pass is a separate veto. Finite differences",
        "remain explanatory only.",
        "",
        "## Results",
        "",
        "| Config | threshold | max iter | contract match | residual pass | max row residual | scalar delta | AD grad delta | BF iterations | BF FD | FF FD |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        comparison = row["comparison"]
        lines.append(
            f"| `{row['config_id']}` | `{row['convergence_threshold']}` | `{row['max_iterations']}` | "
            f"`{row['filterflow_contract_match']}` | "
            f"`{comparison['absolute_residuals_within_tolerance']}` | "
            f"`{row['max_row_residual']}` | `{comparison['scalar_delta']}` | "
            f"`{comparison['gradient_delta']}` | `{row['bayesfilter_max_iterations_used']}` | "
            f"`{row['bayesfilter']['finite_difference_gradient']}` | "
            f"`{row['filterflow']['finite_difference_gradient']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            _interpretation(payload),
            "",
            "## Non-Implications",
            "",
            *[f"- {item}" for item in payload["non_implications"]],
            "",
        ]
    )
    return "\n".join(lines)


def _interpretation(payload: dict[str, Any]) -> str:
    if payload["decision"].endswith("contract_mismatch"):
        return (
            "At least one solver setting broke BayesFilter-vs-FilterFlow parity. "
            "Do not interpret the residual ladder until the contract mismatch is localized."
        )
    if payload["decision"].endswith("residual_resolved"):
        best = payload["summary"]["best_config_id"]
        value = payload["summary"]["min_max_row_residual"]
        return (
            f"The residual veto can be cleared by solver settings on this fixture. "
            f"Best observed config: `{best}` with max row residual `{value}`."
        )
    best = payload["summary"]["best_config_id"]
    value = payload["summary"]["min_max_row_residual"]
    return (
        "BayesFilter and FilterFlow remain matched across this solver ladder, but "
        f"the T4 row-residual veto persists. Best observed config: `{best}` with "
        f"max row residual `{value}`. This weakens the hypothesis that the veto is "
        "only a convergence-threshold or max-iteration setting issue."
    )


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("run_manifest", None)
    clone.pop("created_at_utc", None)
    clone.pop("reproducibility_digest", None)
    return stable_digest(clone)


if __name__ == "__main__":
    raise SystemExit(main())
