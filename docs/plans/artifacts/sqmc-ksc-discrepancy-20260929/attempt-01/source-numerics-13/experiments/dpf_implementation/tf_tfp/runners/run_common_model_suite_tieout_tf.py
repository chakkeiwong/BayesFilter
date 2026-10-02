"""Run common model-suite density tie-outs for BayesFilter and FilterFlow."""

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
from typing import Any

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.fixtures.common_model_suite_tf import (
    common_model_specs,
    evaluate_bayesfilter_density,
)
from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    REPO_ROOT,
    environment_manifest,
    load_json,
    scalar,
    stable_digest,
    utc_now,
    write_json,
    write_text,
)
from experiments.dpf_implementation.tf_tfp.runners.filterflow_reference_policy import (
    FILTERFLOW_BRANCH_MARKER,
    FILTERFLOW_REFERENCE_DTYPE,
    reference_policy,
    validate_filterflow_reference_status,
)


DTYPE = tf.float64
PLAN_PATH = "docs/plans/bayesfilter-dpf-common-model-suite-implementation-plan-2026-06-06.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-common-model-suite-implementation-result-2026-06-06.md"
JSON_PATH = OUTPUT_DIR / "dpf_common_model_suite_tieout_2026-06-06.json"
REPORT_PATH = REPORT_DIR / "dpf-common-model-suite-tieout-2026-06-06.md"
FILTERFLOW_ENV_PYTHON = REPO_ROOT / ".localenv" / "filterflow-py311" / "bin" / "python"
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"
FILTERFLOW_MARKER_PATH = FILTERFLOW_PATH / FILTERFLOW_BRANCH_MARKER
VALUE_TOLERANCE = 5e-10


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
    filterflow_status = _filterflow_checkout_manifest()
    specs = common_model_specs()
    filterflow_payload = _filterflow_common_suite_subprocess([spec.payload() for spec in specs])
    cells = []
    for spec in specs:
        bayesfilter = evaluate_bayesfilter_density(spec)
        filterflow = _filterflow_cell(filterflow_payload, spec.model_id)
        cells.append(_cell(spec, bayesfilter, filterflow))
    decision = _decision(cells)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "question": "Common model-suite density tie-out across BayesFilter and executable float64 FilterFlow",
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "filterflow_reference_policy": reference_policy(),
        "filterflow_status": filterflow_status,
        "filterflow_payload_status": filterflow_payload.get("status"),
        "tolerances": {"value_abs": VALUE_TOLERANCE},
        "cells": cells,
        "summary": _summary(cells),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_common_model_suite_tieout_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_claims": [
            "no filtering-algorithm correctness claim",
            "no implementation is treated as an oracle",
            "density agreement is not full particle-filter path agreement",
            "range-bearing FilterFlow coverage uses a local subprocess adapter, not an upstream built-in model",
            "no TT-filter correctness claim",
            "no paper-scale validation claim",
            "no HMC/DSGE/GPU/production readiness claim",
        ],
    }


def _cell(spec, bayesfilter: dict[str, Any], filterflow: dict[str, Any]) -> dict[str, Any]:
    if filterflow.get("status") != "executed":
        return {
            "model": spec.model_id,
            "family": spec.family,
            "implementations": ["BayesFilter", "FilterFlow"],
            "cell_type": "common_density_components",
            "status": "INTERFACE_BLOCKED",
            "decision": f"{spec.model_id}_filterflow_common_adapter_blocked",
            "primary_criterion": "both adapters must execute the same declared density components",
            "metrics": {},
            "mismatch_class": "filterflow_common_adapter_blocked",
            "reason": filterflow.get("blocker", "missing FilterFlow cell"),
            "bayesfilter": bayesfilter,
            "filterflow": filterflow,
            "spec_checksum": spec.checksum(),
        }
    comparison = _compare_density_payloads(bayesfilter, filterflow)
    matched = (
        comparison["all_components_within_tolerance"]
        and bayesfilter["finite"]
        and filterflow["finite"]
    )
    return {
        "model": spec.model_id,
        "family": spec.family,
        "implementations": ["BayesFilter", "FilterFlow"],
        "cell_type": "common_density_components",
        "status": "MATCHED" if matched else "EXPLAINED_MISMATCH",
        "decision": f"{spec.model_id}_density_matched" if matched else f"{spec.model_id}_density_mismatch",
        "primary_criterion": "initial, transition, observation, and scalar density components match within tolerance",
        "metrics": comparison,
        "mismatch_class": None if matched else "common_density_component_delta",
        "bayesfilter": bayesfilter,
        "filterflow": filterflow,
        "spec": spec.payload(),
        "spec_checksum": spec.checksum(),
        "non_claim": "density-component agreement is not a full filter-path equality claim",
    }


def _filterflow_common_suite_subprocess(spec_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not FILTERFLOW_ENV_PYTHON.exists():
        return {
            "status": "blocked",
            "blocker": f"missing filterflow env python: {FILTERFLOW_ENV_PYTHON}",
            "cells": [],
        }
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = str(FILTERFLOW_PATH)
    env["MPLCONFIGDIR"] = str(REPO_ROOT / ".cache" / "filterflow-mpl")
    completed = subprocess.run(
        [str(FILTERFLOW_ENV_PYTHON), "-c", _filterflow_script(spec_payloads)],
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
            "blocker": "filterflow common model-suite subprocess failed",
            "returncode": completed.returncode,
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
            "cells": [],
        }
    start = completed.stdout.rfind("FILTERFLOW_COMMON_MODEL_SUITE_JSON_BEGIN")
    end = completed.stdout.rfind("FILTERFLOW_COMMON_MODEL_SUITE_JSON_END")
    if start < 0 or end < 0 or end <= start:
        return {
            "status": "blocked",
            "blocker": "filterflow common model-suite JSON sentinels missing",
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
            "cells": [],
        }
    raw = completed.stdout[start + len("FILTERFLOW_COMMON_MODEL_SUITE_JSON_BEGIN") : end].strip()
    payload = json.loads(raw)
    payload["stderr_excerpt"] = completed.stderr[-2000:]
    return payload


def _filterflow_script(spec_payloads: list[dict[str, Any]]) -> str:
    specs_literal = json.dumps(spec_payloads, sort_keys=True)
    return textwrap.dedent(
        f"""
        import inspect
        import json
        import math
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")

        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec

        import tensorflow as tf
        import tensorflow_probability as tfp

        from filterflow.base import State
        from filterflow.models.stochastic_volatility import SVObservationModel, SVTransitionModel
        from filterflow.observation.linear import LinearObservationModel
        from filterflow.transition.random_walk import RandomWalkModel

        tfd = tfp.distributions
        DTYPE = tf.float64
        SPECS = json.loads({specs_literal!r})

        def to_tensor(value):
            return tf.convert_to_tensor(value, dtype=DTYPE)

        def to_json(tensor):
            return tf.cast(tensor, DTYPE).numpy().tolist()

        def scalar(tensor):
            return float(tf.cast(tensor, DTYPE).numpy())

        def mvn_log_prob(values, loc, covariance):
            return tfd.MultivariateNormalTriL(
                loc=to_tensor(loc),
                scale_tril=tf.linalg.cholesky(to_tensor(covariance)),
            ).log_prob(to_tensor(values))

        def state(rows):
            tensor = to_tensor(rows)
            return State(tf.reshape(tensor, [1, int(tensor.shape[0]), int(tensor.shape[1])]))

        def gaussian_logpdf_zero_mean(residuals, covariance):
            residuals = to_tensor(residuals)
            covariance = to_tensor(covariance)
            chol = tf.linalg.cholesky(covariance)
            solved = tf.linalg.cholesky_solve(chol, tf.transpose(residuals))
            quad = tf.reduce_sum(tf.transpose(solved) * residuals, axis=1)
            dim = tf.cast(tf.shape(covariance)[0], DTYPE)
            logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
            return -0.5 * (dim * tf.math.log(tf.constant(2.0 * math.pi, DTYPE)) + logdet + quad)

        def wrap_angle(value):
            pi = tf.constant(math.pi, DTYPE)
            return tf.math.floormod(to_tensor(value) + pi, 2.0 * pi) - pi

        def range_bearing_observation(x):
            x = to_tensor(x)
            px = x[..., 0]
            py = x[..., 1]
            rng = tf.sqrt(px * px + py * py + tf.constant(1e-12, DTYPE))
            bearing = tf.atan2(py, px)
            return tf.stack([rng, bearing], axis=-1)

        def range_bearing_obs_ll(x_obs, observation, covariance):
            predicted = range_bearing_observation(x_obs)
            residual = to_tensor(observation)[tf.newaxis, :] - predicted
            residual = tf.concat([residual[..., :1], wrap_angle(residual[..., 1:2])], axis=-1)
            return gaussian_logpdf_zero_mean(residual, covariance)

        def eval_lgssm(spec):
            p = spec["parameters"]
            initial = mvn_log_prob(spec["x0"], p["m0"], p["P0"])
            transition_model = RandomWalkModel(
                to_tensor(p["A"]),
                tfd.MultivariateNormalTriL(
                    loc=tf.zeros([len(p["m0"])], dtype=DTYPE),
                    scale_tril=tf.linalg.cholesky(to_tensor(p["Q"])),
                ),
            )
            observation_model = LinearObservationModel(
                to_tensor(p["C"]),
                tfd.MultivariateNormalTriL(
                    loc=tf.zeros([len(p["R"])], dtype=DTYPE),
                    scale_tril=tf.linalg.cholesky(to_tensor(p["R"])),
                ),
            )
            transition = tf.reshape(
                transition_model.loglikelihood(state(spec["x_prev"]), state(spec["x_next"]), tf.constant([], dtype=DTYPE)),
                [-1],
            )
            observation = tf.reshape(
                observation_model.loglikelihood(state(spec["x_obs"]), to_tensor(spec["observation"])),
                [-1],
            )
            return pack(spec, initial, transition, observation, "filterflow_builtin_linear_gaussian")

        def eval_sv(spec):
            p = spec["parameters"]
            gamma = tf.constant([[p["gamma"]]], dtype=DTYPE)
            beta = tf.constant([[p["beta"]]], dtype=DTYPE)
            sigma = tf.constant([[p["sigma"]]], dtype=DTYPE)
            initial_scale = tf.constant(p["sigma"], DTYPE) / tf.sqrt(1.0 - tf.square(tf.constant(p["gamma"], DTYPE)))
            initial = tfd.Normal(loc=tf.constant(0.0, DTYPE), scale=initial_scale).log_prob(to_tensor(spec["x0"])[:, 0])
            transition_model = SVTransitionModel(tf.constant([p["mu"]], dtype=DTYPE), gamma, sigma)
            observation_model = SVObservationModel(beta)
            transition = tf.reshape(
                transition_model.loglikelihood(state(spec["x_prev"]), state(spec["x_next"]), tf.constant([], dtype=DTYPE)),
                [-1],
            )
            observation = tf.reshape(
                observation_model.loglikelihood(state(spec["x_obs"]), to_tensor(spec["observation"])),
                [-1],
            )
            return pack(spec, initial, transition, observation, "filterflow_builtin_sv_plus_stationary_initial")

        def eval_range_bearing(spec):
            p = spec["parameters"]
            initial = mvn_log_prob(spec["x0"], p["m0"], p["P0"])
            transition_model = RandomWalkModel(
                to_tensor(p["A"]),
                tfd.MultivariateNormalTriL(
                    loc=tf.zeros([len(p["m0"])], dtype=DTYPE),
                    scale_tril=tf.linalg.cholesky(to_tensor(p["Q"])),
                ),
            )
            transition = tf.reshape(
                transition_model.loglikelihood(state(spec["x_prev"]), state(spec["x_next"]), tf.constant([], dtype=DTYPE)),
                [-1],
            )
            observation = range_bearing_obs_ll(to_tensor(spec["x_obs"]), spec["observation"], p["R"])
            return pack(spec, initial, transition, observation, "filterflow_local_range_bearing_adapter")

        def pack(spec, initial, transition, observation, backend):
            scalar_value = tf.reduce_sum(initial) + tf.reduce_sum(transition) + tf.reduce_sum(observation)
            return {{
                "status": "executed",
                "backend": backend,
                "model_id": spec["model_id"],
                "initial_log_density": to_json(initial),
                "transition_log_density": to_json(transition),
                "observation_log_density": to_json(observation),
                "scalar": scalar(scalar_value),
                "finite": bool(
                    tf.reduce_all(tf.math.is_finite(initial)).numpy()
                    and tf.reduce_all(tf.math.is_finite(transition)).numpy()
                    and tf.reduce_all(tf.math.is_finite(observation)).numpy()
                    and tf.math.is_finite(scalar_value).numpy()
                ),
                "spec_checksum": spec.get("checksum"),
            }}

        cells = []
        for spec in SPECS:
            if spec["model_id"] == "lgssm_2d_linear":
                cells.append(eval_lgssm(spec))
            elif spec["model_id"] == "sv_1d_synthetic":
                cells.append(eval_sv(spec))
            elif spec["model_id"] == "range_bearing_2d_cv":
                cells.append(eval_range_bearing(spec))
            else:
                cells.append({{"status": "blocked", "model_id": spec["model_id"], "blocker": "unknown common spec"}})

        payload = {{
            "status": "executed",
            "dtype": DTYPE.name,
            "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
            "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
            "cells": cells,
        }}
        print("FILTERFLOW_COMMON_MODEL_SUITE_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("FILTERFLOW_COMMON_MODEL_SUITE_JSON_END")
        """
    )


def _filterflow_cell(payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    if payload.get("status") != "executed":
        return {
            "status": "blocked",
            "model_id": model_id,
            "blocker": payload.get("blocker", "filterflow subprocess did not execute"),
        }
    for cell in payload.get("cells", []):
        if cell.get("model_id") == model_id:
            return cell
    return {"status": "blocked", "model_id": model_id, "blocker": "missing model cell in filterflow payload"}


def _compare_density_payloads(bayesfilter: dict[str, Any], filterflow: dict[str, Any]) -> dict[str, Any]:
    component_deltas = {
        "initial": _max_abs_delta(bayesfilter["initial_log_density"], filterflow["initial_log_density"]),
        "transition": _max_abs_delta(bayesfilter["transition_log_density"], filterflow["transition_log_density"]),
        "observation": _max_abs_delta(bayesfilter["observation_log_density"], filterflow["observation_log_density"]),
        "scalar": abs(float(bayesfilter["scalar"]) - float(filterflow["scalar"])),
    }
    max_abs_delta = max(component_deltas.values())
    return {
        "component_max_abs_delta": component_deltas,
        "max_abs_delta": max_abs_delta,
        "all_components_within_tolerance": max_abs_delta <= VALUE_TOLERANCE,
        "filterflow_finite": bool(filterflow.get("finite")),
        "bayesfilter_finite": bool(bayesfilter.get("finite")),
    }


def _decision(cells: list[dict[str, Any]]) -> str:
    if any(cell["status"] == "EXPLAINED_MISMATCH" and not cell.get("mismatch_class") for cell in cells):
        return "common_model_suite_unclassified_mismatch_veto"
    if all(cell["status"] == "MATCHED" for cell in cells):
        return "common_model_suite_density_all_matched"
    if any(cell["status"] == "EXPLAINED_MISMATCH" for cell in cells):
        return "common_model_suite_density_explained_mismatch"
    return "common_model_suite_density_blocked"


def _summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    for cell in cells:
        statuses[cell["status"]] = statuses.get(cell["status"], 0) + 1
    return {
        "num_cells": len(cells),
        "status_counts": statuses,
        "models": [cell["model"] for cell in cells],
        "max_abs_delta": max(float(cell.get("metrics", {}).get("max_abs_delta", 0.0)) for cell in cells),
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("CPU-only pre-import manifest missing")
    if payload["run_manifest"]["gpu_devices_visible"] != []:
        raise RuntimeError("GPU visible in CPU-only run")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing reproducibility digest")
    validate_filterflow_reference_status(payload["filterflow_status"], marker_path=FILTERFLOW_MARKER_PATH)
    if payload["filterflow_reference_policy"]["dtype"] != FILTERFLOW_REFERENCE_DTYPE:
        raise RuntimeError("wrong FilterFlow dtype policy")
    if payload["decision"] != "common_model_suite_density_all_matched":
        raise RuntimeError(payload["decision"])
    if len(payload["cells"]) != 3:
        raise RuntimeError("expected three common model cells")
    for cell in payload["cells"]:
        if cell["status"] != "MATCHED":
            raise RuntimeError(f"non-matched common model cell: {cell['model']}")
        if cell["metrics"]["max_abs_delta"] > VALUE_TOLERANCE:
            raise RuntimeError(f"common model tolerance failed: {cell['model']}")


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# DPF Common Model Suite Tie-Out Result

metadata_date: 2026-06-06

## Decision

`{payload['decision']}`

## Summary

- Cells: `{payload['summary']['num_cells']}`
- Status counts: `{payload['summary']['status_counts']}`
- Max absolute density delta: `{payload['summary']['max_abs_delta']}`

## Cell Table

{_cell_table(payload['cells'])}

## Interpretation

BayesFilter and executable float64 FilterFlow now share a small common model
suite at the density-component level: a 2D LGSSM, a 1D stochastic-volatility
model, and a nonlinear range-bearing model.  This establishes a reusable
contract for later full filter-path value and gradient matching.  It does not
claim full particle-filter correctness, because proposal, resampling,
random-number, scalar-objective, and branch-gradient contracts still need to
be fixed per model.

## Non-Claims

{_bullet_list(payload['non_claims'])}
"""


def _cell_table(cells: list[dict[str, Any]]) -> str:
    lines = [
        "| model | family | status | max abs delta | backend note |",
        "|---|---|---:|---:|---|",
    ]
    for cell in cells:
        backend = cell.get("filterflow", {}).get("backend", "N/A")
        lines.append(
            "| {model} | {family} | `{status}` | `{delta}` | {backend} |".format(
                model=cell["model"],
                family=cell["family"],
                status=cell["status"],
                delta=cell.get("metrics", {}).get("max_abs_delta", "N/A"),
                backend=backend,
            )
        )
    return "\n".join(lines)


def _filterflow_checkout_manifest() -> dict[str, Any]:
    if not FILTERFLOW_PATH.exists():
        return {
            "path": str(FILTERFLOW_PATH),
            "status": "missing",
            "commit": "N/A",
            "branch": "N/A",
            "status_short": "N/A",
        }
    return {
        "path": str(FILTERFLOW_PATH),
        "status": "current_local_float64_reference_checkout",
        "commit": _git_filterflow(["rev-parse", "HEAD"]),
        "branch": _git_filterflow(["rev-parse", "--abbrev-ref", "HEAD"]),
        "status_short": _git_filterflow(["status", "--short"]),
        "marker_exists": FILTERFLOW_MARKER_PATH.exists(),
        "provenance_note": "local float64 FilterFlow comparator, not pristine upstream and not an oracle",
    }


def _git_filterflow(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(FILTERFLOW_PATH), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _digest_payload(payload: dict[str, Any]) -> str:
    comparable = dict(payload)
    comparable["created_at_utc"] = "TIMESTAMP"
    comparable["run_manifest"] = dict(comparable["run_manifest"])
    comparable["run_manifest"]["wall_time_seconds"] = "WALL_TIME"
    comparable["run_manifest"]["dirty_state_summary"] = "DIRTY_STATE"
    comparable["filterflow_status"] = dict(comparable["filterflow_status"])
    comparable["filterflow_status"]["status_short"] = "FILTERFLOW_STATUS"
    return stable_digest(comparable)


def _max_abs_delta(left: Any, right: Any) -> float:
    left_tensor = tf.convert_to_tensor(left, dtype=DTYPE)
    right_tensor = tf.convert_to_tensor(right, dtype=DTYPE)
    return scalar(tf.reduce_max(tf.abs(left_tensor - right_tensor)))


def _bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


if __name__ == "__main__":
    raise SystemExit(main())
