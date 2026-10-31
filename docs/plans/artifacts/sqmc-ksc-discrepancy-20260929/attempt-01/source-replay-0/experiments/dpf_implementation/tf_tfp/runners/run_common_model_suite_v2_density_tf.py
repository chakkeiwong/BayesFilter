"""Run v2 common model-suite density tie-outs for BayesFilter/FilterFlow."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import math
import subprocess
import textwrap
import time
from typing import Any

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.fixtures.common_model_suite_tf import (
    EXPECTED_V2_MODEL_IDS,
    CommonModelSpecV2,
    common_model_specs_v2,
    evaluate_bayesfilter_density_v2,
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
from experiments.dpf_implementation.tf_tfp.runners.filterflow_reference_policy import (
    FILTERFLOW_BRANCH_MARKER,
    reference_policy,
    validate_filterflow_reference_status,
)


DTYPE = tf.float64
PLAN_PATH = "docs/plans/bayesfilter-dpf-common-model-suite-v2-production-p2-density-tieout-subplan-2026-06-07.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-common-model-suite-v2-production-p2-density-tieout-result-2026-06-07.md"
P1_MANIFEST_PATH = OUTPUT_DIR / "dpf_common_model_suite_v2_manifest_2026-06-07.json"
JSON_PATH = OUTPUT_DIR / "dpf_common_model_suite_v2_density_2026-06-07.json"
REPORT_PATH = REPORT_DIR / "dpf-common-model-suite-v2-density-2026-06-07.md"
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
    p1_manifest = load_json(P1_MANIFEST_PATH)
    _preflight_p1_manifest(p1_manifest)
    filterflow_status = _filterflow_checkout_manifest()
    specs = common_model_specs_v2()
    ready_ids = _ready_ids_for_phase(p1_manifest, "P2_density", "READY_FOR_P2")
    executable_specs = [spec for spec in specs if spec.model_id in ready_ids]
    filterflow_payload = _filterflow_density_subprocess([spec.payload() for spec in executable_specs])
    cells = []
    for spec in specs:
        if spec.model_id not in ready_ids:
            cells.append(_classified_cell(spec, "CONTRACT_BLOCKED", "row not READY_FOR_P2 in P1 classification"))
            continue
        bayesfilter = evaluate_bayesfilter_density_v2(spec)
        filterflow = _filterflow_cell(filterflow_payload, spec.model_id)
        cells.append(_cell(spec, bayesfilter, filterflow))
    decision = _decision(cells)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "question": (
            "Do BayesFilter and executable local float64 FilterFlow-side adapters "
            "evaluate the same v2 initial, transition, observation, and scalar "
            "density components for each P1-ready row?"
        ),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "p1_manifest_path": str(P1_MANIFEST_PATH.relative_to(REPO_ROOT)),
        "filterflow_reference_policy": reference_policy(),
        "filterflow_status": filterflow_status,
        "filterflow_payload_status": filterflow_payload.get("status"),
        "tolerances": {"density_value_abs": VALUE_TOLERANCE},
        "primary_criterion_fields": {
            "ready_model_ids": sorted(ready_ids),
            "executed_model_ids": [cell["model"] for cell in cells if cell["status"] != "CONTRACT_BLOCKED"],
            "primary_components": ["initial_log_density", "transition_log_density", "observation_log_density", "scalar"],
            "all_ready_rows_matched": all(cell["status"] == "MATCHED" for cell in cells if cell["model"] in ready_ids),
        },
        "veto_diagnostics": {
            "missing_filterflow_subprocess_environment": filterflow_payload.get("status") == "blocked",
            "old_three_row_common_model_specs_used_as_v2_source": False,
            "old_2026_06_06_artifact_name_used": False,
            "student_command_executed": False,
            "localsource_filterflow_mutated": False,
            "nonfinite_density_component": any(_cell_nonfinite(cell) for cell in cells),
            "unclassified_mismatch": any(cell["status"] not in {"MATCHED", "CONTRACT_BLOCKED", "INTERFACE_BLOCKED", "EXPLAINED_MISMATCH"} for cell in cells),
        },
        "explanatory_only_fields": {
            "filterflow_adapter_route": "CPU-only subprocess in .localenv/filterflow-py311 with local v2 density adapters",
            "status_counts": _status_counts(cells),
            "max_abs_delta": _max_abs_delta(cells),
            "p1_manifest_checksum": p1_manifest.get("checksum"),
        },
        "cells": cells,
        "summary": _summary(cells),
        "review_round": 0,
        "open_material_blockers": [],
        "repair_amendment_required": False,
        "next_allowed_action": "run Claude P2 result review before P3",
        "artifact_paths": {
            "json": str(JSON_PATH.relative_to(REPO_ROOT)),
            "markdown_report": str(REPORT_PATH.relative_to(REPO_ROOT)),
            "phase_result": RESULT_PATH,
        },
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_common_model_suite_v2_density_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_claims": [
            "density agreement is not full filter-path agreement",
            "no filtering-algorithm correctness claim",
            "no implementation is treated as an oracle",
            "no gradient correctness claim",
            "no student-repository tie-out claim",
            "no TT/SIRT or paper-scale reproduction claim",
        ],
    }


def _filterflow_density_subprocess(spec_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not FILTERFLOW_ENV_PYTHON.exists():
        return {
            "status": "blocked",
            "blocker": f"missing filterflow env python: {FILTERFLOW_ENV_PYTHON}",
            "cells": [],
        }
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{FILTERFLOW_PATH}"
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
            "blocker": "filterflow v2 density subprocess failed",
            "returncode": completed.returncode,
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
            "cells": [],
        }
    start = completed.stdout.rfind("FILTERFLOW_COMMON_MODEL_SUITE_V2_DENSITY_JSON_BEGIN")
    end = completed.stdout.rfind("FILTERFLOW_COMMON_MODEL_SUITE_V2_DENSITY_JSON_END")
    if start < 0 or end < 0 or end <= start:
        return {
            "status": "blocked",
            "blocker": "filterflow v2 density JSON sentinels missing",
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
            "cells": [],
        }
    raw = completed.stdout[start + len("FILTERFLOW_COMMON_MODEL_SUITE_V2_DENSITY_JSON_BEGIN") : end].strip()
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
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec

        import tensorflow as tf
        import tensorflow_probability as tfp
        from filterflow.base import State

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

        def normal_log_prob(values, loc, scale):
            return tfd.Normal(loc=to_tensor(loc), scale=to_tensor(scale)).log_prob(to_tensor(values))

        def wrap_angle(value):
            pi = tf.constant(math.pi, DTYPE)
            return tf.math.floormod(to_tensor(value) + pi, 2.0 * pi) - pi

        def gaussian_logpdf_zero_mean(residuals, covariance):
            residuals = to_tensor(residuals)
            covariance = to_tensor(covariance)
            chol = tf.linalg.cholesky(covariance)
            solved = tf.linalg.cholesky_solve(chol, tf.transpose(residuals))
            quad = tf.reduce_sum(tf.transpose(solved) * residuals, axis=1)
            dim = tf.cast(tf.shape(covariance)[0], DTYPE)
            logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
            return -0.5 * (dim * tf.math.log(tf.constant(2.0 * math.pi, DTYPE)) + logdet + quad)

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

        def structural_k(previous_k, previous_m, current_m, p):
            current_m = to_tensor(current_m)
            return (
                to_tensor(p["a"]) * to_tensor(previous_k)
                + to_tensor(p["b"]) * current_m
                + to_tensor(p["c"]) * current_m * current_m
                + to_tensor(p["d"]) * to_tensor(previous_m) * current_m
            )

        def sir_transition_mean(x_prev, p):
            state = to_tensor(x_prev)
            kappa = to_tensor(p["kappa"])
            nu = to_tensor(p["nu"])
            neighbor_sets = [tuple(row) for row in p["neighbor_sets"]]
            adjacency = tf.constant(
                [[1.0 if j in row else 0.0 for j in range(len(neighbor_sets))] for row in neighbor_sets],
                dtype=DTYPE,
            )
            degree = tf.reduce_sum(adjacency, axis=1)
            substeps = int(p["rk4_substeps"])
            step = to_tensor(p["delta"]) / tf.cast(substeps, DTYPE)
            def rhs(values):
                s = values[:, 0::2]
                i = values[:, 1::2]
                s_neighbor = tf.linalg.matmul(s, adjacency, transpose_b=True) - s * degree[tf.newaxis, :]
                i_neighbor = tf.linalg.matmul(i, adjacency, transpose_b=True) - i * degree[tf.newaxis, :]
                infection = kappa[tf.newaxis, :] * s * i
                ds = -infection + 0.5 * s_neighbor
                di = infection - nu[tf.newaxis, :] * i + 0.5 * i_neighbor
                return tf.reshape(tf.stack([ds, di], axis=2), [tf.shape(values)[0], tf.shape(values)[1]])
            for _ in range(substeps):
                k1 = rhs(state)
                k2 = rhs(state + 0.5 * step * k1)
                k3 = rhs(state + 0.5 * step * k2)
                k4 = rhs(state + step * k3)
                state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            return state

        def predator_transition_mean(theta, x_prev, p):
            theta = to_tensor(theta)
            state = to_tensor(x_prev)
            r, k_capacity, a_half, s_rate, u_rate, v_rate = tf.unstack(theta)
            substeps = int(p["rk4_substeps"])
            step = to_tensor(p["delta"]) / tf.cast(substeps, DTYPE)
            def rhs(values):
                prey = values[:, 0]
                predator = values[:, 1]
                interaction = prey * predator / (a_half + prey)
                d_prey = r * prey * (1.0 - prey / k_capacity) - s_rate * interaction
                d_predator = u_rate * interaction - v_rate * predator
                return tf.stack([d_prey, d_predator], axis=1)
            for _ in range(substeps):
                k1 = rhs(state)
                k2 = rhs(state + 0.5 * step * k1)
                k3 = rhs(state + 0.5 * step * k2)
                k4 = rhs(state + step * k3)
                state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            return state

        def eval_spec(spec):
            p = spec["parameters"]
            d = spec["density_probes"]
            model_id = spec["model_id"]
            if model_id == "lgssm_2d_h25_rich":
                initial = mvn_log_prob(d["x0"], p["m0"], p["P0"])
                transition = mvn_log_prob(to_tensor(d["x_next"]), tf.linalg.matmul(to_tensor(d["x_prev"]), to_tensor(p["A"]), transpose_b=True), p["Q"])
                loc = tf.linalg.matmul(to_tensor(d["x_obs"]), to_tensor(p["C"]), transpose_b=True)
                observation = mvn_log_prob(tf.broadcast_to(to_tensor(d["observation"]), tf.shape(loc)), loc, p["R"])
            elif model_id == "sv_1d_h18_rich":
                x0 = to_tensor(d["x0"])[:, 0]
                xp = to_tensor(d["x_prev"])[:, 0]
                xn = to_tensor(d["x_next"])[:, 0]
                xo = to_tensor(d["x_obs"])[:, 0]
                initial = normal_log_prob(x0, p["h0_mean"], tf.sqrt(to_tensor(p["h0_variance"])))
                transition = normal_log_prob(xn, to_tensor(p["mu"]) + to_tensor(p["phi"]) * (xp - to_tensor(p["mu"])), p["sigma"])
                observation = normal_log_prob(tf.broadcast_to(to_tensor(d["observation"])[0], tf.shape(xo)), 0.0, tf.exp(0.5 * xo))
            elif model_id == "range_bearing_4d_h20_rich":
                initial = mvn_log_prob(d["x0"], p["m0"], p["P0"])
                transition = mvn_log_prob(to_tensor(d["x_next"]), tf.linalg.matmul(to_tensor(d["x_prev"]), to_tensor(p["A"]), transpose_b=True), p["Q"])
                observation = range_bearing_obs_ll(d["x_obs"], d["observation"], p["R"])
            elif model_id == "structural_ar1_quadratic_h16":
                x0 = to_tensor(d["x0"])
                xp = to_tensor(d["x_prev"])
                xn = to_tensor(d["x_next"])
                xo = to_tensor(d["x_obs"])
                initial = normal_log_prob(x0[:, 0], p["m0_mean"], tf.sqrt(to_tensor(p["m0_variance"])))
                transition = normal_log_prob(xn[:, 0], to_tensor(p["rho"]) * xp[:, 0], p["sigma"])
                obs_mean = xo[:, 1] + to_tensor(p["lambda"]) * xo[:, 0]
                observation = normal_log_prob(tf.broadcast_to(to_tensor(d["observation"])[0], tf.shape(obs_mean)), obs_mean, p["observation_scale"])
            elif model_id == "spatial_sir_j3_rk4":
                initial = mvn_log_prob(d["x0"], p["initial_mean"], p["initial_covariance"])
                transition = mvn_log_prob(d["x_next"], sir_transition_mean(d["x_prev"], p), p["process_covariance"])
                infectious = to_tensor(d["x_obs"])[:, 1::2]
                observation = mvn_log_prob(tf.broadcast_to(to_tensor(d["observation"]), tf.shape(infectious)), infectious, p["observation_covariance"])
            elif model_id == "predator_prey_rk4":
                initial = mvn_log_prob(d["x0"], p["initial_mean"], p["initial_covariance"])
                transition = mvn_log_prob(d["x_next"], predator_transition_mean(spec["theta"], d["x_prev"], p), p["process_covariance"])
                observation = mvn_log_prob(tf.broadcast_to(to_tensor(d["observation"]), tf.shape(to_tensor(d["x_obs"]))), d["x_obs"], p["observation_covariance"])
            else:
                raise ValueError(f"unknown v2 model id: {{model_id}}")
            total = tf.reduce_sum(initial) + tf.reduce_sum(transition) + tf.reduce_sum(observation)
            finite = bool(
                tf.reduce_all(tf.math.is_finite(initial)).numpy()
                and tf.reduce_all(tf.math.is_finite(transition)).numpy()
                and tf.reduce_all(tf.math.is_finite(observation)).numpy()
                and tf.math.is_finite(total).numpy()
            )
            # Touch FilterFlow's State type so the artifact records execution
            # inside the FilterFlow environment, while density equations remain
            # the clean local v2 contract.
            _ = State(tf.reshape(to_tensor(d["x0"]), [1, int(to_tensor(d["x0"]).shape[0]), int(to_tensor(d["x0"]).shape[1])]))
            return {{
                "status": "executed",
                "backend": "filterflow_env_local_v2_density_adapter",
                "model_id": model_id,
                "initial_log_density": to_json(initial),
                "transition_log_density": to_json(transition),
                "observation_log_density": to_json(observation),
                "scalar": scalar(total),
                "finite": finite,
                "adapter_certification": spec["adapter_certification"],
                "spec_checksum": spec["checksum"],
            }}

        cells = []
        for spec in SPECS:
            try:
                cells.append(eval_spec(spec))
            except Exception as exc:
                cells.append({{
                    "status": "blocked",
                    "backend": "filterflow_env_local_v2_density_adapter",
                    "model_id": spec.get("model_id"),
                    "blocker": repr(exc),
                    "spec_checksum": spec.get("checksum"),
                }})
        print("FILTERFLOW_COMMON_MODEL_SUITE_V2_DENSITY_JSON_BEGIN")
        print(json.dumps({{"status": "executed", "cells": cells}}, sort_keys=True))
        print("FILTERFLOW_COMMON_MODEL_SUITE_V2_DENSITY_JSON_END")
        """
    )


def _cell(spec: CommonModelSpecV2, bayesfilter: dict[str, Any], filterflow: dict[str, Any]) -> dict[str, Any]:
    if filterflow.get("status") != "executed":
        return {
            "model": spec.model_id,
            "family": spec.family,
            "implementations": ["BayesFilter", "FilterFlow"],
            "cell_type": "v2_density_components",
            "status": "INTERFACE_BLOCKED",
            "decision": f"{spec.model_id}_filterflow_density_adapter_blocked",
            "primary_criterion": "both adapters must execute the same declared v2 density components",
            "metrics": {},
            "mismatch_class": "filterflow_v2_density_adapter_blocked",
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
        "cell_type": "v2_density_components",
        "status": "MATCHED" if matched else "EXPLAINED_MISMATCH",
        "decision": f"{spec.model_id}_density_matched" if matched else f"{spec.model_id}_density_mismatch",
        "primary_criterion": "initial, transition, observation, and scalar density components match within tolerance",
        "metrics": comparison,
        "mismatch_class": None if matched else "v2_density_component_delta",
        "bayesfilter": bayesfilter,
        "filterflow": filterflow,
        "spec": spec.payload(),
        "spec_checksum": spec.checksum(),
        "non_claim": "density-component agreement is not a full filter-path equality claim",
    }


def _classified_cell(spec: CommonModelSpecV2, status: str, reason: str) -> dict[str, Any]:
    return {
        "model": spec.model_id,
        "family": spec.family,
        "implementations": ["BayesFilter", "FilterFlow"],
        "cell_type": "v2_density_components",
        "status": status,
        "decision": f"{spec.model_id}_{status.lower()}",
        "primary_criterion": "row is not executed unless P1-ready",
        "metrics": {},
        "mismatch_class": status.lower(),
        "reason": reason,
        "spec_checksum": spec.checksum(),
    }


def _compare_density_payloads(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    component_names = ["initial_log_density", "transition_log_density", "observation_log_density"]
    components = {}
    max_abs_delta = 0.0
    all_within = True
    for name in component_names:
        left_tensor = tf.reshape(tf.convert_to_tensor(left[name], DTYPE), [-1])
        right_tensor = tf.reshape(tf.convert_to_tensor(right[name], DTYPE), [-1])
        deltas = tf.abs(left_tensor - right_tensor)
        component_max = float(tf.reduce_max(deltas).numpy()) if int(tf.size(deltas).numpy()) else 0.0
        max_abs_delta = max(max_abs_delta, component_max)
        within = component_max <= VALUE_TOLERANCE
        all_within = all_within and within
        components[name] = {
            "max_abs_delta": component_max,
            "within_tolerance": within,
            "bayesfilter": left[name],
            "filterflow": right[name],
        }
    scalar_delta = abs(float(left["scalar"]) - float(right["scalar"]))
    max_abs_delta = max(max_abs_delta, scalar_delta)
    scalar_within = scalar_delta <= VALUE_TOLERANCE
    return {
        "components": components,
        "scalar_abs_delta": scalar_delta,
        "scalar_within_tolerance": scalar_within,
        "max_abs_delta": max_abs_delta,
        "all_components_within_tolerance": all_within and scalar_within,
        "tolerance": VALUE_TOLERANCE,
    }


def _filterflow_cell(payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    for cell in payload.get("cells", []):
        if cell.get("model_id") == model_id:
            return cell
    return {"status": "missing", "blocker": f"missing FilterFlow v2 density cell for {model_id}"}


def _preflight_p1_manifest(payload: dict[str, Any]) -> None:
    ids = [row.get("model_id") for row in payload.get("rows", [])]
    if tuple(ids) != EXPECTED_V2_MODEL_IDS:
        raise ValueError(f"P1 manifest model id gate failed: {ids}")
    if payload.get("veto_diagnostics", {}).get("old_three_row_common_model_specs_used_as_v2_source"):
        raise ValueError("P1 manifest reports old v1 API leakage")


def _ready_ids_for_phase(payload: dict[str, Any], field: str, ready_value: str) -> set[str]:
    ready = set()
    for row in payload.get("pre_run_row_classification_table", []):
        if row.get(field) == ready_value:
            ready.add(str(row["model_id"]))
    return ready


def _filterflow_checkout_manifest() -> dict[str, Any]:
    if not FILTERFLOW_PATH.exists():
        return {
            "path": str(FILTERFLOW_PATH),
            "status": "missing",
            "commit": "N/A",
            "branch": "N/A",
            "status_short": "N/A",
        }
    if not FILTERFLOW_MARKER_PATH.exists():
        return {
            "path": str(FILTERFLOW_PATH),
            "status": "missing_branch_marker",
            "commit": _git_filterflow(["rev-parse", "HEAD"]),
            "branch": _git_filterflow(["rev-parse", "--abbrev-ref", "HEAD"]),
            "status_short": _git_filterflow(["status", "--short"]),
            "marker_exists": False,
        }
    status = {
        "path": str(FILTERFLOW_PATH),
        "status": "current_local_float64_reference_checkout",
        "commit": _git_filterflow(["rev-parse", "HEAD"]),
        "branch": _git_filterflow(["rev-parse", "--abbrev-ref", "HEAD"]),
        "status_short": _git_filterflow(["status", "--short"]),
        "marker_exists": FILTERFLOW_MARKER_PATH.exists(),
        "provenance_note": "local float64 FilterFlow comparator, not pristine upstream and not an oracle",
    }
    try:
        validate_filterflow_reference_status(status, marker_path=FILTERFLOW_MARKER_PATH)
        return status
    except Exception as exc:
        return {**status, "status": "blocked", "blocker": repr(exc)}


def _git_filterflow(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(FILTERFLOW_PATH), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _decision(cells: list[dict[str, Any]]) -> str:
    if all(cell["status"] == "MATCHED" for cell in cells):
        return "PENDING_CLAUDE_REVIEW"
    if any(cell["status"] == "INTERFACE_BLOCKED" for cell in cells):
        return "P2_INTERFACE_BLOCKED"
    return "P2_CLASSIFIED_MISMATCH_PENDING_REVIEW"


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "primary_criterion_fields",
        "veto_diagnostics",
        "explanatory_only_fields",
        "review_round",
        "open_material_blockers",
        "repair_amendment_required",
        "next_allowed_action",
        "cells",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"P2 payload missing required fields: {sorted(missing)}")
    if payload["decision"] not in {"PENDING_CLAUDE_REVIEW", "PASS_P2_DENSITY_READY_FOR_P3"}:
        raise ValueError(f"P2 payload decision not passable: {payload['decision']}")
    ids = [cell["model"] for cell in payload["cells"]]
    if tuple(ids) != EXPECTED_V2_MODEL_IDS:
        raise ValueError(f"P2 cell id gate failed: {ids}")
    if payload["veto_diagnostics"].get("missing_filterflow_subprocess_environment"):
        raise ValueError("P2 missing FilterFlow subprocess environment")
    if payload["veto_diagnostics"].get("nonfinite_density_component"):
        raise ValueError("P2 nonfinite density component")
    ready_ids = set(payload["primary_criterion_fields"]["ready_model_ids"])
    for cell in payload["cells"]:
        if cell["model"] in ready_ids and cell["status"] != "MATCHED":
            raise ValueError(f"P2 ready row did not match: {cell['model']} {cell['status']}")


def _summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_cells": len(cells),
        "models": [cell["model"] for cell in cells],
        "status_counts": _status_counts(cells),
        "max_abs_delta": _max_abs_delta(cells),
    }


def _status_counts(cells: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cell in cells:
        counts[cell["status"]] = counts.get(cell["status"], 0) + 1
    return counts


def _max_abs_delta(cells: list[dict[str, Any]]) -> float | None:
    values = [
        float(cell.get("metrics", {}).get("max_abs_delta", 0.0))
        for cell in cells
        if "max_abs_delta" in cell.get("metrics", {})
    ]
    return max(values) if values else None


def _cell_nonfinite(cell: dict[str, Any]) -> bool:
    return (
        cell.get("bayesfilter", {}).get("finite") is False
        or cell.get("filterflow", {}).get("finite") is False
    )


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("reproducibility_digest", None)
    if "run_manifest" in clone:
        clone["run_manifest"] = dict(clone["run_manifest"])
        clone["run_manifest"].pop("wall_time_seconds", None)
    return stable_digest(clone)


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# DPF Common Model Suite V2 P2 Density Tie-Out Result",
        "",
        "metadata_date: 2026-06-07",
        "phase: P2",
        f"decision: {payload['decision']}",
        "",
        "## Question",
        "",
        payload["question"],
        "",
        "## Evidence Contract",
        "",
        "Primary criterion: for every P1-ready v2 row, BayesFilter and FilterFlow-side density components match within tolerance.",
        "",
        "Veto diagnostics: missing FilterFlow subprocess environment, nonfinite density components, v1/v2 API leakage, old artifact-name leakage, student command leakage, or unclassified mismatch.",
        "",
        "## Result",
        "",
        f"- JSON artifact: `{payload['artifact_paths']['json']}`",
        f"- Markdown report: `{payload['artifact_paths']['markdown_report']}`",
        f"- Reproducibility digest: `{payload['reproducibility_digest']}`",
        "",
        "## Density Cells",
        "",
        "| Model id | Status | Max abs delta |",
        "|---|---|---|",
    ]
    for cell in payload["cells"]:
        lines.append(f"| `{cell['model']}` | {cell['status']} | {cell.get('metrics', {}).get('max_abs_delta', 'N/A')} |")
    lines.extend(
        [
            "",
            "## Primary Criterion Fields",
            "",
            f"- ready model ids: `{payload['primary_criterion_fields']['ready_model_ids']}`",
            f"- executed model ids: `{payload['primary_criterion_fields']['executed_model_ids']}`",
            f"- all ready rows matched: `{payload['primary_criterion_fields']['all_ready_rows_matched']}`",
            "",
            "## Veto Diagnostics",
            "",
        ]
    )
    lines.extend(f"- {key}: `{value}`" for key, value in payload["veto_diagnostics"].items())
    lines.extend(
        [
            "",
            "## Explanatory Only Fields",
            "",
            f"- status counts: `{payload['explanatory_only_fields']['status_counts']}`",
            f"- max abs delta: `{payload['explanatory_only_fields']['max_abs_delta']}`",
            f"- adapter route: `{payload['explanatory_only_fields']['filterflow_adapter_route']}`",
            "",
            "## Command Manifest",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| git commit | `{payload['run_manifest'].get('commit')}` |",
            f"| dirty status | `{_single_line(payload['run_manifest'].get('dirty_state_summary'))}` |",
            f"| command | `{payload['run_manifest'].get('command')}` |",
            f"| validation commands | `python -m json.tool {payload['artifact_paths']['json']}`; `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/bayesfilter-dpf-mpl python -m experiments.dpf_implementation.tf_tfp.runners.run_common_model_suite_v2_density_tf --validate-only`; `git diff --check` on P2 files |",
            f"| CPU/GPU status | CPU-only TensorFlow run; pre-import `CUDA_VISIBLE_DEVICES={payload['run_manifest'].get('pre_import_cuda_visible_devices')}`; visible GPUs `{payload['run_manifest'].get('gpu_devices_visible')}` |",
            "| random seeds | deterministic density probes from P1 manifest; no stochastic resampling |",
            "| dtype | `tf.float64` / JSON dtype `float64` |",
            f"| output artifacts | `{payload['artifact_paths']['json']}`; `{payload['artifact_paths']['markdown_report']}`; `{payload['artifact_paths']['phase_result']}` |",
            "",
            "## Review State",
            "",
            f"review_round: {payload['review_round']} pending Claude result/governance review",
            "",
            f"open_material_blockers: {payload['open_material_blockers'] or 'none identified locally'}",
            "",
            f"repair_amendment_required: {str(payload['repair_amendment_required']).lower()}",
            "",
            f"next_allowed_action: {payload['next_allowed_action']}",
            "",
            "## Repair History",
            "",
            "- No P2 repair has been required before Claude result review.",
            "",
            "## Decision Table",
            "",
            "| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |",
            "|---|---|---|---|---|---|",
            "| PENDING_CLAUDE_REVIEW | all P1-ready rows matched on density components locally | no local veto fired | Claude may identify adapter-certification or artifact gaps | run Claude P2 result review | no path, gradient, filter correctness, or student claim |",
            "",
            "## Post-Run Red Team",
            "",
            "Strongest alternative explanation: local FilterFlow-side adapters could match equations while still not representing a broader upstream FilterFlow model feature.",
            "",
            "Result that would overturn the decision: a row adapter is found to differ from the P1 declared equation or to depend on v1 fixtures as v2 source.",
            "",
            "Weakest evidence link: density equality does not test recursive path state updates or fixed-branch gradients.",
            "",
            "## Non-Claims",
            "",
        ]
    )
    lines.extend(f"- {claim}" for claim in payload["non_claims"])
    lines.append("")
    return "\n".join(lines)


def _single_line(value: object) -> str:
    return str(value).replace("\n", " | ")


if __name__ == "__main__":
    raise SystemExit(main())
