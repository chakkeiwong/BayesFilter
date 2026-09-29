"""Localize the LGSSM table smoke residual veto against FilterFlow transport."""

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

from experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf import (
    _transport_active,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_lgssm_matched_cross_audit_tf as legacy,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_lgssm_paper_table_gated_comparator_tf as table,
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
    validate_filterflow_reference_status,
)


DTYPE = tf.float64
PLAN_PATH = "docs/plans/bayesfilter-dpf-lgssm-paper-table-smoke-residual-localization-plan-2026-06-06.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-lgssm-paper-table-smoke-residual-localization-result-2026-06-06.md"
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_lgssm_paper_table_smoke_residual_localization_2026-06-06.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-lgssm-paper-table-smoke-residual-localization-2026-06-06.md"
FILTERFLOW_ENV_PYTHON = REPO_ROOT / ".localenv" / "filterflow-py311" / "bin" / "python"
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"


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
    filterflow_status = legacy._filterflow_status()
    state = _first_failing_state()
    filterflow_probe = _run_filterflow_transport_probe(state)
    comparison = _compare_transport(state, filterflow_probe)
    decision = _decision(comparison)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "question": "Localize LGSSM table-smoke BayesFilter residual veto against executable FilterFlow transport",
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "filterflow_status": filterflow_status,
        "settings": {
            "theta": 0.75,
            "epsilon": 0.25,
            "horizon": table.HORIZON,
            "num_particles": table.NUM_PARTICLES,
            "num_realizations": table.SMOKE_NUM_REALIZATIONS,
            "scaling": table.SCALING,
            "convergence_threshold": table.CONVERGENCE_THRESHOLD,
            "max_iterations": table.MAX_ITERATIONS,
            "residual_tolerance": table.RESIDUAL_TOLERANCE,
            "data_seed": table.DATA_SEED,
            "filter_seed": table.FILTER_SEED,
        },
        "first_failing_state": _state_manifest(state),
        "bayesfilter_probe": state["bayesfilter_probe"],
        "filterflow_probe": filterflow_probe,
        "comparison": comparison,
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_lgssm_paper_table_smoke_residual_localization_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": legacy._non_implications(),
    }


def _first_failing_state() -> dict[str, Any]:
    config = table.TableConfig("smoke", table.SMOKE_THETAS, table.SMOKE_EPSILONS, table.SMOKE_NUM_REALIZATIONS)
    filterflow = table._run_filterflow_subprocess(config)
    spec = legacy.MatchedSpec()
    observations = tf.constant(filterflow["observations"], dtype=DTYPE)
    particles = tf.constant(filterflow["initial_particles"], dtype=DTYPE)
    batch_size = tf.shape(particles)[0]
    log_weights = tf.fill([batch_size, table.NUM_PARTICLES], -tf.math.log(tf.cast(table.NUM_PARTICLES, DTYPE)))
    theta = 0.75
    epsilon = 0.25
    for time_index in range(table.HORIZON):
        weights = tf.exp(log_weights)
        ess_before = legacy._ess_batched(weights)
        do_resample = ess_before <= tf.constant(0.5 * table.NUM_PARTICLES, DTYPE)
        if bool(tf.reduce_any(do_resample).numpy()):
            active_particles = tf.boolean_mask(particles, do_resample)
            active_log_weights = tf.boolean_mask(log_weights, do_resample)
            transported, matrix, diag = _transport_active(
                active_particles,
                active_log_weights,
                epsilon=epsilon,
                scaling=table.SCALING,
                convergence_threshold=table.CONVERGENCE_THRESHOLD,
                max_iterations=table.MAX_ITERATIONS,
                transport_gradient_mode="filterflow_clipped",
            )
            row_residuals = tf.reduce_max(tf.abs(tf.reduce_sum(matrix, axis=2) - 1.0), axis=1)
            column_target = tf.exp(active_log_weights) * tf.cast(table.NUM_PARTICLES, DTYPE)
            column_residuals = tf.reduce_max(
                tf.abs(tf.reduce_sum(matrix, axis=1) - column_target),
                axis=1,
            )
            max_residual = tf.maximum(tf.reduce_max(row_residuals), tf.reduce_max(column_residuals))
            if legacy._float(max_residual) > table.RESIDUAL_TOLERANCE:
                failing_local = tf.where(
                    tf.maximum(row_residuals, column_residuals) > tf.constant(table.RESIDUAL_TOLERANCE, DTYPE)
                )[:, 0]
                active_indices = tf.where(do_resample)[:, 0]
                failing_global = tf.gather(active_indices, failing_local)
                return {
                    "time_index": time_index,
                    "particles": active_particles,
                    "log_weights": active_log_weights,
                    "transport_matrix": matrix,
                    "transported_particles": transported,
                    "active_indices": active_indices,
                    "failing_local_indices": failing_local,
                    "failing_global_indices": failing_global,
                    "ess_before": ess_before,
                    "do_resample": do_resample,
                    "bayesfilter_probe": {
                        "max_row_residual": legacy._float(tf.reduce_max(row_residuals)),
                        "max_column_residual": legacy._float(tf.reduce_max(column_residuals)),
                        "max_residual": legacy._float(max_residual),
                        "row_residuals": tf.cast(row_residuals, DTYPE).numpy().tolist(),
                        "column_residuals": tf.cast(column_residuals, DTYPE).numpy().tolist(),
                        "max_iterations_used": legacy._float(diag["max_iterations_used"]),
                        "max_cost_scale": legacy._float(diag["max_cost_scale"]),
                        "min_cost_scale": legacy._float(diag["min_cost_scale"]),
                        "finite_transport": bool(diag["finite_transport"]),
                        "finite_particles": bool(diag["finite_particles"]),
                    },
                }
            particles = tf.tensor_scatter_nd_update(particles, tf.where(do_resample), transported)
            uniform = tf.fill([batch_size, table.NUM_PARTICLES], -tf.math.log(tf.cast(table.NUM_PARTICLES, DTYPE)))
            log_weights = tf.where(do_resample[:, None], uniform, log_weights)
        particles = legacy._transition_particles(
            particles,
            theta,
            spec,
            "bayesfilter_filterflow_style_transport_ess",
            time_index,
        )
        obs_logp = legacy._observation_log_prob(particles, observations[time_index], spec)
        unnormalized = log_weights + obs_logp
        normalizer = tf.reduce_logsumexp(unnormalized, axis=1)
        log_weights = unnormalized - normalizer[:, None]
    raise RuntimeError("no failing residual state found")


def _run_filterflow_transport_probe(state: dict[str, Any]) -> dict[str, Any]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = str(FILTERFLOW_PATH)
    env["MPLCONFIGDIR"] = str(REPO_ROOT / ".cache" / "filterflow-mpl")
    completed = subprocess.run(
        [str(FILTERFLOW_ENV_PYTHON), "-c", _filterflow_probe_script(state)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "filterflow transport probe failed\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )
    stdout = completed.stdout
    start = stdout.rfind("FILTERFLOW_SMOKE_RESIDUAL_JSON_BEGIN")
    end = stdout.rfind("FILTERFLOW_SMOKE_RESIDUAL_JSON_END")
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(f"filterflow JSON sentinels missing:\n{stdout[-4000:]}")
    payload = json.loads(stdout[start + len("FILTERFLOW_SMOKE_RESIDUAL_JSON_BEGIN"):end].strip())
    payload["stderr_excerpt"] = completed.stderr[-2000:]
    return payload


def _filterflow_probe_script(state: dict[str, Any]) -> str:
    particles = tf.cast(state["particles"], DTYPE).numpy().tolist()
    log_weights = tf.cast(state["log_weights"], DTYPE).numpy().tolist()
    return textwrap.dedent(
        f"""
        import json
        import os
        import tensorflow as tf

        from filterflow.resampling.differentiable.regularized_transport.plan import transport

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        particles = tf.constant({particles!r}, dtype=tf.float64)
        log_weights = tf.constant({log_weights!r}, dtype=tf.float64)
        matrix = transport(
            particles,
            log_weights,
            tf.constant({0.25!r}, tf.float64),
            tf.constant({table.SCALING!r}, tf.float64),
            tf.constant({table.CONVERGENCE_THRESHOLD!r}, tf.float64),
            tf.constant({table.MAX_ITERATIONS}, tf.int32),
            {table.NUM_PARTICLES},
        )
        row_residuals = tf.reduce_max(tf.abs(tf.reduce_sum(matrix, axis=2) - 1.0), axis=1)
        column_target = tf.exp(log_weights) * tf.cast({table.NUM_PARTICLES}, tf.float64)
        column_residuals = tf.reduce_max(tf.abs(tf.reduce_sum(matrix, axis=1) - column_target), axis=1)
        transported = tf.linalg.matmul(matrix, particles)
        payload = {{
            "status": "executed",
            "max_row_residual": float(tf.reduce_max(row_residuals).numpy()),
            "max_column_residual": float(tf.reduce_max(column_residuals).numpy()),
            "max_residual": float(tf.maximum(tf.reduce_max(row_residuals), tf.reduce_max(column_residuals)).numpy()),
            "row_residuals": row_residuals.numpy().astype(float).tolist(),
            "column_residuals": column_residuals.numpy().astype(float).tolist(),
            "transport_matrix": matrix.numpy().astype(float).tolist(),
            "transported_particles": transported.numpy().astype(float).tolist(),
            "finite_transport": bool(tf.reduce_all(tf.math.is_finite(matrix)).numpy()),
            "finite_particles": bool(tf.reduce_all(tf.math.is_finite(transported)).numpy()),
        }}
        print("FILTERFLOW_SMOKE_RESIDUAL_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("FILTERFLOW_SMOKE_RESIDUAL_JSON_END")
        """
    )


def _compare_transport(state: dict[str, Any], filterflow_probe: dict[str, Any]) -> dict[str, Any]:
    bf_matrix = state["transport_matrix"]
    ff_matrix = tf.constant(filterflow_probe["transport_matrix"], dtype=DTYPE)
    bf_particles = state["transported_particles"]
    ff_particles = tf.constant(filterflow_probe["transported_particles"], dtype=DTYPE)
    matrix_delta = tf.reduce_max(tf.abs(bf_matrix - ff_matrix))
    particle_delta = tf.reduce_max(tf.abs(bf_particles - ff_particles))
    return {
        "transport_matrix_max_abs_delta": legacy._float(matrix_delta),
        "transported_particles_max_abs_delta": legacy._float(particle_delta),
        "bayesfilter_residual_pass": state["bayesfilter_probe"]["max_residual"] <= table.RESIDUAL_TOLERANCE,
        "filterflow_residual_pass": filterflow_probe["max_residual"] <= table.RESIDUAL_TOLERANCE,
        "both_residuals_large": (
            state["bayesfilter_probe"]["max_residual"] > table.RESIDUAL_TOLERANCE
            and filterflow_probe["max_residual"] > table.RESIDUAL_TOLERANCE
        ),
        "residual_delta": abs(state["bayesfilter_probe"]["max_residual"] - filterflow_probe["max_residual"]),
        "failing_global_indices": tf.cast(state["failing_global_indices"], tf.int32).numpy().tolist(),
    }


def _decision(comparison: dict[str, Any]) -> str:
    if comparison["both_residuals_large"] and comparison["transport_matrix_max_abs_delta"] <= 1e-10:
        return "lgssm_table_smoke_residual_shared_filterflow_state"
    if comparison["filterflow_residual_pass"]:
        return "lgssm_table_smoke_residual_bayesfilter_transport_mismatch"
    return "lgssm_table_smoke_residual_unclassified"


def _state_manifest(state: dict[str, Any]) -> dict[str, Any]:
    ess = tf.cast(state["ess_before"], DTYPE)
    logw = tf.cast(state["log_weights"], DTYPE)
    return {
        "time_index": state["time_index"],
        "active_count": int(tf.shape(state["particles"])[0].numpy()),
        "failing_count": int(tf.shape(state["failing_local_indices"])[0].numpy()),
        "active_indices": tf.cast(state["active_indices"], tf.int32).numpy().tolist(),
        "failing_global_indices": tf.cast(state["failing_global_indices"], tf.int32).numpy().tolist(),
        "ess_min": legacy._float(tf.reduce_min(ess)),
        "ess_max": legacy._float(tf.reduce_max(ess)),
        "active_log_weight_min": legacy._float(tf.reduce_min(logw)),
        "active_log_weight_max": legacy._float(tf.reduce_max(logw)),
        "active_weight_min": legacy._float(tf.reduce_min(tf.exp(logw))),
        "active_weight_max": legacy._float(tf.reduce_max(tf.exp(logw))),
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    validate_filterflow_reference_status(payload["filterflow_status"])
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only pre-import manifest")
    if payload["run_manifest"]["gpu_devices_visible"] != []:
        raise RuntimeError("GPU devices visible in CPU-only run")
    if payload["filterflow_probe"]["status"] != "executed":
        raise RuntimeError("filterflow probe did not execute")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing digest")


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# LGSSM Paper-Table Smoke Residual Localization

## Decision

`{payload['decision']}`

## First Failing State

| Field | Value |
| --- | ---: |
| time index | `{payload['first_failing_state']['time_index']}` |
| active rows | `{payload['first_failing_state']['active_count']}` |
| failing rows | `{payload['first_failing_state']['failing_count']}` |
| failing global indices | `{payload['first_failing_state']['failing_global_indices']}` |
| active log-weight min | `{payload['first_failing_state']['active_log_weight_min']}` |
| active log-weight max | `{payload['first_failing_state']['active_log_weight_max']}` |

## Residuals

| Probe | max row residual | max column residual | max residual |
| --- | ---: | ---: | ---: |
| BayesFilter | `{payload['bayesfilter_probe']['max_row_residual']}` | `{payload['bayesfilter_probe']['max_column_residual']}` | `{payload['bayesfilter_probe']['max_residual']}` |
| FilterFlow same state | `{payload['filterflow_probe']['max_row_residual']}` | `{payload['filterflow_probe']['max_column_residual']}` | `{payload['filterflow_probe']['max_residual']}` |

## Transport Delta

| Metric | Value |
| --- | ---: |
| transport matrix max abs delta | `{payload['comparison']['transport_matrix_max_abs_delta']}` |
| transported particles max abs delta | `{payload['comparison']['transported_particles_max_abs_delta']}` |
| residual delta | `{payload['comparison']['residual_delta']}` |

## Non-Implications

{legacy._non_implications_markdown()}
"""


def _digest_payload(payload: dict[str, Any]) -> str:
    comparable = dict(payload)
    comparable["created_at_utc"] = "TIMESTAMP"
    comparable["run_manifest"] = dict(comparable["run_manifest"])
    comparable["run_manifest"]["wall_time_seconds"] = "WALL_TIME"
    comparable["run_manifest"]["dirty_state_summary"] = "DIRTY_STATE"
    return stable_digest(comparable)


if __name__ == "__main__":
    raise SystemExit(main())
