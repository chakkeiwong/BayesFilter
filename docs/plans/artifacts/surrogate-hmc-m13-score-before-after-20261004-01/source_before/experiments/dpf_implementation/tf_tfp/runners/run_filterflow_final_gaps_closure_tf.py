"""Final bounded closure audit for remaining filterflow OT-DPF gaps."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import math
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any

import tensorflow as tf

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
    FILTERFLOW_REFERENCE_DTYPE,
    FILTERFLOW_UPSTREAM_BASE_COMMIT,
    validate_filterflow_reference_status,
)


DTYPE = tf.float64
PLAN_PATH = "docs/plans/bayesfilter-dpf-filterflow-final-gaps-closure-plan-2026-05-30.md"
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_final_gaps_closure_2026-05-30.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-final-gaps-closure-2026-05-30.md"
FILTERFLOW_ENV_PYTHON = REPO_ROOT / ".localenv" / "filterflow-py311" / "bin" / "python"
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"
UPSTREAM_FILTERFLOW_COMMIT = FILTERFLOW_UPSTREAM_BASE_COMMIT
SINKHORN_BUDGETS = (25, 50, 100, 200, 500, 1000)
SINKHORN_EPSILONS = (0.25, 0.5, 0.75)
SINKHORN_TOLERANCE = 1e-5
NUM_PARTICLES = 25
MATCHED_HORIZON = 150
MATCHED_BATCH_SIZE = 100
MATCHED_DATA_SEED = 111


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
    filterflow_status = _filterflow_status()
    paper_code_ledger = _paper_code_ledger()
    smoothness = _run_smoothness_subprocess()
    fixed_sinkhorn = _run_fixed_sinkhorn_diagnosis()
    gap_ledger = _gap_ledger(paper_code_ledger, smoothness, fixed_sinkhorn)
    payload = {
        "decision": _decision(gap_ledger),
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "question": "Close or localize the three remaining filterflow OT-DPF audit gaps.",
        "filterflow_status": filterflow_status,
        "source_support_ledger": _source_support_ledger(),
        "paper_code_ledger": paper_code_ledger,
        "smoothness_gradient_replication": smoothness,
        "fixed_target_sinkhorn_diagnosis": fixed_sinkhorn,
        "gap_ledger": gap_ledger,
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_filterflow_final_gaps_closure_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }
    return payload


def _source_support_ledger() -> list[dict[str, Any]]:
    return [
        {
            "source_id": "corenflos_main",
            "classification": "DIRECT_METHOD",
            "local_path": ".localsource/dpf_ot_audit/corenflos2021_differentiable_particle_filtering_eot.txt",
            "inspected_anchors": ["Section 5.1", "Table 1", "Section 3.1", "Section 4.3"],
            "allowed_claims": [
                "Section 5.1 reports a two-dimensional LGSSM likelihood comparison against Kalman.",
                "Table 1 reports PF and DPF per-time log-likelihood error means and standard deviations.",
            ],
            "forbidden_claims": [
                "No production or general nonlinear-SSM correctness follows from the table.",
                "The paper table alone does not validate this repository's fixed-target Sinkhorn implementation.",
            ],
            "metadata_status": "local_full_text_only_no_network_metadata_lookup",
        },
        {
            "source_id": "corenflos_supplement",
            "classification": "DIRECT_METHOD",
            "local_path": ".localsource/dpf_ot_audit/corenflos2021_differentiable_particle_filtering_eot_supp.txt",
            "inspected_anchors": ["Appendix E.1", "Table 4"],
            "allowed_claims": [
                "Appendix E.1 discusses gradient vector fields and learning experiments for an LGSSM.",
                "The supplement separates likelihood-table behavior from gradient/SMLE behavior.",
            ],
            "forbidden_claims": [
                "A bounded mesh smoke run does not reproduce the full Appendix E.1 figure or Table 4.",
            ],
            "metadata_status": "local_full_text_only_no_network_metadata_lookup",
        },
        {
            "source_id": "filterflow_code",
            "classification": "IMPLEMENTATION_OR_SOFTWARE",
            "local_path": ".localsource/filterflow",
            "inspected_anchors": [
                "scripts/simple_linear_comparison.py",
                "scripts/simple_linear_smoothness.py",
                "filterflow/resampling/differentiable/regularized_transport/plan.py",
                "filterflow/resampling/differentiable/regularized_transport/sinkhorn.py",
            ],
            "allowed_claims": [
                "The local compatibility branch provides executable reference code for bounded audits.",
            ],
            "forbidden_claims": [
                "The patched local checkout is not untouched upstream source.",
            ],
            "metadata_status": "commit_recorded_no_network_metadata_lookup",
        },
    ]


def _paper_code_ledger() -> dict[str, Any]:
    paper_table = {
        "pf": {
            "0.25": {"mean": -1.13, "std": 0.20},
            "0.5": {"mean": -0.93, "std": 0.18},
            "0.75": {"mean": -1.05, "std": 0.17},
        },
        "dpf_epsilon_0.25": {
            "0.25": {"mean": -1.14, "std": 0.20},
            "0.5": {"mean": -0.94, "std": 0.18},
            "0.75": {"mean": -1.07, "std": 0.19},
        },
        "dpf_epsilon_0.5": {
            "0.25": {"mean": -1.14, "std": 0.20},
            "0.5": {"mean": -0.94, "std": 0.18},
            "0.75": {"mean": -1.08, "std": 0.18},
        },
        "dpf_epsilon_0.75": {
            "0.25": {"mean": -1.14, "std": 0.20},
            "0.5": {"mean": -0.94, "std": 0.18},
            "0.75": {"mean": -1.08, "std": 0.18},
        },
    }
    rows = [
        {
            "item": "state_dimension",
            "paper_section_5_1": "2",
            "filterflow_simple_linear_comparison": "2",
            "status": "matched",
        },
        {
            "item": "transition_mean",
            "paper_section_5_1": "diag(theta_1, theta_2) x",
            "filterflow_simple_linear_comparison": "diag(theta_1, theta_2) x",
            "status": "matched",
        },
        {
            "item": "transition_covariance",
            "paper_section_5_1": "0.5 I_2",
            "filterflow_simple_linear_comparison": "I_2",
            "status": "mismatch_or_version_ambiguity",
        },
        {
            "item": "observation_covariance",
            "paper_section_5_1": "0.1 I_2",
            "filterflow_simple_linear_comparison": "0.1 I_2",
            "status": "matched",
        },
        {
            "item": "horizon",
            "paper_section_5_1": "T=150",
            "filterflow_simple_linear_comparison": "T=150",
            "status": "matched",
        },
        {
            "item": "particle_count",
            "paper_section_5_1": "N=25",
            "filterflow_simple_linear_comparison": "N=25 in primary table loop",
            "status": "matched",
        },
        {
            "item": "realizations",
            "paper_section_5_1": "100 realizations",
            "filterflow_simple_linear_comparison": "batch_size=100 in script main loop",
            "status": "matched",
        },
        {
            "item": "epsilon_grid",
            "paper_section_5_1": "0.25, 0.5, 0.75",
            "filterflow_simple_linear_comparison": "0.25, 0.5, 0.75",
            "status": "matched",
        },
        {
            "item": "resampling_rule",
            "paper_section_5_1": "not fully specified in main text",
            "filterflow_simple_linear_comparison": "NeffCriterion(0.5, True)",
            "status": "code_specific",
        },
        {
            "item": "regularized_transport_scaling",
            "paper_section_5_1": "not fully specified in main text",
            "filterflow_simple_linear_comparison": "scaling=0.9, convergence_threshold=1e-3",
            "status": "code_specific",
        },
    ]
    return {
        "status": "closed_with_transition_covariance_ambiguity_recorded",
        "paper_table_1_values": paper_table,
        "settings_rows": rows,
        "interpretation": (
            "The executable filterflow Section-5.1-style script matches most reported settings, "
            "but its transition covariance is I_2 while the main paper and supplement text state 0.5 I_2. "
            "A direct bounded rerun showed the published Table 1 scale is consistent with the executable "
            "filterflow I_2 convention and not with 0.5 I_2, so this is likely a paper typo or notation "
            "mismatch. BayesFilter reproduction audits use the executable filterflow I_2 convention, while "
            "recording the paper-text discrepancy explicitly."
        ),
    }


def _run_smoothness_subprocess() -> dict[str, Any]:
    if not FILTERFLOW_ENV_PYTHON.exists():
        return {"status": "blocked_missing_filterflow_env", "python": str(FILTERFLOW_ENV_PYTHON)}
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["MPLCONFIGDIR"] = str(REPO_ROOT / ".cache" / "filterflow-mpl")
    env["PYTHONPATH"] = str(FILTERFLOW_PATH)
    completed = subprocess.run(
        [str(FILTERFLOW_ENV_PYTHON), "-c", _smoothness_subprocess_script()],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if completed.returncode != 0:
        return {
            "status": "blocked_subprocess_failed",
            "returncode": completed.returncode,
            "stdout_excerpt": completed.stdout[-3000:],
            "stderr_excerpt": completed.stderr[-3000:],
        }
    start = completed.stdout.rfind("SMOOTHNESS_JSON_BEGIN")
    end = completed.stdout.rfind("SMOOTHNESS_JSON_END")
    if start < 0 or end < 0 or end <= start:
        return {
            "status": "blocked_missing_json_sentinels",
            "stdout_excerpt": completed.stdout[-3000:],
            "stderr_excerpt": completed.stderr[-3000:],
        }
    raw = completed.stdout[start + len("SMOOTHNESS_JSON_BEGIN"):end].strip()
    payload = json.loads(raw)
    payload["command"] = (
        "CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=.cache/filterflow-mpl "
        "PYTHONPATH=.localsource/filterflow "
        f"{FILTERFLOW_ENV_PYTHON} -c <structured smoothness gradient script>"
    )
    payload["stderr_excerpt"] = completed.stderr[-1500:]
    return payload


def _smoothness_subprocess_script() -> str:
    return textwrap.dedent(
        """
        import json
        import inspect
        import os
        import tensorflow as tf
        np = __import__("numpy")

        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec

        from filterflow.base import State
        from filterflow.models.simple_linear_gaussian import make_filter
        from filterflow.resampling import RegularisedTransform
        from filterflow.resampling.criterion import NeffCriterion
        from scripts.simple_linear_smoothness import get_data, get_surface, get_surface_kf

        T = 100
        MESH_SIZE = 4
        N_PARTICLES = 25
        BATCH_SIZE = 1
        EPSILON = 0.25
        SCALING = 0.85
        CONVERGENCE_THRESHOLD = 1e-6
        MAX_ITER = 200
        DATA_SEED = 123
        FILTER_SEED = 1234
        DIFF_EPSILON = 1e-2

        v = 1.0
        t = 0.1
        transition_matrix = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        transition_covariance = v ** 2 * np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]], dtype=np.float64)
        observation_matrix = np.array([[1.0, 0.0]], dtype=np.float64)
        observation_covariance = np.array([[t ** 2]], dtype=np.float64)
        x_linspace = np.linspace(0.95, 1.0, MESH_SIZE).astype(np.float64)
        y_linspace = np.linspace(0.95, 1.0, MESH_SIZE).astype(np.float64)
        mesh = np.asanyarray([(x, y) for x in x_linspace for y in y_linspace])

        random_state = np.random.RandomState(seed=DATA_SEED)
        data, kf = get_data(
            transition_matrix,
            observation_matrix,
            transition_covariance,
            observation_covariance,
            T,
            random_state,
        )
        observation_dataset = tf.data.Dataset.from_tensor_slices(data)
        initial_particles = random_state.normal(0.0, 0.01, [BATCH_SIZE, N_PARTICLES, 2]).astype(np.float64)
        initial_state = State(tf.constant(initial_particles, dtype=tf.float64))
        modifiable_transition_matrix = tf.Variable(transition_matrix, trainable=False, dtype=tf.float64)
        smc = make_filter(
            tf.convert_to_tensor(observation_matrix, dtype=tf.float64),
            modifiable_transition_matrix,
            tf.linalg.cholesky(tf.convert_to_tensor(observation_covariance, dtype=tf.float64)),
            tf.linalg.cholesky(tf.convert_to_tensor(transition_covariance, dtype=tf.float64)),
            RegularisedTransform(
                epsilon=EPSILON,
                scaling=SCALING,
                convergence_threshold=CONVERGENCE_THRESHOLD,
                max_iter=MAX_ITER,
            ),
            NeffCriterion(0.9999, True),
            optimal_proposal=False,
        )
        log_likelihoods, gradients = get_surface(
            mesh,
            modifiable_transition_matrix,
            smc,
            initial_state,
            False,
            observation_dataset,
            tf.constant(T),
            tf.constant(FILTER_SEED),
            False,
        )
        dpf_ll = log_likelihoods.numpy().astype(float)
        dpf_grad = gradients.numpy().astype(float)
        kalman_ll, kalman_grad = get_surface_kf(mesh, kf, data, DIFF_EPSILON, False)
        finite_ll = bool(np.all(np.isfinite(dpf_ll)) and np.all(np.isfinite(kalman_ll)))
        finite_grad = bool(np.all(np.isfinite(dpf_grad)) and np.all(np.isfinite(kalman_grad)))
        grad_diff = dpf_grad - kalman_grad
        ll_diff = dpf_ll - kalman_ll
        dpf_grad_flat = dpf_grad.reshape(-1)
        kalman_grad_flat = kalman_grad.reshape(-1)
        denom = float(np.linalg.norm(dpf_grad_flat) * np.linalg.norm(kalman_grad_flat))
        cosine = None if denom == 0.0 else float(np.dot(dpf_grad_flat, kalman_grad_flat) / denom)
        sign_agreement = float(np.mean(np.sign(dpf_grad_flat) == np.sign(kalman_grad_flat)))
        payload = {
            "status": "executed",
            "python": os.sys.version.split()[0],
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
            "settings": {
                "T": T,
                "mesh_size": MESH_SIZE,
                "n_particles": N_PARTICLES,
                "batch_size": BATCH_SIZE,
                "epsilon": EPSILON,
                "scaling": SCALING,
                "convergence_threshold": CONVERGENCE_THRESHOLD,
                "max_iter": MAX_ITER,
                "data_seed": DATA_SEED,
                "filter_seed": FILTER_SEED,
                "diff_epsilon": DIFF_EPSILON,
                "resampling_neff": 0.9999,
                "optimal_proposal": False,
            },
            "finite_likelihoods": finite_ll,
            "finite_gradients": finite_grad,
            "likelihood_rmse": float(np.sqrt(np.mean(ll_diff ** 2))),
            "likelihood_mean_delta": float(np.mean(ll_diff)),
            "gradient_rmse": float(np.sqrt(np.mean(grad_diff ** 2))),
            "gradient_max_abs_delta": float(np.max(np.abs(grad_diff))),
            "gradient_cosine_vs_kalman_fd": cosine,
            "gradient_sign_agreement": sign_agreement,
            "dpf_gradient_norm": float(np.linalg.norm(dpf_grad_flat)),
            "kalman_fd_gradient_norm": float(np.linalg.norm(kalman_grad_flat)),
            "mesh": mesh.astype(float).tolist(),
            "dpf_log_likelihoods": dpf_ll.tolist(),
            "kalman_log_likelihoods": kalman_ll.astype(float).tolist(),
            "dpf_gradients": dpf_grad.tolist(),
            "kalman_fd_gradients": kalman_grad.astype(float).tolist(),
            "interpretation": "bounded smoothness gradient replication, not full figure or Table 4 reproduction",
        }
        print("SMOOTHNESS_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("SMOOTHNESS_JSON_END")
        """
    )


def _run_fixed_sinkhorn_diagnosis() -> dict[str, Any]:
    data = _matched_lgssm_data_subprocess()
    if data.get("status") != "executed":
        return data
    particles = tf.constant(data["initial_particles"], DTYPE)
    weights = tf.ones([MATCHED_BATCH_SIZE, NUM_PARTICLES], DTYPE) / tf.cast(NUM_PARTICLES, DTYPE)
    ess = _ess_batched(weights)
    do_resample = ess <= tf.constant(0.5 * NUM_PARTICLES, DTYPE)
    cost = _scaled_cost(particles)
    rows = []
    for epsilon in SINKHORN_EPSILONS:
        for budget in SINKHORN_BUDGETS:
            residual = _fixed_sinkhorn_residual(cost, weights, epsilon, budget)
            rows.append(
                {
                    "epsilon": epsilon,
                    "budget": budget,
                    "max_residual": _float(residual["max_residual"]),
                    "median_residual": _float(residual["median_residual"]),
                    "row_residual": _float(residual["row_residual"]),
                    "column_residual": _float(residual["column_residual"]),
                    "finite_coupling": residual["finite_coupling"],
                    "below_tolerance": _float(residual["max_residual"]) <= SINKHORN_TOLERANCE,
                    "all_rows_nontriggered": bool(tf.reduce_all(~do_resample).numpy()),
                }
            )
    eps025 = [row for row in rows if row["epsilon"] == 0.25]
    final_eps025 = next(row for row in eps025 if row["budget"] == SINKHORN_BUDGETS[-1])
    residual_100 = next(row for row in eps025 if row["budget"] == 100)
    if residual_100["below_tolerance"]:
        status = "epsilon_0.25_prior_veto_not_reproduced_on_initial_cloud"
    elif final_eps025["below_tolerance"] and bool(tf.reduce_all(~do_resample).numpy()):
        status = "epsilon_0.25_unconditional_nontriggered_budget_gap"
    elif final_eps025["below_tolerance"]:
        status = "epsilon_0.25_iteration_budget_gap"
    elif final_eps025["max_residual"] < residual_100["max_residual"]:
        status = "epsilon_0.25_improves_but_not_closed_by_budget_cap"
    else:
        status = "epsilon_0.25_fixed_target_stability_gap"
    return {
        "status": status,
        "tolerance": SINKHORN_TOLERANCE,
        "budget_ladder": list(SINKHORN_BUDGETS),
        "epsilon_grid": list(SINKHORN_EPSILONS),
        "state_stream": {
            "model": "exact matched-audit initial cloud from filterflow seed protocol",
            "theta": 0.5,
            "transition_covariance": "I_2 executable filterflow convention",
            "observation_covariance": "0.1 I_2",
            "horizon": MATCHED_HORIZON,
            "batch_size": MATCHED_BATCH_SIZE,
            "num_particles": NUM_PARTICLES,
            "data_seed": MATCHED_DATA_SEED,
            "resampling_threshold": "ESS <= 0.5 N",
            "initial_ess_min": _float(tf.reduce_min(ess)),
            "initial_ess_max": _float(tf.reduce_max(ess)),
            "initial_resampling_triggered_any": bool(tf.reduce_any(do_resample).numpy()),
            "diagnostic_scope": (
                "fixed-Sinkhorn is probed on the initial cloud because the matched audit computed "
                "it before masking by the ESS trigger"
            ),
        },
        "num_probe_states": int(particles.shape[0]),
        "rows": rows,
        "interpretation": (
            "This diagnoses the BayesFilter fixed-target Sinkhorn lane only. The old veto can occur on "
            "non-triggered rows because the diagnostic path computed Sinkhorn before masking by ESS. "
            "It is not paper-equivalence evidence for filterflow's annealed transport."
        ),
    }


def _matched_lgssm_data_subprocess() -> dict[str, Any]:
    if not FILTERFLOW_ENV_PYTHON.exists():
        return {"status": "blocked_missing_filterflow_env", "python": str(FILTERFLOW_ENV_PYTHON)}
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["MPLCONFIGDIR"] = str(REPO_ROOT / ".cache" / "filterflow-mpl")
    env["PYTHONPATH"] = str(FILTERFLOW_PATH)
    completed = subprocess.run(
        [str(FILTERFLOW_ENV_PYTHON), "-c", _matched_lgssm_data_script()],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0:
        return {
            "status": "blocked_matched_data_subprocess_failed",
            "returncode": completed.returncode,
            "stdout_excerpt": completed.stdout[-2000:],
            "stderr_excerpt": completed.stderr[-3000:],
        }
    start = completed.stdout.rfind("MATCHED_DATA_JSON_BEGIN")
    end = completed.stdout.rfind("MATCHED_DATA_JSON_END")
    if start < 0 or end < 0 or end <= start:
        return {
            "status": "blocked_matched_data_json_missing",
            "stdout_excerpt": completed.stdout[-2000:],
            "stderr_excerpt": completed.stderr[-3000:],
        }
    raw = completed.stdout[start + len("MATCHED_DATA_JSON_BEGIN"):end].strip()
    payload = json.loads(raw)
    payload["command"] = (
        "CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=.cache/filterflow-mpl "
        "PYTHONPATH=.localsource/filterflow "
        f"{FILTERFLOW_ENV_PYTHON} -c <structured matched LGSSM data script>"
    )
    return payload


def _matched_lgssm_data_script() -> str:
    return textwrap.dedent(
        f"""
        import inspect
        import json
        import os
        np = __import__("numpy")

        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec

        from scripts.simple_linear_common import get_data

        T = {MATCHED_HORIZON}
        BATCH_SIZE = {MATCHED_BATCH_SIZE}
        N = {NUM_PARTICLES}
        DATA_SEED = {MATCHED_DATA_SEED}

        transition_matrix = 0.5 * np.eye(2, dtype=np.float64)
        transition_covariance = np.eye(2, dtype=np.float64)
        observation_matrix = np.eye(2, dtype=np.float64)
        observation_covariance = 0.1 * np.eye(2, dtype=np.float64)
        rng = np.random.RandomState(seed=DATA_SEED)
        data, _ = get_data(
            transition_matrix,
            observation_matrix,
            transition_covariance,
            observation_covariance,
            T,
            rng,
            dtype=np.float64,
        )
        initial_particles = rng.normal(0.0, 1.0, [BATCH_SIZE, N, 2]).astype(np.float64)
        payload = {{
            "status": "executed",
            "dtype": "{FILTERFLOW_REFERENCE_DTYPE}",
            "python": os.sys.version.split()[0],
            "numpy": np.__version__,
            "observations_checksum": float(np.sum(data.astype(np.float64))),
            "initial_particles_checksum": float(np.sum(initial_particles.astype(np.float64))),
            "initial_particles": initial_particles.astype(float).tolist(),
        }}
        print("MATCHED_DATA_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("MATCHED_DATA_JSON_END")
        """
    )


def _fixed_sinkhorn_residual(
    cost: tf.Tensor,
    weights: tf.Tensor,
    epsilon: float,
    max_iter: int,
) -> dict[str, Any]:
    weights = weights / tf.reduce_sum(weights, axis=1, keepdims=True)
    batch_size = tf.shape(cost)[0]
    target = tf.ones([batch_size, NUM_PARTICLES], DTYPE) / tf.cast(NUM_PARTICLES, DTYPE)
    log_source = tf.math.log(tf.maximum(weights, tf.constant(1e-300, DTYPE)))
    log_target = tf.math.log(target)
    log_kernel = -cost / tf.constant(epsilon, DTYPE)
    log_u = tf.zeros([batch_size, NUM_PARTICLES], DTYPE)
    log_v = tf.zeros([batch_size, NUM_PARTICLES], DTYPE)
    for _ in range(max_iter):
        log_u = log_source - tf.reduce_logsumexp(log_kernel + log_v[:, None, :], axis=2)
        log_v = log_target - tf.reduce_logsumexp(log_kernel + log_u[:, :, None], axis=1)
    coupling = tf.exp(log_u[:, :, None] + log_kernel + log_v[:, None, :])
    row_residuals = tf.reduce_max(tf.abs(tf.reduce_sum(coupling, axis=2) - weights), axis=1)
    column_residuals = tf.reduce_max(tf.abs(tf.reduce_sum(coupling, axis=1) - target), axis=1)
    residuals = tf.maximum(row_residuals, column_residuals)
    return {
        "max_residual": tf.reduce_max(residuals),
        "median_residual": _median_tensor(residuals),
        "row_residual": tf.reduce_max(row_residuals),
        "column_residual": tf.reduce_max(column_residuals),
        "finite_coupling": bool(tf.reduce_all(tf.math.is_finite(coupling)).numpy()),
    }


def _scaled_cost(particles: tf.Tensor) -> tf.Tensor:
    centered = particles - tf.stop_gradient(tf.reduce_mean(particles, axis=1, keepdims=True))
    std = tf.math.reduce_std(tf.cast(particles, DTYPE), axis=1)
    diameter = tf.reduce_max(std, axis=1)
    diameter = tf.where(diameter == 0.0, tf.ones_like(diameter), diameter)
    scale = diameter * tf.sqrt(tf.constant(2.0, DTYPE))
    scaled = centered / tf.stop_gradient(scale[:, None, None])
    return 0.5 * _pairwise_squared(scaled)


def _gap_ledger(
    paper_code: dict[str, Any],
    smoothness: dict[str, Any],
    fixed_sinkhorn: dict[str, Any],
) -> list[dict[str, Any]]:
    smoothness_status = smoothness.get("status")
    smoothness_decision = (
        "closed_bounded_gradient_smoke_with_severe_unreconciled_magnitude_risk"
        if smoothness_status == "executed"
        and smoothness.get("finite_likelihoods")
        and smoothness.get("finite_gradients")
        else "structured_blocker"
    )
    fixed_status = fixed_sinkhorn.get("status")
    return [
        {
            "gap": "paper_code_section_5_1_settings",
            "status": paper_code["status"],
            "primary_risk": "transition covariance paper/code ambiguity remains explicit",
        },
        {
            "gap": "filterflow_smoothness_gradient",
            "status": smoothness_decision,
            "primary_risk": (
                "finite gradients only; severe scalar/randomness/gradient-magnitude "
                "mismatch remains unresolved and full Figure 1/Table 4 reproduction "
                "remains future work"
                if smoothness_decision.startswith("closed")
                else smoothness_status
            ),
        },
        {
            "gap": "bayesfilter_fixed_target_sinkhorn_epsilon_0.25",
            "status": fixed_status,
            "primary_risk": "diagnostic only; not filterflow annealed transport equivalence",
        },
    ]


def _decision(gap_ledger: list[dict[str, Any]]) -> str:
    if any(row["status"] == "structured_blocker" for row in gap_ledger):
        return "final_gaps_partially_closed_with_structured_blocker"
    fixed = next(row for row in gap_ledger if row["gap"] == "bayesfilter_fixed_target_sinkhorn_epsilon_0.25")
    if fixed["status"] == "epsilon_0.25_unconditional_nontriggered_budget_gap":
        return "final_gaps_closed_unconditional_fixed_sinkhorn_compute_gap_identified"
    if fixed["status"] == "epsilon_0.25_iteration_budget_gap":
        return "final_gaps_closed_fixed_sinkhorn_budget_gap_identified"
    return "final_gaps_closed_except_fixed_sinkhorn_design_risk"


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# Filterflow Final Gaps Closure

## Decision

`{payload['decision']}`

## Gap Ledger

{_gap_table(payload['gap_ledger'])}

## Paper/Code Settings

Status: `{payload['paper_code_ledger']['status']}`

{_settings_table(payload['paper_code_ledger']['settings_rows'])}

Interpretation: {payload['paper_code_ledger']['interpretation']}

## Smoothness Gradient

Status: `{payload['smoothness_gradient_replication']['status']}`

{_smoothness_summary(payload['smoothness_gradient_replication'])}

## Fixed-Target Sinkhorn

Status: `{payload['fixed_target_sinkhorn_diagnosis']['status']}`

{_sinkhorn_table(payload['fixed_target_sinkhorn_diagnosis'].get('rows', []))}

## Filterflow

- Branch: `{payload['filterflow_status']['branch']}`
- Commit: `{payload['filterflow_status']['commit']}`
- Status: `{payload['filterflow_status']['status']}`
- Diff summary: `{payload['filterflow_status']['diff_summary']}`

## Non-Implications

{_items(payload['non_implications'])}
"""


def _gap_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| Gap | Status | Primary risk |", "| --- | --- | --- |"]
    for row in rows:
        lines.append(f"| `{row['gap']}` | `{row['status']}` | {row['primary_risk']} |")
    return "\n".join(lines)


def _settings_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| Item | Paper | Filterflow code | Status |", "| --- | --- | --- | --- |"]
    for row in rows:
        lines.append(
            f"| `{row['item']}` | {row['paper_section_5_1']} | "
            f"{row['filterflow_simple_linear_comparison']} | `{row['status']}` |"
        )
    return "\n".join(lines)


def _smoothness_summary(smoothness: dict[str, Any]) -> str:
    if smoothness.get("status") != "executed":
        return f"Structured blocker: `{smoothness.get('status')}`."
    fields = [
        ("finite_likelihoods", smoothness["finite_likelihoods"]),
        ("finite_gradients", smoothness["finite_gradients"]),
        ("likelihood_rmse", smoothness["likelihood_rmse"]),
        ("gradient_rmse", smoothness["gradient_rmse"]),
        ("gradient_max_abs_delta", smoothness["gradient_max_abs_delta"]),
        ("gradient_cosine_vs_kalman_fd", smoothness["gradient_cosine_vs_kalman_fd"]),
        ("gradient_sign_agreement", smoothness["gradient_sign_agreement"]),
    ]
    lines = ["| Metric | Value |", "| --- | ---: |"]
    for key, value in fields:
        lines.append(f"| `{key}` | {_fmt(value)} |")
    return "\n".join(lines)


def _sinkhorn_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| epsilon | budget | max residual | median residual | below tol |", "| ---: | ---: | ---: | ---: | --- |"]
    for row in rows:
        lines.append(
            f"| {row['epsilon']} | {row['budget']} | {_fmt(row['max_residual'])} | "
            f"{_fmt(row['median_residual'])} | {row['below_tolerance']} |"
        )
    return "\n".join(lines)


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload["plan_path"] != PLAN_PATH:
        raise RuntimeError("wrong plan path")
    validate_filterflow_reference_status(payload["filterflow_status"])
    if payload["paper_code_ledger"]["status"] != "closed_with_transition_covariance_ambiguity_recorded":
        raise RuntimeError("paper/code ledger did not record expected ambiguity")
    smoothness = payload["smoothness_gradient_replication"]
    if smoothness.get("status") == "executed" and not smoothness.get("finite_gradients"):
        raise RuntimeError("executed smoothness run has non-finite gradients")
    fixed = payload["fixed_target_sinkhorn_diagnosis"]
    if fixed.get("status") == "blocked_no_resampling_states":
        raise RuntimeError("fixed Sinkhorn diagnosis did not collect states")
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only manifest")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing digest")


def _filterflow_status() -> dict[str, Any]:
    return {
        "branch": _git(["git", "-C", str(FILTERFLOW_PATH), "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _git(["git", "-C", str(FILTERFLOW_PATH), "rev-parse", "HEAD"]),
        "status": _git(["git", "-C", str(FILTERFLOW_PATH), "status", "--short", "--branch"]),
        "diff_summary": _git(["git", "-C", str(FILTERFLOW_PATH), "diff", "--stat"]) or "clean",
        "upstream_base": UPSTREAM_FILTERFLOW_COMMIT,
    }


def _git(args: list[str]) -> str:
    completed = subprocess.run(args, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _ess_batched(weights: tf.Tensor) -> tf.Tensor:
    return 1.0 / tf.reduce_sum(tf.cast(weights, DTYPE) * tf.cast(weights, DTYPE), axis=1)


def _pairwise_squared(x: tf.Tensor) -> tf.Tensor:
    diff = x[:, :, None, :] - x[:, None, :, :]
    return tf.reduce_sum(diff * diff, axis=3)


def _median_tensor(values: tf.Tensor) -> tf.Tensor:
    values = tf.sort(tf.reshape(tf.cast(values, DTYPE), [-1]))
    count = tf.shape(values)[0]
    mid = count // 2
    return tf.cond(
        tf.equal(count % 2, 1),
        lambda: values[mid],
        lambda: 0.5 * (values[mid - 1] + values[mid]),
    )


def _float(value: tf.Tensor) -> float:
    return float(tf.cast(value, DTYPE).numpy())


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _items(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _non_implications() -> list[str]:
    return [
        "No production readiness is concluded.",
        "No public API readiness is concluded.",
        "No posterior correctness is concluded.",
        "No HMC readiness is concluded.",
        "No general nonlinear-SSM validity is concluded.",
        "No full paper figure or Table 4 reproduction is concluded.",
        "No claim that fixed-target Sinkhorn is filterflow paper-equivalent is concluded.",
    ]


def _digest_payload(payload: dict[str, Any]) -> str:
    comparable = dict(payload)
    comparable["created_at_utc"] = "TIMESTAMP"
    comparable["run_manifest"] = dict(comparable["run_manifest"])
    comparable["run_manifest"]["wall_time_seconds"] = "WALL_TIME"
    comparable["run_manifest"]["dirty_state_summary"] = "DIRTY_STATE"
    return stable_digest(comparable)


if __name__ == "__main__":
    raise SystemExit(main())
