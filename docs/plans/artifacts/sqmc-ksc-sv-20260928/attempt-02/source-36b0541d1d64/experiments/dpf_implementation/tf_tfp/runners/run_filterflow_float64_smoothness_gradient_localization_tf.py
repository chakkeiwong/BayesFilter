"""Localize the float64 FilterFlow/BayesFilter smoothness gradient mismatch."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
from dataclasses import dataclass
import json
import subprocess
import textwrap
import time
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import samplers

from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_lgssm_step_gradient_comparison_tf as step,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_to_smoothness_ladder_tf as continuation,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_float64_smoothness_gradient_surface_tf as surface,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_r3_float64_trace_replay_tf as r3,
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
    FILTERFLOW_REFERENCE_DTYPE,
    reference_policy,
    validate_filterflow_reference_status,
)


tfd = tfp.distributions

DTYPE = tf.float64
PLAN_PATH = "docs/plans/bayesfilter-dpf-filterflow-float64-smoothness-gradient-localization-plan-2026-06-03.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-filterflow-float64-smoothness-gradient-localization-result-2026-06-03.md"
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_float64_smoothness_gradient_localization_2026-06-03.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-float64-smoothness-gradient-localization-2026-06-03.md"
FILTERFLOW_ENV_PYTHON = REPO_ROOT / ".localenv" / "filterflow-py311" / "bin" / "python"
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"
FILTERFLOW_MARKER_PATH = FILTERFLOW_PATH / FILTERFLOW_BRANCH_MARKER

T = 100
BATCH_SIZE = 1
NUM_PARTICLES = 50
DATA_SEED = 123
FILTER_SEED = 1234
EPSILON = 0.25
SCALING = 0.85
CONVERGENCE_THRESHOLD = 1e-6
MAX_ITERATIONS = 500
RESAMPLING_NEFF = 0.9999
THETA = [0.95, 0.95]
SCALAR_TOLERANCE = 5e-5
GRADIENT_ABS_TOLERANCE = 2e-4
GRADIENT_REL_TOLERANCE = 2e-4
EXPLOSION_RATIO = 1e6


@dataclass(frozen=True)
class RunConfig:
    theta: list[float]
    tag: str | None
    mesh_index: int | None
    plan_path: str
    result_path: str
    json_path: Any
    report_path: Any


@tf.custom_gradient
def _clip_upstream_identity(x: tf.Tensor) -> tuple[tf.Tensor, Any]:
    def grad(dy: tf.Tensor) -> tf.Tensor:
        return tf.clip_by_value(dy, -1.0, 1.0)

    return x, grad


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--theta",
        nargs=2,
        type=float,
        default=THETA,
        metavar=("THETA0", "THETA1"),
        help="Two diagonal transition parameters for the smoothness row.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional artifact tag, for example row-173.",
    )
    parser.add_argument(
        "--mesh-index",
        type=int,
        default=None,
        help="Optional mesh row index recorded in the payload.",
    )
    args = parser.parse_args(argv)
    config = _run_config(args)
    if args.validate_only:
        _validate_payload(load_json(config.json_path))
        return 0

    start = time.perf_counter()
    payload = _run(config)
    payload["run_manifest"]["wall_time_seconds"] = time.perf_counter() - start
    payload["reproducibility_digest"] = _digest_payload(payload)
    write_json(config.json_path, payload)
    markdown = _markdown(payload)
    write_text(config.report_path, markdown)
    write_text(REPO_ROOT / config.result_path, markdown)
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run_config(args: argparse.Namespace) -> RunConfig:
    theta = [float(args.theta[0]), float(args.theta[1])]
    tag = _safe_tag(args.tag)
    if tag is None:
        return RunConfig(
            theta=theta,
            tag=None,
            mesh_index=args.mesh_index,
            plan_path=PLAN_PATH,
            result_path=RESULT_PATH,
            json_path=JSON_PATH,
            report_path=REPORT_PATH,
        )
    return RunConfig(
        theta=theta,
        tag=tag,
        mesh_index=args.mesh_index,
        plan_path=(
            "docs/plans/"
            f"bayesfilter-dpf-filterflow-float64-{tag}-gradient-localization-plan-2026-06-03.md"
        ),
        result_path=(
            "docs/plans/"
            f"bayesfilter-dpf-filterflow-float64-{tag}-gradient-localization-result-2026-06-03.md"
        ),
        json_path=OUTPUT_DIR / f"dpf_filterflow_float64_{tag.replace('-', '_')}_gradient_localization_2026-06-03.json",
        report_path=REPORT_DIR / f"dpf-filterflow-float64-{tag}-gradient-localization-2026-06-03.md",
    )


def _safe_tag(tag: str | None) -> str | None:
    if tag is None or tag.strip() == "":
        return None
    lowered = tag.strip().lower()
    allowed = []
    for char in lowered:
        if char.isalnum() or char == "-":
            allowed.append(char)
        elif char in {"_", " "}:
            allowed.append("-")
        else:
            raise ValueError(f"unsupported tag character: {char!r}")
    safe = "".join(allowed).strip("-")
    if not safe:
        raise ValueError("tag must contain at least one alphanumeric character")
    return safe


def _run(config: RunConfig) -> dict[str, Any]:
    reference_status = r3._filterflow_status()
    validate_filterflow_reference_status(reference_status, marker_path=FILTERFLOW_MARKER_PATH)
    initial_fingerprint = continuation._filterflow_fingerprint()
    filterflow = _filterflow_localization_subprocess(config)
    if filterflow.get("status") != "executed":
        return _blocked_payload(
            config,
            "filterflow_float64_smoothness_gradient_localization_filterflow_blocker",
            filterflow.get("blocker", "unknown filterflow blocker"),
            initial_fingerprint,
            filterflow,
            reference_status,
    )

    bayesfilter = _bayesfilter_localization(filterflow)
    comparison = _compare_localization(filterflow, bayesfilter)
    final_fingerprint = continuation._filterflow_fingerprint()
    comparator_drift = continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint)
    decision = _decision(comparison, comparator_drift)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": config.plan_path,
        "result_path": config.result_path,
        "report_path": str(config.report_path.relative_to(REPO_ROOT)),
        "json_path": str(config.json_path.relative_to(REPO_ROOT)),
        "question": (
            "Where does the BayesFilter raw gradient path diverge from local "
            f"float64 FilterFlow on smoothness row theta={config.theta}?"
        ),
        "target_row": _target_row(config),
        "evidence_contract": _evidence_contract(),
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": comparator_drift,
        "model_contract": _model_contract(config),
        "gradient_localization_contract": _gradient_localization_contract(),
        "filterflow_localization": _compact_filterflow(filterflow),
        "bayesfilter_localization": _compact_bayesfilter(bayesfilter),
        "comparison": comparison,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "tolerances": _tolerances(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_float64_smoothness_gradient_localization_tf"
                f"{_command_suffix(config)}"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "decision_table": _decision_table(decision, comparison),
        "non_implications": _non_implications(),
    }


def _blocked_payload(
    config: RunConfig,
    decision: str,
    blocker: str,
    initial_fingerprint: dict[str, Any],
    filterflow: dict[str, Any],
    reference_status: dict[str, Any],
) -> dict[str, Any]:
    final_fingerprint = continuation._filterflow_fingerprint()
    comparison = {"status": "blocked", "blocker": blocker}
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": config.plan_path,
        "result_path": config.result_path,
        "blocker": blocker,
        "target_row": _target_row(config),
        "evidence_contract": _evidence_contract(),
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint),
        "model_contract": _model_contract(config),
        "gradient_localization_contract": _gradient_localization_contract(),
        "filterflow_localization": filterflow,
        "bayesfilter_localization": None,
        "comparison": comparison,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "tolerances": _tolerances(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_float64_smoothness_gradient_localization_tf"
                f"{_command_suffix(config)}"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "decision_table": _decision_table(decision, comparison),
        "non_implications": _non_implications(),
    }


def _target_row(config: RunConfig) -> dict[str, Any]:
    return {
        "tag": config.tag,
        "mesh_index": config.mesh_index,
        "theta": config.theta,
        "comparator": "local_float64_filterflow_branch",
    }


def _command_suffix(config: RunConfig) -> str:
    parts = [f" --theta {config.theta[0]!r} {config.theta[1]!r}"]
    if config.tag is not None:
        parts.append(f" --tag {config.tag}")
    if config.mesh_index is not None:
        parts.append(f" --mesh-index {config.mesh_index}")
    return "".join(parts)


def _filterflow_localization_subprocess(config: RunConfig) -> dict[str, Any]:
    if not FILTERFLOW_ENV_PYTHON.exists():
        return {
            "status": "blocked",
            "blocker": f"missing filterflow env python: {FILTERFLOW_ENV_PYTHON}",
        }
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = str(FILTERFLOW_PATH)
    env["MPLCONFIGDIR"] = str(REPO_ROOT / ".cache" / "filterflow-mpl")
    completed = subprocess.run(
        [str(FILTERFLOW_ENV_PYTHON), "-c", _filterflow_localization_script(config)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    if completed.returncode != 0:
        return {
            "status": "blocked",
            "blocker": "filterflow gradient localization subprocess failed",
            "returncode": completed.returncode,
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    stdout = completed.stdout
    start = stdout.rfind("FILTERFLOW_GRADIENT_LOCALIZATION_JSON_BEGIN")
    end = stdout.rfind("FILTERFLOW_GRADIENT_LOCALIZATION_JSON_END")
    if start < 0 or end < 0 or end <= start:
        return {
            "status": "blocked",
            "blocker": "filterflow gradient localization JSON sentinels missing",
            "stdout_excerpt": stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    payload = json.loads(stdout[start + len("FILTERFLOW_GRADIENT_LOCALIZATION_JSON_BEGIN"):end].strip())
    payload["stderr_excerpt"] = completed.stderr[-2000:]
    return payload


def _filterflow_localization_script(config: RunConfig) -> str:
    return textwrap.dedent(
        f"""
        import inspect
        import json
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")

        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec

        np = __import__("numpy")
        import attr
        import tensorflow as tf
        from tensorflow_probability.python.internal import samplers

        from filterflow.base import State
        from filterflow.models.simple_linear_gaussian import make_filter
        from filterflow.resampling import RegularisedTransform
        from filterflow.resampling.criterion import NeffCriterion
        from scripts.simple_linear_smoothness import get_data, routine

        DTYPE = tf.float64
        NP_DTYPE = np.float64
        T = {T}
        BATCH_SIZE = {BATCH_SIZE}
        N = {NUM_PARTICLES}
        DATA_SEED = {DATA_SEED}
        FILTER_SEED = {FILTER_SEED}
        EPSILON = {EPSILON!r}
        SCALING = {SCALING!r}
        CONVERGENCE_THRESHOLD = {CONVERGENCE_THRESHOLD!r}
        MAX_ITERATIONS = {MAX_ITERATIONS}
        RESAMPLING_NEFF = {RESAMPLING_NEFF!r}
        THETA = np.array({config.theta!r}, dtype=NP_DTYPE)

        def to_json(tensor):
            return tf.cast(tensor, tf.float64).numpy().tolist()

        def scalar(tensor):
            return float(tf.cast(tensor, tf.float64).numpy())

        transition_matrix_np = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=NP_DTYPE)
        transition_covariance_np = np.array([[1.0 / 3.0, 0.5], [0.5, 1.0]], dtype=NP_DTYPE)
        observation_matrix_np = np.array([[1.0, 0.0]], dtype=NP_DTYPE)
        observation_covariance_np = np.array([[0.01]], dtype=NP_DTYPE)

        rng = np.random.RandomState(seed=DATA_SEED)
        observations_np, _kf = get_data(
            transition_matrix_np,
            observation_matrix_np,
            transition_covariance_np,
            observation_covariance_np,
            T=T,
            random_state=rng,
        )
        initial_particles_np = rng.normal(0.0, 0.01, [BATCH_SIZE, N, 2]).astype(NP_DTYPE)

        observations = tf.convert_to_tensor(observations_np, dtype=DTYPE)
        initial_state = State(tf.convert_to_tensor(initial_particles_np, dtype=DTYPE))
        modifiable_transition_matrix = tf.Variable(transition_matrix_np, trainable=False, dtype=DTYPE)
        transition_matrix = tf.linalg.diag(tf.convert_to_tensor(THETA, dtype=DTYPE)) + tf.constant(
            [[0.0, 1.0], [0.0, 0.0]],
            dtype=DTYPE,
        )
        modifiable_transition_matrix.assign(transition_matrix)
        observation_matrix = tf.convert_to_tensor(observation_matrix_np, dtype=DTYPE)
        transition_covariance_chol = tf.linalg.cholesky(
            tf.convert_to_tensor(transition_covariance_np, dtype=DTYPE)
        )
        observation_covariance_chol = tf.linalg.cholesky(
            tf.convert_to_tensor(observation_covariance_np, dtype=DTYPE)
        )
        smc = make_filter(
            observation_matrix,
            modifiable_transition_matrix,
            observation_covariance_chol,
            transition_covariance_chol,
            RegularisedTransform(
                epsilon=EPSILON,
                scaling=SCALING,
                convergence_threshold=CONVERGENCE_THRESHOLD,
                max_iter=MAX_ITERATIONS,
            ),
            NeffCriterion(RESAMPLING_NEFF, True),
            optimal_proposal=True,
        )

        full_ll, full_grad_matrix, full_ess = routine(
            smc,
            initial_state,
            False,
            tf.data.Dataset.from_tensor_slices(observations),
            tf.constant(T),
            modifiable_transition_matrix,
            tf.constant(FILTER_SEED, dtype=tf.int32),
        )

        seed = tf.constant(FILTER_SEED, dtype=tf.int32)
        paddings = tf.stack([[0, 0], [0, 2 - tf.size(seed)]])
        seed = tf.squeeze(tf.pad(tf.reshape(seed, [1, -1]), paddings))

        state = initial_state
        cumulative_rows = []
        for time_index in range(T):
            seed, seed1, seed2 = samplers.split_seed(seed, n=3, salt="update")
            observation = observations[time_index]
            with tf.GradientTape() as tape:
                tape.watch(modifiable_transition_matrix)
                state_for_step = state
                resampling_flag, ess = smc._resampling_criterion.apply(state_for_step)
                float_t = tf.cast(state_for_step.t, DTYPE)
                float_t_1 = float_t + tf.constant(1.0, dtype=DTYPE)
                state_for_step = attr.evolve(
                    state_for_step,
                    ess=ess / float_t_1 + state_for_step.ess * (float_t / float_t_1),
                )
                resampled_state = smc._resampling_method.apply(state_for_step, resampling_flag, seed1)
                new_state = smc.propose_and_weight(
                    resampled_state,
                    observation,
                    tf.convert_to_tensor(time_index, dtype=tf.int32),
                    seed2,
                )
                new_state = smc._resampling_correction_term(
                    resampling_flag,
                    new_state,
                    state_for_step,
                    observation,
                    tf.convert_to_tensor(time_index, dtype=tf.int32),
                )
                new_state = attr.evolve(new_state, t=state_for_step.t + 1)
                scalar_value = tf.reduce_mean(new_state.log_likelihoods)
            grad_matrix = tape.gradient(scalar_value, modifiable_transition_matrix)
            state = new_state
            cumulative_rows.append({{
                "time_index": int(time_index),
                "cumulative_mean_log_likelihood": scalar(scalar_value),
                "gradient_diag": to_json(tf.linalg.diag_part(grad_matrix)),
                "gradient_matrix": to_json(grad_matrix),
                "ess_before_resampling": to_json(ess),
                "resampling_flag": [bool(v) for v in tf.reshape(resampling_flag, [-1]).numpy().tolist()],
                "finite_scalar": bool(tf.math.is_finite(scalar_value).numpy()),
                "finite_gradient": bool(tf.reduce_all(tf.math.is_finite(grad_matrix)).numpy()),
            }})

        payload = {{
            "status": "executed",
            "localization_contract": "first_failing_smoothness_row_per_time_cumulative_gradient",
            "settings": {{
                "T": T,
                "batch_size": BATCH_SIZE,
                "n_particles": N,
                "data_seed": DATA_SEED,
                "filter_seed": FILTER_SEED,
                "epsilon": EPSILON,
                "scaling": SCALING,
                "convergence_threshold": CONVERGENCE_THRESHOLD,
                "max_iter": MAX_ITERATIONS,
                "resampling_neff": RESAMPLING_NEFF,
                "optimal_proposal": True,
                "resampling_correction": False,
                "dtype": "float64",
                "theta": THETA.astype(float).tolist(),
            }},
            "model": {{
                "transition_matrix": to_json(transition_matrix),
                "transition_covariance": transition_covariance_np.astype(float).tolist(),
                "transition_covariance_chol": to_json(transition_covariance_chol),
                "observation_matrix": observation_matrix_np.astype(float).tolist(),
                "observation_covariance": observation_covariance_np.astype(float).tolist(),
                "observation_covariance_chol": to_json(observation_covariance_chol),
                "observations": observations_np.astype(float).tolist(),
                "initial_particles": initial_particles_np.astype(float).tolist(),
                "num_particles": N,
            }},
            "full_routine": {{
                "mean_log_likelihood": scalar(full_ll),
                "gradient_diag": to_json(tf.linalg.diag_part(full_grad_matrix)),
                "gradient_matrix": to_json(full_grad_matrix),
                "final_ess": to_json(full_ess),
                "finite_scalar": bool(tf.math.is_finite(full_ll).numpy()),
                "finite_gradient": bool(tf.reduce_all(tf.math.is_finite(full_grad_matrix)).numpy()),
            }},
            "cumulative_rows": cumulative_rows,
            "observation_checksum": float(np.sum(observations_np.astype(np.float64))),
            "initial_particles_checksum": float(np.sum(initial_particles_np.astype(np.float64))),
            "package_versions": {{
                "python": os.sys.version.split()[0],
                "tensorflow": tf.__version__,
                "numpy": np.__version__,
            }},
            "cpu_only_manifest": {{
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
                "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
            }},
            "source_gradient_note": (
                "FilterFlow RegularisedTransform transport uses @tf.custom_gradient "
                "and clips upstream d_transport to [-1, 1]."
            ),
        }}
        print("FILTERFLOW_GRADIENT_LOCALIZATION_JSON_BEGIN")
        print(json.dumps(payload, sort_keys=True))
        print("FILTERFLOW_GRADIENT_LOCALIZATION_JSON_END")
        """
    )


def _bayesfilter_localization(filterflow: dict[str, Any]) -> dict[str, Any]:
    original_dtype = annealed_transport_tf.DTYPE
    annealed_transport_tf.DTYPE = DTYPE
    try:
        modes = [
            "raw",
            "transport_upstream_clip",
            "transport_matrix_stop_gradient",
            "post_resample_state_stop_gradient",
            "proposal_mean_stop_gradient",
            "proposal_sample_stop_gradient",
            "transport_clip_proposal_sample_stop_gradient",
            "proposal_log_prob_stop_gradient",
            "transition_log_prob_stop_gradient",
            "normalized_weights_stop_gradient",
        ]
        rows = [_bayesfilter_mode_ledger(mode, filterflow) for mode in modes]
        return {
            "status": "executed",
            "backend": "tensorflow_tensorflow_probability",
            "localization_contract": "same fixed data/seeds with gradient-path ablations",
            "modes": rows,
            "finite_values": all(row["finite_values"] for row in rows),
            "cpu_only_manifest": _parent_cpu_manifest(),
        }
    finally:
        annealed_transport_tf.DTYPE = original_dtype


def _bayesfilter_mode_ledger(mode: str, filterflow: dict[str, Any]) -> dict[str, Any]:
    model = filterflow["model"]
    settings = filterflow["settings"]
    theta_variable = tf.Variable(settings["theta"], dtype=DTYPE)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(theta_variable)
        cumulative_scalars, final_scalar, diagnostics = _bayesfilter_cumulative_scalars(
            theta_variable,
            model=model,
            settings=settings,
            mode=mode,
        )
    cumulative_gradients = []
    finite_gradients = True
    for scalar_value in cumulative_scalars:
        gradient = tape.gradient(scalar_value, theta_variable)
        if gradient is None:
            gradient = tf.fill([2], tf.constant(float("nan"), dtype=DTYPE))
        finite_gradients = finite_gradients and bool(
            tf.reduce_all(tf.math.is_finite(gradient)).numpy()
        )
        cumulative_gradients.append(gradient)
    final_gradient = tape.gradient(final_scalar, theta_variable)
    del tape
    if final_gradient is None:
        final_gradient = tf.fill([2], tf.constant(float("nan"), dtype=DTYPE))
    finite_gradients = finite_gradients and bool(
        tf.reduce_all(tf.math.is_finite(final_gradient)).numpy()
    )
    rows = []
    for diag, scalar_value, gradient in zip(diagnostics, cumulative_scalars, cumulative_gradients, strict=True):
        rows.append(
            {
                **diag,
                "cumulative_mean_log_likelihood": _float(scalar_value),
                "gradient_diag": r3._json(gradient),
                "gradient_max_abs": _max_abs_list(r3._json(gradient)),
                "finite_scalar": bool(tf.math.is_finite(scalar_value).numpy()),
                "finite_gradient": bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy()),
            }
        )
    return {
        "mode": mode,
        "mode_description": _mode_description(mode),
        "final_mean_log_likelihood": _float(final_scalar),
        "final_gradient_diag": r3._json(final_gradient),
        "final_gradient_max_abs": _max_abs_list(r3._json(final_gradient)),
        "rows": rows,
        "finite_values": bool(
            tf.math.is_finite(final_scalar).numpy()
            and tf.reduce_all(tf.math.is_finite(final_gradient)).numpy()
            and all(row["finite_scalar"] and row["finite_gradient"] for row in rows)
        ),
        "finite_gradients": finite_gradients,
    }


def _bayesfilter_cumulative_scalars(
    theta: tf.Tensor,
    *,
    model: dict[str, Any],
    settings: dict[str, Any],
    mode: str,
) -> tuple[list[tf.Tensor], tf.Tensor, list[dict[str, Any]]]:
    transition_matrix = _transition_matrix(theta)
    observation_matrix = tf.constant(model["observation_matrix"], dtype=DTYPE)
    transition_chol = tf.constant(model["transition_covariance_chol"], dtype=DTYPE)
    observation_chol = tf.constant(model["observation_covariance_chol"], dtype=DTYPE)
    observations = tf.constant(model["observations"], dtype=DTYPE)
    particles = tf.constant(model["initial_particles"], dtype=DTYPE)
    num_particles = int(settings["n_particles"])
    log_weights = tf.fill(
        [BATCH_SIZE, num_particles],
        -tf.math.log(tf.cast(num_particles, DTYPE)),
    )
    log_likelihoods = tf.zeros([BATCH_SIZE], dtype=DTYPE)
    transition_noise = tfd.MultivariateNormalTriL(
        loc=tf.zeros([2], dtype=DTYPE),
        scale_tril=transition_chol,
    )
    observation_noise = tfd.MultivariateNormalTriL(
        loc=tf.zeros([1], dtype=DTYPE),
        scale_tril=observation_chol,
    )
    transition_cov_inv = tf.linalg.cholesky_solve(transition_chol, tf.eye(2, dtype=DTYPE))
    observation_cov_inv = tf.linalg.cholesky_solve(observation_chol, tf.eye(1, dtype=DTYPE))
    sigma_inv = transition_cov_inv + tf.linalg.matmul(
        tf.linalg.matmul(
            observation_matrix,
            observation_cov_inv,
            transpose_a=True,
        ),
        observation_matrix,
    )
    sigma = tf.linalg.inv(sigma_inv)
    sigma_chol = tf.linalg.cholesky(sigma)
    seed = tf.constant(FILTER_SEED, dtype=tf.int32)
    paddings = tf.stack([[0, 0], [0, 2 - tf.size(seed)]])
    seed = tf.squeeze(tf.pad(tf.reshape(seed, [1, -1]), paddings))

    cumulative_scalars = []
    diagnostics = []
    for time_index in range(T):
        seed, seed1, seed2 = samplers.split_seed(seed, n=3, salt="update")
        del seed1
        ess_log = r3._ess(log_weights)
        flags = ess_log <= tf.math.log(
            tf.cast(num_particles, DTYPE) * tf.constant(RESAMPLING_NEFF, DTYPE)
        )
        bool_flags = tf.reshape(flags, [-1])
        transport_status = "not_triggered"
        row_residual = 0.0
        column_residual = 0.0
        if bool(tf.reduce_any(bool_flags).numpy()):
            transport_gradient_mode = (
                "filterflow_clipped"
                if mode in {
                    "transport_upstream_clip",
                    "transport_clip_proposal_sample_stop_gradient",
                }
                else "raw"
            )
            transported = annealed_transport_tf.annealed_transport_resample_tf(
                particles,
                log_weights,
                epsilon=float(settings["epsilon"]),
                scaling=float(settings["scaling"]),
                convergence_threshold=float(settings["convergence_threshold"]),
                max_iterations=int(settings["max_iter"]),
                ess_mask=bool_flags,
                transport_gradient_mode=transport_gradient_mode,
                application_mode="filterflow_all_rows",
            )
            transport_matrix = tf.cast(transported.transport_matrix, DTYPE)
            if mode in {
                "transport_upstream_clip",
                "transport_clip_proposal_sample_stop_gradient",
            }:
                del transport_matrix
                particles = transported.particles
                log_weights = transported.log_weights
                transport_status = (
                    "computed_with_clipped_upstream_gradient_filterflow_all_rows"
                )
            elif mode == "transport_matrix_stop_gradient":
                particles = tf.linalg.matmul(tf.stop_gradient(transport_matrix), particles)
                log_weights = transported.log_weights
                transport_status = "computed_transport_matrix_stop_gradient"
            elif mode == "post_resample_state_stop_gradient":
                particles = tf.stop_gradient(transported.particles)
                log_weights = tf.stop_gradient(transported.log_weights)
                transport_status = "computed_post_resample_state_stop_gradient"
            else:
                particles = transported.particles
                log_weights = transported.log_weights
                transport_status = "computed_raw_transport_gradient_filterflow_all_rows"
            row_residual = transported.diagnostics["max_row_residual"]
            column_residual = transported.diagnostics["max_column_residual"]
        observation = observations[time_index]
        proposal_mean = _optimal_proposal_mean(
            particles,
            observation,
            transition_matrix,
            observation_matrix,
            transition_cov_inv,
            observation_cov_inv,
            sigma,
        )
        if mode == "proposal_mean_stop_gradient":
            proposal_mean = tf.stop_gradient(proposal_mean)
        proposal_dist = tfd.MultivariateNormalTriL(proposal_mean, sigma_chol)
        proposed_particles = proposal_dist.sample(seed=seed2)
        if mode in {
            "proposal_sample_stop_gradient",
            "transport_clip_proposal_sample_stop_gradient",
        }:
            proposed_particles = tf.stop_gradient(proposed_particles)
        observation_ll = r3._observation_log_prob(
            proposed_particles,
            observation,
            observation_matrix,
            observation_noise,
        )
        transition_ll = r3._transition_log_prob(
            particles,
            proposed_particles,
            transition_matrix,
            transition_noise,
        )
        if mode == "transition_log_prob_stop_gradient":
            transition_ll = tf.stop_gradient(transition_ll)
        proposal_ll = r3._optimal_proposal_log_prob(
            particles,
            proposed_particles,
            observation,
            transition_matrix,
            observation_matrix,
            transition_cov_inv,
            observation_cov_inv,
            sigma,
            sigma_chol,
        )
        if mode == "proposal_log_prob_stop_gradient":
            proposal_ll = tf.stop_gradient(proposal_ll)
        unnormalized = transition_ll + observation_ll - proposal_ll + log_weights
        increment = tf.reduce_logsumexp(unnormalized, axis=1)
        log_likelihoods = log_likelihoods + increment
        normalized = r3._filterflow_normalize(unnormalized, num_particles)
        if mode == "normalized_weights_stop_gradient":
            normalized = tf.stop_gradient(normalized)
        log_weights = normalized
        particles = proposed_particles
        cumulative_scalars.append(tf.reduce_mean(log_likelihoods))
        diagnostics.append(
            {
                "time_index": time_index,
                "resampling_flag": [bool(v) for v in bool_flags.numpy().tolist()],
                "log_neff_before_resampling": r3._json(ess_log),
                "transport_status": transport_status,
                "max_row_residual": row_residual,
                "max_column_residual": column_residual,
                "log_likelihood_increment": r3._json(increment),
            }
        )
    return cumulative_scalars, tf.reduce_mean(log_likelihoods), diagnostics


def _transition_matrix(theta: tf.Tensor) -> tf.Tensor:
    return tf.linalg.diag(tf.cast(theta, DTYPE)) + tf.constant(
        [[0.0, 1.0], [0.0, 0.0]],
        dtype=DTYPE,
    )


def _optimal_proposal_mean(
    prior_particles: tf.Tensor,
    observation: tf.Tensor,
    transition_matrix: tf.Tensor,
    observation_matrix: tf.Tensor,
    transition_cov_inv: tf.Tensor,
    observation_cov_inv: tf.Tensor,
    sigma: tf.Tensor,
) -> tf.Tensor:
    mean = tf.linalg.matvec(
        observation_matrix,
        tf.linalg.matvec(observation_cov_inv, observation),
        transpose_a=True,
    )
    mean = mean + tf.linalg.matvec(
        transition_cov_inv,
        tf.linalg.matvec(transition_matrix, prior_particles),
    )
    return tf.linalg.matvec(sigma, mean)


def _compare_localization(
    filterflow: dict[str, Any],
    bayesfilter: dict[str, Any],
) -> dict[str, Any]:
    if bayesfilter.get("status") != "executed":
        return {
            "status": "blocked",
            "blocker": bayesfilter.get("blocker", "BayesFilter localization did not execute"),
        }
    ff_rows = filterflow["cumulative_rows"]
    ff_full = filterflow["full_routine"]
    mode_summaries = []
    first_raw_gradient_failure = None
    first_raw_scalar_failure = None
    best_ablation = None
    for mode_payload in bayesfilter["modes"]:
        rows = []
        for ff_row, bf_row in zip(ff_rows, mode_payload["rows"], strict=True):
            scalar_delta = abs(
                float(bf_row["cumulative_mean_log_likelihood"])
                - float(ff_row["cumulative_mean_log_likelihood"])
            )
            gradient_delta = [
                float(bf_value) - float(ff_value)
                for bf_value, ff_value in zip(
                    bf_row["gradient_diag"],
                    ff_row["gradient_diag"],
                    strict=True,
                )
            ]
            max_abs_gradient_delta = max(abs(value) for value in gradient_delta)
            max_abs_ff_gradient = max(abs(float(value)) for value in ff_row["gradient_diag"])
            relative_gradient_delta = max_abs_gradient_delta / max(1.0, max_abs_ff_gradient)
            row = {
                "time_index": ff_row["time_index"],
                "scalar_delta": scalar_delta,
                "gradient_delta": gradient_delta,
                "max_abs_gradient_delta": max_abs_gradient_delta,
                "relative_gradient_delta": relative_gradient_delta,
                "filterflow_gradient_max_abs": max_abs_ff_gradient,
                "bayesfilter_gradient_max_abs": bf_row["gradient_max_abs"],
                "scalar_within_tolerance": scalar_delta <= SCALAR_TOLERANCE,
                "gradient_within_tolerance": (
                    max_abs_gradient_delta <= GRADIENT_ABS_TOLERANCE
                    or relative_gradient_delta <= GRADIENT_REL_TOLERANCE
                ),
                "gradient_explosion_ratio": bf_row["gradient_max_abs"] / max(1.0, max_abs_ff_gradient),
                "resampling_flag": bf_row["resampling_flag"],
                "transport_status": bf_row["transport_status"],
            }
            rows.append(row)
        first_scalar_failure = _first_failure(rows, "scalar_within_tolerance")
        first_gradient_failure = _first_failure(rows, "gradient_within_tolerance")
        explosion_row = _first_explosion(rows)
        final_gradient_delta = [
            float(bf_value) - float(ff_value)
            for bf_value, ff_value in zip(
                mode_payload["final_gradient_diag"],
                ff_full["gradient_diag"],
                strict=True,
            )
        ]
        final_max_abs_gradient_delta = max(abs(value) for value in final_gradient_delta)
        final_filterflow_gradient_max_abs = _max_abs_list(ff_full["gradient_diag"])
        final_relative_gradient_delta = final_max_abs_gradient_delta / max(
            1.0,
            final_filterflow_gradient_max_abs,
        )
        final_scalar_delta = abs(
            float(mode_payload["final_mean_log_likelihood"]) - float(ff_full["mean_log_likelihood"])
        )
        summary = {
            "mode": mode_payload["mode"],
            "mode_description": mode_payload["mode_description"],
            "finite_values": mode_payload["finite_values"],
            "first_scalar_failure": first_scalar_failure,
            "first_gradient_failure": first_gradient_failure,
            "first_gradient_explosion": explosion_row,
            "final_scalar_delta": final_scalar_delta,
            "final_gradient_delta": final_gradient_delta,
            "final_max_abs_gradient_delta": final_max_abs_gradient_delta,
            "final_relative_gradient_delta": final_relative_gradient_delta,
            "final_filterflow_gradient_diag": ff_full["gradient_diag"],
            "final_bayesfilter_gradient_diag": mode_payload["final_gradient_diag"],
            "final_filterflow_gradient_max_abs": final_filterflow_gradient_max_abs,
            "final_bayesfilter_gradient_max_abs": mode_payload["final_gradient_max_abs"],
            "final_gradient_within_tolerance": (
                final_max_abs_gradient_delta <= GRADIENT_ABS_TOLERANCE
                or final_relative_gradient_delta <= GRADIENT_REL_TOLERANCE
            ),
            "sample_rows": _sample_rows(rows),
        }
        mode_summaries.append(summary)
        if mode_payload["mode"] == "raw":
            first_raw_gradient_failure = first_gradient_failure
            first_raw_scalar_failure = first_scalar_failure
        if best_ablation is None or summary["final_relative_gradient_delta"] < best_ablation["final_relative_gradient_delta"]:
            best_ablation = summary
    raw_summary = next(row for row in mode_summaries if row["mode"] == "raw")
    transport_clip_summary = next(
        row for row in mode_summaries if row["mode"] == "transport_upstream_clip"
    )
    return {
        "status": "localized",
        "primary_question": "first_raw_gradient_divergence_time",
        "first_raw_scalar_failure": first_raw_scalar_failure,
        "first_raw_gradient_failure": first_raw_gradient_failure,
        "raw_summary": raw_summary,
        "best_ablation": best_ablation,
        "transport_clip_summary": transport_clip_summary,
        "mode_summaries": mode_summaries,
        "interpretive_hint": _interpretive_hint(raw_summary, transport_clip_summary, best_ablation),
    }


def _first_failure(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    for row in rows:
        if not row[field]:
            return {
                "status": "failure",
                "time_index": row["time_index"],
                "scalar_delta": row["scalar_delta"],
                "max_abs_gradient_delta": row["max_abs_gradient_delta"],
                "relative_gradient_delta": row["relative_gradient_delta"],
                "gradient_explosion_ratio": row["gradient_explosion_ratio"],
                "resampling_flag": row["resampling_flag"],
                "transport_status": row["transport_status"],
            }
    return {"status": "no_failure"}


def _first_explosion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if row["gradient_explosion_ratio"] > EXPLOSION_RATIO:
            return {
                "status": "explosion",
                "time_index": row["time_index"],
                "gradient_explosion_ratio": row["gradient_explosion_ratio"],
                "bayesfilter_gradient_max_abs": row["bayesfilter_gradient_max_abs"],
                "filterflow_gradient_max_abs": row["filterflow_gradient_max_abs"],
                "resampling_flag": row["resampling_flag"],
                "transport_status": row["transport_status"],
            }
    return {"status": "no_explosion"}


def _sample_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = {0, 1, 2, 3, 4, 9, 24, 49, 74, 99}
    selected.update(
        row["time_index"]
        for row in rows
        if not row["gradient_within_tolerance"] or row["gradient_explosion_ratio"] > EXPLOSION_RATIO
    )
    return [row for row in rows if row["time_index"] in selected]


def _decision(comparison: dict[str, Any], comparator_drift: bool) -> str:
    if comparator_drift:
        return "filterflow_float64_smoothness_gradient_localization_blocked_by_comparator_drift"
    if comparison.get("status") != "localized":
        return "filterflow_float64_smoothness_gradient_localization_blocked"
    if comparison["first_raw_scalar_failure"].get("status") == "failure":
        return "filterflow_float64_smoothness_gradient_localization_scalar_veto"
    if comparison["first_raw_gradient_failure"].get("status") == "failure":
        return "filterflow_float64_smoothness_gradient_localized"
    return "filterflow_float64_smoothness_gradient_localization_no_gradient_mismatch"


def _interpretive_hint(
    raw_summary: dict[str, Any],
    transport_clip_summary: dict[str, Any],
    best_ablation: dict[str, Any] | None,
) -> dict[str, Any]:
    raw_rel = raw_summary["final_relative_gradient_delta"]
    clip_rel = transport_clip_summary["final_relative_gradient_delta"]
    if clip_rel < raw_rel / 1e6:
        hint = "transport_custom_gradient_clipping_is_primary_suspect"
    elif best_ablation is not None and best_ablation["final_relative_gradient_delta"] < raw_rel / 1e6:
        hint = f"{best_ablation['mode']}_is_primary_suspect"
    else:
        hint = "no_single_ablation_collapsed_gradient_scale"
    return {
        "status": hint,
        "raw_final_relative_gradient_delta": raw_rel,
        "transport_clip_final_relative_gradient_delta": clip_rel,
        "best_ablation_mode": None if best_ablation is None else best_ablation["mode"],
        "best_ablation_final_relative_gradient_delta": (
            None if best_ablation is None else best_ablation["final_relative_gradient_delta"]
        ),
    }


def _evidence_contract() -> dict[str, Any]:
    return {
        "primary_comparator": "local executable float64 filterflow checkout",
        "primary_question": "cross_implementation_gradient_difference_localization_only",
        "primary_pass": "identify first raw gradient divergence time with scalar path still aligned",
        "scalar_tolerance": SCALAR_TOLERANCE,
        "gradient_abs_tolerance": GRADIENT_ABS_TOLERANCE,
        "gradient_rel_tolerance": GRADIENT_REL_TOLERANCE,
        "finite_gradients": "veto_gate_not_correctness_claim",
        "mathematical_correctness": "not_concluded",
    }


def _model_contract(config: RunConfig) -> dict[str, Any]:
    return {
        "model": "filterflow_simple_linear_smoothness_constant_velocity_lgssm",
        "theta": config.theta,
        "mesh_index": config.mesh_index,
        "artifact_tag": config.tag,
        "transition_matrix": "A(theta)=diag(theta_1, theta_2)+[[0,1],[0,0]]",
        "transition_covariance": [[1.0 / 3.0, 0.5], [0.5, 1.0]],
        "observation_matrix": [[1.0, 0.0]],
        "observation_covariance": [[0.01]],
        "T": T,
        "batch_size": BATCH_SIZE,
        "num_particles": NUM_PARTICLES,
        "data_seed": DATA_SEED,
        "filter_seed": FILTER_SEED,
        "epsilon": EPSILON,
        "scaling": SCALING,
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "max_iter": MAX_ITERATIONS,
        "resampling_neff": RESAMPLING_NEFF,
        "optimal_proposal": True,
        "resampling_correction": False,
        "dtype": FILTERFLOW_REFERENCE_DTYPE,
    }


def _gradient_localization_contract() -> dict[str, str]:
    return {
        "filterflow_cumulative_gradient": "tape.gradient(cumulative_log_likelihood_at_t, transition_matrix_variable)",
        "bayesfilter_cumulative_gradient": "tape.gradient(cumulative_log_likelihood_at_t, theta_variable)",
        "filterflow_transport_backward": "RegularisedTransform transport @tf.custom_gradient clips d_transport to [-1,1]",
        "primary_ablation": "transport_upstream_clip",
        "scalar": "tf.reduce_mean(log_likelihoods), batch_size=1",
        "resampling_correction": "False",
        "correctness_status": "difference audit only",
    }


def _tolerances() -> dict[str, float]:
    return {
        "scalar": SCALAR_TOLERANCE,
        "gradient_abs": GRADIENT_ABS_TOLERANCE,
        "gradient_rel": GRADIENT_REL_TOLERANCE,
        "explosion_ratio": EXPLOSION_RATIO,
    }


def _mode_description(mode: str) -> str:
    descriptions = {
        "raw": "BayesFilter raw TensorFlow gradient through annealed transport",
        "transport_upstream_clip": "Clip upstream gradient entering transport matrix to [-1,1]",
        "transport_matrix_stop_gradient": "Stop gradient through transport matrix only",
        "post_resample_state_stop_gradient": "Stop gradient through post-resampling particles and log weights",
        "proposal_mean_stop_gradient": "Stop gradient through optimal proposal mean",
        "proposal_sample_stop_gradient": "Stop gradient through sampled proposal particles",
        "transport_clip_proposal_sample_stop_gradient": (
            "Clip transport upstream gradient and stop gradient through sampled proposal particles"
        ),
        "proposal_log_prob_stop_gradient": "Stop gradient through proposal log probability",
        "transition_log_prob_stop_gradient": "Stop gradient through transition log probability",
        "normalized_weights_stop_gradient": "Stop gradient through normalized weights after each update",
    }
    return descriptions[mode]


def _compact_filterflow(filterflow: dict[str, Any]) -> dict[str, Any]:
    if filterflow.get("status") != "executed":
        return filterflow
    return {
        "status": filterflow["status"],
        "localization_contract": filterflow["localization_contract"],
        "settings": filterflow["settings"],
        "observation_checksum": filterflow["observation_checksum"],
        "initial_particles_checksum": filterflow["initial_particles_checksum"],
        "full_routine": filterflow["full_routine"],
        "first_rows": filterflow["cumulative_rows"][:5],
        "last_row": filterflow["cumulative_rows"][-1],
        "package_versions": filterflow["package_versions"],
        "cpu_only_manifest": filterflow["cpu_only_manifest"],
        "source_gradient_note": filterflow["source_gradient_note"],
        "stderr_excerpt": filterflow.get("stderr_excerpt", ""),
    }


def _compact_bayesfilter(bayesfilter: dict[str, Any] | None) -> dict[str, Any] | None:
    if bayesfilter is None:
        return None
    return {
        "status": bayesfilter["status"],
        "backend": bayesfilter["backend"],
        "localization_contract": bayesfilter["localization_contract"],
        "finite_values": bayesfilter["finite_values"],
        "mode_final_summaries": [
            {
                "mode": mode["mode"],
                "final_mean_log_likelihood": mode["final_mean_log_likelihood"],
                "final_gradient_diag": mode["final_gradient_diag"],
                "final_gradient_max_abs": mode["final_gradient_max_abs"],
                "finite_values": mode["finite_values"],
            }
            for mode in bayesfilter["modes"]
        ],
        "cpu_only_manifest": bayesfilter["cpu_only_manifest"],
    }


def _decision_table(decision: str, comparison: dict[str, Any]) -> list[dict[str, str]]:
    if decision == "filterflow_float64_smoothness_gradient_localized":
        primary = f"localized raw failure: {comparison.get('first_raw_gradient_failure')}"
        veto = "scalar path remained aligned before gradient mismatch"
        clip_summary = comparison.get("transport_clip_summary", {})
        if clip_summary.get("final_gradient_within_tolerance") is False:
            next_action = (
                "localize the remaining clipped/default gradient residual with "
                "per-term VJP diagnostics at the first resampling time"
            )
        else:
            next_action = "rerun the smoothness surface with FilterFlow-style clipped transport backward semantics"
    elif decision == "filterflow_float64_smoothness_gradient_localization_no_gradient_mismatch":
        primary = "no gradient mismatch observed in localization rerun"
        veto = "none"
        next_action = "rerun the prior surface to reconcile the changed outcome"
    elif decision == "filterflow_float64_smoothness_gradient_localization_scalar_veto":
        primary = f"scalar mismatch before gradient localization: {comparison.get('first_raw_scalar_failure')}"
        veto = "scalar mismatch"
        next_action = "repair scalar replay before using gradient diagnostics"
    else:
        primary = comparison.get("blocker", "blocked")
        veto = decision
        next_action = "repair blocker before using this artifact"
    return [
        {
            "decision": decision,
            "primary_criterion_status": primary,
            "veto_diagnostic_status": veto,
            "main_uncertainty": "single theta row only; no analytic-gradient correctness is concluded",
            "next_justified_action": next_action,
            "not_concluded": "correctness of either implementation, analytic gradient correctness, production readiness",
        }
    ]


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "plan_path",
        "result_path",
        "evidence_contract",
        "reference_policy",
        "filterflow_status",
        "filterflow_fingerprint_initial",
        "filterflow_fingerprint_final",
        "model_contract",
        "gradient_localization_contract",
        "filterflow_localization",
        "bayesfilter_localization",
        "comparison",
        "path_boundary_manifest",
        "run_manifest",
        "decision_table",
        "non_implications",
        "reproducibility_digest",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"missing payload keys: {sorted(missing)}")
    validate_filterflow_reference_status(payload["filterflow_status"], marker_path=FILTERFLOW_MARKER_PATH)
    allowed = {
        "filterflow_float64_smoothness_gradient_localization_filterflow_blocker",
        "filterflow_float64_smoothness_gradient_localization_blocked_by_comparator_drift",
        "filterflow_float64_smoothness_gradient_localization_blocked",
        "filterflow_float64_smoothness_gradient_localization_scalar_veto",
        "filterflow_float64_smoothness_gradient_localized",
        "filterflow_float64_smoothness_gradient_localization_no_gradient_mismatch",
    }
    if payload["decision"] not in allowed:
        raise ValueError(f"unexpected decision: {payload['decision']}")
    _validate_cpu(payload["run_manifest"], "parent")
    if any(bool(value) for value in payload["path_boundary_manifest"].values()):
        raise ValueError(f"path boundary violation: {payload['path_boundary_manifest']}")
    filterflow_payload = payload["filterflow_localization"]
    if filterflow_payload.get("status") == "executed":
        _validate_cpu(filterflow_payload["cpu_only_manifest"], "filterflow localization")
        settings = filterflow_payload["settings"]
        if settings.get("optimal_proposal") is not True:
            raise ValueError("filterflow optimal proposal not used")
        if settings.get("resampling_correction") is not False:
            raise ValueError("filterflow correction term unexpectedly enabled")
        if settings.get("dtype") != FILTERFLOW_REFERENCE_DTYPE:
            raise ValueError("filterflow dtype mismatch")
    bayesfilter_payload = payload["bayesfilter_localization"]
    if bayesfilter_payload is not None:
        _validate_cpu(bayesfilter_payload["cpu_only_manifest"], "BayesFilter localization")
    if payload["decision"] == "filterflow_float64_smoothness_gradient_localized":
        if payload["comparison"]["first_raw_gradient_failure"].get("status") != "failure":
            raise ValueError("localized decision without first raw gradient failure")
        if payload["comparison"]["first_raw_scalar_failure"].get("status") != "no_failure":
            raise ValueError("localized decision with scalar failure")
    if "custom_gradient" not in payload["gradient_localization_contract"]["filterflow_transport_backward"]:
        raise ValueError("missing FilterFlow custom-gradient contract")


def _validate_cpu(manifest: dict[str, Any], label: str) -> None:
    if manifest.get("pre_import_cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: pre-import CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("gpu_devices_visible") != []:
        raise ValueError(f"{label}: GPU devices visible {manifest.get('gpu_devices_visible')}")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# FilterFlow Float64 Smoothness Gradient Localization",
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        "## Decision Table",
        "",
        "| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["decision_table"]:
        lines.append(
            "| {decision} | {primary} | {veto} | {uncertainty} | {next_action} | {not_concluded} |".format(
                decision=row["decision"],
                primary=row["primary_criterion_status"],
                veto=row["veto_diagnostic_status"],
                uncertainty=row["main_uncertainty"],
                next_action=row["next_justified_action"],
                not_concluded=row["not_concluded"],
            )
        )
    lines.extend(
        [
            "",
            "## Model Contract",
            "",
            _key_value_table(payload["model_contract"]),
            "",
            "## Gradient Localization Contract",
            "",
            _key_value_table(payload["gradient_localization_contract"]),
            "",
            "## Localization Summary",
            "",
            _json_block(
                {
                    "first_raw_scalar_failure": payload["comparison"].get("first_raw_scalar_failure"),
                    "first_raw_gradient_failure": payload["comparison"].get("first_raw_gradient_failure"),
                    "interpretive_hint": payload["comparison"].get("interpretive_hint"),
                    "best_ablation": {
                        "mode": payload["comparison"].get("best_ablation", {}).get("mode"),
                        "final_relative_gradient_delta": payload["comparison"].get(
                            "best_ablation",
                            {},
                        ).get("final_relative_gradient_delta"),
                    },
                }
            ),
            "",
            "## Mode Summaries",
            "",
            _json_block(payload["comparison"].get("mode_summaries")),
            "",
            "## FilterFlow Reference Rows",
            "",
            _json_block(payload["filterflow_localization"]),
            "",
            "## BayesFilter Mode Finals",
            "",
            _json_block(payload["bayesfilter_localization"]),
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
    decision = payload["decision"]
    comparison = payload["comparison"]
    if decision == "filterflow_float64_smoothness_gradient_localized":
        hint = comparison.get("interpretive_hint", {})
        return (
            "The scalar path stays aligned on the first failing smoothness row, "
            "but the raw BayesFilter gradient diverges. The strongest "
            f"diagnostic hint is `{hint.get('status')}`."
        )
    if decision == "filterflow_float64_smoothness_gradient_localization_scalar_veto":
        return "A scalar mismatch appeared, so gradient localization is vetoed."
    if decision == "filterflow_float64_smoothness_gradient_localization_no_gradient_mismatch":
        return "The localization rerun did not reproduce the prior gradient mismatch."
    return "The localization run did not produce evidence because a blocker fired."


def _key_value_table(values: dict[str, Any]) -> str:
    lines = ["| Key | Value |", "| --- | --- |"]
    for key, value in values.items():
        lines.append(f"| `{key}` | `{value}` |")
    return "\n".join(lines)


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, indent=2, sort_keys=True, default=str) + "\n```"


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("run_manifest", None)
    clone.pop("created_at_utc", None)
    clone.pop("reproducibility_digest", None)
    return stable_digest(clone)


def _parent_cpu_manifest() -> dict[str, Any]:
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pre_import_cuda_visible_devices": PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        "gpu_devices_visible": [str(device) for device in tf.config.list_physical_devices("GPU")],
    }


def _non_implications() -> list[str]:
    return step._non_implications() + [
        "No correctness claim is made for either implementation.",
        "No analytic smoothness-gradient correctness is concluded.",
        "No full mesh_size=20 surface agreement is concluded.",
        "No production dtype default is concluded.",
        "Finite gradients alone are smoke evidence only.",
    ]


def _float(value: tf.Tensor) -> float:
    return float(tf.cast(value, DTYPE).numpy())


def _max_abs_list(values: list[float]) -> float:
    return max(abs(float(value)) for value in values)


if __name__ == "__main__":
    raise SystemExit(main())
