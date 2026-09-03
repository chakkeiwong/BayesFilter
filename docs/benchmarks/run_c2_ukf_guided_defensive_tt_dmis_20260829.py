"""Run the bounded C2 UKF-guided defensive TT-DMIS diagnostic.

The driver loads the frozen JSON fixture, captures a fresh retained-TT snapshot
with the current TensorFlow route, and evaluates all proposal families through
the shared frozen-branch APF program.  It is a diagnostic campaign only:
results do not establish pseudo-marginal exactness, posterior correctness, or
default readiness.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import statistics
import subprocess
import sys
import time
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/benchmarks/fixtures/c2_sv_n4_seed52_obs42_t20_frozen_v1.json"
PF_REFERENCE = ROOT / "docs/benchmarks/artifacts/c2_completion_20260824/attempt05/reference_n4_s42.json"
PF_PREFIX_REFERENCE = ROOT / "docs/benchmarks/artifacts/c2_n4_root_cause_20260828/attempt03/pf_per_step_reference.json"
PLAN = ROOT / "docs/plans/bayesfilter-c2-ukf-guided-defensive-tt-dmis-implementation-test-plan-2026-08-29.md"
PLAN_REVIEW = ROOT / "docs/plans/bayesfilter-c2-ukf-guided-defensive-tt-dmis-plan-review-2026-08-29.md"
LATEX = ROOT / "docs/benchmarks/artifacts/c2_completion_20260824/attempt05/ukf_guided_defensive_tt_dmis_analytical_gradient.tex"
N = 4
GAMMA = 0.6
BETA = 0.4
SIGMA = 1.0
DEGREE = 6
RANK = 6
FIT_ROWS = 8192
SWEEPS = 32
RIDGE = 1e-10
TAU = 1e-6
ALPHA = 0.5
NU = 8.0
DTYPE_NAME = "float64"
TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"
FAMILIES = (
    "retained_tt",
    "bootstrap_conditional",
    "defensive_student",
    "dmis_half",
    "gaussian_hint_marginal",
    "stationary_independence",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "serious"), default="smoke")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--fixture", default=str(FIXTURE))
    parser.add_argument("--particle-count", type=int)
    parser.add_argument("--branch-seeds", type=int, nargs="+")
    parser.add_argument("--skip-focused-tests", action="store_true")
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: object) -> object:
    if hasattr(value, "numpy"):
        materialized = value.numpy()
        if hasattr(materialized, "tolist"):
            return _jsonable(materialized.tolist())
        return _jsonable(materialized)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"refusing to serialize non-finite float {value!r}")
        return value
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError(f"unsupported JSON value {type(value).__name__}")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _maximum_difference(tf, left, right) -> float:
    left_tensor = tf.convert_to_tensor(left)
    right_tensor = tf.convert_to_tensor(right, dtype=left_tensor.dtype)
    return float(tf.reduce_max(tf.abs(left_tensor - right_tensor)).numpy())


def _run_focused_tests(output_root: Path) -> Mapping[str, object]:
    paths = (
        "tests/highdim/test_zhao_cui_frozen_proposal_apf_tf.py",
        "tests/highdim/test_c2_sv_frozen_proposal_apf_tf.py",
        "tests/highdim/test_c2_transformed_observation_student_proposal_tf.py",
        "tests/highdim/test_c2_ukf_guided_tt_dmis_tf.py",
    )
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *paths]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    environment["MPLCONFIGDIR"] = "/tmp/matplotlib-c2-ukf-guided-dmis"
    started = time.perf_counter()
    process = subprocess.run(
        command, cwd=ROOT, env=environment, capture_output=True, text=True
    )
    wall = time.perf_counter() - started
    log_path = output_root / "focused_tests.log"
    log_path.write_text(process.stdout + process.stderr, encoding="utf-8")
    record = {
        "classification": "cpu_only_focused_engineering_tests",
        "command": shlex.join(command),
        "cuda_visible_devices": "-1",
        "returncode": process.returncode,
        "passed": process.returncode == 0,
        "wall_seconds": wall,
        "log_path": str(log_path.relative_to(ROOT)),
        "test_paths": paths,
    }
    _write_json(output_root / "engineering_tests.json", record)
    if process.returncode != 0:
        raise RuntimeError("focused engineering tests failed; see focused_tests.log")
    return record


def _frozen_adapter(tf, adapter_class, fixture: Mapping[str, object]):
    dtype = tf.float64
    transition = tf.constant(fixture["transition_matrix"], dtype)
    process_chol = tf.linalg.cholesky(tf.constant(fixture["process_covariance"], dtype))
    initial_chol = tf.linalg.cholesky(tf.constant(fixture["stationary_covariance"], dtype))
    beta = tf.constant(float(fixture["beta"]), dtype)

    def _mvn(states, means, chol):
        difference = states - means
        whitened = tf.transpose(
            tf.linalg.triangular_solve(chol, tf.transpose(difference), lower=True)
        )
        return -0.5 * (
            tf.constant(N * math.log(2.0 * math.pi), dtype)
            + tf.reduce_sum(tf.square(whitened), axis=1)
        ) - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))

    def transition_log_density(current, previous):
        means = tf.linalg.matmul(previous, transition, transpose_b=True)
        return _mvn(current, means, process_chol)

    def observation_log_density(current, observation):
        observed = tf.ensure_shape(tf.convert_to_tensor(observation, dtype), [N])
        return tf.reduce_sum(
            -0.5 * tf.constant(math.log(2.0 * math.pi), dtype)
            - tf.math.log(beta)
            - 0.5 * current
            - 0.5 * tf.square(observed)[None, :] * tf.exp(-current) / tf.square(beta),
            axis=1,
        )

    def initial_log_density(current):
        return _mvn(current, tf.zeros_like(current), initial_chol)

    return adapter_class(
        state_dim=N,
        transition_log_density=transition_log_density,
        observation_log_density=observation_log_density,
        initial_log_density=initial_log_density,
    )


def _frozen_hint_factory(tf, fixture: Mapping[str, object], horizon: int):
    hints = fixture["moment_hints"][:horizon]
    next_index = {"value": 0}

    def initial_hint(_observation):
        if next_index["value"] != 0:
            raise RuntimeError("frozen hints were consumed out of order")
        next_index["value"] = 1
        return tf.constant(hints[0]["mean"], tf.float64), tf.constant(
            hints[0]["covariance"], tf.float64
        )

    def predictive_hint(time_index, _observation):
        if int(time_index) != next_index["value"]:
            raise RuntimeError("frozen hints were consumed out of order")
        next_index["value"] += 1
        return tf.constant(hints[time_index]["mean"], tf.float64), tf.constant(
            hints[time_index]["covariance"], tf.float64
        )

    return initial_hint, predictive_hint


def _save_snapshot(tf, snapshot_api, snapshot, output_root: Path) -> Mapping[str, object]:
    metadata, tensors = snapshot_api.gaussian_xla_retained_proposal_snapshot_parts(snapshot)
    step_root = output_root / "snapshots" / f"t{snapshot.time_index:02d}"
    step_root.mkdir(parents=True)
    tensor_rows = {}
    for name, tensor in tensors.items():
        path = step_root / f"{name}.tensor"
        tf.io.write_file(str(path), tf.io.serialize_tensor(tensor))
        tensor_rows[name] = {
            "path": str(path.relative_to(output_root)),
            "sha256": _sha256_file(path),
        }
    metadata = dict(metadata)
    metadata["snapshot_fingerprint"] = snapshot_api.gaussian_xla_retained_proposal_snapshot_fingerprint(snapshot)
    metadata["tensor_files"] = tensor_rows
    metadata_path = step_root / "metadata.json"
    _write_json(metadata_path, metadata)
    return {
        "time_index": int(snapshot.time_index),
        "snapshot_fingerprint": metadata["snapshot_fingerprint"],
        "metadata_path": str(metadata_path.relative_to(output_root)),
        "metadata_sha256": _sha256_file(metadata_path),
        "tensor_count": len(tensors),
    }


def _evaluate(tf, model, compilation, theta_reference, *, family: str, seed: int, branch_wall: float):
    from bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf import prepare_frozen_proposal_apf_program

    program = prepare_frozen_proposal_apf_program(model, compilation.branch)
    eager_started = time.perf_counter()
    eager = program.evaluate(theta_reference)
    eager_wall = time.perf_counter() - eager_started
    compiled_function = program.compiled(jit_compile=True)
    first_started = time.perf_counter()
    compiled = compiled_function(theta_reference)
    xla_first_wall = time.perf_counter() - first_started
    repeat_started = time.perf_counter()
    repeated = compiled_function(theta_reference)
    xla_repeat_wall = time.perf_counter() - repeat_started
    fd_values = []
    fd_started = time.perf_counter()
    for index in range(model.parameter_dim()):
        direction = tf.one_hot(index, model.parameter_dim(), dtype=tf.float64)
        plus = compiled_function(theta_reference + 1e-5 * direction)["log_likelihood"]
        minus = compiled_function(theta_reference - 1e-5 * direction)["log_likelihood"]
        fd_values.append((plus - minus) / (2e-5))
    fd = tf.stack(fd_values)
    fd_wall = time.perf_counter() - fd_started
    fields = (
        "log_likelihood", "score", "log_increments", "increment_scores",
        "ess_by_time", "log_weight_spread_by_time",
        "maximum_normalized_weight_by_time", "final_log_weights",
    )
    eager_xla = {name: _maximum_difference(tf, eager[name], compiled[name]) for name in fields}
    repeated_differences = {name: _maximum_difference(tf, compiled[name], repeated[name]) for name in fields}
    score_error = tf.abs(compiled["score"] - fd)
    score_scale = tf.maximum(tf.maximum(tf.abs(compiled["score"]), tf.abs(fd)), tf.ones_like(score_error))
    diagnostics = [_jsonable(item) for item in compilation.proposal_diagnostics]
    checks = {
        "finite": bool(compiled["finite"].numpy()),
        "eager_xla_parity": max(eager_xla.values()) <= 2e-10,
        "repeat_parity": max(repeated_differences.values()) <= 2e-10,
        "same_scalar_score": float(tf.reduce_max(score_error).numpy()) <= 2e-5
        or float(tf.reduce_max(score_error / score_scale).numpy()) <= 2e-5,
        "single_trace": int(compiled_function.experimental_get_tracing_count()) == 1,
        "proposal_finite": all(bool(row.get("finite", True)) for row in diagnostics),
        "gpu_output": "GPU" in str(compiled["log_likelihood"].device).upper(),
    }
    return {
        "family": family,
        "seed": int(seed),
        "branch_id": compilation.branch.branch_id,
        "compiler_id": compilation.compiler_id,
        "program_id": program.program_id,
        "log_likelihood": float(compiled["log_likelihood"].numpy()),
        "score": compiled["score"],
        "finite_difference_score": fd,
        "maximum_score_absolute_error": float(tf.reduce_max(score_error).numpy()),
        "maximum_score_relative_error": float(tf.reduce_max(score_error / score_scale).numpy()),
        "log_increments": compiled["log_increments"],
        "increment_scores": compiled["increment_scores"],
        "ess_by_time": compiled["ess_by_time"],
        "normalized_ess_by_time": compiled["ess_by_time"] / float(compilation.branch.particle_count),
        "maximum_normalized_weight_by_time": compiled[
            "maximum_normalized_weight_by_time"
        ],
        "log_weight_spread_by_time": compiled["log_weight_spread_by_time"],
        "minimum_normalized_ess": float((compiled["minimum_ess"] / float(compilation.branch.particle_count)).numpy()),
        "maximum_log_weight_spread": float(compiled["maximum_log_weight_spread"].numpy()),
        "proposal_diagnostics": diagnostics,
        "checks": checks,
        "all_engineering_checks_pass": all(checks.values()),
        "eager_xla_max_absolute_differences": eager_xla,
        "repeated_call_max_absolute_differences": repeated_differences,
        "tracing_count": int(compiled_function.experimental_get_tracing_count()),
        "output_device": str(compiled["log_likelihood"].device),
        "wall_seconds": {
            "branch_generation": branch_wall,
            "eager_evaluation": eager_wall,
            "xla_first_call": xla_first_wall,
            "xla_repeated_call": xla_repeat_wall,
            "finite_difference_calls": fd_wall,
        },
        "manifest": compilation.manifest,
    }


def _sample_stats(values: Sequence[float]) -> Mapping[str, object]:
    rows = [float(value) for value in values]
    mean = statistics.fmean(rows)
    if len(rows) < 2:
        return {"n": len(rows), "values": rows, "mean": mean, "standard_deviation": 0.0, "standard_error": 0.0, "interval_is_statistical": False}
    sd = statistics.stdev(rows)
    return {"n": len(rows), "values": rows, "mean": mean, "standard_deviation": sd, "standard_error": sd / math.sqrt(len(rows)), "interval_is_statistical": True}


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a quantile of an empty sequence")
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _exact_two_sided_sign_pvalue(positive: int, total: int) -> float:
    if total < 1 or not 0 <= positive <= total:
        raise ValueError("invalid sign-test counts")
    negative_or_equal = total - positive
    tail = min(
        sum(math.comb(total, k) for k in range(0, negative_or_equal + 1)),
        sum(math.comb(total, k) for k in range(positive, total + 1)),
    )
    return min(1.0, 2.0 * tail / (2.0**total))


def _paired_ess_diagnostics(
    branches: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    by_family = {
        family: {
            int(row["seed"]): row
            for row in branches
            if row["family"] == family
        }
        for family in FAMILIES
    }
    seeds = tuple(sorted(by_family["retained_tt"]))
    if len(seeds) < 2 or any(tuple(sorted(by_family[family])) != seeds for family in FAMILIES):
        return {
            "available": False,
            "reason": "paired diagnostic requires at least two identical family seed maps",
        }
    contrasts = tuple(
        math.log(
            float(by_family["dmis_half"][seed]["minimum_normalized_ess"])
            / float(by_family["retained_tt"][seed]["minimum_normalized_ess"])
        )
        for seed in seeds
    )
    generator = random.Random(20260830)
    bootstrap_means = [
        statistics.fmean(
            contrasts[generator.randrange(len(contrasts))] for _ in contrasts
        )
        for _ in range(20_000)
    ]
    interval = [_quantile(bootstrap_means, 0.025), _quantile(bootstrap_means, 0.975)]
    positive = sum(value > 0.0 for value in contrasts)
    return {
        "available": True,
        "seeds": seeds,
        "values": contrasts,
        "mean": statistics.fmean(contrasts),
        "standard_error": statistics.stdev(contrasts) / math.sqrt(len(contrasts)),
        "bootstrap_replicates": 20_000,
        "bootstrap_seed": 20260830,
        "percentile_95_interval": interval,
        "positive_count": positive,
        "exact_two_sided_sign_pvalue": _exact_two_sided_sign_pvalue(
            positive, len(contrasts)
        ),
        "criterion_pass": bool(interval[0] > 0.0 and positive >= 10),
    }


def _summarize(branches: Sequence[Mapping[str, object]], horizon: int) -> Mapping[str, object]:
    result = {}
    for family in FAMILIES:
        rows = [row for row in branches if row["family"] == family]
        result[family] = {
            "total": _sample_stats([row["log_likelihood"] for row in rows]),
            "per_step": [_sample_stats([row["log_increments"][t] for row in rows]) for t in range(horizon)],
            "mean_normalized_ess_by_time": [statistics.fmean(float(row["normalized_ess_by_time"][t]) for row in rows) for t in range(horizon)],
            "mean_maximum_normalized_weight_by_time": [
                statistics.fmean(
                    float(row["maximum_normalized_weight_by_time"][t])
                    for row in rows
                )
                for t in range(horizon)
            ],
            "minimum_observed_normalized_ess": min(float(row["minimum_normalized_ess"]) for row in rows),
            "maximum_observed_log_weight_spread": max(float(row["maximum_log_weight_spread"]) for row in rows),
            "all_engineering_checks_pass": all(bool(row["all_engineering_checks_pass"]) for row in rows),
        }
    return result


def _reference(horizon: int) -> Mapping[str, object]:
    record = json.loads(PF_REFERENCE.read_text(encoding="utf-8"))
    if not bool(record["valid"]):
        raise ValueError("PF reference is not valid")
    arm = record["arms"]["800000"]
    steps = [float(value) for value in arm["per_step_mean"][:horizon]]
    prefix = json.loads(PF_PREFIX_REFERENCE.read_text(encoding="utf-8"))
    if any(abs(float(prefix["per_step_mean"][i]) - steps[i]) > 1e-12 for i in range(min(5, horizon))):
        raise ValueError("PF T=5 prefix does not match T=20 reference")
    if horizon == 20:
        return {"pf_total": float(arm["mean_total"]), "pf_total_se": float(arm["se_total"]), "pf_per_step_mean": steps, "source": str(PF_REFERENCE.relative_to(ROOT))}
    if horizon == 5:
        return {"pf_total": float(prefix["mean_total"]), "pf_total_se": float(prefix["se_total"]), "pf_per_step_mean": steps, "source": str(PF_PREFIX_REFERENCE.relative_to(ROOT))}
    raise ValueError("only T=5 and T=20 references are screened")


def _result_markdown(result: Mapping[str, object]) -> str:
    reference = result["reference"]
    decision = result["decision"]
    lines = [
        "# C2 UKF-Guided Defensive TT-DMIS Result", "",
        f"Mode: `{result['mode']}`; horizon: {result['horizon']}; particles: {result['particle_count']}; branches per family: {len(result['branch_seeds'])}.", "",
        f"Engineering checks: `{'PASS' if decision['engineering_correctness_pass'] else 'FAIL'}`. DMIS mechanism screen: `{'PASS' if decision['mechanism_screen_pass'] else 'FAIL'}`. No promotion verdict is issued.", "",
        "| Family | Mean total | Mean minimum normalized ESS | Engineering |", "| --- | ---: | ---: | --- |",
    ]
    for family in FAMILIES:
        row = result["family_summary"][family]
        lines.append(f"| `{family}` | {row['total']['mean']:.10f} | {row['minimum_observed_normalized_ess']:.6g} | {'PASS' if row['all_engineering_checks_pass'] else 'FAIL'} |")
    lines.extend([
        "", "## Decision", "",
        f"The PF reference total is `{reference['pf_total']:.10f}`; the fixed-half DMIS total gap is descriptive and does not establish exactness. The selected-alpha/tail calibration phase was `{result['calibration']['status']}`.", "",
        f"Heuristic-dominance veto: `{'FIRED' if decision['heuristic_dominance_veto'] else 'CLEAR'}`. Statistical ranking: `NOT SUPPORTED`.", "",
    ])
    paired = result.get("paired_ess_diagnostics", {})
    if paired.get("available"):
        lines.extend([
            f"Paired DMIS/retained-TT log minimum-ESS ratio: mean `{paired['mean']:.10f}`, 95% bootstrap interval `[{paired['percentile_95_interval'][0]:.10f}, {paired['percentile_95_interval'][1]:.10f}]`, positive contrasts `{paired['positive_count']}/{len(paired['seeds'])}`, exact sign-test p-value `{paired['exact_two_sided_sign_pvalue']:.10g}`.", "",
        ])
    lines.extend([
        "This artifact does not claim an exact pseudo-marginal likelihood, posterior correctness, HMC readiness, default readiness, source-faithful Zhao-Cui reproduction, or superiority.", "",
        "## Red Team", "",
        f"Strongest alternative explanation: {decision['strongest_alternative_explanation']}", "",
        f"Overturning evidence: {decision['overturning_evidence']}", "",
        f"Weakest evidence: {decision['weakest_evidence']}", "",
    ])
    return "\n".join(lines)


def _run(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root {output_root}")
    output_root.mkdir(parents=True)
    mode = str(args.mode)
    horizon = 5 if mode == "smoke" else 20
    particle_count = int(args.particle_count or (1024 if mode == "smoke" else 8192))
    branch_seeds = tuple(args.branch_seeds or ((9201,) if mode == "smoke" else tuple(range(9201, 9213))))
    if mode == "smoke" and (particle_count != 1024 or len(branch_seeds) != 1):
        raise ValueError("smoke scope is fixed to N=1024 and one branch seed")
    if mode == "serious" and (particle_count != 8192 or len(branch_seeds) != 12):
        raise ValueError("serious scope is fixed to N=8192 and twelve branch seeds")
    fixture_path = Path(args.fixture).resolve()
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    if fixture.get("schema_id") != "bayesfilter.c2_sv_frozen_fixture.v1" or int(fixture["state_dimension"]) != N or int(fixture["horizon"]) < horizon:
        raise ValueError("fixture identity does not match the C2 scope")
    run_log = output_root / "run.log"
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()

    def log(message: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
        print(line, flush=True)
        with run_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    if args.skip_focused_tests:
        tests_record = {"passed": True, "skipped_in_this_process": True}
    else:
        tests_record = _run_focused_tests(output_root)
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-c2-ukf-guided-dmis")
    sys.path.insert(0, str(ROOT))
    import tensorflow as tf
    import tensorflow_probability as tfp
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
    if not logical_gpus:
        raise RuntimeError("no logical TensorFlow GPU")
    with tf.device("/GPU:0"):
        placement_probe = tf.reduce_sum(tf.ones([32], tf.float64))
    if "GPU" not in placement_probe.device.upper():
        raise RuntimeError("GPU placement probe did not execute on GPU")
    from bayesfilter.highdim.c2_gaussian_hermite_proposal_tf import retained_proposal_from_transition_snapshot
    from bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf import (
        C2StochasticVolatilityFrozenAPFModel,
        FrozenGaussianStateProposal,
        compile_c2_bootstrap_proposal_branch,
        compile_c2_dmis_proposal_branch,
        compile_c2_independent_proposal_branch,
        compile_c2_transformed_student_proposal_branch,
        stationary_gaussian_proposals,
        transformed_student_proposals,
    )
    from bayesfilter.highdim.squared_tt_engine_v0_tf import DensityKernelAdapter, EngineConfig
    import bayesfilter.highdim.squared_tt_engine_gaussian_xla_tf as snapshot_api

    observations = tf.constant(fixture["observations"][:horizon], tf.float64)
    transition_matrix = tf.constant(fixture["transition_matrix"], tf.float64)
    theta_reference = tf.constant([GAMMA, math.log(BETA)], tf.float64)
    coupling = transition_matrix - GAMMA * tf.eye(N, dtype=tf.float64)
    model = C2StochasticVolatilityFrozenAPFModel(coupling_matrix=coupling, sigma=SIGMA)
    if _maximum_difference(tf, model.transition_matrix(theta_reference), transition_matrix) > 2e-14:
        raise ValueError("C2 parameterization does not reconstruct the fixture transition")
    if _maximum_difference(tf, model.stationary_covariance_and_derivative(theta_reference)[0], tf.constant(fixture["stationary_covariance"], tf.float64)) > 2e-12:
        raise ValueError("C2 Lyapunov solve does not reconstruct fixture covariance")
    adapter = _frozen_adapter(tf, DensityKernelAdapter, fixture)
    initial_hint, predictive_hint = _frozen_hint_factory(tf, fixture, horizon)
    config = EngineConfig(
        basis_degree=DEGREE, rank=RANK, row_count=FIT_ROWS, sweeps=SWEEPS,
        ridge=RIDGE, tau=TAU, coordinate_half_width=3.0,
        seed=98000 + 100 * N + 10 * DEGREE + RANK, row_design="sobol",
    )
    run_identity_payload = {
        "fixture_sha256": _sha256_file(fixture_path), "horizon": horizon,
        "degree": DEGREE, "rank": RANK, "fit_rows": FIT_ROWS, "sweeps": SWEEPS,
        "ridge": RIDGE, "tau": TAU, "alpha": ALPHA, "nu": NU,
    }
    run_identity = hashlib.sha256(json.dumps(run_identity_payload, sort_keys=True).encode("utf-8")).hexdigest()
    log(f"starting {mode}: horizon={horizon}, particles={particle_count}, seeds={branch_seeds}")
    fit_started = time.perf_counter()
    direct_value, direct_diagnostics, snapshots = snapshot_api.run_value_filter_branch_axis_gaussian_xla_retained_proposal_diagnostic(
        adapter, observations, config, predictive_moment_hint=predictive_hint,
        initial_moment_hint=initial_hint, capture_steps=tuple(range(1, horizon)),
        run_identity=run_identity, defensive_nu=NU,
    )
    fit_wall = time.perf_counter() - fit_started
    if set(snapshots) != set(range(1, horizon)):
        raise ValueError("fresh retained snapshot set is incomplete")
    snapshot_rows = [_save_snapshot(tf, snapshot_api, snapshots[t], output_root) for t in range(1, horizon)]
    tt_proposals = tuple(retained_proposal_from_transition_snapshot(snapshots[t]) for t in range(1, horizon))
    hint_proposals = tuple(FrozenGaussianStateProposal(mean=snapshots[t].coordinate_offset, chol=snapshots[t].coordinate_matrix, time_index=t, family="gaussian_hint_marginal") for t in range(1, horizon))
    stationary_proposals = stationary_gaussian_proposals(model, theta_reference, horizon)
    defensive_proposals = transformed_student_proposals(model=model, observations=observations, theta_reference=theta_reference, nu=NU)
    proposal_manifest = {
        "schema_id": "bayesfilter.c2_ukf_guided_defensive_tt_dmis_proposal_manifest.v1",
        "run_identity": run_identity, "snapshots": snapshot_rows,
        "retained_tt": [proposal.manifest_payload() for proposal in tt_proposals],
        "transformed_student": [proposal.manifest_payload() for proposal in defensive_proposals],
        "alpha": ALPHA, "nu": NU,
    }
    _write_json(output_root / "proposal_manifest.json", proposal_manifest)
    branches = []
    for family in FAMILIES:
        for seed in branch_seeds:
            log(f"compiling family={family}, seed={seed}")
            branch_started = time.perf_counter()
            if family == "retained_tt":
                compilation = compile_c2_independent_proposal_branch(model=model, observations=observations, theta_reference=theta_reference, transition_proposals=tt_proposals, particle_count=particle_count, seed=seed, family=family, jit_compile_sampler=True)
            elif family == "bootstrap_conditional":
                compilation = compile_c2_bootstrap_proposal_branch(model=model, observations=observations, theta_reference=theta_reference, particle_count=particle_count, seed=seed, jit_compile_sampler=True)
            elif family == "defensive_student":
                compilation = compile_c2_transformed_student_proposal_branch(model=model, observations=observations, theta_reference=theta_reference, nu=NU, particle_count=particle_count, seed=seed, jit_compile_sampler=True)
            elif family == "dmis_half":
                compilation = compile_c2_dmis_proposal_branch(model=model, observations=observations, theta_reference=theta_reference, transition_proposals=tt_proposals, defensive_proposals=defensive_proposals, particle_count=particle_count, seed=seed, alpha=ALPHA, nu=NU, jit_compile_sampler=True)
            elif family == "gaussian_hint_marginal":
                compilation = compile_c2_independent_proposal_branch(model=model, observations=observations, theta_reference=theta_reference, transition_proposals=hint_proposals, particle_count=particle_count, seed=seed, family=family, jit_compile_sampler=True)
            else:
                compilation = compile_c2_independent_proposal_branch(model=model, observations=observations, theta_reference=theta_reference, transition_proposals=stationary_proposals, particle_count=particle_count, seed=seed, family=family, jit_compile_sampler=True)
            record = _evaluate(tf, model, compilation, theta_reference, family=family, seed=seed, branch_wall=time.perf_counter() - branch_started)
            branches.append(_jsonable(record))
            _write_json(output_root / "branch_results.partial.json", branches)
            log(f"complete family={family}, seed={seed}, total={record['log_likelihood']:.8f}, min_ness={record['minimum_normalized_ess']:.6g}")
    _write_json(output_root / "branch_results.json", branches)
    family_summary = _summarize(branches, horizon)
    reference = _reference(horizon)
    engineering_pass = bool(tests_record["passed"]) and all(bool(row["all_engineering_checks_pass"]) for row in branches)
    dmis_total = family_summary["dmis_half"]["total"]["mean"]
    dmis_gap = abs(dmis_total - reference["pf_total"])
    tt_total = family_summary["retained_tt"]["total"]["mean"]
    tt_ess = family_summary["retained_tt"]["minimum_observed_normalized_ess"]
    dmis_ess = family_summary["dmis_half"]["minimum_observed_normalized_ess"]
    paired_ess = _paired_ess_diagnostics(branches)
    if bool(paired_ess.get("available", False)):
        ess_criterion_pass = bool(paired_ess["criterion_pass"])
    else:
        ess_criterion_pass = dmis_ess > tt_ess
    mechanism_pass = (
        engineering_pass
        and ess_criterion_pass
        and dmis_gap <= max(1.0, 3.0 * reference["pf_total_se"])
    )
    dominance_rows = []
    for t in (3, 4, min(range(horizon), key=lambda i: family_summary["retained_tt"]["mean_normalized_ess_by_time"][i])):
        if t >= horizon or any(row["situation"] == f"t={t}" for row in dominance_rows):
            continue
        candidate_error = abs(
            float(family_summary["dmis_half"]["per_step"][t]["mean"])
            - reference["pf_per_step_mean"][t]
        )
        heuristic_families = (
            "bootstrap_conditional",
            "defensive_student",
            "gaussian_hint_marginal",
            "stationary_independence",
        )
        heuristic_errors = {
            family: abs(
                float(family_summary[family]["per_step"][t]["mean"])
                - reference["pf_per_step_mean"][t]
            )
            for family in heuristic_families
        }
        better = [
            family for family in heuristic_families if heuristic_errors[family] < candidate_error
        ]
        dominance_rows.append(
            {
                "situation": f"t={t}",
                "candidate": "dmis_half",
                "candidate_mean_absolute_per_step_error": candidate_error,
                "heuristic_mean_absolute_per_step_errors": heuristic_errors,
                "heuristics_better_than_candidate": better,
            }
        )
    decision = {
        "engineering_correctness_pass": engineering_pass,
        "mechanism_screen_pass": mechanism_pass,
        "heuristic_dominance_veto": bool(
            any(row["heuristics_better_than_candidate"] for row in dominance_rows)
        ),
        "tt_total": tt_total, "dmis_total": dmis_total, "dmis_absolute_pf_gap": dmis_gap,
        "tt_minimum_observed_normalized_ess": tt_ess, "dmis_minimum_observed_normalized_ess": dmis_ess,
        "paired_ess_criterion_pass": ess_criterion_pass,
        "statistically_supported_ranking": False,
        "strongest_alternative_explanation": "an ESS change can come entirely from the defensive component and does not establish that retained TT is useful",
        "overturning_evidence": "same-scalar mismatch, proposal-law failure, invalid snapshot identity, or replicated reference incompatibility",
        "weakest_evidence": (
            "the fixed-half result has twelve branch replicates but no alpha/nu calibration, "
            "and no selected allocation is supported"
            if mode == "serious"
            else "smoke has one branch and all stochastic differences are descriptive"
        ),
    }
    result = {
        "schema_id": "bayesfilter.c2_ukf_guided_defensive_tt_dmis_result.v1",
        "mode": mode, "horizon": horizon, "particle_count": particle_count,
        "branch_seeds": branch_seeds, "fixture_sha256": _sha256_file(fixture_path),
        "reference": reference, "family_summary": family_summary,
        "paired_ess_diagnostics": paired_ess,
        "heuristic_dominance": {"rows": dominance_rows, "veto": decision["heuristic_dominance_veto"]},
        "decision": decision,
        "calibration": {"status": "fixed_half_only_smoke" if mode == "smoke" else "fixed_half_only_pending_full_pilot", "alpha": ALPHA, "nu": NU, "selected_alpha": None, "selected_nu": None},
        "fit": {"direct_value": direct_value, "diagnostics": direct_diagnostics, "wall_seconds": fit_wall},
        "nonclaims": ["exact_pseudo_marginal_likelihood", "exact_posterior", "HMC_readiness", "default_readiness", "source_faithful_full_Zhao_Cui_solver", "statistical_superiority"],
    }
    _write_json(output_root / "result.json", result)
    (output_root / "result.md").write_text(_result_markdown(result), encoding="utf-8")
    try:
        allocator = tf.config.experimental.get_memory_info("GPU:0")
    except (ValueError, RuntimeError):
        allocator = {"unavailable": True}
    manifest = {
        "schema_id": "bayesfilter.c2_ukf_guided_defensive_tt_dmis_run_manifest.v1",
        "started_at_utc": started_at.isoformat(), "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started, "fit_wall_seconds": fit_wall,
        "command": shlex.join([sys.executable, *sys.argv]), "git_commit": subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "git_status_short": subprocess.run(("git", "status", "--short"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "unset"), "python_version": sys.version,
        "tensorflow_version": tf.__version__, "tensorflow_probability_version": tfp.__version__,
        "tensorflow_gpu_memory_policy": memory_policy, "logical_gpus": [str(device) for device in logical_gpus],
        "placement_probe_device": placement_probe.device, "allocator_memory_info_bytes": allocator,
        "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()), "dtype": DTYPE_NAME,
        "jit_compile": True, "xla_required": True, "trust_basis": TRUST_BASIS,
        "fixture_sha256": _sha256_file(fixture_path), "run_identity": run_identity, "run_identity_payload": run_identity_payload,
        "plan_file": str(PLAN.relative_to(ROOT)), "plan_review_file": str(PLAN_REVIEW.relative_to(ROOT)), "latex_file": str(LATEX.relative_to(ROOT)),
        "result_file": str((output_root / "result.json").relative_to(ROOT)), "proposal_manifest": str((output_root / "proposal_manifest.json").relative_to(ROOT)),
        "source_sha256": {str(path.relative_to(ROOT)): _sha256_file(path) for path in (PLAN, PLAN_REVIEW, LATEX, Path(__file__).resolve(), ROOT / "bayesfilter/highdim/c2_transformed_observation_student_proposal_tf.py", ROOT / "bayesfilter/highdim/c2_sv_frozen_proposal_apf_tf.py", ROOT / "bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py")},
    }
    _write_json(output_root / "run_manifest.json", manifest)
    log(f"completed engineering_pass={engineering_pass}, mechanism_pass={mechanism_pass}")


def main() -> None:
    _run(_parse_args())


if __name__ == "__main__":
    main()
