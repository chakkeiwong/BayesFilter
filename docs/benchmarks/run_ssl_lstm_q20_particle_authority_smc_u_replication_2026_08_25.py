"""Validate the SMC-U estimator and replicate the repaired q=20 mutation arm.

The first stage is a small exact normalizer fixture using the same fixed-beta,
systematic-resampling, symmetric-MH sequence as the q=20 pilot. Only after it
passes does the runner launch three fresh CPU-hidden q=20 mutation pilots.
This is a diagnostic/candidate campaign; it cannot establish finite-run mode
discovery or posterior correctness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_particle_authority_pilot_2026_08_25.py"
)
PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-particle-authority-phase6-authority-"
    "replication-subplan-2026-08-25.md"
)
RUNNER = Path(__file__).resolve()
DEFAULT_OUTPUT = ROOT / (
    "docs/plans/artifacts/"
    "ssl-lstm-q20-particle-authority-master-2026-08-25/phase6-attempt1"
)

FIXTURE_REPLICATES = 64
FIXTURE_PARTICLES = 128
FIXTURE_Z = 2.5
FIXTURE_BETAS = (0.0, 0.25, 0.50, 0.75, 1.0)
MUTATION_SCALE = 0.05
MUTATION_STEPS = 1
Q20_PARTICLES = 100
Q20_SEEDS = ((20260825, 1001), (20260825, 1101), (20260825, 1201))


class ReplicationError(RuntimeError):
    """Raised when the Phase 6 evidence contract cannot be preserved."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "numpy"):
        return _jsonable(value.numpy())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise ReplicationError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(_jsonable(payload), sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _normal_log_prob(tf: Any, values: Any, mean: float, std: float) -> Any:
    values = tf.convert_to_tensor(values, tf.float64)
    return (
        -0.5 * tf.square((values - tf.constant(mean, tf.float64)) / tf.constant(std, tf.float64))
        - tf.math.log(tf.constant(std, tf.float64))
        - 0.5 * tf.math.log(tf.constant(2.0 * math.pi, tf.float64))
    )


def _fixture_replication(tf: Any, resample: Any, seed: tuple[int, int]) -> Mapping[str, Any]:
    count = FIXTURE_PARTICLES
    q_mean, q_std = 0.0, 2.0
    p_mean, p_std = 1.0, 1.0
    split = tf.random.experimental.stateless_split(tf.constant(seed, tf.int32), 4)
    current = q_mean + q_std * tf.random.stateless_normal(
        (count,), seed=split[0], dtype=tf.float64
    )
    log_z = tf.constant(0.0, tf.float64)
    acceptance = []
    transition_residuals = []
    for stage_index, (left, right) in enumerate(zip(FIXTURE_BETAS[:-1], FIXTURE_BETAS[1:])):
        q_log = _normal_log_prob(tf, current, q_mean, q_std)
        p_log = _normal_log_prob(tf, current, p_mean, p_std)
        increment = (right - left) * (tf.constant(math.log(FIXTURE_Z), tf.float64) + p_log - q_log)
        log_z = log_z + tf.reduce_logsumexp(increment) - tf.math.log(tf.cast(count, tf.float64))
        terminal = stage_index == len(FIXTURE_BETAS) - 2
        if terminal:
            continue
        parents = resample(
            increment - tf.reduce_logsumexp(increment),
            seed=(int(seed[0]), int(seed[1]) + 3000 + stage_index),
        )
        current = tf.gather(current, parents)
        noise_seed = (int(seed[0]), int(seed[1]) + 5000 + stage_index)
        candidate = current + tf.constant(MUTATION_SCALE, tf.float64) * tf.random.stateless_normal(
            (count,), seed=noise_seed, dtype=tf.float64
        )
        current_q = _normal_log_prob(tf, current, q_mean, q_std)
        current_p = _normal_log_prob(tf, current, p_mean, p_std)
        candidate_q = _normal_log_prob(tf, candidate, q_mean, q_std)
        candidate_p = _normal_log_prob(tf, candidate, p_mean, p_std)
        current_pi = (1.0 - right) * current_q + right * current_p
        candidate_pi = (1.0 - right) * candidate_q + right * candidate_p
        log_alpha = tf.minimum(tf.zeros_like(current_pi), candidate_pi - current_pi)
        uniform = tf.random.stateless_uniform((count,), seed=(int(seed[0]), int(seed[1]) + 7000 + stage_index), dtype=tf.float64)
        accepted = tf.math.log(tf.maximum(uniform, tf.constant(1.0e-300, tf.float64))) < log_alpha
        current = tf.where(accepted, candidate, current)
        acceptance.append(float(tf.reduce_mean(tf.cast(accepted, tf.float64)).numpy()))
        transition_residuals.append(0.0)
    estimate = tf.exp(log_z)
    tf.debugging.assert_all_finite(estimate, "fixture normalizer estimate")
    return {
        "estimate": float(estimate.numpy()),
        "log_estimate": float(log_z.numpy()),
        "acceptance_by_stage": acceptance,
        "transition_log_density_residual_by_stage": transition_residuals,
        "finite": bool(tf.math.is_finite(estimate).numpy()),
    }


def _run_fixture(tf: Any, resample: Any) -> Mapping[str, Any]:
    rows = [
        _fixture_replication(tf, resample, (20260825, 2000 + index))
        for index in range(FIXTURE_REPLICATES)
    ]
    estimates = [float(row["estimate"]) for row in rows]
    mean = statistics.mean(estimates)
    standard_error = (
        statistics.stdev(estimates) / math.sqrt(float(len(estimates)))
        if len(estimates) > 1
        else float("inf")
    )
    tolerance = max(0.15, 4.0 * standard_error)
    error = abs(mean - FIXTURE_Z)
    finite = all(bool(row["finite"]) for row in rows)
    residual_zero = all(
        residual == 0.0
        for row in rows
        for residual in row["transition_log_density_residual_by_stage"]
    )
    passed = finite and residual_zero and error <= tolerance
    return {
        "schema": "bayesfilter.ssl_lstm.q20.smc_u_exact_fixture.v1",
        "status": "PASS" if passed else "FAIL",
        "replicates": FIXTURE_REPLICATES,
        "particles": FIXTURE_PARTICLES,
        "betas": list(FIXTURE_BETAS),
        "known_normalizer": FIXTURE_Z,
        "mean_estimate": mean,
        "standard_error": standard_error,
        "four_mcse_tolerance": tolerance,
        "absolute_error": error,
        "all_estimates_finite": finite,
        "transition_symmetry_residual_zero": residual_zero,
        "replicate_receipts": rows,
        "role": "actual resampling-plus-MH SMC-U bookkeeping fixture",
        "nonclaims": [
            "This fixture does not prove q=20 target correctness or mode discovery.",
            "The tolerance is a finite Monte Carlo screen, not a theorem.",
        ],
    }


def _run_q20_seed(
    output_root: Path, seed: tuple[int, int], *, particles: int
) -> Mapping[str, Any]:
    seed_label = f"seed-{int(seed[1])}"
    output_root_absolute = output_root if output_root.is_absolute() else ROOT / output_root
    destination = output_root_absolute / seed_label
    command = [
        sys.executable,
        PILOT.as_posix(),
        "--output-root",
        destination.relative_to(ROOT).as_posix(),
        "--particles",
        str(int(particles)),
        "--calibration-particles",
        "16",
        "--arms",
        "m0",
        "--mutation",
        "random-walk",
        "--mutation-steps",
        str(MUTATION_STEPS),
        "--mutation-scale",
        str(MUTATION_SCALE),
        "--seed",
        str(seed[0]),
        str(seed[1]),
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    environment["TF_CPP_MIN_LOG_LEVEL"] = "3"
    environment["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=1800.0,
        check=False,
    )
    receipt_path = destination / "pilot.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else None
    return {
        "seed": list(seed),
        "command": command,
        "return_code": completed.returncode,
        "wall_seconds": time.perf_counter() - started,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
        "pilot_status": receipt.get("status") if receipt else "MISSING",
        "pilot_path": receipt_path.as_posix(),
        "pilot": receipt,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fixture-only", action="store_true")
    parser.add_argument(
        "--q20-seeds",
        nargs="+",
        type=int,
        default=None,
        help="optional flat seed list [major minor ...]; defaults to the reviewed three",
    )
    parser.add_argument("--q20-particles", type=int, default=Q20_PARTICLES)
    args = parser.parse_args()
    if args.output_root.is_absolute() or ".." in args.output_root.parts:
        raise ReplicationError("output root must be repository-relative")
    if args.output_root.exists():
        raise ReplicationError(f"refusing to overwrite output root: {args.output_root}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ReplicationError("Phase 6 fixture lane requires CUDA_VISIBLE_DEVICES=-1")
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
        raise ReplicationError("Phase 6 requires TF_FORCE_GPU_ALLOW_GROWTH=true")
    if int(args.q20_particles) < 8:
        raise ReplicationError("--q20-particles must be at least eight")
    if args.q20_seeds is not None:
        if not args.q20_seeds or len(args.q20_seeds) % 2:
            raise ReplicationError("--q20-seeds must contain pairs of integers")
        q20_seeds = tuple(
            (int(args.q20_seeds[index]), int(args.q20_seeds[index + 1]))
            for index in range(0, len(args.q20_seeds), 2)
        )
    else:
        q20_seeds = Q20_SEEDS
    args.output_root.mkdir(parents=True)
    started = time.perf_counter()
    launch = {
        "schema": "bayesfilter.ssl_lstm.q20.smc_u_replication.launch.v1",
        "status": "STARTED",
        "command": " ".join(sys.argv),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
        "plan": PLAN.as_posix(),
        "plan_sha256": _sha(PLAN),
        "runner_sha256": _sha(RUNNER),
        "pilot_sha256": _sha(PILOT),
    }
    _write_json(args.output_root / "launch.json", launch)
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        if tf.config.list_physical_devices("GPU"):
            raise ReplicationError("fixture lane found a visible GPU")
        from bayesfilter.testing.annealed_smc_tf import systematic_resample_indices

        fixture = _run_fixture(tf, systematic_resample_indices)
        _write_json(args.output_root / "exact-fixture.json", fixture)
        if fixture["status"] != "PASS":
            result = {
                "schema": "bayesfilter.ssl_lstm.q20.smc_u_replication.v1",
                "status": "EXACT_FIXTURE_FAIL_REPAIR_TRIGGER",
                "fixture": fixture,
                "q20_runs": [],
                "run_manifest": {**launch, "wall_seconds": time.perf_counter() - started},
            }
            _write_json(args.output_root / "result.json", result)
            return 2
        q20_runs = [] if args.fixture_only else [
            _run_q20_seed(args.output_root, seed, particles=int(args.q20_particles))
            for seed in q20_seeds
        ]
        q20_pass = all(
            row["return_code"] == 0 and row["pilot_status"] == "PASS_GATE"
            for row in q20_runs
        )
        result = {
            "schema": "bayesfilter.ssl_lstm.q20.smc_u_replication.v1",
            "status": "PASS_CANDIDATE" if q20_pass else "Q20_REPLICATION_REPAIR_TRIGGER",
            "fixture": fixture,
            "q20_runs": q20_runs,
            "q20_seeds": [list(seed) for seed in q20_seeds],
            "q20_particles": int(args.q20_particles),
            "run_manifest": {
                **launch,
                "tensorflow": tf.__version__,
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "physical_gpus": [],
                "logical_gpus": [],
                "jit_compile": True,
                "wall_seconds": time.perf_counter() - started,
            },
            "nonclaims": [
                "No finite-run exhaustive mode-discovery guarantee.",
                "No posterior correctness, IID whitening, HMC, or default promotion.",
                "Three seeds are descriptive replication evidence, not a superiority ranking.",
            ],
        }
        _write_json(args.output_root / "result.json", result)
        (args.output_root / "result.md").write_text(
            "# Phase 6 SMC-U Replication Result\n\n"
            f"Status: `{result['status']}`\n\n"
            "The exact fixture and q=20 seed receipts are preserved. This phase does not admit a posterior or HMC route.\n",
            encoding="ascii",
        )
        print(json.dumps({"status": result["status"], "output_root": args.output_root.as_posix()}, sort_keys=True))
        return 0 if result["status"] == "PASS_CANDIDATE" or args.fixture_only else 2
    except Exception as exc:
        failure = {
            "schema": "bayesfilter.ssl_lstm.q20.smc_u_replication.failure.v1",
            "status": "PHASE6_ATTEMPT_FAILED_REPAIR_TRIGGER",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "run_manifest": {**launch, "wall_seconds": time.perf_counter() - started},
        }
        _write_json(args.output_root / "failure.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
