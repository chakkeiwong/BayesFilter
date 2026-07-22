"""Run the P3 KSC principal-square-root-UKF target-admission gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROGRAM_ID = "multimodel-neutra-filter-posterior-20260715"
CELL_ID = "KSC-UKF"
PREFIX_HORIZON = 20
DENSE_RADIUS = 8.0
DENSE_ORDERS = (401, 601)
FD_STEPS = (1.0e-4, 3.0e-5, 1.0e-5)
DENSE_VALUE_ORDER_GAP_PER_OBSERVATION_MAX = 2.0e-4
DENSE_FD_STEP_GAP_MAX = 2.0e-3
DENSE_FD_ORDER_GAP_MAX = 2.0e-3
UKF_DENSE_VALUE_GAP_PER_OBSERVATION_MAX = 1.0e-3
UKF_DENSE_SCORE_GAP_MAX = 1.0e-2
PLAN_PATH = Path(
    "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-"
    "p3-ksc-ukf-subplan-2026-07-15.md"
)
NONCLAIMS = (
    "KSC-UKF target-admission gate only",
    "no exact-SV or exact latent-state filtering claim",
    "no HMC or NeuTra training executed",
    "no calibration, superiority, or readiness claim",
)


def build_audit_points(tf: Any, tfp: Any) -> Any:
    fixed = tf.constant(
        [[-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [1.0, 1.0]],
        tf.float64,
    )
    normal = tfp.distributions.Normal(
        loc=tf.constant(0.0, tf.float64),
        scale=tf.constant(1.0, tf.float64),
    )
    truth_probabilities = tf.constant(
        [(0.6 - 0.1) / 0.8, (0.4 - 0.1) / 0.8], tf.float64
    )
    return tf.concat((fixed, normal.quantile(truth_probabilities)[None, :]), axis=0)


def run_admission(output_root: Path) -> Mapping[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"P3 admission output root must be fresh: {output_root}")
    output_root.mkdir(parents=True)
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()

    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)

    import tensorflow_probability as tfp

    from bayesfilter.highdim.sv_mixture_cut4 import (
        StochasticVolatilitySSM,
        scalar_sv_mixture_dense_reference,
    )
    from bayesfilter.inference.neutra_campaign import (
        CampaignCellLedger,
        admit_independent_posterior_recomposition,
        issue_typed_neutra_target_identity,
        require_typed_neutra_target,
    )
    from bayesfilter.ssm import stable_ssm_target_signature
    from bayesfilter.testing.exact_sv_sgqf_neutra_target_tf import (
        generate_frozen_exact_sv_dataset_tf,
        source_chart_physical_parameters,
        source_two_probit_jacobian_value_score,
        source_uniform_prior_value_score,
    )
    from bayesfilter.testing.ksc_ukf_neutra_target_tf import (
        KSC_UKF_DATASET_SEED,
        KSC_UKF_HORIZON,
        KSC_UKF_RAW_OBSERVATION_SHA256,
        KSC_UKF_STATE_SHA256,
        KSCUKFLikelihoodRecomposer,
        make_ksc_ukf_neutra_adapter,
        transformed_ksc_observations,
    )

    states, raw_observations = generate_frozen_exact_sv_dataset_tf()
    raw_hash = _tensor_hash(tf, raw_observations)
    state_hash = _tensor_hash(tf, states)
    if raw_hash != KSC_UKF_RAW_OBSERVATION_SHA256:
        raise RuntimeError("frozen KSC raw-observation hash mismatch")
    if state_hash != KSC_UKF_STATE_SHA256:
        raise RuntimeError("frozen KSC state hash mismatch")
    transformed = transformed_ksc_observations(raw_observations)
    transformed_hash = _tensor_hash(tf, transformed)
    audit_points = build_audit_points(tf, tfp)
    full_adapter = make_ksc_ukf_neutra_adapter(
        raw_observations=raw_observations
    )
    prefix_adapter = make_ksc_ukf_neutra_adapter(
        raw_observations=raw_observations[:PREFIX_HORIZON]
    )

    @tf.function(
        input_signature=[tf.TensorSpec([None, 2], tf.float64)],
        jit_compile=True,
        reduce_retracing=True,
    )
    def full_program(theta):
        return full_adapter.neutra_batch_log_prob_and_grad_status(theta)

    compile_started = time.monotonic()
    full_value, full_score, full_status = full_program(audit_points)
    compile_seconds = time.monotonic() - compile_started
    warm_started = time.monotonic()
    reversed_value, reversed_score, reversed_status = full_program(
        tf.reverse(audit_points, axis=(0,))
    )
    warm_seconds = time.monotonic() - warm_started
    permutation_value_gap = float(
        tf.reduce_max(
            tf.abs(full_value - tf.reverse(reversed_value, axis=(0,)))
        ).numpy()
    )
    permutation_score_gap = float(
        tf.reduce_max(
            tf.abs(full_score - tf.reverse(reversed_score, axis=(0,)))
        ).numpy()
    )
    permutation_status_equal = bool(
        tf.reduce_all(
            tf.equal(
                full_status["status_code"],
                tf.reverse(reversed_status["status_code"], axis=(0,)),
            )
        ).numpy()
    )

    prefix_posterior_value, prefix_posterior_score = (
        prefix_adapter.log_prob_and_grad(audit_points)
    )
    prefix_prior_value, prefix_prior_score = source_uniform_prior_value_score(
        audit_points
    )
    prefix_jacobian_value, prefix_jacobian_score = (
        source_two_probit_jacobian_value_score(audit_points)
    )
    prefix_likelihood_value = (
        prefix_posterior_value - prefix_prior_value - prefix_jacobian_value
    )
    prefix_likelihood_score = (
        prefix_posterior_score - prefix_prior_score - prefix_jacobian_score
    )

    dense_model = StochasticVolatilitySSM(sigma=1.0)

    def dense_value(theta_row: Any, *, order: int) -> float:
        gamma, beta = source_chart_physical_parameters(theta_row[None, :])
        legacy_theta = tf.stack(
            (
                tfp.distributions.Normal(
                    loc=tf.constant(0.0, tf.float64),
                    scale=tf.constant(1.0, tf.float64),
                ).quantile(gamma[0]),
                tf.math.log(beta[0]),
            )
        )
        with tf.device("/CPU:0"):
            result = scalar_sv_mixture_dense_reference(
                dense_model,
                legacy_theta,
                raw_observations[:PREFIX_HORIZON],
                order=int(order),
                radius=DENSE_RADIUS,
            )
        return float(result.log_likelihood.numpy())

    dense_values: dict[int, Any] = {}
    for order in DENSE_ORDERS:
        dense_values[order] = tf.constant(
            [dense_value(row, order=order) for row in tf.unstack(audit_points)],
            tf.float64,
        )

    def dense_fd(order: int, epsilon: float) -> Any:
        rows = []
        for row in tf.unstack(audit_points):
            coordinates = []
            for coordinate in range(2):
                basis = tf.one_hot(coordinate, 2, dtype=tf.float64)
                plus = dense_value(
                    row + tf.constant(epsilon, tf.float64) * basis, order=order
                )
                minus = dense_value(
                    row - tf.constant(epsilon, tf.float64) * basis, order=order
                )
                coordinates.append((plus - minus) / (2.0 * epsilon))
            rows.append(coordinates)
        return tf.constant(rows, tf.float64)

    dense_fd_estimates: dict[tuple[int, float], Any] = {}
    for step in FD_STEPS:
        dense_fd_estimates[(401, step)] = dense_fd(401, step)
    dense_fd_estimates[(601, 3.0e-5)] = dense_fd(601, 3.0e-5)

    dense_value_order_gap_per_observation = float(
        tf.reduce_max(tf.abs(dense_values[401] - dense_values[601])).numpy()
        / PREFIX_HORIZON
    )
    dense_fd_step_gap = float(
        tf.reduce_max(
            tf.abs(
                dense_fd_estimates[(401, 3.0e-5)]
                - dense_fd_estimates[(401, 1.0e-5)]
            )
        ).numpy()
    )
    dense_fd_order_gap = float(
        tf.reduce_max(
            tf.abs(
                dense_fd_estimates[(401, 3.0e-5)]
                - dense_fd_estimates[(601, 3.0e-5)]
            )
        ).numpy()
    )
    ukf_dense_value_gap_per_observation = float(
        tf.reduce_max(tf.abs(prefix_likelihood_value - dense_values[601])).numpy()
        / PREFIX_HORIZON
    )
    ukf_dense_score_gap = float(
        tf.reduce_max(
            tf.abs(
                prefix_likelihood_score - dense_fd_estimates[(601, 3.0e-5)]
            )
        ).numpy()
    )
    all_dense_finite = all(
        bool(tf.reduce_all(tf.math.is_finite(value)).numpy())
        for value in (*dense_values.values(), *dense_fd_estimates.values())
    )
    graph_valid = bool(
        tf.reduce_all(full_status["valid_pre_regularized_score"]).numpy()
    ) and bool(tf.reduce_all(tf.equal(full_status["status_code"], 0)).numpy())
    engineering_passed = bool(
        graph_valid
        and permutation_value_gap <= 1.0e-12
        and permutation_score_gap <= 1.0e-12
        and permutation_status_equal
    )
    reference_passed = bool(
        all_dense_finite
        and dense_value_order_gap_per_observation
        <= DENSE_VALUE_ORDER_GAP_PER_OBSERVATION_MAX
        and dense_fd_step_gap <= DENSE_FD_STEP_GAP_MAX
        and dense_fd_order_gap <= DENSE_FD_ORDER_GAP_MAX
    )
    filter_passed = bool(
        reference_passed
        and ukf_dense_value_gap_per_observation
        <= UKF_DENSE_VALUE_GAP_PER_OBSERVATION_MAX
        and ukf_dense_score_gap <= UKF_DENSE_SCORE_GAP_MAX
    )

    identity_payload = None
    recomposition_payload = None
    repaired_registry = None
    if engineering_passed and filter_passed:
        target_signature = stable_ssm_target_signature(full_adapter.contract)
        repaired_registry = {
            "schema": "bayesfilter.multimodel_neutra_p3_target_repair_registry.v1",
            "program_id": PROGRAM_ID,
            "cell_id": CELL_ID,
            "previous_state": "TARGET_BLOCKED",
            "state": "VALUE_SCORE_ADMITTED",
            "target_signature": target_signature,
            "dataset_sha256": raw_hash,
            "transformed_dataset_sha256": transformed_hash,
            "repair_evidence": str(output_root / "result.json"),
            "remaining_blockers": (),
        }
        registry_path = output_root / "repaired_registry.json"
        _write_new_json(registry_path, repaired_registry)
        registry_sha256 = hashlib.sha256(registry_path.read_bytes()).hexdigest()
        recomposition = admit_independent_posterior_recomposition(
            adapter=full_adapter,
            points=audit_points,
            prior_value_score_fn=source_uniform_prior_value_score,
            likelihood_value_score_fn=KSCUKFLikelihoodRecomposer(full_adapter),
            jacobian_value_score_fn=source_two_probit_jacobian_value_score,
            value_tolerance=1.0e-10,
            score_tolerance=1.0e-10,
        )
        identity = issue_typed_neutra_target_identity(
            program_id=PROGRAM_ID,
            scope_kind="model_cell",
            scope_id=CELL_ID,
            adapter=full_adapter,
            recomposition=recomposition,
            registry_row=repaired_registry,
            registry_artifact_sha256=registry_sha256,
        )
        require_typed_neutra_target(identity, adapter=full_adapter)
        ledger = CampaignCellLedger(
            {"cells": [repaired_registry]},
            event_path=output_root / "cell_events.jsonl",
        )
        ledger.transition(
            cell_id=CELL_ID,
            new_state="POSTERIOR_IDENTITY_ADMITTED",
            evidence_path=str(output_root / "target_identity.json"),
            target_identity=identity,
        )
        identity_payload = identity.payload()
        recomposition_payload = recomposition.payload()
        _write_new_json(output_root / "target_identity.json", identity_payload)
        _write_new_json(output_root / "recomposition.json", recomposition_payload)
        _write_new_json(output_root / "cell_ledger.json", ledger.payload())

    passed = bool(identity_payload is not None)
    result = {
        "schema": "bayesfilter.multimodel_neutra_p3_ksc_ukf_admission.v1",
        "program_id": PROGRAM_ID,
        "cell_id": CELL_ID,
        "passed": passed,
        "decision": (
            "ADMIT_KSC_UKF_POSTERIOR_IDENTITY"
            if passed
            else (
                "KEEP_KSC_UKF_TARGET_BLOCKED_FILTER_GATE"
                if engineering_passed and reference_passed
                else (
                    "KEEP_KSC_UKF_TARGET_BLOCKED_REFERENCE_INVALID"
                    if engineering_passed
                    else "KEEP_KSC_UKF_IMPLEMENTATION_BLOCKED"
                )
            )
        ),
        "dataset": {
            "seed": KSC_UKF_DATASET_SEED,
            "horizon": KSC_UKF_HORIZON,
            "raw_observation_sha256": raw_hash,
            "state_sha256": state_hash,
            "transformed_observation_sha256": transformed_hash,
            "transform": "log(y^2+1e-8)",
            "truth_physical": (0.6, 0.4, 1.0),
            "truth_role": "explanatory_only",
        },
        "audit_points": _json_ready(audit_points),
        "thresholds": {
            "dense_value_order_gap_per_observation": DENSE_VALUE_ORDER_GAP_PER_OBSERVATION_MAX,
            "dense_fd_step_gap": DENSE_FD_STEP_GAP_MAX,
            "dense_fd_order_gap": DENSE_FD_ORDER_GAP_MAX,
            "ukf_dense_value_gap_per_observation": UKF_DENSE_VALUE_GAP_PER_OBSERVATION_MAX,
            "ukf_dense_score_gap": UKF_DENSE_SCORE_GAP_MAX,
        },
        "engineering": {
            "passed": engineering_passed,
            "full_status_all_valid": graph_valid,
            "permutation_value_gap": permutation_value_gap,
            "permutation_score_gap": permutation_score_gap,
            "permutation_status_equal": permutation_status_equal,
            "compile_and_first_seconds": compile_seconds,
            "warm_seconds": warm_seconds,
            "full_value": _json_ready(full_value),
            "full_score": _json_ready(full_score),
            "minimum_innovation_variance": _json_ready(
                full_status["min_innovation_eigenvalue"]
            ),
            "minimum_state_variance": _json_ready(
                full_status["minimum_state_variance"]
            ),
            "maximum_mixture_weight_sum_error": _json_ready(
                full_status["maximum_mixture_weight_sum_error"]
            ),
        },
        "dense_reference": {
            "passed": reference_passed,
            "orders": DENSE_ORDERS,
            "radius": DENSE_RADIUS,
            "fd_steps": FD_STEPS,
            "all_finite": all_dense_finite,
            "values": {
                str(order): _json_ready(value)
                for order, value in dense_values.items()
            },
            "fd_scores": {
                f"order-{order}-step-{step:.0e}": _json_ready(value)
                for (order, step), value in dense_fd_estimates.items()
            },
            "value_order_gap_per_observation": dense_value_order_gap_per_observation,
            "fd_step_gap": dense_fd_step_gap,
            "fd_order_gap": dense_fd_order_gap,
        },
        "filter_admission": {
            "passed": filter_passed,
            "ukf_dense_value_gap_per_observation": ukf_dense_value_gap_per_observation,
            "ukf_dense_score_gap": ukf_dense_score_gap,
            "ukf_prefix_likelihood_value": _json_ready(prefix_likelihood_value),
            "ukf_prefix_likelihood_score": _json_ready(prefix_likelihood_score),
        },
        "target_identity": identity_payload,
        "recomposition": recomposition_payload,
        "elapsed_seconds": time.monotonic() - started,
        "nonclaims": NONCLAIMS,
    }
    _write_new_json(output_root / "result.json", result)
    manifest = _run_manifest(
        output_root=output_root,
        started_at=started_at,
        tensorflow_version=tf.__version__,
        tfp_version=tfp.__version__,
        memory_policy=memory_policy,
        target_signature=(
            None if identity_payload is None else identity_payload["target_signature"]
        ),
        wall_time=time.monotonic() - started,
    )
    _write_new_json(output_root / "run_manifest.json", manifest)
    hashes = {
        str(path.relative_to(output_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _write_new_json(
        output_root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.multimodel_neutra_p3_artifact_hashes.v1",
            "artifacts": hashes,
        },
    )
    return result


def _run_manifest(
    *,
    output_root: Path,
    started_at: datetime,
    tensorflow_version: str,
    tfp_version: str,
    memory_policy: Mapping[str, Any],
    target_signature: str | None,
    wall_time: float,
) -> Mapping[str, Any]:
    git_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "schema": "bayesfilter.multimodel_neutra_p3_run_manifest.v1",
        "program_id": PROGRAM_ID,
        "cell_id": CELL_ID,
        "git_commit": git_commit,
        "dirty_worktree_disclosure": "shared dirty worktree; scoped paths only",
        "command": (
            "TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/bayesfilter-mpl "
            "/home/chakwong/anaconda3/envs/tf-gpu/bin/python "
            "docs/benchmarks/run_multimodel_neutra_p3_ksc_ukf_admission.py "
            f"--output-root {output_root}"
        ),
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "python_executable": sys.executable,
        "tensorflow_version": tensorflow_version,
        "tensorflow_probability_version": tfp_version,
        "gpu_memory_policy": memory_policy,
        "jit_compile": True,
        "dtype": "float64",
        "tf32_execution_enabled": True,
        "cpu_reference": "T20 dense Legendre orders 401 and 601",
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "data_version": "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000 seed 81101",
        "random_seeds": {"dataset": 81101, "filter": "deterministic"},
        "target_signature": target_signature,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_time_seconds": float(wall_time),
        "output_root": str(output_root),
        "plan_file": str(PLAN_PATH),
        "result_file": str(output_root / "result.json"),
        "nonclaims": NONCLAIMS,
    }


def _tensor_hash(tf: Any, value: Any) -> str:
    return hashlib.sha256(bytes(tf.io.serialize_tensor(value).numpy())).hexdigest()


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "numpy"):
        return _json_ready(value.numpy())
    if hasattr(value, "tolist") and hasattr(value, "shape"):
        return _json_ready(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_admission(args.output_root)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "decision": result["decision"],
                "output_root": str(args.output_root),
            },
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
