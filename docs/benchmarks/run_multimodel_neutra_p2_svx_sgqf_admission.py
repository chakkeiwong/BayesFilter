"""Run the P2 exact-SV fixed-SGQF level and typed-target admission ladder."""

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
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROGRAM_ID = "multimodel-neutra-filter-posterior-20260715"
CELL_ID = "SVX-SGQF"
CANDIDATE_LEVELS = (2, 4, 6, 8)
REFERENCE_LEVEL = 10
PREFIX_HORIZON = 20
DENSE_ORDER = 401
DENSE_RADIUS = 8.0
VALUE_PREFIX_PER_OBSERVATION_MAX = 1.0e-3
SCORE_PREFIX_FD_MAX = 1.0e-5
VALUE_FULL_PER_OBSERVATION_MAX = 1.0e-4
SCORE_FULL_MAX = 1.0e-3
PLAN_PATH = Path(
    "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-"
    "p2-target-repair-skeptical-audit-2026-07-15.md"
)
NONCLAIMS = (
    "fixed-SGQF target-admission ladder only",
    "no HMC or NeuTra training executed",
    "no exact-likelihood or filter superiority claim",
    "no Zhao-Cui production-route claim",
    "no posterior convergence or scientific readiness claim",
)


def _build_audit_points(tf: Any, tfp: Any) -> Any:
    audit_points = tf.constant(
        [[-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [1.0, 1.0]],
        tf.float64,
    )
    standard_normal = tfp.distributions.Normal(
        loc=tf.constant(0.0, tf.float64),
        scale=tf.constant(1.0, tf.float64),
    )
    truth_probabilities = tf.constant(
        [(0.6 - 0.1) / 0.8, (0.4 - 0.1) / 0.8], tf.float64
    )
    truth_point = standard_normal.quantile(truth_probabilities)[None, :]
    return tf.concat((audit_points, truth_point), axis=0)


def run_admission(output_root: Path) -> Mapping[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"P2 admission output root must be fresh: {output_root}")
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
        exact_transformed_sv_independent_panel_dense_reference,
    )
    from bayesfilter.inference.neutra_campaign import (
        CampaignCellLedger,
        admit_independent_posterior_recomposition,
        issue_typed_neutra_target_identity,
        require_typed_neutra_target,
    )
    from bayesfilter.ssm import stable_ssm_target_signature
    from bayesfilter.testing.exact_sv_sgqf_neutra_target_tf import (
        SVX_SGQF_OBSERVATION_SHA256,
        SVX_SGQF_STATE_SHA256,
        SVX_SGQF_HORIZON,
        ExactSVSGQFLikelihoodRecomposer,
        generate_frozen_exact_sv_dataset_tf,
        make_exact_sv_sgqf_neutra_adapter,
        source_chart_physical_parameters,
        source_two_probit_jacobian_value_score,
        source_uniform_prior_value_score,
    )

    states, observations = generate_frozen_exact_sv_dataset_tf()
    observation_hash = hashlib.sha256(
        bytes(tf.io.serialize_tensor(observations).numpy())
    ).hexdigest()
    state_hash = hashlib.sha256(
        bytes(tf.io.serialize_tensor(states).numpy())
    ).hexdigest()
    if observation_hash != SVX_SGQF_OBSERVATION_SHA256:
        raise RuntimeError("frozen exact-SV observation hash mismatch")
    if state_hash != SVX_SGQF_STATE_SHA256:
        raise RuntimeError("frozen exact-SV state hash mismatch")
    audit_points = _build_audit_points(tf, tfp)
    prefix_observations = observations[:PREFIX_HORIZON]
    gamma, beta = source_chart_physical_parameters(audit_points)
    dense_prefix_values = []
    with tf.device("/CPU:0"):
        for index in range(int(audit_points.shape[0])):
            dense_prefix_values.append(
                exact_transformed_sv_independent_panel_dense_reference(
                    prefix_observations,
                    gamma=gamma[index],
                    beta=beta[index],
                    sigma=1.0,
                    order=DENSE_ORDER,
                    radius=DENSE_RADIUS,
                ).log_likelihood
            )
    dense_prefix_values = tf.stack(dense_prefix_values)

    level_results: dict[int, Mapping[str, Any]] = {}
    level_outputs: dict[int, tuple[tf.Tensor, tf.Tensor]] = {}
    for level in (*CANDIDATE_LEVELS, REFERENCE_LEVEL):
        adapter = make_exact_sv_sgqf_neutra_adapter(
            sparse_level=level, observations=observations
        )
        prefix_adapter = make_exact_sv_sgqf_neutra_adapter(
            sparse_level=level, observations=prefix_observations
        )

        @tf.function(
            input_signature=[tf.TensorSpec([None, 2], tf.float64)],
            jit_compile=True,
            reduce_retracing=True,
        )
        def full_program(theta):
            return adapter.neutra_batch_log_prob_and_grad_status(theta)

        compile_started = time.monotonic()
        full_value, full_score, status = full_program(audit_points)
        compile_seconds = time.monotonic() - compile_started
        warm_started = time.monotonic()
        full_program(tf.reverse(audit_points, axis=(0,)))
        warm_seconds = time.monotonic() - warm_started
        prefix_likelihood, prefix_score = prefix_adapter.log_prob_and_grad(
            audit_points
        )
        prefix_prior, prefix_prior_score = source_uniform_prior_value_score(
            audit_points
        )
        prefix_jacobian, prefix_jacobian_score = (
            source_two_probit_jacobian_value_score(audit_points)
        )
        prefix_likelihood = prefix_likelihood - prefix_prior - prefix_jacobian
        prefix_score = prefix_score - prefix_prior_score - prefix_jacobian_score
        epsilon = tf.constant(1.0e-5, tf.float64)
        fd_columns = []
        for coordinate in range(2):
            basis = tf.one_hot(coordinate, 2, dtype=tf.float64)[None, :]
            plus = prefix_adapter.log_prob(audit_points + epsilon * basis)
            minus = prefix_adapter.log_prob(audit_points - epsilon * basis)
            fd_columns.append((plus - minus) / (2.0 * epsilon))
        finite_difference = tf.stack(fd_columns, axis=1)
        prefix_score_fd_gap = float(
            tf.reduce_max(tf.abs(prefix_score + prefix_prior_score + prefix_jacobian_score - finite_difference)).numpy()
        )
        level_outputs[level] = (full_value, full_score)
        level_results[level] = {
            "level": level,
            "target_signature": stable_ssm_target_signature(adapter.contract),
            "point_count": int(adapter.nodes.shape[0]),
            "prefix_dense_value_gap_per_observation": float(
                tf.reduce_max(
                    tf.abs(prefix_likelihood - dense_prefix_values)
                    / tf.constant(PREFIX_HORIZON, tf.float64)
                ).numpy()
            ),
            "prefix_posterior_score_fd_gap": prefix_score_fd_gap,
            "status_all_valid": bool(
                tf.reduce_all(status["valid_pre_regularized_score"]).numpy()
            ),
            "status_code_all_zero": bool(
                tf.reduce_all(tf.equal(status["status_code"], 0)).numpy()
            ),
            "full_value": tuple(float(item) for item in full_value.numpy().tolist()),
            "full_score": tuple(
                tuple(float(item) for item in row)
                for row in full_score.numpy().tolist()
            ),
            "compile_and_first_seconds": compile_seconds,
            "warm_seconds": warm_seconds,
        }

    selection_rows = []
    selected_level = None
    ordered_next = {2: 4, 4: 6, 6: 8, 8: REFERENCE_LEVEL}
    reference_value, reference_score = level_outputs[REFERENCE_LEVEL]
    for level in CANDIDATE_LEVELS:
        value, score = level_outputs[level]
        next_value, next_score = level_outputs[ordered_next[level]]
        row = dict(level_results[level])
        row.update(
            {
                "next_level": ordered_next[level],
                "full_value_gap_to_next_per_observation": float(
                    tf.reduce_max(tf.abs(value - next_value)).numpy()
                    / SVX_SGQF_HORIZON
                ),
                "full_value_gap_to_reference_per_observation": float(
                    tf.reduce_max(tf.abs(value - reference_value)).numpy()
                    / SVX_SGQF_HORIZON
                ),
                "full_score_gap_to_next": float(
                    tf.reduce_max(tf.abs(score - next_score)).numpy()
                ),
                "full_score_gap_to_reference": float(
                    tf.reduce_max(tf.abs(score - reference_score)).numpy()
                ),
            }
        )
        row["passed"] = bool(
            row["status_all_valid"]
            and row["status_code_all_zero"]
            and row["prefix_dense_value_gap_per_observation"]
            <= VALUE_PREFIX_PER_OBSERVATION_MAX
            and row["prefix_posterior_score_fd_gap"] <= SCORE_PREFIX_FD_MAX
            and row["full_value_gap_to_next_per_observation"]
            <= VALUE_FULL_PER_OBSERVATION_MAX
            and row["full_value_gap_to_reference_per_observation"]
            <= VALUE_FULL_PER_OBSERVATION_MAX
            and row["full_score_gap_to_next"] <= SCORE_FULL_MAX
            and row["full_score_gap_to_reference"] <= SCORE_FULL_MAX
        )
        selection_rows.append(row)
        if selected_level is None and row["passed"]:
            selected_level = level

    identity_payload = None
    recomposition_payload = None
    repaired_registry = None
    if selected_level is not None:
        adapter = make_exact_sv_sgqf_neutra_adapter(
            sparse_level=selected_level, observations=observations
        )
        mathematical_signature = stable_ssm_target_signature(adapter.contract)
        repaired_registry = {
            "schema": "bayesfilter.multimodel_neutra_p2_target_repair_registry.v1",
            "program_id": PROGRAM_ID,
            "cell_id": CELL_ID,
            "previous_state": "TARGET_BLOCKED",
            "state": "VALUE_SCORE_ADMITTED",
            "target_signature": mathematical_signature,
            "selected_sparse_level": selected_level,
            "dataset_sha256": observation_hash,
            "repair_evidence": str(output_root / "result.json"),
            "remaining_blockers": (),
        }
        registry_path = output_root / "repaired_registry.json"
        _write_new_json(registry_path, repaired_registry)
        registry_sha256 = hashlib.sha256(registry_path.read_bytes()).hexdigest()
        likelihood_recomposer = ExactSVSGQFLikelihoodRecomposer(adapter)
        recomposition = admit_independent_posterior_recomposition(
            adapter=adapter,
            points=audit_points,
            prior_value_score_fn=source_uniform_prior_value_score,
            likelihood_value_score_fn=likelihood_recomposer.__call__,
            jacobian_value_score_fn=source_two_probit_jacobian_value_score,
            value_tolerance=1.0e-10,
            score_tolerance=1.0e-10,
        )
        identity = issue_typed_neutra_target_identity(
            program_id=PROGRAM_ID,
            scope_kind="model_cell",
            scope_id=CELL_ID,
            adapter=adapter,
            recomposition=recomposition,
            registry_row=repaired_registry,
            registry_artifact_sha256=registry_sha256,
        )
        require_typed_neutra_target(identity, adapter=adapter)
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

    passed = selected_level is not None and identity_payload is not None
    result = {
        "schema": "bayesfilter.multimodel_neutra_p2_svx_sgqf_admission.v1",
        "program_id": PROGRAM_ID,
        "cell_id": CELL_ID,
        "passed": bool(passed),
        "decision": (
            "ADMIT_SVX_SGQF_POSTERIOR_IDENTITY"
            if passed
            else "KEEP_SVX_SGQF_TARGET_BLOCKED_NO_LEVEL_PASSED"
        ),
        "dataset": {
            "seed": 81101,
            "horizon": 1000,
            "observation_sha256": observation_hash,
            "state_sha256": state_hash,
            "truth_physical": (0.6, 0.4, 1.0),
            "truth_role": "explanatory_only",
        },
        "source_target": {
            "physical_prior": "independent Uniform(0.1,0.9) for gamma,beta",
            "chart": "gamma=0.1+0.8*Phi(u0), beta=0.1+0.8*Phi(u1)",
            "fixed_sigma": 1.0,
            "observation": "z=log(y^2), exact log-chi-square density",
        },
        "audit_points": tuple(tuple(float(item) for item in row) for row in audit_points.numpy().tolist()),
        "thresholds": {
            "prefix_dense_value_gap_per_observation": VALUE_PREFIX_PER_OBSERVATION_MAX,
            "prefix_posterior_score_fd_gap": SCORE_PREFIX_FD_MAX,
            "full_value_gap_per_observation": VALUE_FULL_PER_OBSERVATION_MAX,
            "full_score_gap": SCORE_FULL_MAX,
        },
        "selection_rows": selection_rows,
        "reference_level": level_results[REFERENCE_LEVEL],
        "selected_level": selected_level,
        "target_identity": identity_payload,
        "recomposition": recomposition_payload,
        "zhao_cui_cell_status": "TARGET_BLOCKED_SOURCE_ROUTE_MISMATCH",
        "enhanced_family_status": "UNAVAILABLE_CAPABILITY_NOT_EXECUTED",
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
            "schema": "bayesfilter.multimodel_neutra_p2_artifact_hashes.v1",
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
        "schema": "bayesfilter.multimodel_neutra_p2_run_manifest.v1",
        "program_id": PROGRAM_ID,
        "cell_id": CELL_ID,
        "git_commit": git_commit,
        "dirty_worktree_disclosure": "shared dirty worktree; scoped paths only",
        "command": (
            "TF_FORCE_GPU_ALLOW_GROWTH=true /home/chakwong/anaconda3/envs/tf-gpu/bin/python "
            "docs/benchmarks/run_multimodel_neutra_p2_svx_sgqf_admission.py "
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
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "data_version": "zhao_cui_sv_actual_nongaussian_T1000 seed 81101",
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
                "selected_level": result["selected_level"],
                "output_root": str(args.output_root),
            },
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
