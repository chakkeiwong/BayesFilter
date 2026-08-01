"""Run the frozen P4 PP-SGQF level and posterior-identity admission gate."""

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
CELL_ID = "PP-SGQF"
LEVELS = (2, 3, 4, 5)
CANDIDATE_LEVELS = (2, 3, 4)
VALUE_CONVERGENCE_LIMIT = 0.25
SCORE_CONVERGENCE_LIMIT = 0.5
PF_PRACTICAL_MARGIN = 1.0
MINIMUM_VARIANCE = 1.0e-12
PLAN_PATH = Path(
    "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-"
    "p4-pp-sgqf-level-design-2026-07-15.md"
)
PF_REFERENCE_ROOT = Path(
    "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/"
    "phase-p4/PP-UKF/pf-target-admission/attempt-03-20260715T121908Z"
)
PF_RESULT_SHA256 = "3771961abd01d3d75a964b5568706f706a56e71aa19f4e8e4a87e1a56b43c8c4"
NONCLAIMS = (
    "fixed-SGQF approximate predator-prey filter posterior only",
    "bootstrap PF is a stochastic value diagnostic, not exact likelihood",
    "PF supplies no score authority",
    "level convergence is an admission screen, not superiority evidence",
    "no HMC, NeuTra training, calibration, or readiness claim",
)


def run_campaign(output_root: Path) -> Mapping[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"P4 SGQF output root must be fresh: {output_root}")
    output_root.mkdir(parents=True)
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()

    pf_payload, pf_reference_record = _load_and_verify_pf_reference()
    selected_rung = next(
        row for row in pf_payload["pf_rungs"] if row["rung_id"] == "PF1"
    )
    if not pf_payload["reference_stabilized"] or pf_payload["selected_rung"] != "PF1":
        raise RuntimeError("frozen PP-UKF artifact does not admit stabilized PF1")

    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)

    import tensorflow_probability as tfp

    from bayesfilter.inference.neutra_campaign import (
        CampaignCellLedger,
        admit_independent_posterior_recomposition,
        issue_typed_neutra_target_identity,
        require_typed_neutra_target,
    )
    from bayesfilter.ssm import stable_ssm_target_signature
    from bayesfilter.testing.predator_prey_sgqf_neutra_target_tf import (
        PredatorPreySGQFLikelihoodRecomposer,
        make_predator_prey_sgqf_neutra_adapter,
    )
    from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
        PP_OBSERVATION_SHA256,
        PP_STATE_SHA256,
        generate_frozen_predator_prey_dataset_tf,
        source_six_probit_jacobian_value_score,
        source_uniform_prior_value_score,
    )

    states, observations = generate_frozen_predator_prey_dataset_tf()
    state_hash = _tensor_hash(tf, states)
    observation_hash = _tensor_hash(tf, observations)
    if state_hash != PP_STATE_SHA256 or observation_hash != PP_OBSERVATION_SHA256:
        raise RuntimeError("frozen predator-prey dataset hash mismatch")
    if observation_hash != pf_payload["dataset"]["observation_sha256"]:
        raise RuntimeError("SGQF and PF reference observation hashes differ")

    audit_points = tf.constant(pf_payload["audit_points"], tf.float64)
    prior_value, _prior_score = source_uniform_prior_value_score(audit_points)
    jacobian_value, _jacobian_score = source_six_probit_jacobian_value_score(
        audit_points
    )
    adapters: dict[int, Any] = {}
    level_results: dict[int, Mapping[str, Any]] = {}
    for level in LEVELS:
        adapter = make_predator_prey_sgqf_neutra_adapter(
            sparse_level=level, observations=observations
        )
        adapters[level] = adapter

        @tf.function(
            input_signature=[tf.TensorSpec([4, 6], tf.float64)],
            jit_compile=True,
        )
        def compiled_sgqf(theta):
            return adapter.neutra_batch_log_prob_and_grad_status(theta)

        compile_started = time.monotonic()
        posterior_value, posterior_score, status = compiled_sgqf(audit_points)
        _synchronize(tf, posterior_value, posterior_score, status)
        compile_seconds = time.monotonic() - compile_started
        runtime_started = time.monotonic()
        posterior_value, posterior_score, status = compiled_sgqf(audit_points)
        _synchronize(tf, posterior_value, posterior_score, status)
        runtime_seconds = time.monotonic() - runtime_started
        likelihood_value = posterior_value - prior_value - jacobian_value

        finite = tf.logical_and(
            tf.math.is_finite(likelihood_value),
            tf.reduce_all(tf.math.is_finite(posterior_score), axis=1),
        )
        status_pass = tf.logical_and(
            tf.equal(status["status_code"], 0),
            status["valid_pre_regularized_score"],
        )
        covariance_pass = tf.logical_and(
            status["min_predictive_eigenvalue"] > MINIMUM_VARIANCE,
            tf.logical_and(
                status["min_innovation_eigenvalue"] > MINIMUM_VARIANCE,
                status["min_filtered_eigenvalue"] > MINIMUM_VARIANCE,
            ),
        )
        no_floor = tf.equal(status["floor_count_value"], 0)
        engineering_by_point = tf.logical_and(
            finite, tf.logical_and(status_pass, tf.logical_and(covariance_pass, no_floor))
        )
        pf_rows = []
        for point_index, (value, point_row) in enumerate(
            zip(likelihood_value.numpy().tolist(), selected_rung["point_rows"], strict=True)
        ):
            summary = point_row["summary"]
            lower = float(summary["interval_lower"]) - PF_PRACTICAL_MARGIN
            upper = float(summary["interval_upper"]) + PF_PRACTICAL_MARGIN
            pf_rows.append(
                {
                    "point_index": point_index,
                    "sgqf_likelihood_value": float(value),
                    "expanded_pf1_interval_lower": lower,
                    "expanded_pf1_interval_upper": upper,
                    "passed": bool(lower <= float(value) <= upper),
                }
            )
        level_results[level] = {
            "level": level,
            "target_signature": stable_ssm_target_signature(adapter.contract),
            "adapter_signature": adapter.adapter_signature(),
            "point_count": int(adapter.nodes.shape[0]),
            "negative_weight_count": int(
                tf.math.count_nonzero(adapter.weights < 0.0).numpy()
            ),
            "likelihood_value": _json_ready(likelihood_value),
            "posterior_score": _json_ready(posterior_score),
            "status": _json_ready(status),
            "engineering_by_point": _json_ready(engineering_by_point),
            "engineering_passed": bool(tf.reduce_all(engineering_by_point).numpy()),
            "pf_rows": pf_rows,
            "pf_passed": all(row["passed"] for row in pf_rows),
            "xla_compile_and_first_execution_seconds": compile_seconds,
            "xla_warm_execution_seconds": runtime_seconds,
        }

    candidate_rows = []
    selected_level = None
    for level in CANDIDATE_LEVELS:
        next_level = level + 1
        value = tf.constant(level_results[level]["likelihood_value"], tf.float64)
        score = tf.constant(level_results[level]["posterior_score"], tf.float64)
        next_value = tf.constant(
            level_results[next_level]["likelihood_value"], tf.float64
        )
        next_score = tf.constant(level_results[next_level]["posterior_score"], tf.float64)
        reference_value = tf.constant(level_results[5]["likelihood_value"], tf.float64)
        reference_score = tf.constant(level_results[5]["posterior_score"], tf.float64)
        next_value_gap = tf.abs(value - next_value)
        reference_value_gap = tf.abs(value - reference_value)
        next_score_gap = tf.abs(score - next_score)
        reference_score_gap = tf.abs(score - reference_score)
        convergence_passed = bool(
            tf.reduce_all(next_value_gap <= VALUE_CONVERGENCE_LIMIT).numpy()
            and tf.reduce_all(reference_value_gap <= VALUE_CONVERGENCE_LIMIT).numpy()
            and tf.reduce_all(next_score_gap <= SCORE_CONVERGENCE_LIMIT).numpy()
            and tf.reduce_all(reference_score_gap <= SCORE_CONVERGENCE_LIMIT).numpy()
        )
        passed = bool(
            level_results[level]["engineering_passed"]
            and level_results[level]["pf_passed"]
            and convergence_passed
        )
        candidate_rows.append(
            {
                "level": level,
                "next_level": next_level,
                "engineering_passed": level_results[level]["engineering_passed"],
                "pf_passed": level_results[level]["pf_passed"],
                "maximum_next_value_gap": float(tf.reduce_max(next_value_gap).numpy()),
                "maximum_level5_value_gap": float(
                    tf.reduce_max(reference_value_gap).numpy()
                ),
                "maximum_next_score_component_gap": float(
                    tf.reduce_max(next_score_gap).numpy()
                ),
                "maximum_level5_score_component_gap": float(
                    tf.reduce_max(reference_score_gap).numpy()
                ),
                "value_convergence_limit": VALUE_CONVERGENCE_LIMIT,
                "score_convergence_limit": SCORE_CONVERGENCE_LIMIT,
                "convergence_passed": convergence_passed,
                "passed": passed,
            }
        )
        if passed and selected_level is None:
            selected_level = level

    identity_payload = None
    recomposition_payload = None
    if selected_level is not None:
        adapter = adapters[selected_level]
        target_signature = stable_ssm_target_signature(adapter.contract)
        repaired_registry = {
            "schema": "bayesfilter.multimodel_neutra_p4_target_repair_registry.v1",
            "program_id": PROGRAM_ID,
            "cell_id": CELL_ID,
            "previous_state": "TARGET_BLOCKED",
            "state": "VALUE_SCORE_ADMITTED",
            "target_signature": target_signature,
            "dataset_sha256": observation_hash,
            "selected_sparse_level": selected_level,
            "repair_evidence": str(output_root / "result.json"),
            "remaining_blockers": (),
        }
        registry_path = output_root / "repaired_registry.json"
        _write_new_json(registry_path, repaired_registry)
        registry_sha256 = hashlib.sha256(registry_path.read_bytes()).hexdigest()
        recomposer = PredatorPreySGQFLikelihoodRecomposer(adapter)
        recomposition = admit_independent_posterior_recomposition(
            adapter=adapter,
            points=audit_points,
            prior_value_score_fn=source_uniform_prior_value_score,
            likelihood_value_score_fn=recomposer.__call__,
            jacobian_value_score_fn=source_six_probit_jacobian_value_score,
            value_tolerance=1.0e-9,
            score_tolerance=1.0e-8,
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
            required_candidate_families=("plain_dense_iaf",),
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

    result = {
        "schema": "bayesfilter.multimodel_neutra_p4_predator_prey_sgqf.v1",
        "program_id": PROGRAM_ID,
        "cell_id": CELL_ID,
        "completed": True,
        "passed": selected_level is not None,
        "decision": (
            "ADMIT_PP_SGQF_POSTERIOR_IDENTITY"
            if selected_level is not None
            else "KEEP_PP_SGQF_TARGET_BLOCKED_LEVEL_GATE"
        ),
        "terminal_state": (
            "POSTERIOR_IDENTITY_ADMITTED"
            if selected_level is not None
            else "TARGET_BLOCKED_FILTER_GATE"
        ),
        "selected_level": selected_level,
        "dataset": {
            "seed": 81104,
            "horizon": 20,
            "state_sha256": state_hash,
            "observation_sha256": observation_hash,
            "time_order": "y0_initial_observation_then_transition_y1_onward",
        },
        "pf_reference": pf_reference_record,
        "audit_points": _json_ready(audit_points),
        "level_results": [level_results[level] for level in LEVELS],
        "candidate_rows": candidate_rows,
        "target_identity": identity_payload,
        "recomposition": recomposition_payload,
        "elapsed_seconds": time.monotonic() - started,
        "nonclaims": NONCLAIMS,
    }
    _write_new_json(output_root / "result.json", result)
    _write_new_json(
        output_root / "run_manifest.json",
        _run_manifest(
            output_root=output_root,
            started_at=started_at,
            tensorflow_version=tf.__version__,
            tfp_version=tfp.__version__,
            memory_policy=memory_policy,
            target_signature=(
                None if identity_payload is None else identity_payload["target_signature"]
            ),
            wall_time=time.monotonic() - started,
        ),
    )
    hashes = {
        str(path.relative_to(output_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _write_new_json(
        output_root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.multimodel_neutra_p4_sgqf_artifact_hashes.v1",
            "artifacts": hashes,
        },
    )
    return result


def _load_and_verify_pf_reference() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    result_path = PF_REFERENCE_ROOT / "result.json"
    hash_path = PF_REFERENCE_ROOT / "artifact_hashes.json"
    result_hash = hashlib.sha256(result_path.read_bytes()).hexdigest()
    if result_hash != PF_RESULT_SHA256:
        raise RuntimeError("frozen PF reference result hash mismatch")
    declared_hashes = json.loads(hash_path.read_text(encoding="utf-8"))["artifacts"]
    for relative_path, expected_hash in declared_hashes.items():
        actual_hash = hashlib.sha256(
            (PF_REFERENCE_ROOT / relative_path).read_bytes()
        ).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(f"PF reference artifact hash mismatch: {relative_path}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return payload, {
        "root": str(PF_REFERENCE_ROOT),
        "result_sha256": result_hash,
        "artifact_hashes_sha256": hashlib.sha256(hash_path.read_bytes()).hexdigest(),
        "selected_rung": payload["selected_rung"],
        "reference_stabilized": payload["reference_stabilized"],
        "role": "common_stochastic_value_diagnostic_only",
    }


def _synchronize(tf: Any, *values: Any) -> None:
    for value in values:
        if isinstance(value, Mapping):
            _synchronize(tf, *value.values())
        else:
            tf.convert_to_tensor(value).numpy()


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
        "schema": "bayesfilter.multimodel_neutra_p4_sgqf_run_manifest.v1",
        "program_id": PROGRAM_ID,
        "cell_id": CELL_ID,
        "git_commit": git_commit,
        "dirty_worktree_disclosure": "shared dirty worktree; scoped paths only",
        "command": (
            "TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/bayesfilter-mpl "
            "/home/chakwong/anaconda3/envs/tf-gpu/bin/python "
            "docs/benchmarks/run_multimodel_neutra_p4_predator_prey_sgqf_admission.py "
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
        "data_version": "zhao_cui_predator_prey_T20 seed 81104",
        "random_seeds": "N/A; deterministic fixed SGQF",
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
    result = run_campaign(args.output_root)
    print(
        json.dumps(
            {
                "completed": result["completed"],
                "passed": result["passed"],
                "decision": result["decision"],
                "selected_level": result["selected_level"],
                "output_root": str(args.output_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
