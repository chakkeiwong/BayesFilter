"""Run the P4 predator-prey PF reference ladder and PP-UKF admission gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
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
CELL_ID = "PP-UKF"
PLAN_PATH = Path(
    "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-"
    "p4-bootstrap-pf-reference-design-2026-07-15.md"
)
RUNGS = (
    {"id": "PF0", "particles": 4096, "seeds": 8, "offset": 0, "half_width": 0.5},
    {"id": "PF1", "particles": 16384, "seeds": 12, "offset": 100, "half_width": 0.5},
    {"id": "PF2", "particles": 65536, "seeds": 16, "offset": 200, "half_width": 0.35},
)
T_CRITICAL_95 = {7: 2.364624, 11: 2.200985, 15: 2.131450}
PRACTICAL_MARGIN = 1.0
MINIMUM_ESS = 16.0
NONCLAIMS = (
    "bootstrap-PF stochastic filter diagnostic, not exact likelihood",
    "PF supplies no score or HMC target authority",
    "PP-UKF target-admission gate only",
    "no HMC, NeuTra training, calibration, superiority, or readiness claim",
)


def build_audit_points(tf: Any) -> Any:
    from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
        PP_PARAMETER_LOWER,
        PP_PARAMETER_UPPER,
        PP_TRUTH_PHYSICAL,
    )

    probability = (PP_TRUTH_PHYSICAL - PP_PARAMETER_LOWER) / (
        PP_PARAMETER_UPPER - PP_PARAMETER_LOWER
    )
    truth = tf.sqrt(tf.constant(2.0, tf.float64)) * tf.math.erfinv(
        2.0 * probability - 1.0
    )
    return tf.stack(
        (
            truth,
            tf.zeros([6], tf.float64),
            tf.constant([-0.35, 0.20, -0.20, 0.25, -0.25, 0.20], tf.float64),
            tf.constant([0.30, -0.30, 0.25, -0.20, 0.20, -0.25], tf.float64),
        ),
        axis=0,
    )


def run_campaign(output_root: Path) -> Mapping[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"P4 output root must be fresh: {output_root}")
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

    from bayesfilter.inference.neutra_campaign import (
        CampaignCellLedger,
        admit_independent_posterior_recomposition,
        issue_typed_neutra_target_identity,
        require_typed_neutra_target,
    )
    from bayesfilter.ssm import stable_ssm_target_signature
    from bayesfilter.testing.predator_prey_bootstrap_pf_reference_tf import (
        predator_prey_bootstrap_pf_reference,
    )
    from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
        PP_OBSERVATION_SHA256,
        PP_STATE_SHA256,
        PredatorPreyUKFLikelihoodRecomposer,
        generate_frozen_predator_prey_dataset_tf,
        make_predator_prey_ukf_neutra_adapter,
        source_six_probit_jacobian_value_score,
        source_uniform_prior_value_score,
    )

    states, observations = generate_frozen_predator_prey_dataset_tf()
    state_hash = _tensor_hash(tf, states)
    observation_hash = _tensor_hash(tf, observations)
    if state_hash != PP_STATE_SHA256 or observation_hash != PP_OBSERVATION_SHA256:
        raise RuntimeError("frozen predator-prey dataset hash mismatch")
    audit_points = build_audit_points(tf)
    adapter = make_predator_prey_ukf_neutra_adapter(observations=observations)

    rung_results: list[Mapping[str, Any]] = []
    previous_summary = None
    reference_stabilized = False
    selected_rung = None
    for rung_index, rung in enumerate(RUNGS):
        if rung_index == 2 and reference_stabilized:
            break
        seeds = tf.constant(
            [81400 + int(rung["offset"]) + index for index in range(int(rung["seeds"]))],
            tf.int32,
        )

        @tf.function(
            input_signature=[tf.TensorSpec([1, 6], tf.float64)],
            jit_compile=True,
        )
        def compiled_pf(theta):
            result = predator_prey_bootstrap_pf_reference(
                theta,
                observations=observations,
                seeds=seeds,
                num_particles=int(rung["particles"]),
            )
            return (
                result["log_likelihood"],
                result["minimum_ess"],
                result["minimum_state"],
                result["finite"],
            )

        point_rows = []
        rung_started = time.monotonic()
        for point_index, point in enumerate(tf.unstack(audit_points)):
            point_started = time.monotonic()
            log_likelihood, minimum_ess, minimum_state, finite = compiled_pf(
                point[None, :]
            )
            samples = [float(item) for item in log_likelihood.numpy().tolist()]
            summary = _summary(samples, half_width_max=float(rung["half_width"]))
            point_rows.append(
                {
                    "point_index": point_index,
                    "source_point": [float(item) for item in point.numpy().tolist()],
                    "samples": samples,
                    "minimum_ess_by_seed": [
                        float(item) for item in minimum_ess.numpy().tolist()
                    ],
                    "minimum_state_by_seed": [
                        float(item) for item in minimum_state.numpy().tolist()
                    ],
                    "finite_by_seed": [bool(item) for item in finite.numpy().tolist()],
                    "summary": summary,
                    "elapsed_seconds": time.monotonic() - point_started,
                }
            )
        current_summary = tuple(row["summary"] for row in point_rows)
        stability_rows = []
        if previous_summary is not None:
            for point_index, (previous, current) in enumerate(
                zip(previous_summary, current_summary, strict=True)
            ):
                shift_limit = max(
                    0.5,
                    2.0
                    * math.sqrt(
                        float(previous["standard_error"]) ** 2
                        + float(current["standard_error"]) ** 2
                    ),
                )
                mean_shift = abs(float(current["mean"]) - float(previous["mean"]))
                passed = bool(
                    mean_shift <= shift_limit
                    and float(current["interval_half_width"])
                    <= float(rung["half_width"])
                    and min(point_rows[point_index]["minimum_ess_by_seed"])
                    >= MINIMUM_ESS
                    and all(point_rows[point_index]["finite_by_seed"])
                )
                stability_rows.append(
                    {
                        "point_index": point_index,
                        "mean_shift": mean_shift,
                        "mean_shift_limit": shift_limit,
                        "interval_half_width": current["interval_half_width"],
                        "interval_half_width_limit": rung["half_width"],
                        "minimum_ess": min(
                            point_rows[point_index]["minimum_ess_by_seed"]
                        ),
                        "passed": passed,
                    }
                )
            reference_stabilized = all(row["passed"] for row in stability_rows)
            if reference_stabilized:
                selected_rung = str(rung["id"])
        rung_results.append(
            {
                "rung_id": rung["id"],
                "particles": rung["particles"],
                "seeds": [int(item) for item in seeds.numpy().tolist()],
                "point_rows": point_rows,
                "stability_rows": stability_rows,
                "reference_stabilized": reference_stabilized,
                "elapsed_seconds": time.monotonic() - rung_started,
            }
        )
        previous_summary = current_summary

    selected = next(
        (row for row in rung_results if row["rung_id"] == selected_rung), None
    )
    filter_rows = []
    ukf_value = None
    ukf_score = None
    ukf_status = None
    filter_passed = False
    if selected is not None:
        @tf.function(
            input_signature=[tf.TensorSpec([4, 6], tf.float64)],
            jit_compile=True,
        )
        def compiled_ukf(theta):
            return adapter.neutra_batch_log_prob_and_grad_status(theta)

        posterior_value, ukf_score, ukf_status = compiled_ukf(audit_points)
        prior_value, _prior_score = source_uniform_prior_value_score(audit_points)
        jacobian_value, _jacobian_score = source_six_probit_jacobian_value_score(
            audit_points
        )
        ukf_value = posterior_value - prior_value - jacobian_value
        for point_index, (value, point_row) in enumerate(
            zip(ukf_value.numpy().tolist(), selected["point_rows"], strict=True)
        ):
            summary = point_row["summary"]
            lower = float(summary["interval_lower"]) - PRACTICAL_MARGIN
            upper = float(summary["interval_upper"]) + PRACTICAL_MARGIN
            passed = bool(lower <= float(value) <= upper)
            filter_rows.append(
                {
                    "point_index": point_index,
                    "ukf_log_likelihood": float(value),
                    "expanded_interval_lower": lower,
                    "expanded_interval_upper": upper,
                    "passed": passed,
                }
            )
        filter_passed = bool(
            all(row["passed"] for row in filter_rows)
            and tf.reduce_all(ukf_status["valid_pre_regularized_score"]).numpy()
        )

    identity_payload = None
    recomposition_payload = None
    if reference_stabilized and filter_passed:
        target_signature = stable_ssm_target_signature(adapter.contract)
        repaired_registry = {
            "schema": "bayesfilter.multimodel_neutra_p4_target_repair_registry.v1",
            "program_id": PROGRAM_ID,
            "cell_id": CELL_ID,
            "previous_state": "TARGET_BLOCKED",
            "state": "VALUE_SCORE_ADMITTED",
            "target_signature": target_signature,
            "dataset_sha256": observation_hash,
            "repair_evidence": str(output_root / "result.json"),
            "remaining_blockers": (),
        }
        registry_path = output_root / "repaired_registry.json"
        _write_new_json(registry_path, repaired_registry)
        registry_sha256 = hashlib.sha256(registry_path.read_bytes()).hexdigest()
        likelihood_recomposer = PredatorPreyUKFLikelihoodRecomposer(adapter)
        recomposition = admit_independent_posterior_recomposition(
            adapter=adapter,
            points=audit_points,
            prior_value_score_fn=source_uniform_prior_value_score,
            likelihood_value_score_fn=likelihood_recomposer.__call__,
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

    passed = identity_payload is not None
    result = {
        "schema": "bayesfilter.multimodel_neutra_p4_predator_prey_pf_ukf.v1",
        "program_id": PROGRAM_ID,
        "cell_id": CELL_ID,
        "passed": bool(passed),
        "decision": (
            "ADMIT_PP_UKF_POSTERIOR_IDENTITY"
            if passed
            else (
                "KEEP_PP_UKF_TARGET_BLOCKED_FILTER_GATE"
                if reference_stabilized
                else "KEEP_PP_UKF_TARGET_BLOCKED_REFERENCE_INCONCLUSIVE"
            )
        ),
        "dataset": {
            "seed": 81104,
            "horizon": 20,
            "state_sha256": state_hash,
            "observation_sha256": observation_hash,
            "time_order": "y0_initial_observation_then_transition_y1_onward",
        },
        "audit_points": _json_ready(audit_points),
        "pf_rungs": rung_results,
        "reference_stabilized": reference_stabilized,
        "selected_rung": selected_rung,
        "filter_rows": filter_rows,
        "filter_passed": filter_passed,
        "ukf_likelihood_value": _json_ready(ukf_value),
        "ukf_score": _json_ready(ukf_score),
        "ukf_status": _json_ready(ukf_status),
        "target_identity": identity_payload,
        "recomposition": recomposition_payload,
        "pp_zc_status": "TARGET_BLOCKED_SOURCE_ROUTE_MISMATCH",
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
            "schema": "bayesfilter.multimodel_neutra_p4_artifact_hashes.v1",
            "artifacts": hashes,
        },
    )
    return result


def _summary(samples: Sequence[float], *, half_width_max: float) -> Mapping[str, Any]:
    values = tuple(float(item) for item in samples)
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    standard_error = standard_deviation / math.sqrt(len(values))
    critical = T_CRITICAL_95[len(values) - 1]
    half_width = critical * standard_error
    return {
        "count": len(values),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "t_critical_95": critical,
        "interval_half_width": half_width,
        "interval_lower": mean - half_width,
        "interval_upper": mean + half_width,
        "half_width_limit": half_width_max,
    }


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
        "schema": "bayesfilter.multimodel_neutra_p4_run_manifest.v1",
        "program_id": PROGRAM_ID,
        "cell_id": CELL_ID,
        "git_commit": git_commit,
        "dirty_worktree_disclosure": "shared dirty worktree; scoped paths only",
        "command": (
            "TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/bayesfilter-mpl "
            "/home/chakwong/anaconda3/envs/tf-gpu/bin/python "
            "docs/benchmarks/run_multimodel_neutra_p4_predator_prey_pf_ukf_admission.py "
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
        "principal_sqrt_backend": "tensorflow_eigh_xla_portable",
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "data_version": "zhao_cui_predator_prey_T20 seed 81104",
        "random_seeds": "PF rung seed ledgers embedded in result",
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
                "passed": result["passed"],
                "decision": result["decision"],
                "selected_rung": result["selected_rung"],
                "output_root": str(args.output_root),
            },
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
