"""Run one P4 predator-prey same-target plain-HMC comparator campaign."""

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
PLAN_PATH = Path(
    "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-"
    "p4-plain-hmc-comparator-subplan-2026-07-15.md"
)
PHASE_ROOT = Path(
    "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p4"
)
IDENTITY_ROOTS = {
    "PP-UKF": PHASE_ROOT
    / "PP-UKF/pf-target-admission/attempt-03-20260715T121908Z",
    "PP-SGQF": PHASE_ROOT
    / "PP-SGQF/target-admission/attempt-01-20260715T123720Z",
}
EXPECTED_TYPED_SIGNATURES = {
    "PP-UKF": "036948f0faaf028d159d7b70337214f01514d732112c2d10e9f7eea1e13b8e30",
    "PP-SGQF": "8e0a9582fd30643b2e77e7615a21c0d44cc6c1827865ea52c841cc6dbfdde1ad",
}
STEP_SIZES = (0.001, 0.002, 0.004, 0.008, 0.016, 0.032)
NUM_LEAPFROG_STEPS = 8
PROBE_RESULTS = 128
PROBE_BURNIN = 64
PROBE_SEEDS = {"PP-UKF": (20260715, 8400), "PP-SGQF": (20260715, 8500)}
WARMUP_SEEDS = {"PP-UKF": (20260715, 8601), "PP-SGQF": (20260715, 8701)}
RETAINED_SEEDS = {"PP-UKF": (20260715, 8801), "PP-SGQF": (20260715, 8901)}
INITIAL_OFFSETS = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.10, -0.10, 0.08, -0.08, 0.06, -0.06),
    (-0.10, 0.10, -0.08, 0.08, -0.06, 0.06),
    (0.16, 0.08, -0.12, -0.10, 0.12, 0.04),
)
NONCLAIMS = (
    "same-target plain-HMC comparator for one frozen filter posterior only",
    "probe ESS nominates a kernel but does not establish convergence",
    "acceptance is explanatory only",
    "no NeuTra quality, filter exactness, superiority, calibration, or readiness claim",
)


class P4PlainHMCError(RuntimeError):
    """Raised when the frozen comparator contract cannot be replayed."""


def run_campaign(*, cell_id: str, output_root: Path) -> Mapping[str, Any]:
    cell = str(cell_id)
    if cell not in IDENTITY_ROOTS:
        raise P4PlainHMCError(f"unsupported P4 comparator cell: {cell}")
    if output_root.exists():
        raise FileExistsError(f"P4 comparator output root must be fresh: {output_root}")
    output_root.mkdir(parents=True)
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()

    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )
    from bayesfilter.runtime import atomic_write_json

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)

    import tensorflow_probability as tfp

    from bayesfilter.inference.hmc_convergence import (
        RankNormalizedHMCThresholds,
        rank_normalized_hmc_diagnostics,
    )
    from bayesfilter.inference.neutra_campaign import (
        CampaignCellLedger,
        SeparateCampaignArchive,
        admit_independent_posterior_recomposition,
        issue_typed_neutra_target_identity,
        require_typed_neutra_target,
        run_campaign_plain_hmc,
    )
    from bayesfilter.inference.neutra_hmc import (
        BatchedHMCConfig,
        SequentialNeuTraHMCConfig,
        run_sequential_neutra_hmc,
    )
    from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
        PP_PARAMETER_NAMES,
        generate_frozen_predator_prey_dataset_tf,
        source_six_probit_jacobian_value_score,
        source_uniform_prior_value_score,
    )

    source_root = IDENTITY_ROOTS[cell]
    source_reference = _verify_source_root(source_root)
    source_result = _read_mapping(source_root / "result.json")
    expected_identity = _read_mapping(source_root / "target_identity.json")
    states, observations = generate_frozen_predator_prey_dataset_tf()
    del states
    audit_points = tf.constant(source_result["audit_points"], tf.float64)
    adapter, likelihood_recomposer = _build_adapter_and_recomposer(
        cell=cell, observations=observations, source_result=source_result
    )
    repaired_registry = _read_mapping(source_root / "repaired_registry.json")
    registry_sha256 = _file_sha256(source_root / "repaired_registry.json")
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
        scope_id=cell,
        adapter=adapter,
        recomposition=recomposition,
        registry_row=repaired_registry,
        registry_artifact_sha256=registry_sha256,
    )
    require_typed_neutra_target(identity, adapter=adapter)
    if _json_ready(identity.payload()) != expected_identity:
        raise P4PlainHMCError("reconstructed typed identity payload drifted")
    if identity.target_signature != EXPECTED_TYPED_SIGNATURES[cell]:
        raise P4PlainHMCError("reconstructed typed target signature mismatch")

    initial_state = audit_points[0][None, :] + tf.constant(
        INITIAL_OFFSETS, tf.float64
    )
    probe_rows = []
    for grid_index, step_size in enumerate(STEP_SIZES):
        probe = run_campaign_plain_hmc(
            identity=identity,
            adapter=adapter,
            initial_state=initial_state,
            config=BatchedHMCConfig(
                num_results=PROBE_RESULTS,
                num_burnin_steps=PROBE_BURNIN,
                step_size=step_size,
                num_leapfrog_steps=NUM_LEAPFROG_STEPS,
                seed=(PROBE_SEEDS[cell][0], PROBE_SEEDS[cell][1] + grid_index),
            ),
        )
        diagnostics = rank_normalized_hmc_diagnostics(
            probe["samples"],
            parameter_names=PP_PARAMETER_NAMES,
            thresholds=RankNormalizedHMCThresholds(
                rhat_max=1.01, bulk_ess_min=1000.0, tail_ess_min=400.0
            ),
        )
        health = probe["diagnostics"]
        eligible = bool(
            health["health_passed"] is True
            and diagnostics["input_all_finite"] is True
            and diagnostics["diagnostics_all_finite"] is True
        )
        probe_rows.append(
            {
                "grid_index": grid_index,
                "step_size": step_size,
                "num_leapfrog_steps": NUM_LEAPFROG_STEPS,
                "trajectory_length": step_size * NUM_LEAPFROG_STEPS,
                "seed": (PROBE_SEEDS[cell][0], PROBE_SEEDS[cell][1] + grid_index),
                "eligible": eligible,
                "nomination_metric": "minimum_rank_normalized_bulk_ess",
                "minimum_bulk_ess": diagnostics.get("min_bulk_ess"),
                "minimum_tail_ess": diagnostics.get("min_tail_ess"),
                "maximum_modern_rhat": diagnostics.get("max_rhat"),
                "acceptance_rate": health["acceptance_rate"],
                "health": health,
                "short_chain_diagnostics": diagnostics,
            }
        )
        atomic_write_json(
            output_root / "probe_progress.json",
            {
                "schema": "bayesfilter.multimodel_neutra_p4_plain_hmc_probe_progress.v1",
                "cell_id": cell,
                "completed_probe_count": len(probe_rows),
                "total_probe_count": len(STEP_SIZES),
                "probe_rows": probe_rows,
                "scientific_role": "checkpoint_only_not_comparator_admission",
            },
        )
    selected = _select_probe(probe_rows)
    atomic_write_json(
        output_root / "tuning_selection.json",
        {
            "schema": "bayesfilter.multimodel_neutra_p4_plain_hmc_tuning_selection.v1",
            "cell_id": cell,
            "probe_rows": probe_rows,
            "selected_probe": selected,
            "selection_rule": "health_valid_then_maximum_minimum_rank_normalized_bulk_ess_grid_order_tie_break",
            "acceptance_role": "explanatory_only",
        },
    )

    sequential = None
    if selected is not None:
        archive = SeparateCampaignArchive(
            output_root=output_root / "samples", identity=identity, adapter=adapter
        )

        def archive_with_progress(**kwargs: Any) -> Mapping[str, Any]:
            payload = archive(**kwargs)
            atomic_write_json(
                output_root / "sequential_progress.json",
                {
                    "schema": "bayesfilter.multimodel_neutra_p4_plain_hmc_sequential_progress.v1",
                    "cell_id": cell,
                    "latest_stage": kwargs["stage"],
                    "latest_chunk_index": kwargs["chunk_index"],
                    "latest_seed": kwargs["seed"],
                    "latest_cumulative": kwargs["cumulative"],
                    "latest_archive": payload,
                    "scientific_role": "checkpoint_only_not_comparator_admission",
                },
            )
            return payload

        sequential = run_sequential_neutra_hmc(
            adapter=adapter,
            initial_state=initial_state,
            parameter_names=PP_PARAMETER_NAMES,
            config=SequentialNeuTraHMCConfig(
                step_size=float(selected["step_size"]),
                num_leapfrog_steps=NUM_LEAPFROG_STEPS,
                warmup_seed=WARMUP_SEEDS[cell],
                retained_seed=RETAINED_SEEDS[cell],
                warmup_chunk_results=1000,
                warmup_min_results=2000,
                warmup_check_window_results=1000,
                warmup_max_results=10000,
                warmup_rhat_max=1.05,
                retained_chunk_results=2000,
                retained_min_results=4000,
                retained_max_results=10000,
                retained_rhat_max=1.01,
            ),
            retained_diagnostic_fn=lambda draws: rank_normalized_hmc_diagnostics(
                draws,
                parameter_names=PP_PARAMETER_NAMES,
                thresholds=RankNormalizedHMCThresholds(
                    rhat_max=1.01, bulk_ess_min=1000.0, tail_ess_min=400.0
                ),
            ),
            archive_callback=archive_with_progress,
        )

    passed = bool(sequential is not None and sequential["passed"] is True)
    public_sequential = (
        None
        if sequential is None
        else {key: value for key, value in sequential.items() if not key.startswith("private_")}
    )
    posterior_summary = (
        _posterior_summary(tf, sequential["private_retained_raw"], PP_PARAMETER_NAMES)
        if sequential is not None and sequential["retained_results_per_chain"] > 0
        else None
    )
    ledger = CampaignCellLedger(
        {
            "cells": [
                {
                    "cell_id": cell,
                    "state": "POSTERIOR_IDENTITY_ADMITTED",
                    "target_signature": identity.mathematical_target_signature,
                }
            ]
        },
        required_candidate_families=("plain_dense_iaf",),
        event_path=output_root / "cell_events.jsonl",
    )
    ledger.transition(
        cell_id=cell,
        new_state="COMPARATOR_ADMITTED" if passed else "COMPARATOR_BLOCKED",
        evidence_path=str(output_root / "result.json"),
        target_identity=identity if passed else None,
    )
    _write_new_json(output_root / "cell_ledger.json", ledger.payload())

    result = {
        "schema": "bayesfilter.multimodel_neutra_p4_predator_prey_plain_hmc.v1",
        "program_id": PROGRAM_ID,
        "cell_id": cell,
        "completed": True,
        "passed": passed,
        "decision": (
            f"ADMIT_{cell.replace('-', '_')}_PLAIN_HMC_COMPARATOR"
            if passed
            else f"BLOCK_{cell.replace('-', '_')}_PLAIN_HMC_COMPARATOR"
        ),
        "terminal_state": "COMPARATOR_ADMITTED" if passed else "COMPARATOR_BLOCKED",
        "source_identity": source_reference,
        "target_identity": identity.payload(),
        "initial_state": _json_ready(initial_state),
        "kernel_grid": {
            "step_sizes": STEP_SIZES,
            "num_leapfrog_steps": NUM_LEAPFROG_STEPS,
            "probe_results": PROBE_RESULTS,
            "probe_burnin": PROBE_BURNIN,
            "selection_rule": "health_valid_then_maximum_minimum_rank_normalized_bulk_ess_grid_order_tie_break",
            "acceptance_role": "explanatory_only",
        },
        "probe_rows": probe_rows,
        "selected_probe": selected,
        "sequential_run": public_sequential,
        "posterior_summary": posterior_summary,
        "elapsed_seconds": time.monotonic() - started,
        "nonclaims": NONCLAIMS,
    }
    _write_new_json(output_root / "result.json", result)
    _write_new_json(
        output_root / "run_manifest.json",
        _run_manifest(
            cell=cell,
            output_root=output_root,
            started_at=started_at,
            tensorflow_version=tf.__version__,
            tfp_version=tfp.__version__,
            memory_policy=memory_policy,
            target_signature=identity.target_signature,
            wall_time=time.monotonic() - started,
        ),
    )
    hashes = {
        str(path.relative_to(output_root)): _file_sha256(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _write_new_json(
        output_root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.multimodel_neutra_p4_plain_hmc_artifact_hashes.v1",
            "artifacts": hashes,
        },
    )
    return result


def _build_adapter_and_recomposer(
    *, cell: str, observations: Any, source_result: Mapping[str, Any]
) -> tuple[Any, Any]:
    if cell == "PP-UKF":
        from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
            PredatorPreyUKFLikelihoodRecomposer,
            make_predator_prey_ukf_neutra_adapter,
        )

        adapter = make_predator_prey_ukf_neutra_adapter(observations=observations)
        return adapter, PredatorPreyUKFLikelihoodRecomposer(adapter)
    from bayesfilter.testing.predator_prey_sgqf_neutra_target_tf import (
        PredatorPreySGQFLikelihoodRecomposer,
        make_predator_prey_sgqf_neutra_adapter,
    )

    selected_level = source_result.get("selected_level")
    if selected_level != 2:
        raise P4PlainHMCError("PP-SGQF frozen selected level is not 2")
    adapter = make_predator_prey_sgqf_neutra_adapter(
        sparse_level=selected_level, observations=observations
    )
    return adapter, PredatorPreySGQFLikelihoodRecomposer(adapter)


def _select_probe(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    eligible = [row for row in rows if row["eligible"] is True]
    if not eligible:
        return None
    return dict(
        max(
            eligible,
            key=lambda row: (
                float(row["minimum_bulk_ess"]), -int(row["grid_index"])
            ),
        )
    )


def _posterior_summary(
    tf: Any, samples: Any, parameter_names: Sequence[str]
) -> Mapping[str, Any]:
    import tensorflow_probability as tfp

    values = tf.convert_to_tensor(samples, tf.float64)
    pooled = tf.reshape(values, (-1, int(values.shape[-1])))
    quantiles = tfp.stats.percentile(
        pooled, (5.0, 50.0, 95.0), axis=0, interpolation="linear"
    )
    return {
        "parameter_names": tuple(parameter_names),
        "draws_per_chain": int(values.shape[0]),
        "chain_count": int(values.shape[1]),
        "all_finite": bool(tf.reduce_all(tf.math.is_finite(values)).numpy()),
        "mean": _json_ready(tf.reduce_mean(pooled, axis=0)),
        "sd": _json_ready(tf.math.reduce_std(pooled, axis=0)),
        "q05": _json_ready(quantiles[0]),
        "q50": _json_ready(quantiles[1]),
        "q95": _json_ready(quantiles[2]),
        "role": "descriptive_same_target_comparator_summary_only",
    }


def _verify_source_root(root: Path) -> Mapping[str, Any]:
    hash_path = root / "artifact_hashes.json"
    declared = _read_mapping(hash_path)["artifacts"]
    for relative_path, expected_hash in declared.items():
        actual_hash = _file_sha256(root / relative_path)
        if actual_hash != expected_hash:
            raise P4PlainHMCError(f"source artifact hash mismatch: {relative_path}")
    return {
        "root": str(root),
        "artifact_hashes_sha256": _file_sha256(hash_path),
        "target_identity_sha256": _file_sha256(root / "target_identity.json"),
        "result_sha256": _file_sha256(root / "result.json"),
    }


def _run_manifest(
    *,
    cell: str,
    output_root: Path,
    started_at: datetime,
    tensorflow_version: str,
    tfp_version: str,
    memory_policy: Mapping[str, Any],
    target_signature: str,
    wall_time: float,
) -> Mapping[str, Any]:
    git_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "schema": "bayesfilter.multimodel_neutra_p4_plain_hmc_run_manifest.v1",
        "program_id": PROGRAM_ID,
        "cell_id": cell,
        "git_commit": git_commit,
        "dirty_worktree_disclosure": "shared dirty worktree; scoped paths only",
        "command": (
            "TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/bayesfilter-mpl "
            "/home/chakwong/anaconda3/envs/tf-gpu/bin/python "
            "docs/benchmarks/run_multimodel_neutra_p4_predator_prey_plain_hmc.py "
            f"--cell {cell} --output-root {output_root}"
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
        "random_seeds": {
            "probe_root": PROBE_SEEDS[cell],
            "warmup_root": WARMUP_SEEDS[cell],
            "retained_root": RETAINED_SEEDS[cell],
        },
        "target_signature": target_signature,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_time_seconds": float(wall_time),
        "output_root": str(output_root),
        "plan_file": str(PLAN_PATH),
        "result_file": str(output_root / "result.json"),
        "nonclaims": NONCLAIMS,
    }


def _read_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise P4PlainHMCError(f"artifact is not a mapping: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    parser.add_argument("--cell", choices=tuple(IDENTITY_ROOTS), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_campaign(cell_id=args.cell, output_root=args.output_root)
    print(
        json.dumps(
            {
                "cell_id": result["cell_id"],
                "completed": result["completed"],
                "passed": result["passed"],
                "decision": result["decision"],
                "selected_step_size": (
                    None
                    if result["selected_probe"] is None
                    else result["selected_probe"]["step_size"]
                ),
                "output_root": str(args.output_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
