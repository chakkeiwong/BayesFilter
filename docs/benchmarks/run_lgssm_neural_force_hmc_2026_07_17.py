#!/usr/bin/env python3
"""Run the reviewed exact-likelihood LGSSM corrected neural-force HMC pilot."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = "docs/plans/bayesfilter-hnn-surrogate-hmc-p3-lgssm-pilot-subplan-2026-07-17.md"
TRANSPORT = ROOT / (
    "docs/plans/artifacts/neutra-batch-native-training-2026-07-14/"
    "long-training-attempt-01/phase4/training_jobs/dense_seed1201/"
    "attempt_1_graph_native/training/frozen_transport.json"
)
SOURCE_ROOT = ROOT / (
    "docs/plans/artifacts/lgssm-neutra-gap-closure-2026-07-15/"
    "sequential-repair-attempt-01/confirmation-attempt-01/dense_seed1201/samples"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"output root must be fresh: {output_root}")
    output_root.mkdir(parents=True)
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)

    from bayesfilter.inference.neural_force_hmc import FrozenPositionOnlyForce
    from bayesfilter.testing import lgssm_neural_force_hmc_pilot_tf as pilot

    context = pilot.load_pilot_context(transport_payload_path=TRANSPORT)
    supervision = pilot.prepare_supervision(
        context=context,
        warmup_archive_sidecar=SOURCE_ROOT / "warmup/all_z.tftensor.json",
        retained_archive_sidecar=SOURCE_ROOT / "retained/all_z.tftensor.json",
        train_rows=128 if args.smoke else 2048,
        heldout_rows=64 if args.smoke else 1024,
    )
    if args.smoke:
        from bayesfilter.inference.neural_force_hmc import (
            FrozenPositionOnlyForce,
            NeuralForceHMCConfig,
            sample_neural_force_hmc,
        )

        target = context["target_binding"].hmc_target()
        initial = _initial_positions(tf, supervision["heldout"].positions)
        potential = target.function(initial)
        config = NeuralForceHMCConfig(
            0.4, 2, (1.0,) * 18, dtype="float64"
        )
        force = FrozenPositionOnlyForce(
            lambda position: position, "lgssm-smoke-zero-residual"
        )

        @tf.function(jit_compile=True, reduce_retracing=True)
        def smoke_chain(position, current, seed):
            return sample_neural_force_hmc(
                position,
                current,
                force,
                target,
                config,
                num_warmup=0,
                num_results=4,
                seed=seed,
            )

        chain = smoke_chain(initial, potential, tf.constant((20260717, 53999), tf.int32))
        full_energy_error = tf.reduce_max(
            tf.abs(
                chain.delta_h
                - (
                    chain.final_potential
                    + chain.final_kinetic
                    - chain.initial_potential
                    - chain.initial_kinetic
                )
            )
        )
        result = {
            "schema": "bayesfilter.lgssm_neural_force_hmc_p3_smoke.v1",
            "passed": True,
            "value_only_endpoint_parity": supervision["parity"],
            "source_shapes": supervision["source_shapes"],
            "memory_policy": memory_policy,
            "model_specific_corrected_kernel_xla": {
                "passed": bool(
                    tf.reduce_all(chain.finite_status).numpy()
                    and (full_energy_error <= tf.constant(1.0e-12, tf.float64)).numpy()
                ),
                "transition_count_per_chain": 4,
                "chain_count": 4,
                "full_energy_identity_max_error": float(full_energy_error.numpy()),
                "acceptance_rate": float(
                    tf.reduce_mean(tf.cast(chain.accepted, tf.float64)).numpy()
                ),
            },
            "nonclaims": ("identity and value-only parity smoke only",),
        }
        _write_json(output_root / "result.json", result)
        print(json.dumps({"passed": True, "smoke": True}))
        return 0

    training_started = time.monotonic()
    training = pilot.train_recipe_grid(
        supervision=supervision,
        output_root=output_root / "training",
    )
    training_seconds = time.monotonic() - training_started
    target = context["target_binding"].hmc_target()
    transform = context["loaded_transport"].transport
    initial_position = _initial_positions(tf, supervision["heldout"].positions)

    def true_force(position):
        _potential, force_value = context["target_binding"].potential_and_force(position)
        return force_value

    arms = {
        "zero_residual": FrozenPositionOnlyForce(
            lambda position: position, "lgssm-zero-residual-gaussian-force"
        ),
        "learned_residual": training["final"].frozen.hmc_force(),
        "true_gradient": FrozenPositionOnlyForce(
            true_force, "lgssm-complete-transformed-true-gradient"
        ),
    }
    tuning_results = {}
    run_results = {}
    for index, (arm_id, force) in enumerate(arms.items()):
        tuning_results[arm_id] = pilot.tune_force(
            force=force,
            target=target,
            initial_position=initial_position,
            transform=transform,
            step_sizes=(0.4, 0.6, 0.8, 1.0),
            leapfrog_steps=(6, 10),
            seed_offset=53000 + index * 100,
        )
        selected = tuning_results[arm_id]["selected"]
        run_results[arm_id] = pilot.run_sequential_arm(
            arm_id=arm_id,
            force=force,
            target=target,
            initial_position=initial_position,
            transform=transform,
            parameter_names=context["bundle"].parameter_names,
            step_size=selected.step_size,
            num_leapfrog_steps=selected.num_leapfrog_steps,
            output_root=output_root / "sampling",
            seed_base=54000 + index * 1000,
        )

    learned = run_results["learned_residual"]
    learned_samples = learned["private_retained_raw"]
    posterior = (
        pilot.posterior_summary(
            candidate_samples=learned_samples,
            parameter_names=context["bundle"].parameter_names,
            comparator=context["plain_hmc_comparator"],
        )
        if learned["passed"]
        else None
    )
    truth_tail = (
        pilot.truth_tail_summary(
            learned_samples,
            context["bundle"].raw_truth,
            context["bundle"].parameter_names,
        )
        if learned["passed"]
        else None
    )
    validity = bool(
        learned["passed"]
        and posterior is not None
        and posterior["posterior_agreement_passed"]
        and posterior["recovery_passed"]
        and truth_tail is not None
        and truth_tail["passed"]
    )
    performance = _performance(training_seconds, tuning_results, run_results)
    decision = (
        "HNN_VALIDITY_CONFIRMED_ONE_EXACT_LGSSM_FIXTURE"
        if validity
        else "LGSSM_NEURAL_FORCE_CANDIDATE_NOT_CONFIRMED"
    )
    result = {
        "schema": "bayesfilter.lgssm_neural_force_hmc_p3_result.v1",
        "passed": validity,
        "decision": decision,
        "performance_decision": performance["decision"],
        "target_signature": pilot.EXPECTED_TARGET_SIGNATURE,
        "transport_signature": pilot.EXPECTED_TRANSPORT_SIGNATURE,
        "value_only_endpoint_parity": supervision["parity"],
        "training": {
            "screen_recipes": training["recipes"],
            "selected_recipe": training["selected_recipe"],
            "final_result_path": str(training["final"].result_path),
            "final_artifact_path": str(training["final"].artifact_path),
            "final_metrics": training["final"].metrics,
            "shell_tail": training["shell_tail"],
            "elapsed_seconds": training_seconds,
        },
        "tuning": {
            arm: {
                "selected": {
                    "candidate_id": value["selected"].candidate_id,
                    "step_size": value["selected"].step_size,
                    "num_leapfrog_steps": value["selected"].num_leapfrog_steps,
                },
                "rows": value["rows"],
            }
            for arm, value in tuning_results.items()
        },
        "arms": {arm: pilot.json_ready(value) for arm, value in run_results.items()},
        "posterior_summary": posterior,
        "truth_tail": truth_tail,
        "performance": performance,
        "run_manifest": _manifest(
            output_root=output_root,
            started_at=started_at,
            elapsed=time.monotonic() - started,
            memory_policy=memory_policy,
        ),
        "decision_table": {
            "primary_criterion": "corrected learned-force arm passes convergence, plain-HMC agreement, recovery, and no severe truth tail",
            "primary_status": validity,
            "hard_vetoes": learned["hard_vetoes"],
            "main_uncertainty": "one favorably truth-centered exact-likelihood LGSSM fixture",
            "next_action": "continue Tier A nonlinear cells" if validity else "classify failure and repair only the failed boundary",
            "not_concluded": "no statistical superiority, broad validity, or default readiness",
        },
        "inference_status": {
            "hard_veto_screen": "clear" if validity else "failed_or_inconclusive",
            "statistically_supported_ranking": False,
            "descriptive_only_differences": "all runtime and ESS differences",
            "default_readiness": False,
            "next_evidence": "nonlinear Tier A cells under the same prospective gates",
        },
        "nonclaims": (
            "one exact-likelihood LGSSM fixture only",
            "runtime differences are descriptive only",
            "no broad HNN-HMC validity or superiority claim",
            "no default readiness claim",
        ),
    }
    _write_json(output_root / "result.json", result)
    _write_json(output_root / "run_manifest.json", result["run_manifest"])
    _write_markdown(output_root / "result.md", result)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "decision": decision,
                "performance_decision": result["performance_decision"],
                "result": str(output_root / "result.json"),
            },
            sort_keys=True,
        )
    )
    return 0 if validity else 1


def _initial_positions(tf: Any, heldout: Any) -> Any:
    positions = tf.convert_to_tensor(heldout, tf.float64)
    indices = tf.constant((0, 17, 33, 49), tf.int32)
    return tf.gather(positions, indices)


def _minimum_bulk_ess(run: Mapping[str, Any]) -> float | None:
    checks = run.get("retained_checks")
    if not checks:
        return None
    diagnostic = checks[-1]["full_convergence"]
    values = [float(row["bulk_ess"]) for row in diagnostic["parameter_diagnostics"]]
    return min(values)


def _performance(training_seconds, tuning, runs):
    rows = {}
    for arm, run in runs.items():
        minimum = _minimum_bulk_ess(run)
        tuning_seconds = sum(float(row["elapsed_seconds"]) for row in tuning[arm]["rows"])
        total = float(run["elapsed_seconds"]) + tuning_seconds
        if arm == "learned_residual":
            total += float(training_seconds)
        rows[arm] = {
            "valid": bool(run["passed"]),
            "training_seconds": float(training_seconds) if arm == "learned_residual" else 0.0,
            "tuning_seconds": tuning_seconds,
            "sampling_seconds": float(run["elapsed_seconds"]),
            "reuse_scenario_seconds": total,
            "minimum_bulk_ess": minimum,
            "sampling_seconds_per_minimum_bulk_ess": (
                None if not minimum else float(run["elapsed_seconds"]) / minimum
            ),
            "reuse_seconds_per_minimum_bulk_ess": None if not minimum else total / minimum,
            "endpoint_scalar_values": run["endpoint_scalar_values"],
            "force_batch_invocations": run["force_batch_invocations"],
        }
    learned = rows["learned_residual"]
    true_gradient = rows["true_gradient"]
    passed = bool(
        learned["valid"]
        and true_gradient["valid"]
        and learned["reuse_seconds_per_minimum_bulk_ess"]
        <= true_gradient["reuse_seconds_per_minimum_bulk_ess"]
        and learned["sampling_seconds_per_minimum_bulk_ess"]
        < true_gradient["sampling_seconds_per_minimum_bulk_ess"]
    )
    return {
        "decision": (
            "DESCRIPTIVE_PERFORMANCE_SCREEN_PASS"
            if passed
            else "PERFORMANCE_NOT_DEMONSTRATED"
        ),
        "rows": rows,
        "ranking_statistically_supported": False,
        "differences_are_descriptive_only": True,
    }


def _manifest(*, output_root, started_at, elapsed, memory_policy):
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    import tensorflow as tf
    import tensorflow_probability as tfp

    return {
        "schema": "bayesfilter.lgssm_neural_force_hmc_p3_manifest.v1",
        "git_commit": commit,
        "command": sys.argv,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "tf-gpu"),
        "tensorflow_version": tf.__version__,
        "tensorflow_probability_version": tfp.__version__,
        "cpu_gpu_status": "trusted RTX 4080 SUPER GPU/XLA; memory growth; TF32 enabled",
        "gpu_memory_policy": memory_policy,
        "data_version": "registered fixture T120 seed [20260709,301] and preserved target-matched NeuTra chart coordinates",
        "seeds": {
            "training_screen": [20260717, 51000],
            "fresh_training": [20260717, 52000],
            "tuning": [20260717, 53000],
            "warmup": [20260717, 54000],
            "retained": [20260717, 54100],
        },
        "started_at_utc": started_at.isoformat(),
        "wall_time_seconds": elapsed,
        "output_artifact_paths": [str(output_root / "result.json"), str(output_root / "result.md")],
        "plan_file": PLAN,
        "result_file": "docs/plans/bayesfilter-hnn-surrogate-hmc-p3-lgssm-pilot-result-2026-07-17.md",
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
    }


def _json_ready(value):
    from bayesfilter.testing.lgssm_neural_force_hmc_pilot_tf import json_ready
    return json_ready(value)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, sort_keys=True, indent=2)
        handle.write("\n")


def _write_markdown(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# LGSSM Corrected Neural-Force HMC P3 Result",
        "",
        f"Decision: `{result['decision']}`",
        "",
        f"Performance: `{result['performance_decision']}`",
        "",
        f"Value-only endpoint maximum parity error: `{result['value_only_endpoint_parity']['maximum_absolute_error']:.6g}`.",
        "",
        "| Arm | Passed | Acceptance | Warm-up | Retained | Seconds |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, arm in result["arms"].items():
        lines.append(
            f"| `{name}` | `{arm['passed']}` | {arm['acceptance_rate']:.6g} | "
            f"{arm['warmup_results_per_chain']} | {arm['retained_results_per_chain']} | "
            f"{arm['elapsed_seconds']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
