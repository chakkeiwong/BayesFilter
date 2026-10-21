"""Diagnostic particle-resolution ladder for the existing M13 force target."""
import json
from pathlib import Path
import time

import tensorflow as tf

from bayesfilter.highdim.ledh_range_bearing_target_tf import make_range_bearing_target
from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash, write_json
from bayesfilter.testing.surrogate_hmc_ar2_controls import describe, finite_json
from bayesfilter.testing.surrogate_hmc_m13_controls import control_points, partition_fixture
from bayesfilter.testing.surrogate_hmc_m13_correction_localize import SAVED, verify_saved_binding

PLAN = "docs/plans/surrogate-hmc-m13-particle-recovery-2026-09-24.md"
BASELINE = ROOT/"docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-controls-12/m13-moment-localize/repaired-fp32.json"
COUNTS = (1024, 4096)


def verify_resolution_binding(current, saved, count):
    current = json.loads(json.dumps(dict(current)))
    for key in saved:
        if key not in {"particle_count", "chunk_policy", "input_sha256"} and current[key] != saved[key]:
            raise ValueError("particle resolution changed unrelated field: "+key)
    if current["particle_count"] != count:
        raise ValueError("wrong particle resolution")
    if current["input_sha256"]["observations"] != saved["input_sha256"]["observations"]:
        raise ValueError("particle resolution changed observations")


def bounded_ladder(evaluate):
    arms = []
    for count in COUNTS:
        arm = evaluate(count)
        arms.append(arm)
        if not arm["all_failures_flagged"]:
            raise ValueError("unflagged numerical failure")
        if arm["all_healthy"]:
            return arms, count
    return arms, None


def particle_recovery_stage(output):
    out = Path(output).parent
    out.mkdir(parents=True, exist_ok=True)
    if not tf.config.experimental.tensor_float_32_execution_enabled():
        raise ValueError("FP32/TF32 default required")
    saved = json.loads((SAVED/"arm-0.json").read_text())
    data = json.loads((SAVED/"data.json").read_text())
    baseline = json.loads(BASELINE.read_text())
    fixture = partition_fixture("calibration", attempt=3)
    if (data["attempt"], data["partition"], data["design_partition"]) != (3, "calibration", 6):
        raise ValueError("wrong saved calibration partition")
    if (fixture.model_definition() != data["model"] or
            fixture.observations.numpy().tolist() != data["observations"] or
            fixture.states.numpy().tolist() != data["states"]):
        raise ValueError("saved fixture changed")
    base = make_range_bearing_target(fixture, particle_count=256, controls=saved["controls"],
        dtype=tf.float32, design_partition=6)
    verify_saved_binding(base, baseline["target"])
    points = tf.constant(saved["points"]["nominal"], tf.float32)
    tf.debugging.assert_equal(points, control_points()["nominal"])
    write_json(out/"binding.json", dict(baseline=str(BASELINE), baseline_sha256=file_hash(BASELINE),
        baseline_attempt_status="budget_terminated_after_report_saved", data=str(SAVED/"data.json"),
        data_sha256=file_hash(SAVED/"data.json"), points=points.numpy().tolist(),
        base_target=dict(base.manifest), plan=PLAN))

    def evaluate(count):
        start = time.monotonic()
        target = make_range_bearing_target(fixture, particle_count=count, controls=saved["controls"],
            dtype=tf.float32, design_partition=6)
        verify_resolution_binding(target.manifest, baseline["target"], count)
        prefix_equal = bool(tf.reduce_all(target.inputs["initial_states"][:256] == base.inputs["initial_states"]))
        kernel = target.compile(9, filter_moments=True, all_correction_diagnostics=True)
        result = kernel(points)
        result[0].numpy()
        diagnostic = describe(result)
        if "GPU" not in result[0].device:
            raise ValueError("wrong numerical device")
        status = diagnostic["status"]
        required = ("filtering_ess", "higher_moment_affine_restoration_value_valid",
                    "higher_moment_output_standardization_valid")
        if any(key not in status for key in required):
            raise ValueError("missing correction diagnostic")
        path = out/f"n{count}.json"
        write_json(path, finite_json(dict(target=dict(target.manifest), diagnostic=diagnostic,
            wall_seconds=time.monotonic()-start, initial_design_prefix_equal=prefix_equal,
            device=result[0].device)))
        summary = dict(particle_count=count, all_healthy=diagnostic["all_healthy"],
            all_failures_flagged=diagnostic["all_failures_flagged"],
            healthy_rows=diagnostic["healthy_rows"], first_failure_time=status["first_failure_time"],
            first_failure_stage=status["first_failure_stage"], filtering_ess_time0=[x[0] for x in status["filtering_ess"]],
            path=str(path), sha256=file_hash(path), initial_design_prefix_equal=prefix_equal)
        write_json(out/"progress.json", summary)
        print(json.dumps(summary), flush=True)
        return summary

    arms, nominated = bounded_ladder(evaluate)
    return dict(status="M13_PARTICLE_RESOLUTION_NOMINATED" if nominated else "M13_PARTICLE_RESOLUTION_REPAIR_REQUIRED",
        qualified=False, diagnostic_completed=True, nominated_particle_count=nominated, arms=arms,
        posterior_qualified=False, endpoint_qualified=False, controls_selected=False,
        heuristic_dominance="NOT_EVALUATED", statistical_ranking="NONE",
        continuation_veto=False, inference_launched=False, all_models_validated=False,
        next_action="Existing master: fresh scope-specific tuning for a nominated resolution; otherwise inspect proposal concentration. Independent endpoint work remains.")
