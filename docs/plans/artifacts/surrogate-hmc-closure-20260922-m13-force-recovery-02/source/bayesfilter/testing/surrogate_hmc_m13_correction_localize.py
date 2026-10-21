"""Exact-input arithmetic diagnostic for saved M13 correction failures.

This module issues no tuning or posterior authority and changes no default.
All arms call the same canonical analytical batch implementation.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from types import MappingProxyType

import tensorflow as tf

from bayesfilter.highdim.ledh_range_bearing_target_tf import (
    make_range_bearing_model, make_range_bearing_target,
)
from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash, write_json
from bayesfilter.testing.surrogate_hmc_ar2_controls import describe, finite_json
from bayesfilter.testing.surrogate_hmc_m13_controls import (
    PARTICLE_COUNT, control_points, fixed_protection_controls, m13_scope, partition_fixture,
)

SAVED = ROOT/"docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-controls-07/m13-controls-bounded/calibration"
PLAN = "docs/plans/surrogate-hmc-m13-correction-localize-2026-09-24.md"


def verify_saved_binding(target, saved):
    """A saved-run tolerance cannot excuse comparing different inputs."""
    for key in ("fixture", "horizon", "particle_count", "dtype", "tf32", "jit_compile",
                "settings", "design_partition", "seeds", "input_sha256", "source_sha256"):
        if json.loads(json.dumps(target.manifest[key])) != saved[key]:
            raise ValueError("saved M13 binding mismatch: "+key)


def arithmetic_target(base, transition_matrix, *, dtype, tf32):
    """Lift already-realized FP32 tensors, including all model coefficients."""
    dtype = tf.as_dtype(dtype)
    if base.inputs["initial_states"].dtype != tf.float32:
        raise ValueError("the diagnostic baseline must be FP32")
    if dtype not in (tf.float32, tf.float64) or (dtype == tf.float64 and tf32):
        raise ValueError("unsupported diagnostic arithmetic")
    inputs = {key: tf.cast(value, dtype) for key, value in base.inputs.items()}
    settings = dict(base.controls)
    settings["reset_design"] = tf.cast(settings["reset_design"], dtype)
    a32 = tf.cast(transition_matrix, tf.float32)
    model = (base.model if dtype == tf.float32 else make_range_bearing_model(
        tf.cast(a32, dtype), tf.cast(base.model.process_covariance, dtype),
        tf.cast(base.model.observation_covariance, dtype), dtype=dtype))
    tensors = inputs | {"reset_design": settings["reset_design"],
                        "transition_matrix": tf.cast(a32, dtype),
                        "process_covariance": model.process_covariance,
                        "observation_covariance": model.observation_covariance}
    manifest = dict(base.manifest) | dict(dtype=dtype.name, tf32=tf32,
        diagnostic_role="exact_lift_of_saved_FP32_inputs", base_identity=base.identity,
        input_sha256={key: hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()
                      for key, value in tensors.items()},
        diagnostic_source_sha256={str(Path(__file__).relative_to(ROOT)): file_hash(Path(__file__))})
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return replace(base, model=model, inputs=MappingProxyType(inputs),
                   controls=MappingProxyType(settings), manifest=MappingProxyType(manifest), identity=identity)


def correction_localization_stage(output):
    out = Path(output).parent
    out.mkdir(parents=True, exist_ok=True)
    saved = json.loads((SAVED/"arm-0.json").read_text())
    data = json.loads((SAVED/"data.json").read_text())
    if (data["attempt"], data["partition"], data["design_partition"]) != (3, "calibration", 6):
        raise ValueError("wrong saved calibration partition")
    if data["scope"] != m13_scope(bounded=True) or saved["controls"] != fixed_protection_controls(1.):
        raise ValueError("saved control scope changed")
    fixture = partition_fixture("calibration", attempt=data["attempt"])
    if (fixture.model_definition() != data["model"] or
            fixture.observations.numpy().tolist() != data["observations"] or
            fixture.states.numpy().tolist() != data["states"]):
        raise ValueError("saved calibration data changed")
    if not tf.config.experimental.tensor_float_32_execution_enabled():
        raise ValueError("start in the declared FP32/TF32 default")
    base = make_range_bearing_target(fixture, particle_count=PARTICLE_COUNT,
        controls=saved["controls"], dtype=tf.float32, design_partition=data["design_partition"])
    verify_saved_binding(base, saved["target"])
    points = tf.constant(saved["points"]["nominal"], tf.float32)
    tf.debugging.assert_equal(points, control_points()["nominal"])
    binding = dict(saved_root=str(SAVED.relative_to(ROOT)),
        saved_sha256={name: file_hash(SAVED/name) for name in ("arm-0.json", "data.json")},
        source_closure_verified=True, realized_inputs_verified=True, scope=data["scope"],
        base_target=dict(base.manifest), points=points.numpy().tolist(),
        saved_healthy_rows=saved["situations"]["nominal"]["healthy_rows"])
    write_json(out/"saved-binding.json", binding)
    arms = {}
    original_tf32 = tf.config.experimental.tensor_float_32_execution_enabled()
    try:
        for name, dtype, tf32 in (("fp32_tf32", tf.float32, True),
                                 ("fp32_no_tf32", tf.float32, False),
                                 ("fp64_lift", tf.float64, False)):
            tf.config.experimental.enable_tensor_float_32_execution(tf32)
            target = arithmetic_target(base, fixture.A, dtype=dtype, tf32=tf32)
            evaluate = target.compile(9, filter_moments=True, all_correction_diagnostics=True)
            begin = time.monotonic()
            result = evaluate(tf.cast(points, dtype))
            result[0].numpy()
            elapsed = time.monotonic()-begin
            description = describe(result)
            missing = {"higher_moment_target_skew", "higher_moment_target_kurtosis",
                       "higher_moment_value_valid", "filtering_ess"}-set(description["status"])
            if missing:
                raise ValueError("missing correction diagnostics: "+str(sorted(missing)))
            nonfinite = {key: int(tf.reduce_sum(tf.cast(~tf.math.is_finite(value), tf.int32)))
                         for key, value in result[2].items() if value.dtype.is_floating}
            write_json(out/(name+".json"), finite_json(dict(target=dict(target.manifest),
                wall_seconds=elapsed, device=result[0].device, diagnostic=description,
                nonfinite_diagnostic_counts=nonfinite)))
            if "GPU" not in result[0].device:
                raise ValueError("correction diagnostic must run on GPU")
            if not description["all_failures_flagged"]:
                raise ValueError("unflagged nonfinite correction output")
            arms[name] = {key: description[key] for key in
                          ("healthy_rows", "all_healthy", "all_failures_flagged", "values", "scores")}
            arms[name].update({key: description["status"][key] for key in
                              ("first_failure_time", "first_failure_stage", "value_failure_code")})
            print(name, description["healthy_rows"], elapsed, flush=True)
    finally:
        tf.config.experimental.enable_tensor_float_32_execution(original_tf32)
    return finite_json(dict(status="M13_CORRECTION_PRECISION_DIAGNOSTIC_COMPLETED",
        qualified=False, diagnostic_completed=True, plan=PLAN, arms=arms,
        saved_validity_reproduced=arms["fp32_tf32"]["healthy_rows"] == binding["saved_healthy_rows"],
        requires_repair=not arms["fp32_tf32"]["all_healthy"], continuation_veto=False,
        posterior_reference_qualified=False, endpoint_qualified=False, inference_launched=False,
        default_changed=False, all_models_validated=False,
        next_action="Existing master: localize the arithmetic or geometric cause using saved correction diagnostics before the smallest repair."))
