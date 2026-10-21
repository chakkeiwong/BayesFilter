"""Diagnostic non-harm evaluation of weighted-moment multiplication order."""
import json
from pathlib import Path
import time
from types import SimpleNamespace

import tensorflow as tf

from bayesfilter.highdim.higher_moment_contract_e import (
    _weighted_diag_moments_jvp, _weighted_pair_moments_jvp,
)
from bayesfilter.highdim.ledh_range_bearing_target_tf import make_range_bearing_target
from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash, write_json
from bayesfilter.testing.surrogate_hmc_ar2_controls import describe, finite_json
from bayesfilter.testing.surrogate_hmc_m13_controls import partition_fixture
from bayesfilter.testing.surrogate_hmc_m13_correction_localize import SAVED, verify_saved_binding

PLAN = "docs/plans/surrogate-hmc-m13-weighted-moment-repair-2026-09-24.md"
BASELINE = ROOT/"docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-controls-10/m13-correction-localize/fp32_tf32.json"
REPAIRED_BASELINE = ROOT/"docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-controls-11/m13-moment-repair/repaired-fp32.json"
CHANGED_SOURCE = "bayesfilter/highdim/higher_moment_contract_e.py"


def bind_repaired_source(target, saved, *, localization=False):
    old, new = saved["source_sha256"], target.manifest["source_sha256"]
    changed = {key for key in set(old) | set(new) if old.get(key) != new.get(key)}
    expected = {CHANGED_SOURCE}
    if localization:
        expected.add("bayesfilter/highdim/ledh_unified_correction_tf.py")
    if changed != expected:
        raise ValueError("undeclared source change: "+str(sorted(changed)))
    adjusted = dict(target.manifest) | {"source_sha256": old}
    verify_saved_binding(SimpleNamespace(manifest=adjusted), saved)
    return {key: dict(before=old[key], after=new[key]) for key in sorted(changed)}


def moment_primitive(z, w, dz, dw):
    d = z.shape[1]
    zero = tf.zeros([d], z.dtype)
    diagonal = _weighted_diag_moments_jvp(z, w, dz, dw, zero,
        tf.zeros([d, 1], z.dtype), tf.eye(d, dtype=z.dtype), tf.zeros([d, d, 1], z.dtype))
    return diagonal + _weighted_pair_moments_jvp(z, w, dz, dw)


def direct_reference(z, w, dz, dw):
    """Independent FP64 polynomial formulas; diagnostic reference only."""
    z, w, dz, dw = [tf.cast(x, tf.float64) for x in (z, w, dz, dw)]
    cubes, fourths = z**3, z**4
    diagonal = (tf.reduce_sum(w[:, None]*cubes, 0), tf.reduce_sum(w[:, None]*fourths, 0),
        tf.reduce_sum(dw[:, None, :]*cubes[:, :, None]+3*w[:, None, None]*z[:, :, None]**2*dz, 0),
        tf.reduce_sum(dw[:, None, :]*fourths[:, :, None]+4*w[:, None, None]*z[:, :, None]**3*dz, 0))
    pair3 = z[:, :, None]**2*z[:, None, :]
    pair4 = z[:, :, None]**2*z[:, None, :]**2
    dpair3 = (2*z[:, :, None, None]*dz[:, :, None, :]*z[:, None, :, None]
              +z[:, :, None, None]**2*dz[:, None, :, :])
    dpair4 = (2*z[:, :, None, None]*dz[:, :, None, :]*z[:, None, :, None]**2
              +2*z[:, :, None, None]**2*z[:, None, :, None]*dz[:, None, :, :])
    mask = 1-tf.eye(z.shape[1], dtype=tf.float64)
    pair = (tf.reduce_sum(w[:, None, None]*pair3, 0)*mask,
        tf.reduce_sum(w[:, None, None]*pair4, 0)*mask,
        tf.reduce_sum(dw[:, None, None, :]*pair3[:, :, :, None]+w[:, None, None, None]*dpair3, 0)*mask[:, :, None],
        tf.reduce_sum(dw[:, None, None, :]*pair4[:, :, :, None]+w[:, None, None, None]*dpair4, 0)*mask[:, :, None])
    return diagonal+pair


def primitive_inputs(extreme=True, zero_weight=False):
    z = tf.constant([[1e10, -2e10], [.3, .5], [-.4, .2]] if extreme else
                    [[1., -2.], [.3, .5], [-.4, .2]], tf.float32)
    w = tf.constant([0. if zero_weight else 1e-20 if extreme else .2, .4,
                     .6 if extreme or zero_weight else .4], tf.float32)
    dz = z[:, :, None]*.01
    dw = tf.constant([[1e-22], [-1e-22], [0.]], tf.float32)
    return z, w, dz, dw


def relative_error(actual, reference):
    return float(tf.reduce_max(tf.abs(tf.cast(actual, tf.float64)-reference)/(1+tf.abs(reference))))


def repair_stage(output, *, localization=False):
    out = Path(output).parent
    out.mkdir(parents=True, exist_ok=True)
    if not tf.config.experimental.tensor_float_32_execution_enabled():
        raise ValueError("FP32/TF32 default required")
    saved = json.loads((SAVED/"arm-0.json").read_text())
    data = json.loads((SAVED/"data.json").read_text())
    if (data["attempt"], data["partition"], data["design_partition"]) != (3, "calibration", 6):
        raise ValueError("wrong saved partition")
    fixture = partition_fixture("calibration", attempt=3)
    if fixture.model_definition() != data["model"] or fixture.observations.numpy().tolist() != data["observations"]:
        raise ValueError("saved calibration data changed")
    target = make_range_bearing_target(fixture, particle_count=256, controls=saved["controls"],
        dtype=tf.float32, design_partition=6)
    changes = bind_repaired_source(target, saved["target"], localization=localization)
    write_json(out/"saved-binding.json", dict(source_changes=changes, target=dict(target.manifest),
        saved_inputs_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in
                            (SAVED/"arm-0.json", SAVED/"data.json", BASELINE)}))
    if localization:
        path = BASELINE.parents[1].parent/"surrogate-hmc-closure-20260922-m13-controls-11/m13-moment-repair/primitive.json"
        primitive = json.loads(path.read_text())
        if not all(row["passed"] for row in primitive.values()):
            raise ValueError("prior primitive parity missing")
        write_json(out/"primitive-reuse.json", dict(path=str(path), sha256=file_hash(path)))
    else:
        kernel = tf.function(moment_primitive, input_signature=[tf.TensorSpec([3, 2], tf.float32),
            tf.TensorSpec([3], tf.float32), tf.TensorSpec([3, 2, 1], tf.float32),
            tf.TensorSpec([3, 1], tf.float32)], jit_compile=True, autograph=False)
        primitive = {}
        for name, extreme, zero in (("healthy", False, False), ("extreme", True, False),
                                   ("zero_weight_extreme", True, True)):
            inputs = primitive_inputs(extreme, zero)
            result = kernel(*inputs)
            with tf.device("/CPU:0"):
                reference = direct_reference(*inputs)
            errors = [relative_error(x, y) for x, y in zip(result, reference)]
            finite = all(bool(tf.reduce_all(tf.math.is_finite(x))) for x in result)
            primitive[name] = dict(finite=finite, errors=errors, device=result[0].device,
                passed=finite and max(errors) <= 2e-4 and "GPU" in result[0].device)
        write_json(out/"primitive.json", finite_json(primitive))
        if not all(p["passed"] for p in primitive.values()):
            return dict(status="WEIGHTED_MOMENT_PRIMITIVE_REPAIR_FAILED", qualified=False,
                primitive=finite_json(primitive), inference_launched=False, all_models_validated=False)
    evaluate = target.compile(9, filter_moments=True, all_correction_diagnostics=True)
    start = time.monotonic()
    result = evaluate(tf.constant(saved["points"]["nominal"], tf.float32))
    result[0].numpy()
    description = describe(result)
    if localization:
        required = ["target_covariance_valid", "initial_standardization_valid", "diagonal_iterations_valid",
                    "pairwise_iterations_valid", "affine_restoration_value_valid", "output_standardization_valid"]
        if any("higher_moment_"+key not in description["status"] for key in required):
            raise ValueError("missing correction substep diagnostic")
        write_json(out/"substeps.json", {key: description["status"]["higher_moment_"+key] for key in required})
    write_json(out/"repaired-fp32.json", finite_json(dict(target=dict(target.manifest),
        diagnostic=description, wall_seconds=time.monotonic()-start, device=result[0].device)))
    if not description["all_failures_flagged"] or "GPU" not in result[0].device:
        raise ValueError("unflagged numerical failure or wrong device")
    comparison_path = REPAIRED_BASELINE if localization else BASELINE
    before = json.loads(comparison_path.read_text())["diagnostic"]
    comparisons = {}
    for key, left, right in [("value", description["values"], before["values"]),
                             ("score", description["scores"], before["scores"])]+[
            (k, description["moments"][k], v) for k, v in before["moments"].items()]:
        # Invalid baseline rows are excluded before conversion: JSON nulls are not numbers.
        selected = [i for i, ok in enumerate(before["healthy_rows"]) if ok]
        x = tf.constant([left[i] for i in selected], tf.float64)
        y = tf.constant([right[i] for i in selected], tf.float64)
        comparisons[key] = relative_error(x, y)
    healthy_preserved = all(not old or new for old, new in zip(before["healthy_rows"], description["healthy_rows"]))
    return finite_json(dict(status="M13_CORRECTION_SUBSTEPS_LOCALIZED" if localization else "M13_WEIGHTED_MOMENT_REPAIR_ASSESSED", qualified=False,
        diagnostic_completed=True, source_changes=changes, primitive=primitive,
        comparison_baseline=dict(path=str(comparison_path), sha256=file_hash(comparison_path)),
        baseline_healthy_rows=before["healthy_rows"], repaired_healthy_rows=description["healthy_rows"],
        repair_passed=description["all_healthy"], healthy_preserved=healthy_preserved,
        healthy_row_changes=comparisons, nonharm_screen_passed=healthy_preserved and max(comparisons.values()) <= 2e-4,
        first_failure_time=description["status"]["first_failure_time"],
        first_failure_stage=description["status"]["first_failure_stage"],
        continuation_veto=False, endpoint_qualified=False, posterior_reference_qualified=False,
        inference_launched=False, all_models_validated=False,
        next_action="Existing master: inspect remaining typed failure or proceed to fresh scoped controls; endpoint reference remains unqualified."))
