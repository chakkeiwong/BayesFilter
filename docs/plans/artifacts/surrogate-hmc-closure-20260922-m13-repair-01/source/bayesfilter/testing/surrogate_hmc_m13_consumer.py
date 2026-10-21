"""Bounded M13 canonical-likelihood engineering and numerical diagnostics.

No HMC, prior selection, scope-tuning artifact, or posterior claim is made here.
"""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time

import tensorflow as tf

from bayesfilter.highdim.ledh_range_bearing_target_tf import make_range_bearing_target
from bayesfilter.inference.surrogate_hmc_program import ROOT, write_json
from bayesfilter.testing.surrogate_hmc_lg1_tf import warm_start_controls
from experiments.dpf_implementation.tf_tfp.fixtures.range_bearing_tf import build_range_bearing_fixture_tf

REFERENCE = ROOT/"docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-reference-03/m13-reference-trajectory/report.json"


def serial(value):
    if tf.is_tensor(value):
        return serial(value.numpy().tolist())
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def consumer_stage(output):
    started = time.monotonic()
    reference = json.loads(REFERENCE.read_text())
    if not reference["likelihood_reference_qualified"]:
        raise ValueError("qualified independent likelihood reference required")
    fixture = build_range_bearing_fixture_tf()
    for key in ("A", "Q", "R", "m0", "P0"):
        if serial(getattr(fixture, key)) != reference["scope"]["fixture"][key]:
            raise ValueError("M13 fixture law differs from the qualified reference")
    # Consume the saved observations exactly, avoiding accidental data regeneration.
    fixture = replace(fixture, observations=tf.constant(reference["scope"]["observations"],tf.float64),
                      observation_checksum=reference["scope"]["fixture"]["observation_checksum"])
    report = dict(status="M13_CONSUMER_RUNNING", qualified=False,
        plan="docs/plans/surrogate-hmc-m13-consumer-2026-09-23.md",
        quantity="finite_canonical_log_likelihood_only", prior=None, inference_launched=False,
        control_status="diagnostic_warm_start_not_scope_tuned", posterior_qualified=False,
        all_models_validated=False, reference_path=str(REFERENCE.relative_to(ROOT)),
        reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        tests={}, next_action="continue existing-master M13 scoped control and posterior prerequisites",
        nonclaims=["no posterior reference", "no admitted HMC force or tuning artifact",
                   "no speed ranking", "fixed-point likelihood errors are explanatory only"])
    output = Path(output)
    write_json(output, report)
    small = replace(fixture, observations=fixture.observations[:2], states=fixture.states[:2])
    target = make_range_bearing_target(small, particle_count=16, controls=warm_start_controls(), dtype=tf.float64)
    evaluate = target.compile(2)
    theta = tf.constant([[0.,0.],[.2,-.2]],tf.float64)
    begin = time.monotonic()
    value, score, status = evaluate(theta)
    valid = bool(tf.reduce_all(status["program_valid"]))
    small_result = dict(manifest=dict(target.manifest), identity=target.identity,
        value=serial(value), score=serial(score), status=serial(status),
        compilation_and_first_call_seconds=time.monotonic()-begin, kernel_device=value.device)
    report["tests"]["small_total_derivative"] = small_result
    if valid:
        h = tf.constant(1e-5,tf.float64)
        fd, fd_valid = [], True
        for direction in tf.unstack(tf.eye(2,dtype=tf.float64)):
            plus, _, sp = evaluate(theta+h*direction)
            minus, _, sm = evaluate(theta-h*direction)
            fd.append((plus-minus)/(2*h))
            fd_valid &= bool(tf.reduce_all(sp["program_valid"] & sm["program_valid"]))
        fd = tf.stack(fd,-1)
        scaled = tf.abs(score-fd)/(1+tf.abs(fd))
        again, score_again, status_again = evaluate(theta)
        deterministic = bool(tf.reduce_all(again == value) & tf.reduce_all(score_again == score)
                             & tf.reduce_all(status_again["program_valid"] == status["program_valid"]))
        derivative_pass = fd_valid and bool(tf.reduce_all(scaled <= 3e-5))
        small_result.update(finite_difference=serial(fd),maximum_scaled_error=float(tf.reduce_max(scaled)),
            finite_difference_step=float(h),scaled_error_threshold=3e-5,
            finite_difference_valid=fd_valid,deterministic=deterministic,derivative_pass=derivative_pass)
    else:
        derivative_pass,deterministic = False,False
        small_result.update(derivative_pass=False,finite_difference_valid=False,deterministic=False)
    write_json(output,report)
    if not (valid and derivative_pass and deterministic):
        report.update(status=("M13_CONSUMER_TOTAL_DERIVATIVE_MISMATCH" if valid and not derivative_pass
                              else "M13_WARM_START_NUMERICAL_INVALID"),wall_seconds=time.monotonic()-started)
        return report

    full = make_range_bearing_target(fixture,particle_count=64,controls=warm_start_controls(),dtype=tf.float32)
    full_evaluate = full.compile(9)
    points = tf.constant([r["theta"] for r in reference["numerical_rows"]],tf.float32)
    begin = time.monotonic()
    values,scores,status = full_evaluate(points)
    first_wall = time.monotonic()-begin
    begin = time.monotonic()
    repeat,repeat_score,repeat_status = full_evaluate(points)
    steady_wall = time.monotonic()-begin
    valid_rows = serial(status["program_valid"])
    rows = []
    for i, ref in enumerate(reference["numerical_rows"]):
        interval = ref["particle_levels"]["65536"]["full"]
        val = float(values[i])
        delta = val-interval["log_mean"] if valid_rows[i] else None
        rows.append(dict(index=i,theta=ref["theta"],program_valid=valid_rows[i],
            log_likelihood=serial(val),score=serial(scores[i]),reference_log_mean=interval["log_mean"],
            reference_log_interval=[interval["log_lower"],interval["log_upper"]],
            log_likelihood_error=delta,error_role="explanatory_only",posterior_assessment=None))
    nominal_error = rows[4]["log_likelihood_error"]
    for row in rows:
        delta = row["log_likelihood_error"]
        row["error_centered_at_nominal"] = None if delta is None or nominal_error is None else delta-nominal_error
    # Compare only finite rows for bitwise repeatability; identical failure masks
    # remain necessary. Numerical failure is not silently counted as deterministic success.
    same = (tf.reduce_all(repeat_status["program_valid"] == status["program_valid"])
            & tf.reduce_all(tf.where(status["program_valid"],repeat == values,True))
            & tf.reduce_all(tf.where(status["program_valid"][:,None],repeat_score == scores,True)))
    full_valid = bool(tf.reduce_all(status["program_valid"]))
    report["tests"]["full_horizon"] = dict(manifest=dict(full.manifest),identity=full.identity,
        compilation_and_first_call_seconds=first_wall,steady_call_seconds=steady_wall,
        kernel_device=values.device,status=serial(status),rows=rows,deterministic_on_valid_rows=bool(same))
    report.update(status=("M13_CONSUMER_MECHANICS_VERIFIED_SCOPE_TUNING_REQUIRED" if full_valid and bool(same)
                          else "M13_FULL_HORIZON_WARM_START_REQUIRES_REPAIR"),
        qualified=full_valid and bool(same),wall_seconds=time.monotonic()-started,
        engineering_derivative_parity=True,scope_controls_qualified=False)
    try:
        report["allocator"]=tf.config.experimental.get_memory_info("GPU:0")
    except ValueError:
        report["allocator"]=None
    return report

LOCALIZATION_ARMS = ((64, "float64"), (256, "float32"), (1024, "float32"))


def _covariance_summary(covariances):
    finite = tf.reduce_all(tf.math.is_finite(covariances),[-2,-1])
    safe = tf.where(finite[...,None,None],covariances,tf.eye(4,dtype=covariances.dtype))
    eigenvalues = tf.linalg.eigvalsh((safe+tf.linalg.matrix_transpose(safe))*.5)
    eigenvalues = tf.where(finite[...,None],eigenvalues,tf.constant(float("nan"),eigenvalues.dtype))
    return dict(finite=serial(finite),minimum_eigenvalue=serial(tf.reduce_min(eigenvalues,-1)),
                maximum_eigenvalue=serial(tf.reduce_max(eigenvalues,-1)))


def localization_stage(output):
    started = time.monotonic()
    reference = json.loads(REFERENCE.read_text())
    if not reference["likelihood_reference_qualified"]:
        raise ValueError("qualified independent likelihood reference required")
    fixture = build_range_bearing_fixture_tf()
    for key in ("A","Q","R","m0","P0"):
        if serial(getattr(fixture,key)) != reference["scope"]["fixture"][key]:
            raise ValueError("M13 fixture law differs from the qualified reference")
    fixture = replace(fixture,observations=tf.constant(reference["scope"]["observations"],tf.float64),
                      observation_checksum=reference["scope"]["fixture"]["observation_checksum"])
    report = dict(status="M13_CONSUMER_LOCALIZATION_RUNNING",qualified=False,
        plan="docs/plans/surrogate-hmc-m13-consumer-2026-09-23.md",
        quantity="finite_canonical_log_likelihood_only",prior=None,inference_launched=False,
        control_status="diagnostic_warm_start_not_scope_tuned",posterior_qualified=False,
        all_models_validated=False,reference_path=str(REFERENCE.relative_to(ROOT)),
        reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),arms=[],
        next_action="continue existing-master M13 scope-specific repair based on validity diagnostics",
        nonclaims=["no numerical default promotion","no tuning artifact","no posterior assessment",
                   "likelihood differences and timings are descriptive only"])
    write_json(Path(output),report)
    for n,dtype_name in LOCALIZATION_ARMS:
        dtype = tf.as_dtype(dtype_name)
        target = make_range_bearing_target(fixture,particle_count=n,controls=warm_start_controls(),dtype=dtype)
        evaluate = target.compile(9,filter_moments=True)
        points = tf.constant([r["theta"] for r in reference["numerical_rows"]],dtype)
        begin = time.monotonic()
        values,scores,status = evaluate(points)
        values_list=serial(values)
        rows=[]
        for i,ref in enumerate(reference["numerical_rows"]):
            valid=bool(status["program_valid"][i])
            interval=ref["particle_levels"]["65536"]["full"]
            rows.append(dict(index=i,theta=ref["theta"],program_valid=valid,log_likelihood=values_list[i],
                score=serial(scores[i]),reference_log_mean=interval["log_mean"],
                reference_log_interval=[interval["log_lower"],interval["log_upper"]],
                log_likelihood_error=float(values[i])-interval["log_mean"] if valid else None))
        arm=dict(particle_count=n,dtype=dtype_name,manifest=dict(target.manifest),identity=target.identity,
            kernel_device=values.device,wall_seconds=time.monotonic()-begin,status=serial(status),rows=rows,
            all_points_valid=bool(tf.reduce_all(status["program_valid"])),
            filtering_covariance_summary=_covariance_summary(status["filtering_covariance"]),
            reset_covariance_summary=_covariance_summary(status["post_reset_covariance"]))
        report["arms"].append(arm)
        report["wall_seconds"]=time.monotonic()-started
        write_json(Path(output),report)
    report.update(status="M13_CONSUMER_LOCALIZATION_COMPLETE",
        qualified=any(arm["all_points_valid"] for arm in report["arms"]),
        scope_controls_qualified=False,wall_seconds=time.monotonic()-started,
        interpretation="viability screen only; no ranking or posterior promotion")
    report["allocator"]=tf.config.experimental.get_memory_info("GPU:0")
    return report

PREVIOUS_LOCALIZATION = ROOT/"docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-consumer-03/m13-consumer-localize/report.json"


def full_parity_stage(output):
    started = time.monotonic()
    reference = json.loads(REFERENCE.read_text())
    previous = json.loads(PREVIOUS_LOCALIZATION.read_text())
    if not reference["likelihood_reference_qualified"] or previous["status"] != "M13_CONSUMER_LOCALIZATION_COMPLETE":
        raise ValueError("completed reference and localization required")
    fixture = build_range_bearing_fixture_tf()
    for key in ("A","Q","R","m0","P0"):
        if serial(getattr(fixture,key)) != reference["scope"]["fixture"][key]:
            raise ValueError("M13 fixture law differs from the qualified reference")
    fixture = replace(fixture,observations=tf.constant(reference["scope"]["observations"],tf.float64),
                      observation_checksum=reference["scope"]["fixture"]["observation_checksum"])
    report = dict(status="M13_FULL_PARITY_RUNNING",qualified=False,
        plan="docs/plans/surrogate-hmc-m13-consumer-2026-09-23.md",
        prior=None,inference_launched=False,scope_controls_qualified=False,posterior_qualified=False,
        all_models_validated=False,parameter_points=reference["scope"]["parameter_grid"],
        reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        previous_localization_sha256=hashlib.sha256(PREVIOUS_LOCALIZATION.read_bytes()).hexdigest(),
        checks={},finite_difference_checks=[],
        nonclaims=["parity tests finite-program derivatives, not likelihood accuracy",
                   "no posterior or numerical-default promotion"],
        next_action="continue existing-master M13 scoped controls after numerical qualification")
    write_json(Path(output),report)
    points = tf.constant([r["theta"] for r in reference["numerical_rows"]],tf.float64)
    target = make_range_bearing_target(fixture,particle_count=256,controls=warm_start_controls(),dtype=tf.float64)
    plain = target.compile(9)
    traced = target.compile(9,filter_moments=True,all_correction_diagnostics=True)
    value,score,status = plain(points)
    trace_value,trace_score,trace_status = traced(points)
    scaled_value = tf.abs(value-trace_value)/(1+tf.abs(value))
    scaled_score = tf.abs(score-trace_score)/(1+tf.abs(score))
    trace_parity = bool(tf.reduce_all(scaled_value <= 3e-8) & tf.reduce_all(scaled_score <= 3e-8)
        & tf.reduce_all(status["program_valid"] == trace_status["program_valid"]))
    report.update(manifest=dict(target.manifest),identity=target.identity,value=serial(value),score=serial(score),
        status_tensors=serial(trace_status),kernel_device=value.device,
        trace_value_max_scaled_error=float(tf.reduce_max(scaled_value)),
        trace_score_max_scaled_error=float(tf.reduce_max(scaled_score)))
    report["checks"].update(all_fp64_points_valid=bool(tf.reduce_all(status["program_valid"])),
        trace_value_score_parity=trace_parity,
        correction_condition_diagnostic_present="higher_moment_maximum_diagonal_scaled_system_condition" in trace_status,
        correction_feasibility_diagnostic_present="higher_moment_minimum_pearson_feasibility_margin" in trace_status)
    write_json(Path(output),report)
    for step in (1e-4,1e-5):
        h=tf.constant(step,tf.float64)
        derivatives,all_valid=[],True
        for direction in tf.unstack(tf.eye(2,dtype=tf.float64)):
            plus,_,sp=plain(points+h*direction)
            minus,_,sm=plain(points-h*direction)
            all_valid &= bool(tf.reduce_all(sp["program_valid"] & sm["program_valid"]))
            derivatives.append((plus-minus)/(2*h))
        fd=tf.stack(derivatives,-1)
        error=tf.abs(score-fd)/(1+tf.abs(fd))
        report["finite_difference_checks"].append(dict(step=step,all_points_valid=all_valid,
            finite_difference=serial(fd),scaled_error=serial(error),
            maximum_scaled_error=serial(float(tf.reduce_max(error))),
            parity_pass=all_valid and bool(tf.reduce_all(error <= 3e-5))))
        write_json(Path(output),report)
    repeat,repeat_score,repeat_status=plain(points)
    report["checks"]["deterministic"]=bool(tf.reduce_all(repeat==value) & tf.reduce_all(repeat_score==score)
        & tf.reduce_all(repeat_status["program_valid"]==status["program_valid"]))
    report["checks"]["full_total_derivative_parity"]=report["finite_difference_checks"][-1]["parity_pass"]

    f32=make_range_bearing_target(fixture,particle_count=256,controls=warm_start_controls(),dtype=tf.float32)
    f32value,f32score,f32status=f32.compile(9,filter_moments=True)(tf.cast(points,tf.float32))
    previous_arm=next(a for a in previous["arms"] if a["particle_count"]==256 and a["dtype"]=="float32")
    old_value=tf.constant([r["log_likelihood"] for r in previous_arm["rows"]],tf.float32)
    old_score=tf.constant([r["score"] for r in previous_arm["rows"]],tf.float32)
    scaled_v=tf.abs(f32value-old_value)/(1+tf.abs(old_value))
    scaled_s=tf.abs(f32score-old_score)/(1+tf.abs(old_score))
    no_change=bool(tf.reduce_all(f32status["program_valid"]) & tf.reduce_all(scaled_v<=3e-5)
                   & tf.reduce_all(scaled_s<=3e-5))
    report["checks"]["fp32_observability_no_harm"]=no_change
    report["fp32_replay"]=dict(manifest=dict(f32.manifest),value=serial(f32value),score=serial(f32score),
        status_tensors=serial(f32status),value_max_scaled_error=float(tf.reduce_max(scaled_v)),
        score_max_scaled_error=float(tf.reduce_max(scaled_s)))
    qualified=all(report["checks"].values())
    report.update(qualified=qualified,status=("M13_FULL_CONSUMER_PARITY_VERIFIED_TUNING_PENDING" if qualified
        else "M13_FULL_CONSUMER_PARITY_REQUIRES_LOCALIZATION"),wall_seconds=time.monotonic()-started,
        allocator=tf.config.experimental.get_memory_info("GPU:0"))
    return report
