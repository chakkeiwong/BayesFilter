#!/usr/bin/env python3
"""Resumable SIR NeuTra training and same-endpoint surrogate-HMC campaign.

The supervisor is standard-library only. Numerical workers use the shared
TensorFlow/XLA authorities; the HMC consumer delegates to the existing public
sequential controller. Loss nominates a map, never certifies posterior quality.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import threading
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAN = "docs/plans/sir-neutra-training-execution-2026-10-21.md"
OUTPUT = ROOT / "docs/plans/artifacts/sir-neutra-training-20261022-neutra-only01"
GPU = "GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a"
ASSESSMENT_POLICY = "sir_neutra_rhat_convergence_precision_diagnostic_v2"
SOURCE_FILES = (
    "scripts/run_sir_neutra_training.py",
    "bayesfilter/testing/surrogate_hmc_sir_neutra.py",
    "bayesfilter/testing/surrogate_hmc_five_model.py",
    "bayesfilter/testing/surrogate_hmc_scope_tuning.py",
    "bayesfilter/testing/surrogate_hmc_literature_priors.py",
    "bayesfilter/highdim/covariance_proposal_score_batch_tf.py",
    "bayesfilter/highdim/covariance_proposal_native_models_tf.py",
    "bayesfilter/highdim/covariance_proposal_nonshared_gaussian_tf.py",
    "bayesfilter/highdim/ledh_range_bearing_target_tf.py",
    "bayesfilter/inference/neutra_transport.py",
    "bayesfilter/inference/neutra_transport_core.py",
    "bayesfilter/inference/neutra_hmc.py",
    "bayesfilter/inference/hmc_tensorflow_verified.py",
    "bayesfilter/inference/hmc_tensorflow_resume.py",
    "bayesfilter/inference/hmc_sequential_verification.py",
)


def ready(value):
    if hasattr(value, "numpy"):
        return ready(value.numpy())
    if hasattr(value, "tolist"):
        return ready(value.tolist())
    if isinstance(value, dict):
        return {str(k): ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(ready(payload), indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def learning_rate(base, accepted):
    return base * (1. if accepted < 128 else .5 if accepted < 384 else .25)


def latent_seed(seed, attempt, stream=1):
    # The stream is separate from both initialization and accepted-step count.
    if min(seed, attempt, stream) < 0 or attempt >= 1_000_000:
        raise ValueError("invalid latent stream index")
    return [261021 + seed * 100, stream * 1_000_000 + attempt]


def logabsdet3(tf, matrix):
    """Independent 3x3 cofactor diagnostic; compatible with GPU XLA.

    LogMatrixDeterminant lacks an XLA-GPU kernel in the campaign environment.
    Singular input deliberately returns -inf and fails the geometry check.
    """
    if matrix.shape[-2:] != (3,3):
        raise ValueError("the SIR geometry diagnostic requires 3x3 Jacobians")
    a,b,c = matrix[...,0,0],matrix[...,0,1],matrix[...,0,2]
    d,e,f = matrix[...,1,0],matrix[...,1,1],matrix[...,1,2]
    g,h,i = matrix[...,2,0],matrix[...,2,1],matrix[...,2,2]
    determinant=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)
    return tf.math.log(tf.math.abs(determinant))


def map_geometry_function(tf, transport):
    @tf.function(input_signature=[tf.TensorSpec([32,3],tf.float64)],
                 jit_compile=True,autograph=False)
    def geometry(z):
        q,ld=transport.forward_and_logdet(z)
        inverse,ild=transport.inverse_and_forward_logdet(q)
        jac=tf.stack([transport.pullback_score_batch(z,tf.broadcast_to(
            tf.one_hot(i,3,dtype=tf.float64),[32,3])) for i in range(3)],1)
        return dict(q=q,logdet=ld,inverse_error=tf.reduce_max(tf.abs(inverse-z)),
            logdet_error=tf.reduce_max(tf.abs(ild-ld)),
            determinant_error=tf.reduce_max(tf.abs(logabsdet3(tf,jac)-ld)))
    return geometry


def remaining(state, budget):
    used = sum(a["wall_seconds"] if "wall_seconds" in a else
               min(a["cap_seconds"], max(0., time.time()-a["started"]))
               for a in state["attempts"]
               if a["budget"] == budget)
    return max(0., state["budgets_seconds"][budget] - used)


def worker_alive(attempt):
    """Check an ordinary PID against its exact job, avoiding PID reuse."""
    try:
        command = Path(f"/proc/{attempt['pid']}/cmdline").read_bytes().split(b"\0")
        return str(Path(attempt["output"])/"job.json").encode() in command
    except (FileNotFoundError, ProcessLookupError, KeyError):
        return False


def nominate(rows, count=1):
    # This deterministic tie-break is a resource allocation rule, not ranking
    # evidence. Independent heldout/replicated/downstream checks remain needed.
    good = [r for r in rows if r and r.get("valid") and
            isinstance(r.get("selection_loss"), (float, int)) and
            math.isfinite(r["selection_loss"])]
    return sorted(good, key=lambda r: (r["selection_loss"], r["key"]))[:count]


def posterior_assessment(result):
    """Assess cumulative R-hat completion; ESS and MCSE are precision diagnostics."""
    result=result or {}
    diagnostic=result.get("diagnostics") or {}
    def available(values):
        return isinstance(values,(list,tuple)) and len(values)==3 and all(
            type(x) in (float,int) and math.isfinite(x) and x>=0 for x in values)
    rhat=(diagnostic.get("rank_normalized_split_rhat") or {}).get("maximum")
    checks=dict(rhat=available(rhat) and all(0<x<=1.01 for x in rhat),
        sampler_complete=result.get("posterior_sampling_complete") is True
            and result.get("status")=="VERIFIED")
    precision={key:(diagnostic.get(section) or {}).get(field)
        for key,section,field in (
            ("bulk_ess","rank_normalized_ess","bulk"),
            ("tail_ess","rank_normalized_ess","tail"),
            ("mcse_sd_ratio","mean","mcse_sd_ratio"))}
    qualified=all(checks.values())
    return dict(method="neutra",status=result.get("status","MISSING"),checks=checks,
        assessment_policy=ASSESSMENT_POLICY,
        precision_diagnostic_role="diagnostic_only",precision_diagnostics=precision,
        precision_diagnostics_available={key:available(value) for key,value in precision.items()},
        qualified=qualified,acceptance=result.get("acceptance_per_chain"),
        mean=result.get("q_mean"),wall_seconds=result.get("wall_seconds"),
        primary_status="NEUTRA_POSTERIOR_CRITERIA_MET" if qualified
            else "NEUTRA_HMC_INCOMPLETE",
        default_ready=False,comparative_claim=False,
        uncertainty="Diagnostics concern the fixed numerical endpoint; exact SIR posterior accuracy is not established")


def reassess_saved_result(root):
    """CPU-only reporting update; preserve runtime provenance and launch no workers."""
    root=Path(root)
    with (root/"supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("cannot reassess while a supervisor is active") from None
        state=read(root/"campaign.json")
        if state["status"]=="RUNNING" or any(
            a.get("status") in ("RUNNING","LAUNCHING") for a in state["attempts"]):
            raise RuntimeError("cannot reassess an active campaign")
        report_path=root/"hmc-result.json"
        report=read(report_path)
        assessment=posterior_assessment(report["result"])
        records=root/"reassessments"
        number=1
        while (records/f"{number:03d}.json").exists():
            number+=1
        record=records/f"{number:03d}.json"
        save(record,dict(schema="sir_neutra_saved_result_reassessment_v1",
            time_utc=time.time(),command=[sys.executable,*sys.argv],plan=PLAN,
            assessment_policy=ASSESSMENT_POLICY,
            assessment_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            previous_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
            previous_report=report,previous_status=state["status"],
            previous_phase=read(root/"next-phase.json"),assessment=assessment,
            runtime_source_sha256=state["source_sha256"],
            remaining_seconds={key:remaining(state,key) for key in state["budgets_seconds"]},
            cpu_only=True,tensorflow_imported="tensorflow" in sys.modules,workers_launched=0))
        save(report_path,{**report,"assessment":assessment,"assessment_record":str(record)})
        state.update(status=assessment["primary_status"],assessment_record=str(record))
        save(root/"campaign.json",state)
        save(root/"next-phase.json",dict(phase=state["status"],
            explanation="Saved result reassessed: cumulative R-hat controls completion; ESS and MCSE are diagnostic only",
            training_seconds_remaining=remaining(state,"training"),
            hmc_seconds_remaining=remaining(state,"hmc"),time_utc=time.time(),
            assessment_record=str(record)))
        return assessment


def initialize_gpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("BAYESFILTER_PRELOAD_CUSTOM_OP", "0")
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    return tf, memory


def make_case(tf, tuning=None):
    from bayesfilter.testing import surrogate_hmc_five_model as five
    from bayesfilter.testing.surrogate_hmc_literature_priors import PRIOR_ID
    from bayesfilter.highdim.covariance_proposal_tf import ProposalControls
    warm = ROOT / ("docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/"
        "literature-prior-other-four-advance-01/sir_d18/filter-tuning/sir_d18/filter-selection.json")
    case = replace(five.load_case("sir_d18"), prior_choice=PRIOR_ID,
        precision_policy="uniform", tuning_seed_offset=1000,
        controls=ProposalControls(**read(warm)["selected"]["controls"]))
    if tuning:
        record = five.load_selection(case, Path(tuning)/"batch-2/filter-selection.json")
        b4 = read(Path(tuning)/"batch-4/filter-selection.json")
        body = {k: v for k, v in b4.items() if k != "selection_hash"}
        if (five.hashed(body) != b4["selection_hash"] or
                body["scope"] != {**case.scope(), "force_batch_size": 4} or
                body["selected"] != record["selected"] or
                body["status"] != "FILTER_SCOPE_SELECTED_FOR_EXPLORATORY_HMC"):
            raise ValueError("joint native training/HMC filter scope mismatch")
    return case


def target_and_trainer(tf, job):
    from bayesfilter.testing.surrogate_hmc_sir_neutra import SIRNeuTraTarget
    from bayesfilter.inference.neutra_transport import (
        NeuTraTransport, NeuTraTransportConfig, NeuTraOptimizerConfig, NeuTraTransportTrainer)
    case = make_case(tf, job["tuning"])
    target = SIRNeuTraTarget(case, 4)
    config = replace(NeuTraTransportConfig.hoffman_author_iaf(3,
        conditional_scale_cap=2., seed=(261021, job["seed"]), dtype="float32"),
        hidden_layers=(job["width"],)*2, affine_center=(0.,)*3,
        affine_scale=(job["scale"],)*3)
    optimizer = NeuTraOptimizerConfig(batch_size=8, estimator="standard",
        learning_rate=job["rate"], beta1=.9, beta2=.999, epsilon=1e-7,
        gradient_clip_norm=job.get("clip", 1.), jit_compile=True,
        target_dtype="float32", shard_size=4)
    trainer = NeuTraTransportTrainer(NeuTraTransport(config), target.value_score,
        optimizer, target_signature=target.target_signature)
    return case, target, trainer


def monitor_function(tf, target, transport, size=32):
    """Independent batched post-update density checks, not optimizer outputs."""
    @tf.function(input_signature=[tf.TensorSpec([size, 3], tf.float32)],
                 jit_compile=True, autograph=False)
    def evaluate(z):
        q, ld = transport.forward_and_logdet(z)
        values = tf.TensorArray(tf.float32, size=size//4, element_shape=[4])
        scores = tf.TensorArray(tf.float32, size=size//4, element_shape=[4, 3])
        valid = tf.TensorArray(tf.bool, size=size//4, element_shape=[4])
        def body(i, values, scores, valid):
            v, s, good = target.value_score(tf.slice(q,[4*i,0],[4,3]))
            return i+1, values.write(i,v), scores.write(i,s), valid.write(i,good)
        _, values, scores, valid = tf.while_loop(lambda i,*_: i<size//4, body,
            (0,values,scores,valid), parallel_iterations=1)
        v, s, good = values.concat(), scores.concat(), valid.concat()
        jac = tf.stack([transport.pullback_score_batch(z,
            tf.broadcast_to(tf.one_hot(i,3), [size,3])) for i in range(3)], axis=1)
        singular = tf.linalg.svd(jac, compute_uv=False)
        losses = -v-ld
        from bayesfilter.inference import neutra_transport_core as core
        activation, saturation, stage_scales = z, [], []
        for layer in transport.components:
            if any(layer is stage for stage in transport.stages):
                scale, _, logits, _ = core.iaf_network_diagnostics(layer, activation)
                capped = (logits[:, :3]-layer.biases[-1][:3])/2.
                saturation.append(tf.reduce_mean(tf.cast(tf.abs(tf.tanh(capped))>.99,tf.float32)))
                stage_scales.append(scale)
            activation, _ = layer.forward_and_logdet(activation)
        return dict(valid=tf.reduce_all(good)&tf.reduce_all(tf.math.is_finite(losses))
                        &tf.reduce_all(tf.math.is_finite(jac)),
            row_valid=good, losses=losses, loss=tf.reduce_mean(losses),
            loss_se=tf.math.reduce_std(losses)/tf.sqrt(tf.constant(float(size-1))),
            q=q, values=v, score=s, logdet=ld,
            condition=singular[:,0]/singular[:,-1],
            saturation_fraction=tf.stack(saturation),stage_log_scales=tf.stack(stage_scales),
            free_biases=tf.stack([s.biases[-1][:3] for s in transport.stages]))
    return evaluate


def preflight(tf, job, out):
    from bayesfilter.testing.surrogate_hmc_sir_neutra import SIRNeuTraTarget
    case = make_case(tf, job["tuning"])
    target = SIRNeuTraTarget(case, 4)
    bank = tf.random.stateless_normal([32,3], latent_seed(0,0,10))
    scales = []
    for scale in (1., .35, .1, .02):
        evaluations = [target.evaluate(bank[i:i+4]*scale) for i in range(0,32,4)]
        good = tf.concat([e["valid"] for e in evaluations],0)
        row = dict(scale=scale,valid=bool(tf.reduce_all(good)),row_valid=good,
            values=tf.concat([e["value"] for e in evaluations],0),
            scores=tf.concat([e["score"] for e in evaluations],0))
        scales.append(ready(row))
        save(out/"initialization.json",scales)
    viable = [r["scale"] for r in scales if r["valid"]]
    if not viable:
        return dict(status="INITIALIZATION_REPAIR_REQUIRED", valid=False,
                    scales=scales, continuation_veto=False)
    scale = max(viable)
    # FD is a diagnostic/reference exception. The actual training derivative
    # remains the analytical filter score. Quantization bounds scale with ll/h.
    fd, contradictions = [], []
    for q in (tf.zeros([3]), tf.constant([1.,-1.,.5])*scale,
              tf.constant([-2.,1.,2.])*scale):
        center = target.evaluate(tf.broadcast_to(q,[4,3]))
        if not bool(center["valid"][0]):
            fd.append(dict(q=q,status="invalid_probe")); continue
        for axis in range(3):
            entries = []
            for h in (.03,.01,.003):
                delta = tf.one_hot(axis,3)*h
                result = target.evaluate(tf.stack([q+delta,q-delta,q+2*delta,q-2*delta]))
                v = result["value"]
                estimate = (v[0]-v[1])/(2*h)
                coarse = (v[2]-v[3])/(4*h)
                roundoff = 64.*2.**-23*tf.reduce_max(tf.abs(v))/h
                uncertainty = tf.abs(estimate-coarse)+roundoff
                difference = tf.abs(estimate-center["score"][0,axis])
                entries.append(dict(h=h,finite=bool(tf.reduce_all(result["valid"])),
                    estimate=estimate,coarse=coarse,score=center["score"][0,axis],
                    quantization_allowance=roundoff,discrepancy=difference,
                    compatible=bool(difference <= 4.*uncertainty)))
            contradicted = all(e["finite"] and not e["compatible"] for e in entries)
            contradictions.append(contradicted)
            fd.append(dict(q=q,axis=axis,entries=entries,consistent_mismatch=contradicted))
            save(out/"derivative-diagnostic.json",fd)
    z = tf.random.stateless_normal([8,3],latent_seed(100,0))
    map_checks=[]
    for scale in sorted(viable,reverse=True):
        pilot_job = {**job,"width":3,"rate":.001,"seed":100,"scale":scale}
        _,_,trainer = target_and_trainer(tf,pilot_job)
        result = trainer.evaluate(z)
        map_checks.append(dict(scale=scale,valid=bool(result["valid"])))
        if bool(result["valid"]):break
    if not bool(result["valid"]):
        return dict(status="INITIALIZATION_REPAIR_REQUIRED",valid=False,scales=scales,
                    map_checks=map_checks,continuation_veto=False)
    norm = tf.linalg.global_norm(result["gradients"])
    clip_rows=[]
    for clip in (1.,10.,None):
        _,_,arm = target_and_trainer(tf,{**pilot_job,"clip":clip})
        update = arm.train_step(z)
        post = arm.evaluate(tf.random.stateless_normal([8,3],latent_seed(100,0,2)))
        clip_rows.append(dict(clip=clip,valid=update["valid"],
            gradient_norm=update["gradient_norm"],post_valid=post["valid"],post_loss=post["loss"]))
    save(out/"clipping-calibration.json",clip_rows)
    return dict(status="PREREQUISITES_PASSED" if not any(contradictions) and bool(result["valid"])
        else "DERIVATIVE_OR_INITIALIZATION_REPAIR_REQUIRED",
        valid=not any(contradictions) and bool(result["valid"]),
        continuation_veto=any(contradictions), scale=scale, scales=scales,
        map_checks=map_checks,
        initial_gradient_norm=norm,clipping=clip_rows,target_scope=target.scope,
        target_signature=target.target_signature,derivative_diagnostics=str(out/"derivative-diagnostic.json"),
        numerical_score_claim="analytical same finite LEDH density including chart and prior",
        fd_role="quantization-aware derivative diagnostic, not independent score oracle")


def train(tf, job, out):
    _,target,trainer = target_and_trainer(tf,job)
    attempts=accepted=consecutive=0
    monitor_failed=False
    history=[]
    if job.get("resume"):
        saved=read(job["resume"])
        if saved["seed"] != job["seed"] or saved["target_scope"] != target.scope:
            raise ValueError("training checkpoint stream/scope mismatch")
        trainer.restore(saved["trainer"])
        attempts,accepted=saved["attempted"],saved["accepted"]
        consecutive=saved["consecutive_rejections"]
        history=saved["history"]
        monitor_failed=saved.get("monitor_failed",False)
        if accepted != int(trainer.optimizer.iterations):
            raise ValueError("accepted-step/Adam iteration mismatch")
    monitor=monitor_function(tf,target,trainer.transport)
    monitor_z=tf.random.stateless_normal([32,3],latent_seed(job["seed"],0,2))
    # All candidate selection uses common random numbers, independent of training.
    selection_z=tf.random.stateless_normal([32,3],latent_seed(0,0,3))
    deadline=float(os.environ["BAYESFILTER_WORKER_WALL_DEADLINE"])
    next_monitor=((accepted//64)+1)*64
    def checkpoint():
        payload=dict(schema="sir_neutra_training_checkpoint_v1",seed=job["seed"],
            accepted=accepted,attempted=attempts,next_latent_seed=latent_seed(job["seed"],attempts),
            consecutive_rejections=consecutive,schedule_rate=learning_rate(job["rate"],accepted),
            history=history,target_scope=target.scope,trainer=trainer.checkpoint(),
            monitor_failed=monitor_failed,job=job)
        save(out/"checkpoint.json",payload)
    checkpoint()
    while accepted<job["updates"] and attempts<job["updates"]*2+32:
        if monitor_failed:break
        if time.monotonic()>deadline-300:
            break
        trainer.optimizer.learning_rate.assign(learning_rate(job["rate"],accepted))
        z=tf.random.stateless_normal([8,3],latent_seed(job["seed"],attempts))
        attempts+=1
        start=time.monotonic()
        result=trainer.train_step(z)
        valid=bool(result["valid"])
        if valid:
            accepted+=1; consecutive=0
        else:
            consecutive+=1
        history.append(ready(dict(attempt=attempts,accepted=accepted,valid=valid,
            loss=result["loss"],gradient_norm=result["gradient_norm"],
            clipped_gradient_norm=result["clipped_gradient_norm"],
            learning_rate=float(trainer.optimizer.learning_rate),seconds=time.monotonic()-start)))
        # Checkpoint every update: small flow/Adam states; no lost stream position.
        checkpoint()
        save(out/"progress.json",dict(accepted=accepted,attempted=attempts,
            requested=job["updates"],latest=history[-1]))
        if accepted>=next_monitor and valid:
            check=monitor(monitor_z)
            save(out/f"monitor-{accepted:05d}.json",check)
            next_monitor+=64
            if not bool(check["valid"]):
                monitor_failed=True
                break  # Failed candidate; supervisor can execute the other arms.
        if consecutive>=8:
            break
    checkpoint()
    checked=monitor(selection_z)
    save(out/"selection.json",checked)
    valid=accepted>0 and bool(checked["valid"]) and consecutive<8 and not monitor_failed
    return dict(status="RUNG_COMPLETED" if valid and accepted>=job["updates"] else
        "TRAINING_INCOMPLETE" if valid else "CANDIDATE_REPAIR_REQUIRED",
        valid=valid,key=job["key"],accepted=accepted,attempted=attempts,
        selection_loss=checked["loss"],selection_se=checked["loss_se"],
        checkpoint=str(out/"checkpoint.json"),target_scope=target.scope,
        target_signature=target.target_signature,job=job,
        loss_role="nomination only; no supported superiority ranking",
        posterior_qualified=False,default_ready=False)


def geometry_smoke(tf,job,out):
    """GPU-XLA mechanics regression on the actual saved map; not inference."""
    _,_,trainer=target_and_trainer(tf,job)
    trainer.restore(read(job["resume"])["trainer"])
    converted=trainer.transport.as_dtype("float64")
    z=tf.cast(tf.random.stateless_normal([32,3],latent_seed(0,0,4)),tf.float64)
    audit=map_geometry_function(tf,converted)(z)
    # Independent eager diagnostic only: the reference is deliberately outside
    # XLA because its unsupported LogMatrixDeterminant caused the incident.
    jac=tf.stack([converted.pullback_score_batch(z,tf.broadcast_to(
        tf.one_hot(i,3,dtype=tf.float64),[32,3])) for i in range(3)],1)
    reference=tf.linalg.slogdet(jac)[1]
    reference_error=tf.reduce_max(tf.abs(logabsdet3(tf,jac)-reference))
    valid=all(float(audit[k])<1e-7 for k in
              ("inverse_error","logdet_error","determinant_error"))
    valid &= float(reference_error)<1e-7
    return dict(status="GEOMETRY_SMOKE_PASSED" if valid else "GEOMETRY_REPAIR_REQUIRED",
        valid=valid,geometry=audit,reference_error=reference_error,
        independent_reference="eager TensorFlow slogdet; diagnostic exception",
        jit_compile=True,checkpoint=job["resume"],posterior_qualified=False)


def freeze(tf,job,out):
    from bayesfilter.testing import surrogate_hmc_five_model as five
    case,target,trainer=target_and_trainer(tf,job)
    trainer.restore(read(job["resume"])["trainer"])
    heldout=tf.random.stateless_normal([32,3],latent_seed(0,0,4))
    check=monitor_function(tf,target,trainer.transport)(heldout)
    converted=trainer.transport.as_dtype("float64")
    z=tf.cast(heldout,tf.float64)
    audit=map_geometry_function(tf,converted)(z)
    converted_checks=[target.evaluate(tf.cast(audit["q"][i:i+4],tf.float32))
                      for i in range(0,32,4)]
    save(out/"converted-target.json",dict(
        valid=tf.concat([r["valid"] for r in converted_checks],0),
        values=tf.concat([r["value"] for r in converted_checks],0),
        scores=tf.concat([r["score"] for r in converted_checks],0)))
    q32,ld32=trainer.transport.forward_and_logdet(heldout)
    audit.update(cast_q_max_difference=tf.reduce_max(tf.abs(audit["q"]-tf.cast(q32,tf.float64))),
                 cast_logdet_max_difference=tf.reduce_max(tf.abs(audit["logdet"]-tf.cast(ld32,tf.float64))))
    payload=converted.frozen_payload(target_signature=target.target_signature,
        training_state_hash=trainer.checkpoint()["checkpoint_hash"])
    save(out/"frozen-map.json",payload)
    components=five.bindings(case,neutra_payload=payload,
        training_target_signature=target.target_signature,training_target_scope=target.scope)
    # Independent endpoint values at the *same physical q*. This diagnoses the
    # training/MH density gap; it is never used to change the training gradient.
    independent=[]
    for q in tf.unstack(audit["q"][:8]):
        theta,_=case.chart(q)
        value,valid=components["independent"](theta)
        prior,_=case.log_prior(q)
        independent.append(dict(q=q,value=value+prior,valid=valid))
    save(out/"heldout.json",check)
    save(out/"geometry.json",audit)
    save(out/"endpoint-diagnostic.json",independent)
    valid=bool(check["valid"]) and all(bool(r["valid"]) for r in independent)
    valid &= all(bool(tf.reduce_all(r["valid"])) for r in converted_checks)
    valid &= all(float(audit[k])<1e-7 for k in ("inverse_error","logdet_error","determinant_error"))
    return dict(status="MAP_FROZEN_FOR_EXPLORATORY_HMC" if valid else "MAP_REPAIR_REQUIRED",
        valid=valid,frozen_map=str(out/"frozen-map.json"),target_scope=target.scope,
        target_signature=target.target_signature,checkpoint=job["resume"],job=job,
        geometry=audit,posterior_qualified=False,default_ready=False)


def mechanics(tf,job,out):
    """Cheap geometry adversaries, by regime; no posterior promotion."""
    from types import SimpleNamespace
    from bayesfilter.testing import surrogate_hmc_five_model as five
    case=make_case(tf,job["tuning"])
    frozen=read(job["frozen"])
    # Covariance is a proposal-map hypothesis, not estimated posterior geometry.
    map_payload=read(frozen["frozen_map"])
    mapped=five.bindings(case,neutra_payload=map_payload,
        training_target_signature=frozen["target_signature"],training_target_scope=frozen["target_scope"])
    z=tf.random.stateless_normal([256,3],[261021,700],dtype=tf.float64)
    q=mapped["transport"].forward_batch(z)
    center=tf.reduce_mean(q,0)
    centered=q-center
    scales=tf.sqrt(tf.reduce_mean(tf.square(centered),0))
    rows=[]
    for name in ("identity","diagonal","neutra"):
        comp=mapped if name=="neutra" else five.bindings(case)
        factor=tf.linalg.diag(scales) if name=="diagonal" else tf.eye(3,dtype=tf.float64)
        origin=center if name=="diagonal" else tf.zeros([3],tf.float64)
        adapter=SimpleNamespace(transform=SimpleNamespace(factor=factor,center=origin),
            latent_to_position=lambda x:origin+tf.linalg.matmul(x,factor,transpose_b=True))
        for regime,spread in (("central",.25),("dispersed",1.),("tail",2.)):
            standardized=tf.constant([[-.75,-.25,.25],[.75,.25,-.25],[-.25,.75,-.75],[.25,-.75,.75]],tf.float64)*spread
            physical=center+scales*standardized
            start=(mapped["transport"].inverse_theta_to_z_batch(physical) if name=="neutra"
                   else standardized if name=="diagonal" else physical)
            kernel=comp["binding"].build_tensor_kernel(adapter,step_size=.01,
                num_leapfrog_steps=1,target_scope=comp["scope"],target_status_trace_policy="none",
                chain_execution_mode="tf_function",use_xla=True)
            @tf.function(input_signature=[tf.TensorSpec([4,3],tf.float64)],
                         jit_compile=True,autograph=False)
            def run(x):
                result=kernel.bootstrap_results(x)
                diagnostics=tf.TensorArray(tf.float64,size=8,element_shape=[4,4])
                def body(i,x,result,diagnostics):
                    x,result=kernel.one_step(x,result,seed=tf.stack([261021,800+i]))
                    row=tf.stack([tf.cast(result.finite_status,tf.float64),
                        tf.cast(result.force_fallback,tf.float64),
                        tf.cast(result.is_accepted,tf.float64),result.delta_h],axis=1)
                    return i+1,x,result,diagnostics.write(i,row)
                _,x,result,diagnostics=tf.while_loop(lambda i,*_:i<8,body,
                    (0,x,result,diagnostics),parallel_iterations=1)
                return x,diagnostics.stack()
            begin=time.monotonic()
            positions,health=run(start)
            rows.append(dict(arm=name,regime=regime,seconds=time.monotonic()-begin,
                common_physical_start=physical,finite=health[:,:,0],force_fallback=health[:,:,1],
                accepted=health[:,:,2],delta_h=health[:,:,3],positions=positions))
            save(out/"geometry-adversaries.json",rows)
    return dict(status="DIAGNOSTICS_COMPLETE",rows=rows,valid=True,
        heuristic_dominance_verdict="NOT_ESTABLISHED_SHORT_DIAGNOSTICS_ONLY",
        posterior_qualified=False)


def hmc(tf,job,out):
    if job.get("arm")!="neutra":
        raise ValueError("this campaign runs only frozen NeuTra HMC")
    from bayesfilter.testing import surrogate_hmc_five_model as five
    case=make_case(tf,job["tuning"])
    frozen=read(job["frozen"])
    kwargs=dict(neutra_payload=read(frozen["frozen_map"]),
        training_target_signature=frozen["target_signature"],training_target_scope=frozen["target_scope"])
    if job.get("resume_manifest"):
        kwargs["resume_manifest"]=job["resume_manifest"]
    else:
        kwargs.update(target_accept_prob=.7,initial_step_size=job.get("initial_step_size",.01))
    return five.inference(case,out,Path(job["tuning"])/"batch-2/filter-selection.json",**kwargs)


def worker(path):
    # The worker retains its own hard wall cap if its supervisor disappears.
    deadline=float(os.environ["BAYESFILTER_WORKER_WALL_DEADLINE"])
    watchdog=threading.Timer(max(1.,deadline-time.monotonic()+55),lambda:os._exit(124))
    watchdog.daemon=True
    watchdog.start()
    job=read(path);out=Path(path).parent
    begin=time.monotonic()
    manifest=dict(command=sys.argv,git_commit=subprocess.check_output(
        ["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        python=sys.executable,plan=PLAN,job=job,source_sha256=hashes(),
        backend="TensorFlow GPU XLA",dtype="float32",tf32=False,jit_compile=True,
        gpu_access="escalated",logical_batch=8 if job["kind"]=="train" else None,
        native_batch=2 if job["kind"] in ("hmc","mechanics") else 4,
        map_and_hmc_dtype="float64" if job["kind"] in ("geometry_smoke","freeze","hmc","mechanics") else "float32",
        native_batch_sizes=[4,2] if job["kind"]=="tuning" else None,
        sample_wise_target_loop=False,result=str(out/"result.json"))
    save(out/"manifest.json",manifest)
    try:
        tf,memory=initialize_gpu();manifest["memory_policy"]=memory
        save(out/"manifest.json",manifest)
        if job["kind"]=="tuning":
            from bayesfilter.testing.surrogate_hmc_scope_tuning import tune
            result=tune(make_case(tf),out,fixed_ot_five=True,
                native_batch_sizes=(4,2),precision_warning_only=True)
            result["valid"]=result["selected"] is not None
        else:
            result=globals()[job["kind"]](tf,job,out)
        result.update(allocator=tf.config.experimental.get_memory_info("GPU:0"))
        save(out/"result.json",result)
        manifest["status"]="FINISHED"
    except BaseException as error:
        manifest.update(status="FAILED",error=repr(error))
        (out/"error.txt").write_text(traceback.format_exc())
        raise
    finally:
        manifest["wall_seconds"]=time.monotonic()-begin
        save(out/"manifest.json",manifest)


class Campaign:
    def __init__(self,root,resume_from=None,repair_note=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.lock=(self.root/"supervisor.lock").open("a")
        try:
            fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError("this campaign already has a live supervisor") from None
        path=self.root/"campaign.json"
        if resume_from is not None:
            if path.exists() or self.root.resolve()==Path(resume_from).resolve():
                raise ValueError("repair requires a fresh, versioned output root")
            if not repair_note or not (ROOT/repair_note).is_file():
                raise ValueError("repair requires its recorded scientific/code review note")
            source=Path(resume_from).resolve()
            with (source/"supervisor.lock").open("a") as parent_lock:
                try:
                    fcntl.flock(parent_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RuntimeError("parent campaign still has a live supervisor") from None
                previous=read(source/"campaign.json")
                if any(a.get("status") in ("RUNNING","LAUNCHING")
                       for a in previous["attempts"]):
                    raise RuntimeError("reconcile parent workers before creating a repaired continuation")
            old_hashes=previous["source_sha256"]
            self.state=previous
            self.state["source_sha256"]=hashes()
            self.state.setdefault("repairs",[]).append(dict(
                review_note=str(repair_note),parent_campaign=str(source),
                parent_source_sha256=old_hashes,
                repaired_source_sha256=self.state["source_sha256"],
                prior_attempts_preserved=True,budgets_carried_forward=True))
            self.state["status"]="RUNNING"
        else:
            self.state=read(path) if path.exists() else dict(schema="sir_neutra_campaign_v1",
                status="RUNNING",plan=PLAN,source_sha256=hashes(),
                budgets_seconds=dict(training=16*3600,hmc=24*3600),attempts=[])
        if self.state["source_sha256"] != hashes():
            raise ValueError("source change: record a repair in a fresh campaign root")
        self.reconcile()
        self.persist()

    def reconcile(self):
        for attempt in self.state["attempts"]:
            if attempt.get("status") not in ("RUNNING","LAUNCHING"):
                continue
            if not attempt.get("pid"):
                for path in Path("/proc").glob("[0-9]*"):
                    candidate={**attempt,"pid":int(path.name)}
                    if worker_alive(candidate):
                        attempt["pid"]=candidate["pid"];break
            print(f"Recovering {attempt['key']} from {attempt['output']}",flush=True)
            while worker_alive(attempt):
                if time.time()>attempt["started"]+attempt["cap_seconds"]+30:
                    os.killpg(attempt["pid"],signal.SIGKILL)
                    break
                time.sleep(5)
            path=Path(attempt["output"])/"manifest.json"
            manifest=read(path) if path.exists() else {}
            completed=(manifest.get("status")=="FINISHED" and
                       (Path(attempt["output"])/"result.json").exists())
            elapsed=max(0.,time.time()-attempt["started"])
            attempt.update(status="FINISHED",exit_code=0 if completed else 124,
                wall_seconds=manifest.get("wall_seconds",min(elapsed,attempt["cap_seconds"])),
                failure_class="RECOVERED_COMPLETE" if completed else "INTERRUPTED_WORKER")
            self.persist()

    def persist(self):
        save(self.root/"campaign.json",self.state)

    def next(self,phase,explanation):
        save(self.root/"next-phase.json",dict(phase=phase,explanation=explanation,
            training_seconds_remaining=remaining(self.state,"training"),
            hmc_seconds_remaining=remaining(self.state,"hmc"),time_utc=time.time()))

    def task(self,key,kind,*,budget="training",allowance=3600,**kwargs):
        old=[a for a in self.state["attempts"] if a["key"]==key]
        for a in old:
            result=Path(a["output"])/"result.json"
            if a.get("exit_code")==0 and result.exists():
                return read(result),Path(a["output"])
        if len(old)>=2 or remaining(self.state,budget)<300:
            return None,None
        # Orphaned GPU workers must be reconciled, never duplicated on resume.
        if any(a.get("status")=="RUNNING" for a in self.state["attempts"]):
            raise RuntimeError("unreconciled running attempt; inspect PID/log before resuming")
        cap=min(allowance,remaining(self.state,budget))
        directory=self.root/f"{len(self.state['attempts'])+1:03d}-{key}"
        directory.mkdir(exist_ok=False)
        job=dict(kind=kind,key=key,**kwargs)
        if old and kind=="train" and (Path(old[-1]["output"])/"checkpoint.json").exists():
            job["resume"]=str(Path(old[-1]["output"])/"checkpoint.json")
        save(directory/"job.json",job)
        command=[sys.executable,str(Path(__file__).resolve()),"worker","--job",str(directory/"job.json")]
        env=dict(os.environ,BAYESFILTER_WORKER_WALL_DEADLINE=str(time.monotonic()+cap-60))
        attempt=dict(key=key,budget=budget,output=str(directory),pid=None,
            status="LAUNCHING",started=time.time(),cap_seconds=cap,command=command)
        self.state["attempts"].append(attempt);self.persist()
        with (directory/"worker.log").open("w") as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                     start_new_session=True)
            attempt.update(pid=process.pid,status="RUNNING");self.persist()
            begin=time.monotonic()
            try:
                code=process.wait(timeout=cap)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try: code=process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);code=process.wait()
                attempt["failure_class"]="WALL_BUDGET_STOP"
            attempt.update(status="FINISHED",exit_code=code,wall_seconds=time.monotonic()-begin)
            self.persist()
        if code==0 and (directory/"result.json").exists():
            return read(directory/"result.json"),directory
        self.next("repair",f"{key} failed; inspect {directory}/error.txt and worker.log")
        return None,directory

    def completed(self,key):
        return any(a["key"]==key and a.get("exit_code")==0 and
                   (Path(a["output"])/"result.json").exists() for a in self.state["attempts"])

    def run(self):
        self.next("FP32_scope_tuning","Joint B4/B2 calibration and heldout numerical checks")
        tune,path=self.task("tuning","tuning",allowance=2*3600)
        if not tune or not tune["valid"]:
            return self.finish("PREREQUISITE_REPAIR_REQUIRED")
        tuning=str(path)
        self.next("preflight","Broad initialization, actual-target derivative and clipping checks")
        pre,_=self.task("preflight","preflight",allowance=2*3600,tuning=tuning)
        if not pre or not pre["valid"]:
            return self.finish("PREREQUISITE_REPAIR_REQUIRED")
        base=dict(tuning=tuning,scale=pre["scale"])
        rows=[]
        self.next("training_screen","Six target-specific width/rate hypotheses, 64 accepted updates")
        for width in (3,6,9):
            for rate in (.001,.01):
                key=f"w{width}-lr{rate}-s100-r64"
                if not self.completed(key) and remaining(self.state,"training")<5400:
                    continue
                row,_=self.task(key,"train",allowance=2400,**base,width=width,rate=rate,seed=100,updates=64)
                if row:rows.append(row)
        promoted=[]
        self.next("training_extension","Independent selection bank nominates two configurations only")
        for row in nominate(rows,2):
            key=row["key"].replace("r64","r256")
            if not self.completed(key) and remaining(self.state,"training")<5400:continue
            job={k:row["job"][k] for k in ("tuning","scale","width","rate","seed")}
            extended,_=self.task(key,"train",allowance=5400,
                **job,updates=256,resume=row["checkpoint"])
            if extended:promoted.append(extended)
        candidates=nominate(promoted or rows)
        if not candidates:
            return self.finish("TRAINING_REPAIR_REQUIRED")
        prototype=candidates[0]
        job={k:prototype["job"][k] for k in ("tuning","scale","width","rate")}
        replicas=[]
        self.next("training_replication","Three fresh initialization/noise seeds, bounded 512-update fits")
        for seed in (201,202,203):
            available=remaining(self.state,"training")-5400
            if not self.completed(f"replica-s{seed}") and available<600:continue
            replicas_left=204-seed
            allocation=min(3*3600,available/replicas_left)
            row,_=self.task(f"replica-s{seed}","train",allowance=allocation,
                **job,seed=seed,updates=512)
            if row:replicas.append(row)
        viable=nominate(replicas or candidates)
        if not viable:
            return self.finish("TRAINING_REPAIR_REQUIRED")
        selected=viable[0]
        save(self.root/"training-summary.json",dict(screen=rows,extensions=promoted,
            replicas=replicas,selected=selected,statistically_supported_ranking=False,
            status="REPLICATED_UPDATE_BUDGET_COMPLETED" if len(replicas)==3 and all(
                r["valid"] and r["accepted"]>=512 for r in replicas) else "TRAINING_INCOMPLETE",
            training_convergence="NOT_ESTABLISHED_BY_UPDATE_COUNT",
            choice_role="independent loss nomination, no posterior or superiority claim"))
        self.next("freeze","Fresh heldout check, FP64 map geometry and independent endpoint diagnostics")
        freeze_job={k:selected["job"][k] for k in ("tuning","scale","width","rate","seed")}
        geometry_check,_=self.task("geometry-smoke","geometry_smoke",allowance=600,
            **freeze_job,resume=selected["checkpoint"])
        if not geometry_check or not geometry_check["valid"]:
            return self.finish("GEOMETRY_REPAIR_REQUIRED")
        frozen,frozen_path=self.task("freeze","freeze",allowance=5400,
            **freeze_job,resume=selected["checkpoint"])
        if not frozen or not frozen["valid"]:
            return self.finish("FROZEN_MAP_REPAIR_REQUIRED")
        frozen_file=str(frozen_path/"result.json")
        return self.sample_neutra(tuning,frozen_file)

    def sample_neutra(self,tuning,frozen_file):
        self.next("adaptive_hmc","NeuTra only: four chains, adaptive warmup and cumulative retained R-hat")
        result,_=self.task("hmc-neutra","hmc",budget="hmc",
            allowance=remaining(self.state,"hmc"),tuning=tuning,frozen=frozen_file,arm="neutra")
        if (result and result.get("manifest") and
                result.get("status")=="BUDGET_EXHAUSTED" and remaining(self.state,"hmc")>300):
            self.next("adaptive_hmc_continuation","Continue NeuTra from saved state under remaining allowance")
            continued,_=self.task("hmc-neutra-continue","hmc",budget="hmc",
                allowance=remaining(self.state,"hmc"),tuning=tuning,frozen=frozen_file,
                arm="neutra",resume_manifest=result["manifest"])
            if continued:result=continued
        assessment=posterior_assessment(result)
        save(self.root/"hmc-result.json",dict(result=result,assessment=assessment))
        return self.finish(assessment["primary_status"])

    def finish(self,status):
        self.state["status"]=status;self.persist()
        self.next(status,"Review saved results; distinguish candidate failure from campaign invalidity")
        print(json.dumps(dict(status=status,output=str(self.root),
            remaining={k:remaining(self.state,k) for k in self.state["budgets_seconds"]})),flush=True)
        return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("run","status","worker","reassess"))
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--resume-from",type=Path,
        help="carry previous attempts and charged time into a fresh repair root")
    parser.add_argument("--repair-note",help="plan/result note auditing the localized repair")
    parser.add_argument("--job",type=Path)
    args=parser.parse_args()
    if args.command=="worker":
        worker(args.job)
    elif args.command=="reassess":
        print(json.dumps(reassess_saved_result(args.output),indent=2))
    elif args.command=="status":
        path=args.output/"campaign.json"
        if not path.exists():print("NOT_STARTED");return
        state=read(path)
        print(json.dumps(dict(status=state["status"],next=read(args.output/"next-phase.json"),
            latest=state["attempts"][-1] if state["attempts"] else None),indent=2))
    else:
        Campaign(args.output,resume_from=args.resume_from,repair_note=args.repair_note).run()


if __name__=="__main__":
    main()
