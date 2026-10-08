#!/usr/bin/env python3
"""Read-only numerical diagnostic of SIR's actual LEDH proposal and reset.

Uses the shared analytical executor; no runtime filter is implemented here.
Gaussian integrations are independent reference checks on its empirical clouds.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PLAN = ROOT / "docs/plans/ledh-sir-proposal-diagnosis-20261007.md"
BASE = ROOT / "docs/plans/artifacts/ledh-zhao-horizons-20261006-01/013-ledh-sir_d18-T10/worker-0002-sir_d18"


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=261006101)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    # Preserve complete framework/compiler messages in the unique attempt log.
    log = (args.output / "run.log").open("w", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    started = time.monotonic()
    os.environ.update(TF_FORCE_GPU_ALLOW_GROWTH="true", TF_CPP_MIN_LOG_LEVEL="2",
                      TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="2",
                      OMP_NUM_THREADS="2")
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    from bayesfilter.highdim import sqmc_campaign_tf as common
    from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
    from bayesfilter.highdim.ledh_canonical_score_tf import (
        canonical_value_and_analytical_score, _flow_substeps_with_tangent)
    from bayesfilter.highdim.ledh_marginal_weights_tf import marginal_prior_ratio_tangent

    saved = next(row for row in json.loads((BASE / "rows.json").read_text())
                 if row["design_seed"] == args.seed)
    data = json.loads((BASE / "dataset.json").read_text())
    assert data["observation_sha256"] == saved["observation_sha256"]
    spec = NonlinearSQMCSpec("sir_d18")
    n, d, horizon = saved["particles"], 18, 10
    dtype = tf.float64
    theta = spec.default_theta(dtype)
    obs = tf.constant(data["observations"], dtype)
    controls = common.numerical_settings(saved["controls"])
    controls.update(common.route_settings(saved["route"]))
    design = common.reset_design(n, d, dtype, saved["reset_design"])
    inputs = common.random_inputs(saved["route"], args.seed, n, d, horizon, dtype,
                                  jit_compile=True)
    direction = tf.constant([1., 0., 0.], dtype)
    paths = [Path(__file__), PLAN,
             ROOT / "bayesfilter/highdim/ledh_canonical_score_tf.py",
             ROOT / "bayesfilter/highdim/ledh_canonical_models_tf.py",
             ROOT / "bayesfilter/highdim/ledh_marginal_weights_tf.py"]
    manifest = dict(git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                      cwd=ROOT, text=True).strip(), command=[sys.executable, *sys.argv],
                    created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    environment=sys.executable, tensorflow=tf.__version__,
                    cpu_only=False, device="GPU", memory_policy=memory,
                    jit_compile=True, tf32=False, dtype="float64", particles=n,
                    seed=args.seed, horizon=horizon, controls=controls,
                    observation_sha256=data["observation_sha256"],
                    dataset_file=str(BASE / "dataset.json"), plan=str(PLAN),
                    output=str(args.output), results=str(args.output / "result.json"),
                    role="conditional_cloud_proposal_diagnostic_only",
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in paths})
    write(args.output / "manifest.json", manifest)

    @tf.function(input_signature=[tf.TensorSpec([n,d],dtype),
                                  tf.TensorSpec([horizon,n,d],dtype),
                                  tf.TensorSpec([horizon,n],dtype),
                                  tf.TensorSpec([horizon,9],dtype)],
                 jit_compile=True, autograph=False)
    def replay(initial, noise, uniforms, observations):
        model, _ = spec.model(theta, direction)
        states, covs, ds, dc = spec.initial_cloud(theta, initial, direction)
        value, score, trace = canonical_value_and_analytical_score(
            model, theta, states, covs, noise, observations, with_score=True,
            return_trace=True, initial_state_tangent=ds, initial_covariance_tangent=dc,
            reset_design=design, process_ancestor_uniforms=uniforms, **controls)
        keep = {"pre_flow", "children", "posterior_logits", "d_posterior_logits",
                "incoming_log_weights", "d_incoming_log_weights", "predicted_covariances",
                "states_after_reset", "d_states_after_reset", "program_valid"}
        return value, score, tuple({k:v for k,v in row.items() if k in keep} for row in trace)

    with tf.device("/GPU:0"):
        value, score, trace = replay(*inputs, obs)
        value, score = float(value), float(score[0])
    replay_check = dict(value=value, saved_value=saved["log_likelihood"],
                        value_gap=value-saved["log_likelihood"],
                        score=score, saved_score=saved["score"][0],
                        score_gap=score-saved["score"][0],
                        valid=all(bool(row["program_valid"]) for row in trace))
    write(args.output / "replay.json", replay_check)
    if not replay_check["valid"] or abs(replay_check["value_gap"]) > 1e-7:
        raise RuntimeError("saved value replay failed; inspect replay.json")
    print("Replay passed", replay_check, flush=True)

    sig = [tf.TensorSpec([n,d],dtype), tf.TensorSpec([n,d],dtype),
           tf.TensorSpec([n,d],dtype), tf.TensorSpec([n,d],dtype),
           tf.TensorSpec([n,d,d],dtype), tf.TensorSpec([n],dtype),
           tf.TensorSpec([n],dtype), tf.TensorSpec([n],dtype),
           tf.TensorSpec([n],dtype), tf.TensorSpec([9],dtype)]

    @tf.function(input_signature=sig, jit_compile=True, autograph=False)
    def diagnose(states, d_states, pre, children, p, logits, dlogits, lw, dlw, y):
        model, _ = spec.model(theta, direction)
        means = model.transition_mean_fn(theta, states)
        dm = model.transition_mean_tangent_fn(theta, states, d_states)
        residual = y[None] - means[:, 1::2]
        two_pi = tf.constant(2.*math.pi, dtype)
        ell = -.5*(9.*tf.math.log(two_pi*101.)+tf.reduce_sum(residual**2,axis=-1)/101.)
        dell = tf.reduce_sum(residual*dm[:,1::2],axis=-1)/101.
        exact = tf.reduce_logsumexp(lw+ell)
        dexact = tf.reduce_sum(tf.nn.softmax(lw+ell)*(dlw+dell))
        boot = -.5*(9.*tf.math.log(two_pi*100.)+tf.reduce_sum((y-pre[:,1::2])**2,axis=-1)/100.)
        boot_z = tf.reduce_logsumexp(lw+boot)
        actual = tf.reduce_logsumexp(logits)
        da = tf.reduce_sum(tf.nn.softmax(logits)*dlogits)
        flow = _flow_substeps_with_tangent(model, pre, tf.zeros_like(pre),
                    means, tf.zeros_like(means), p, tf.zeros_like(p), y,
                    model.observation_covariance, None, tf.eye(9,dtype=dtype)/100.,
                    None, substeps=controls["flow_substeps"], eye=tf.eye(d,dtype=dtype),
                    return_affine=True)
        qmean, matrix = flow[4], flow[6]
        qcov = tf.matmul(matrix, matrix, transpose_b=True)
        chol = tf.linalg.cholesky(qcov)
        precision = tf.linalg.cholesky_solve(chol, tf.broadcast_to(tf.eye(d,dtype=dtype),[n,d,d]))
        exact_mean = means + tf.reshape(tf.stack([tf.zeros_like(residual),residual/101.],axis=-1),[n,d])
        cdiag = tf.tile(tf.constant([1.,100./101.],dtype),[9])
        delta = qmean-exact_mean
        mahal = tf.reduce_sum(delta*tf.linalg.matvec(precision,delta),axis=-1)
        kl = .5*(tf.reduce_sum(tf.linalg.diag_part(precision)*cdiag[None],axis=-1)
                 +mahal-d+2.*tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)),axis=-1)
                 -tf.reduce_sum(tf.math.log(cdiag)))
        # Q=I: exact closest-component transition distance, without a rank-3 pair array.
        squared = (tf.reduce_sum(children**2,axis=-1)[:,None]
                   +tf.reduce_sum(means**2,axis=-1)[None]
                   -2.*tf.matmul(children,means,transpose_b=True))
        weights = tf.nn.softmax(logits)
        pdiag = tf.linalg.diag_part(p)[:,1::2]
        # Controlled diagnostic: change only the covariance used to design the
        # shared flow. Keep the same transition samples and marginal correction.
        q = model.process_covariance
        qbatch = tf.broadcast_to(q, [n,d,d])
        shadow = _flow_substeps_with_tangent(model, pre, tf.zeros_like(pre),
                    means, tf.zeros_like(means), qbatch, tf.zeros_like(qbatch), y,
                    model.observation_covariance, None, tf.eye(9,dtype=dtype)/100.,
                    None, substeps=controls["flow_substeps"], eye=tf.eye(d,dtype=dtype),
                    return_affine=True)
        sx, sm, sb = shadow[0], shadow[4], shadow[6]
        ratio, _, valid, force_valid = marginal_prior_ratio_tangent(
                    sx, tf.zeros_like(sx), means, tf.zeros_like(means), q, tf.zeros_like(q),
                    sm, tf.zeros_like(sm), sb, tf.zeros_like(sb),
                    lw, tf.zeros_like(lw), component_policy="marginal_mixture")
        sell = -.5*(9.*tf.math.log(two_pi*100.)+tf.reduce_sum((y-sx[:,1::2])**2,axis=-1)/100.)
        slogits = lw+ratio+sell
        shadow_z = tf.reduce_logsumexp(slogits)
        return dict(actual_log_increment=actual, exact_incoming_cloud_log_increment=exact,
                    flow_log_error=actual-exact,
                    bootstrap_same_cloud_log_increment=boot_z, bootstrap_log_error=boot_z-exact,
                    ledh_ess=1./tf.reduce_sum(weights**2),
                    bootstrap_ess=1./tf.reduce_sum(tf.nn.softmax(lw+boot)**2),
                    exact_conditioned_ancestor_ess=1./tf.reduce_sum(tf.nn.softmax(lw+ell)**2),
                    actual_log_kappa_score_increment=da,
                    exact_incoming_cloud_log_kappa_score_increment=dexact,
                    flow_log_kappa_score_error=da-dexact,
                    ukf_infectious_variance_mean=tf.reduce_mean(pdiag),
                    ukf_infectious_variance_max=tf.reduce_max(pdiag),
                    proposal_infectious_variance_mean=tf.reduce_mean(tf.linalg.diag_part(qcov)[:,1::2]),
                    exact_component_posterior_infectious_variance=tf.constant(100./101.,dtype),
                    component_mean_displacement_rms=tf.sqrt(tf.reduce_mean(delta**2)),
                    true_component_to_flow_component_kl_mean=tf.reduce_mean(kl),
                    true_component_to_flow_component_kl_min=tf.reduce_min(kl),
                    nearest_transition_squared_distance_min=tf.reduce_min(squared),
                    nearest_transition_squared_distance_weighted=tf.reduce_sum(weights*tf.reduce_min(squared,axis=1)),
                    conditional_covariance_shadow_log_increment=shadow_z,
                    conditional_covariance_shadow_log_error=shadow_z-exact,
                    conditional_covariance_shadow_ess=1./tf.reduce_sum(tf.nn.softmax(slogits)**2),
                    conditional_covariance_shadow_valid=tf.cast(valid & force_valid,dtype),
                    conditional_covariance_shadow_mean_error=tf.reduce_max(tf.abs(sm-exact_mean)),
                    conditional_covariance_shadow_covariance_error=tf.reduce_max(tf.abs(
                        tf.matmul(sb,sb,transpose_b=True)-tf.linalg.diag(cdiag)[None])),
                    affine_replay_max_error=tf.reduce_max(tf.abs(flow[0]-children)))

    rows = []
    states, _, ds, _ = spec.initial_cloud(theta, inputs[0], direction)
    with tf.device("/GPU:0"):
        for t, row in enumerate(trace):
            result = diagnose(states,ds,row["pre_flow"],row["children"],row["predicted_covariances"],
                              row["posterior_logits"],row["d_posterior_logits"],
                              row["incoming_log_weights"],row["d_incoming_log_weights"],obs[t])
            record = dict(time=t+1, **{k:float(v) for k,v in result.items()})
            if record["affine_replay_max_error"] > 1e-8:
                raise RuntimeError("affine reconstruction failed")
            if not all(math.isfinite(v) for v in record.values()):
                raise RuntimeError("nonfinite diagnostic")
            if record["conditional_covariance_shadow_valid"] != 1.:
                raise RuntimeError("conditional-covariance shadow invalid")
            rows.append(record)
            write(args.output / "steps.json", rows)
            print(json.dumps(record), flush=True)
            states, ds = row["states_after_reset"], row["d_states_after_reset"]

    result = dict(replay=replay_check, steps=rows,
                  sum_flow_log_error=sum(r["flow_log_error"] for r in rows),
                  sum_bootstrap_log_error=sum(r["bootstrap_log_error"] for r in rows),
                  sum_conditional_covariance_shadow_log_error=sum(
                      r["conditional_covariance_shadow_log_error"] for r in rows),
                  wall_seconds=time.monotonic()-started,
                  ranking="not established; one saved design, conditional-cloud diagnosis",
                  next_action="interpret proposal integration and accumulated cloud error separately")
    manifest.update(wall_seconds=result["wall_seconds"],
                    allocator=tf.config.experimental.get_memory_info("GPU:0"))
    write(args.output / "manifest.json", manifest)
    write(args.output / "result.json", result)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
