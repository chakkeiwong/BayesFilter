"""Exact Gaussian mechanism reference; no DCTR or model-inference claim.

The shared sampler alone implements proposals, acceptance, tuning and archives.
This module supplies deterministic fields and an analytic comparison target.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import sys

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference import (
    FrozenPositionOnlyForce, FrozenTargetPotential,
    DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS,
    bind_neural_force_hmc_tuning_runner, TensorFlowHMCKernelTuningConfig,
    FourChainMeanBandAcceptancePolicy, VerifiedTensorFlowHMCConfig)

ARMS = ("exact", "half", "rotational", "zero")
PLAN = "docs/plans/surrogate-hmc-gaussian-controls-2026-09-20.md"
SCOPE = "C01-anisotropic-gaussian-d3-attempt1"
PRECISION = (1., 4., 9.)
MAX_TRANSITIONS_PER_ARM = 5216
SEEDS = {arm: dict(nomination=(20260920, 7000+100*i), fresh=(20260920, 7001+100*i),
                  retained=[(20260920, 7010+100*i+j) for j in range(8)])
         for i, arm in enumerate(ARMS)}


def source_hashes():
    from bayesfilter.inference.hmc_tensorflow_verified import _source_hashes
    root = Path(__file__).resolve().parents[2]
    names = ["bayesfilter/inference/"+name for name in _source_hashes()]
    names += ["bayesfilter/testing/surrogate_hmc_gaussian_control.py",
              "bayesfilter/inference/surrogate_hmc_equivalence.py",
              "scripts/qualify_surrogate_hmc_gaussian_controls.py",
              "scripts/run_surrogate_hmc_gaussian_controls.py", PLAN]
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def write_report(path, report):
    from bayesfilter.inference.surrogate_hmc_equivalence import finite_report
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(finite_report(report), indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime+usage.ru_stime


def run_provenance(output, memory, purpose):
    root = Path(__file__).resolve().parents[2]
    hashes = source_hashes()
    for name in hashes:
        copy = output/"source"/name
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/name, copy)
    return dict(schema="surrogate_hmc_c01_v1", scope=SCOPE, purpose=purpose,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        command=[sys.executable, *sys.argv], environment=sys.prefix,
        tensorflow=tf.__version__, tensorflow_probability=tfp.__version__,
        source_hashes=hashes, plan=PLAN, result_file=str(output/"report.json"),
        output=str(output), seeds=SEEDS, data_version="analytic Gaussian; no observation data",
        device=tf.config.list_logical_devices("GPU")[0].name,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), gpu=True,
        memory_policy=memory, dtype="float64", jit_compile=True, tf32=False,
        exception="FP64 analytic mechanism control; rank diagnostics use non-XLA graph",
        classification="Gaussian mechanism control; not DCTR model-inference evidence")


def cost_forecast(timings):
    if set(timings) != set(ARMS):
        raise ValueError("cost qualification needs all four fields")
    arms = {}
    for arm, values in timings.items():
        row = {}
        for measure in ("wall_seconds", "cpu_worker_seconds"):
            compile_cost = values["compile"][measure]
            steady = [v[measure]/256 for v in values["steady"]]
            if len(steady) != 2 or not all(math.isfinite(v) and v > 0
                                         for v in [compile_cost, *steady]):
                raise ValueError("invalid measured control cost")
            row[measure] = 1.5*(MAX_TRANSITIONS_PER_ARM*max(steady)+compile_cost+90.)
        arms[arm] = row
    total = {key: sum(row[key] for row in arms.values())
             for key in ("wall_seconds", "cpu_worker_seconds")}
    return dict(arms=arms, total=total, maximum_transitions_per_arm=MAX_TRANSITIONS_PER_ARM,
                graph_analysis_allowance_per_arm_seconds=90., safety_factor=1.5,
                affordable=total["wall_seconds"] <= 900 and total["cpu_worker_seconds"] <= 1000,
                classification="planning estimate, not a measured runtime bound")


def initial_positions():
    return tf.constant([[-2., -1., -1.5], [-1., 2., .5], [1., -2., -.5], [2., 1., 1.5]],
                       tf.float64)/tf.sqrt(tf.constant(PRECISION, tf.float64))


def bind_control(arm):
    if arm not in ARMS:
        raise ValueError("unknown Gaussian control arm")
    precision = tf.constant(PRECISION, tf.float64)
    rotation = tf.constant([[0., -1., 0.], [1., 0., -1.], [0., 1., 0.]], tf.float64)

    @tf.function(input_signature=[tf.TensorSpec([4, 3], tf.float64)], jit_compile=True, autograph=False)
    def field(q):
        if arm == "zero":
            return tf.zeros_like(q)
        value = q*precision
        if arm == "half":
            return .5*value
        if arm == "rotational":
            return value + .2*tf.linalg.matmul(q, rotation, transpose_b=True)
        return value

    @tf.function(input_signature=[tf.TensorSpec([4, 3], tf.float64)], jit_compile=True, autograph=False)
    def potential(q):
        return .5*tf.reduce_sum(q*q*precision, axis=-1)

    signature = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    target = FrozenTargetPotential(potential, identity=SCOPE+":endpoint:"+signature)
    force = FrozenPositionOnlyForce(field, identity=SCOPE+":"+arm+":"+signature,
                                   semantics=DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS)
    binding = bind_neural_force_hmc_tuning_runner(force=force, target=target, target_scope=SCOPE+":"+arm)

    class Adapter:
        target_scope = SCOPE+":"+arm

        def adapter_signature(self):
            return self.target_scope+":"+signature

    return force, target, binding, Adapter()


def tuning_config(arm):
    config = TensorFlowHMCKernelTuningConfig(
        parameter_dimension=3, evidence_role="diagnostic_candidate_screen", mass_window_results=(64, 128),
        step_adaptation_results=128, verification_results=128, max_leapfrog_steps=4, initial_step_size=.2,
        budget_provenance=PLAN+"; bounded C01 mechanism control",
        initial_step_size_provenance="0.2 target-scale hypothesis; own dual averaging follows",
        geometry_provenance="exact marginal scales; own windowed full covariance adaptation",
        target_scope=SCOPE+":"+arm,
        acceptance_policy=FourChainMeanBandAcceptancePolicy(overall_band=(0., 1.), per_chain_band=(0., 1.)),
        target_accept_prob=.7, verification_repair_rounds=0, step_repair_factor=2., mass_shrinkage=.1,
        covariance_jitter=1e-9, eigenvalue_floor=1e-9, max_condition_number=1e8,
        seed=SEEDS[arm]["nomination"], use_xla=True)
    return VerifiedTensorFlowHMCConfig(config, tuple(tuple(row) for row in initial_positions().numpy().tolist()),
        "four dispersed near/far standardized starts frozen in C01 plan", 256, 2000, SEEDS[arm]["fresh"], 1000.)


def reference_contract():
    normal = tfp.distributions.Normal(tf.constant(0., tf.float64), tf.constant(1., tf.float64))
    thresholds = normal.quantile(tf.constant([.1, .5, .9], tf.float64))
    names = ([f"mean_w{j}" for j in range(3)] + [f"second_w{j}" for j in range(3)]
             + [f"cdf_{p}_w{j}" for p in ("10", "50", "90") for j in range(3)])
    return dict(names=names, thresholds=thresholds.numpy().tolist(),
                expectations=[0.]*3+[1.]*3+[.1]*3+[.5]*3+[.9]*3,
                margins=[.1]*6+[.02]*9, allowance=[0.]*15,
                reference="analytic Gaussian moments and marginal CDFs", independent_seed=[20260920, 7999])


@tf.function(input_signature=[tf.TensorSpec([250, 4, 3], tf.float64)], jit_compile=True, autograph=False)
def functions_for_chunk(samples):
    w = samples*tf.sqrt(tf.constant(PRECISION, tf.float64))
    normal = tfp.distributions.Normal(tf.constant(0., tf.float64), tf.constant(1., tf.float64))
    thresholds = normal.quantile(tf.constant([.1, .5, .9], tf.float64))
    functions = tf.concat([w, w*w]+[tf.cast(w <= threshold, tf.float64)
                                   for threshold in tf.unstack(thresholds)], axis=-1)
    return w, functions
