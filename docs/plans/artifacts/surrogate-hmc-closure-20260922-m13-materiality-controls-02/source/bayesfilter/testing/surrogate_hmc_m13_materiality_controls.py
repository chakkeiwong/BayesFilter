"""M13 prospective force-control experiment, not posterior admission.

Numerics use the canonical batch-native TensorFlow force and the existing
endpoint-corrected HMC kernel. Saved arrays are diagnostic evidence only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import tensorflow as tf

from bayesfilter.inference import hmc_tensorflow_tuning as native
from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash, write_json
from bayesfilter.inference.surrogate_hmc_dctr_selection import MULTIPLIERS, protection_controls
from bayesfilter.inference.surrogate_hmc_materiality_selection import (
    PARTITIONS, choose, issue_selection, load_selection,
)
from bayesfilter.testing import surrogate_hmc_m13_endpoint_mechanics as mechanics
from bayesfilter.testing.surrogate_hmc_m13_controls import partition_fixture
from bayesfilter.testing.surrogate_hmc_m13_endpoint_materiality import _load
from bayesfilter.testing.surrogate_hmc_ar2_controls import describe, finite_json
from bayesfilter.testing.surrogate_hmc_m13_four_chain_cost import (
    STARTS, native_force, batched_endpoint, bind_components, chunk_health,
)

PLAN = "docs/plans/surrogate-hmc-m13-materiality-controls-2026-09-25.md"
F64, F32 = tf.float64, tf.float32
PROBE_STEPS = (.05, .2)
PROBE_SEEDS = ((2026092501, 100), (2026092501, 101))


def selection_scope(fixture):
    return dict(model="M13", horizon=20, state_dimension=4, observation_dimension=2,
        parameter_dimension=2, particle_count=4096, chunk_extent=2048,
        dtype="float32", tf32=True, jit_compile=True,
        route="canonical_batch_fused_value_score", reset_contract_id="contract_e_chol_v1",
        parameterization="Q=exp(2*lambda_Q)*Q0; R=exp(2*lambda_R)*R0",
        prior="independent N(0,log(2)^2)", prior_support="R^2",
        initial_law="fixed fixture Gaussian", force_design_seed=20260923,
        pilot_data_seed=1704, pilot_design_partition=9,
        pilot_observation_checksum=fixture.observation_checksum,
        control_family="full_canonical_geometric_protection_curve_v1",
        endpoint="full_support_wrapped_trajectory_importance",
        endpoint_particles=4096, endpoint_replicates=8,
        endpoint_seed=[2026092402, 9101], endpoint_dtype="float64",
        coordinate_dtype="float64", calibration_validation=dict(PARTITIONS),
        data_regime="independent T20 calibration/validation; once-rounded FP32 observations",
        use="exploratory_data1704_surrogate_hmc_pilot", confirmation_data_used=False)


def source_hashes(target):
    values = mechanics._source_hashes(target)
    for name in ("bayesfilter/inference/surrogate_hmc_materiality_selection.py",
                 "bayesfilter/testing/surrogate_hmc_m13_materiality_controls.py",
                 "bayesfilter/testing/surrogate_hmc_m13_four_chain_cost.py",
                 "bayesfilter/inference/hmc_tensorflow_tuning.py",
                 "bayesfilter/inference/hmc_tensorflow_verified.py",
                 "bayesfilter/inference/hmc_tuning_dispatch.py",
                 "bayesfilter/inference/tuning_contract.py"):
        values[name] = file_hash(ROOT/name)
    return values


def probe_graph(binding, scope):
    scale = tf.constant(math.log(2.), F64)
    transform = native._TensorFlowAffineTransform(tf.zeros([2], F64), tf.eye(2, dtype=F64)*scale)
    adapter = native._TensorFlowAffineAdapter(transform, scope, scope)

    @tf.function(input_signature=[tf.TensorSpec([4, 2], F64),
        tf.TensorSpec([2], tf.int32), tf.TensorSpec([], F64)], jit_compile=True, autograph=False)
    def run(raw, seed, step):
        kernel = binding.build_tensor_kernel(adapter, step_size=step, num_leapfrog_steps=2,
            target_scope=scope, target_status_trace_policy="none",
            chain_execution_mode="tf_function", use_xla=True)
        return native._run_tensor_steps(kernel=kernel, initial_state=raw/scale,
            num_results=1, seed=seed, adaptive=False)
    return run


def materiality_controls_stage(output):
    out = Path(output).parent
    out.mkdir(parents=True, exist_ok=True)
    if not tf.config.experimental.tensor_float_32_execution_enabled():
        raise ValueError("FP32/TF32 required")
    _, pilot, _ = _load()
    scope = json.loads(json.dumps(selection_scope(pilot)))
    starts = tf.constant(STARTS, F64)
    stress = tf.constant([[a*3*math.log(2.), b*3*math.log(2.)]
                          for a,b in ((-1,-1),(-1,1),(1,-1),(1,1))], F32)
    sources = None
    curves, data_paths = {}, {}
    for partition, (seed, design) in PARTITIONS.items():
        fixture = partition_fixture(partition, attempt=5)
        if fixture.fixture_generation_seed != seed:
            raise ValueError("wrong independent data stream")
        folder = out/partition
        folder.mkdir()
        data_paths[partition] = folder/"data.json"
        write_json(data_paths[partition], dict(scope=scope, partition=partition,
            data_seed=seed, design_partition=design, model=fixture.model_definition(),
            observation_checksum=fixture.observation_checksum,
            observations=fixture.observations.numpy().tolist(), states=fixture.states.numpy().tolist()))
        _, evaluator = mechanics.endpoint_components(fixture)
        endpoint = batched_endpoint(evaluator)
        endpoint_values = endpoint(starts)
        if not bool(tf.reduce_all(endpoint_values.status["value_valid"])):
            raise ValueError("invalid independent endpoint at control starts")
        arms = []
        for index, multiplier in enumerate(MULTIPLIERS):
            begin = time.monotonic()
            target = mechanics.make_range_bearing_target(fixture, particle_count=4096,
                controls=protection_controls(multiplier), dtype=F32,
                design_seed=20260923, design_partition=design)
            current_sources = source_hashes(target)
            if sources is not None and sources != current_sources:
                raise ValueError("control source drift")
            sources = current_sources
            evaluate = target.compile(4, filter_moments=True, all_correction_diagnostics=False)
            nominal = describe(evaluate(tf.cast(starts, F32)))
            stressed = describe(evaluate(stress))
            if not nominal["all_failures_flagged"] or not stressed["all_failures_flagged"]:
                raise ValueError("unflagged numerical failure in control curve")
            identity = f"m13-control-{partition}-{index}:{fixture.observation_checksum}"
            binding = bind_components(native_force(target), endpoint, identity)
            graph = probe_graph(binding, identity)
            probabilities, probes = [], []
            for step in PROBE_STEPS:
                for probe_seed in PROBE_SEEDS:
                    chunk = graph(starts, tf.constant(probe_seed, tf.int32), tf.constant(step, F64))
                    alpha = tf.where(tf.math.is_finite(chunk.log_accept_ratio),
                        tf.exp(tf.minimum(chunk.log_accept_ratio, tf.constant(0., F64))),
                        tf.zeros_like(chunk.log_accept_ratio))
                    probabilities.extend(tf.reshape(alpha, [-1]).numpy().tolist())
                    health = chunk_health(chunk, starts/tf.constant(math.log(2.), F64))
                    # A single valid proposal can reject; movement is a chain-level check.
                    healthy = all(bool(v) for k,v in health.items() if k != "all_chains_moved")
                    fallback = bool(tf.reduce_any(chunk.force_fallback))
                    records = {field: native._write_tensor(folder/f"arm-{index}-step-{step}-seed-{probe_seed[1]}-{field}.tfbin",
                               getattr(chunk, field)) for field in chunk._fields}
                    probes.append(dict(step=step, seed=list(probe_seed), healthy=healthy and not fallback,
                                       force_fallback=fallback, tensors=records))
            arm = dict(multiplier=multiplier, controls=protection_controls(multiplier),
                target=dict(target.manifest), nominal_healthy=nominal["all_healthy"],
                stress_failures_flagged=stressed["all_failures_flagged"],
                stress_healthy_rows=stressed["healthy_rows"], nominal=nominal, stress=stressed,
                proposal_healthy=all(p["healthy"] for p in probes), probes=probes,
                acceptance_probabilities=probabilities, wall_seconds=time.monotonic()-begin)
            write_json(folder/f"arm-{index}.json", finite_json(arm))
            arms.append(arm)
            print(partition, multiplier, "nominal", arm["nominal_healthy"],
                  "proposals", arm["proposal_healthy"], "seconds", arm["wall_seconds"], flush=True)
        curves[partition] = folder/"curve.json"
        write_json(curves[partition], finite_json(dict(scope=scope, source_hashes=sources,
            data_sha256=file_hash(data_paths[partition]), arms=arms,
            probe_steps=list(PROBE_STEPS), probe_leapfrogs=2)))
    index, decisions = choose({p: json.loads(path.read_text()) for p,path in curves.items()})
    report = dict(stage="m13-materiality-controls", plan=PLAN, scope=scope,
        source_hashes=sources, decisions=decisions, posterior_qualified=False,
        exact_score_admission=False, continuation_veto=False, claim_data_evaluated=False,
        heuristic_dominance="posterior/reference and random-walk comparison pending",
        inference_launched=False, all_models_validated=False)
    if index is None:
        return dict(report, status="M13_MATERIALITY_CONTROL_REPAIR_REQUIRED", qualified=False,
                    next_action="Existing master: inspect failure and apply bounded independent-partition repair")
    selection_path = out/"selection.json"
    issue_selection(selection_path, scope=scope, source_hashes=sources, curves=curves, data=data_paths)
    selection = load_selection(selection_path, expected_scope=scope, expected_source_hashes=sources)
    return dict(report, status="M13_MATERIALITY_CONTROLS_SELECTED", qualified=True,
                selection=str(selection_path), multiplier=selection["multiplier"],
                next_action="Existing master: public tuning then retained sampling and cheap baseline")
