"""Measured full-chain XLA qualification for the exact fixed-beta bridge.

Qualification concerns graph compatibility and checked endpoint numerics. It
does not qualify a kernel, a learned map or a posterior. Only the measured
repository procedure below issues the receipt consumed by the adapter.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time

import tensorflow as tf

from bayesfilter.inference.tempered_target_tf import FixedBetaBridgeAdapter
from bayesfilter.inference.q20_production_config import digest, scoped_seed, write_json
from bayesfilter.inference.q20_production_training import source_snapshot


def check_full_chain_health(result):
    """Check real public-runner endpoint/proposal telemetry, including rejects."""
    from bayesfilter.inference.hmc_verification import target_status_telemetry_has_failure
    trace = result.trace
    for key in ("log_accept_ratio", "target_log_prob", "proposed_target_log_prob",
                "proposed_state"):
        tf.debugging.assert_all_finite(trace[key], "HMC qualification: " + key)
    tf.debugging.assert_all_finite(result.samples, "HMC qualification samples")
    tf.debugging.assert_equal(tf.reduce_all(trace["target_score_finite"]), True)
    for key in ("target_status_telemetry", "proposed_target_status_telemetry"):
        if target_status_telemetry_has_failure(trace[key], expected_shape=tuple(result.samples.shape[:2])):
            raise ValueError("HMC qualification target status failed")


def qualify_bridge(config, bridge, root, *, betas=None):
    from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import build_fixed_transport_one_step_transition
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    shape = (config["posterior"]["chains"], bridge.parameter_dim)
    count = config["execution"]["pricing_transitions"]
    evidence = {}
    temperatures = config["training"]["betas"][1:] if betas is None else betas
    if not temperatures or any(b not in config["training"]["betas"][1:] for b in temperatures):
        raise ValueError("qualification requires declared positive temperatures")
    for beta in temperatures:
        adapter = FixedBetaBridgeAdapter(bridge, beta=beta)
        # Diagnostic starts use the same prior proposal policy as all methods.
        from bayesfilter.inference.q20_production_hmc import draw_start_bank
        starts, _ = draw_start_bank(config, bridge, beta, "qualification")
        step = build_fixed_transport_one_step_transition(adapter, state_shape=shape,
            step_size=config["tuning"]["initial_epsilon"], num_leapfrog_steps=config["tuning"]["l_grid"][0],
            use_xla=config["jit_compile"], capture_health=True)
        @tf.function(input_signature=(tf.TensorSpec(shape,tf.float64),tf.TensorSpec([2],tf.int32)),
                     jit_compile=config["jit_compile"], reduce_retracing=False)
        def program(initial, seed):
            def body(i,state,healthy):
                next_state,accepted,log_accept,value,score,proposal_health = step(state,tf.random.experimental.stateless_fold_in(seed,i))
                _,_,status = adapter.log_prob_and_grad_status(next_state)
                valid = (tf.reduce_all(status["valid_pre_regularized_score"]) &
                         tf.reduce_all(tf.math.is_finite(log_accept)) & tf.reduce_all(tf.math.is_finite(score)) &
                         tf.reduce_all(tf.math.is_finite(value)) & tf.reduce_all(tf.math.is_finite(next_state)))
                return i+1,next_state,healthy & valid & proposal_health
            return tf.while_loop(lambda i,*_:i<count,body,(tf.constant(0),initial,tf.constant(True)),parallel_iterations=1)
        started=time.monotonic()
        _,endpoint,valid=program(starts,tf.constant(scoped_seed(config,"qualification",beta),tf.int32))
        endpoint.numpy()
        first=time.monotonic()-started
        if not bool(valid):
            raise ValueError("q20 XLA qualification trajectory has invalid target/transition")
        # Compare the enclosing graph's final endpoint evaluation with a direct
        # batch-native call. Independent derivative checks remain separate.
        value,score,status=adapter.log_prob_and_grad_status(endpoint)
        reference_value,reference_score,reference_status=bridge._value_score_status_impl(endpoint,tf.constant(beta,tf.float64))
        tf.debugging.assert_near(value,reference_value,rtol=config["validation"]["reliability_rtol"],atol=config["validation"]["reliability_atol"])
        tf.debugging.assert_near(score,reference_score,rtol=config["validation"]["reliability_rtol"],atol=config["validation"]["reliability_atol"])
        started=time.monotonic()
        _,second,valid=program(endpoint,tf.constant(scoped_seed(config,"qualification-repeat",beta),tf.int32))
        second.numpy()
        elapsed=time.monotonic()-started
        if not bool(valid):
            raise ValueError("q20 steady qualification trajectory invalid")
        # The preliminary enclosing-graph check licenses only this diagnostic
        # use. No external receipt is issued until the actual scalar-chain
        # public runner and its proposal/status traces have also passed.
        from bayesfilter.inference.hmc import FullChainHMCConfig, build_independent_chain_tfp_hmc_runner
        diagnostic_adapter = _QualifiedAdapter(bridge, beta=beta)
        runner = build_independent_chain_tfp_hmc_runner(diagnostic_adapter, starts,
            FullChainHMCConfig(num_results=count, num_burnin_steps=0,
                step_size=config["tuning"]["initial_epsilon"],
                num_leapfrog_steps=config["tuning"]["l_grid"][0],
                seed=scoped_seed(config,"qualification-public",beta),
                use_xla=config["jit_compile"], target_scope=adapter.target_scope,
                target_status_trace_policy="per_chain_step",capture_candidate_health=True))
        begin=time.monotonic()
        public=runner.run(mode="serial")
        check_full_chain_health(public)
        public_seconds=time.monotonic()-begin
        evidence[str(beta)]={"passed":True,"adapter_signature":adapter.adapter_signature(),
            "first_seconds":first,"steady_seconds":elapsed,"transitions_per_chain":count,
            "chains":shape[0],"leapfrog_steps":config["tuning"]["l_grid"][0],"step_size":config["tuning"]["initial_epsilon"],
            "endpoint_device":endpoint.device,"traces":program.experimental_get_tracing_count(),
            "public_serial_runner_passed":True,"public_serial_runner_seconds":public_seconds}
    body={"schema":"bayesfilter.q20.bridge_hmc_qualification.v1","bridge_signature":bridge.signature,
        "sources":source_snapshot(),"jit_compile":config["jit_compile"],"cpu_reference":config["cpu_reference"],
        "betas":evidence,"full_chain_role":"numerical_graph_diagnostic_only","passed":True}
    write_json(root/"result.json",{**body,"checksum":digest(body)})
    return body


class _QualifiedAdapter(FixedBetaBridgeAdapter):
    def value_score_capability(self):
        base=super().value_score_capability()
        return replace(base,full_chain_xla_diagnostic_ready=base.xla_hmc_ready,
                       evidence_path="docs/plans/bayesfilter-ssl-lstm-q20-executable-master-repair-plan-2026-09-16.md")


def attach_qualification(bridge, path, config):
    """Validate measured receipt before exposing the full-chain capability."""
    record=json.loads(Path(path).read_text())
    checksum=record.pop("checksum")
    if checksum!=digest(record) or record["sources"]!=source_snapshot() or record["bridge_signature"]!=bridge.signature:
        raise ValueError("stale/mismatched q20 HMC qualification")
    if record["jit_compile"]!=config["jit_compile"] or record["cpu_reference"]!=config["cpu_reference"]:
        raise ValueError("qualification backend differs")
    if set(record["betas"]) != {str(float(b)) for b in config["training"]["betas"][1:]}:
        raise ValueError("qualification temperature inventory differs")
    if not record["passed"] or any(not v["passed"] or v["traces"]!=1 or not v.get("public_serial_runner_passed") for v in record["betas"].values()):
        raise ValueError("qualification failed")
    if not config["cpu_reference"] and any("GPU" not in v["endpoint_device"] for v in record["betas"].values()):
        raise ValueError("serious qualification requires actual GPU endpoints")
    def fixed(beta):
        row=record["betas"].get(str(float(beta)))
        adapter=_QualifiedAdapter(bridge,beta=beta)
        if row is None or row["adapter_signature"]!=adapter.adapter_signature():
            raise ValueError("unqualified temperature scope")
        return adapter
    bridge.fixed_beta_adapter=fixed
    return bridge
