"""Actual numerical stage dispatch for the q20 supervised master."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

from bayesfilter.inference.q20_production_config import digest, validate_protocol, write_json


def run_worker(request, output):
    """Runs inside a time-bounded process; writes early and terminal manifests."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config = validate_protocol(request["config"])
    manifest = {"schema": "bayesfilter.q20.worker_manifest.v1", "stage": request["stage"],
        "started_at": datetime.now(timezone.utc).isoformat(), "request": request,
        "command": sys.argv, "python": sys.executable, "config_hash": digest(config),
        "plan_file": request.get("plan_file"), "status": "initializing",
        "gpu_intentionally_hidden": config["cpu_reference"], "pid": os.getpid(),
        "result_file": str(output / "worker-result.json")}
    write_json(output / "manifest.json", manifest)
    try:
        if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
            raise ValueError("worker requires TF_FORCE_GPU_ALLOW_GROWTH=true before import")
        if config["cpu_reference"]:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=not config["cpu_reference"])
        if request.get("fixture"):
            if config["role"] != "smoke":
                raise ValueError("a known-target fixture cannot replace a serious q20 target")
            from tests.test_q20_production_repair import four_dimensional_bridge
            bridge = four_dimensional_bridge(config["jit_compile"])
        else:
            from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge
            bridge = make_q20_tempered_bridge(config["target"]["q"], jit_compile=config["jit_compile"],
                principal_sqrt_backend=config["target"]["principal_sqrt_backend"])
        from bayesfilter.inference.q20_production_training import source_snapshot
        manifest.update(status="running", tensorflow=tf.__version__, memory_policy=memory,
            target_signature=bridge.target_signature, bridge_signature=bridge.signature,
            sources=source_snapshot(), seeds=config["seed"], jit_compile=config["jit_compile"],
            tf32=tf.config.experimental.tensor_float_32_execution_enabled(),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
        write_json(output / "manifest.json", manifest, exclusive=False)
        if request.get("qualification_path") and config["jit_compile"]:
            from bayesfilter.inference.q20_hmc_qualification import attach_qualification
            bridge = attach_qualification(bridge, request["qualification_path"], config)
        initialization_seconds=time.monotonic()-started
        result = dispatch(config, bridge, output / "data", request, memory)
        if request["stage"] in {"price", "price-training"}:
            result["worker_initialization_seconds"] = initialization_seconds
            write_json(output / "data/result.json", result, exclusive=False)
        final = {"completed": True, "status": result.get("status", "computed"),
                 "result": result, "result_path": str(output / "data/result.json"),
                 "wall_seconds": time.monotonic()-started, "production_qualified": False}
        if not (output / "data/result.json").exists():
            write_json(output / "data/result.json", result)
        write_json(output / "worker-result.json", final)
        manifest.update(status="completed", wall_seconds=final["wall_seconds"])
        write_json(output / "manifest.json", manifest, exclusive=False)
        return final
    except BaseException as error:
        failure = {"completed": False, "status": "worker_failure", "type": type(error).__name__,
                   "message": str(error), "traceback": traceback.format_exc(),
                   "wall_seconds": time.monotonic()-started, "production_qualified": False}
        write_json(output / "worker-result.json", failure)
        manifest.update(status="failed", wall_seconds=failure["wall_seconds"], failure=failure)
        write_json(output / "manifest.json", manifest, exclusive=False)
        raise


def dispatch(config, bridge, root, request, memory):
    stage, root = request["stage"], Path(root)
    if stage == "qualify":
        from bayesfilter.inference.q20_hmc_qualification import qualify_bridge
        return qualify_bridge(config, bridge, root, betas=request.get("betas"))
    if stage == "price":
        return price_complete(config, bridge, root, memory,
            reservation_limit_seconds=request.get("reservation_limit_seconds"))
    if stage == "price-training":
        from bayesfilter.inference.q20_production_training import price_training
        return price_training(config, bridge, root, memory_policy=memory,
            reservation_limit_seconds=request.get("reservation_limit_seconds"), calibration_only=True)
    if stage == "train":
        from bayesfilter.inference.q20_production_training import run_training_cohort
        return run_training_cohort(config, bridge, root, memory_policy=memory,
            max_seconds=request["cooperative_seconds"], resume=request.get("resume_checkpoint"),
            calibration_only=request.get("calibration_only", False))
    if stage == "reference":
        from bayesfilter.inference.q20_production_reference import run_reference
        return run_reference(config, bridge, root, resume_chunks=request.get("resume_chunks"))
    if stage in {"tune", "reverify"}:
        from bayesfilter.inference.q20_production_hmc import tune_scope, reverify_member, draw_start_bank
        beta = request.get("beta", 1.)
        initial, _ = draw_start_bank(config, bridge, beta, request["start_label"])
        if stage == "reverify":
            return reverify_member(config, bridge, root, parent_member_path=request["member_path"],
                beta=beta, label=request["label"], initial_position=initial)
        return tune_scope(config, bridge, root, method=request["method"], beta=beta,
            training_export=request.get("training_export"), initial_position=initial,
            resume=request.get("resume_checkpoint"))
    if stage in {"sample", "ensemble", "replica_exchange"}:
        from bayesfilter.inference.q20_production_hmc import sample_member, sample_ensemble, draw_start_bank
        if stage == "sample":
            sample_member(config, bridge, root, member_path=request["member_path"],
                          label=request["label"], resume_chunks=request.get("resume_chunks"))
        else:
            starts = {str(beta): draw_start_bank(config, bridge, beta, request["start_label"])[0]
                      for beta in config["training"]["betas"]}
            sample_ensemble(config, bridge, root, members_by_beta=request["members_by_beta"],
                label=request["label"], physical_baseline=stage=="replica_exchange",
                initial_by_beta=starts, resume_chunks=request.get("resume_chunks"))
        return json.loads((root / "result.json").read_text())
    if stage == "compare":
        from bayesfilter.inference.q20_production_comparison import compare_campaign
        return compare_campaign(config, root, reference_path=request["reference_path"],
                                posterior_paths=request["posterior_paths"])
    raise ValueError("unknown executable master stage: " + stage)


def price_complete(config, bridge, root, memory, *, reservation_limit_seconds=None):
    """Measure real training, target, HMC/chart/exchange and preparation work.

    Tiny pricing maps measure work only; they are never exported as admitted
    trained maps or tuned kernels. The full-budget calculation is separate.
    """
    import tensorflow as tf
    from bayesfilter.inference.q20_production_training import (
        price_training, training_quote, new_session, scope_for, source_snapshot)
    from bayesfilter.inference.q20_production_hmc import draw_start_bank
    from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import (
        build_fixed_transport_one_step_transition, build_fixed_transport_value_score_adapter)
    from bayesfilter.inference.neutra_training_protocol import export_weighted_transport
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
    from bayesfilter.inference.hmc_kernel_tuning import HMCKernelTuningConfig, prepare_operational_windowed_mass_handoff
    from bayesfilter.inference.hmc import FullChainHMCConfig, ReusableFullChainHMCRunner
    from bayesfilter.inference.q20_hmc_qualification import check_full_chain_health
    from bayesfilter.inference.q20_production_config import scoped_seed
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    priced = price_training(config, bridge, root / "training", memory_policy=memory,
                            reservation_limit_seconds=reservation_limit_seconds)
    if priced["status"] == "training_reservation_exceeds_allowance":
        result = {"status": "unaffordable_under_declared_reservation", "config_hash": digest(config),
            "sources": source_snapshot(), "training": priced, "training_quote": priced["reservation"],
            "reservation_limit_seconds": reservation_limit_seconds,
            "wall_seconds": time.monotonic()-started, "full_campaign_priced": False,
            "production_qualified": False}
        write_json(root / "result.json", result)
        return result
    quote = training_quote(config, priced)
    shape = (config["posterior"]["chains"], bridge.parameter_dim)
    rows = []
    exchange_bindings=[]
    exchange_initial=[]
    from bayesfilter.inference.tempered_transitions_tf import (
        BoundWithinTemperatureKernel,ProperBridgeReplicaExchange,ProperReplicaExchangeTransitionProgram)
    @tf.function(input_signature=(tf.TensorSpec(shape,tf.float64),tf.TensorSpec([2],tf.int32)),
                 jit_compile=config["jit_compile"])
    def prior_refresh(state,seed):
        return bridge.prior_center+tf.sqrt(tf.constant(bridge.prior_variance,tf.float64))*tf.random.stateless_normal(shape,seed,dtype=tf.float64)
    exchange_bindings.append(BoundWithinTemperatureKernel(beta=0.,bridge_signature=bridge.signature,
        kernel_signature=digest(["pricing-prior",bridge.signature]),kernel=prior_refresh,mechanics_role="pricing_only_exact_prior"))
    exchange_initial.append(tf.broadcast_to(bridge.prior_center,shape))
    for beta in config["training"]["betas"][1:]:
        starts, _ = draw_start_bank(config, bridge, beta, "pricing")
        adapter = bridge.fixed_beta_adapter(beta)
        targets = [("physical", adapter, starts)]
        for width in config["training"]["widths"]:
            candidate={"id":f"price-hmc-w{width}","width":width,"learning_rate":config["training"]["learning_rates"][0],"root":0}
            session=new_session(config,bridge,candidate,scope=scope_for(config,bridge,candidate,sources=source_snapshot(),memory_policy=memory))
            # Shape/cost measurement uses the actual supported map composition.
            payload,_=export_weighted_transport(session.trainer.transport, target_signature=adapter.adapter_signature(),
                training_state_hash=session.checkpoint()["state_hash"], transport_id=candidate["id"])
            loaded=load_frozen_neutra_artifact(payload,expected_target_signature=adapter.adapter_signature())
            transformed=build_fixed_transport_value_score_adapter(base_adapter=adapter,fixed_transport=loaded.transport,
                target_scope=adapter.target_scope,evidence_path=adapter.value_score_capability().evidence_path,
                xla_hmc_ready=config["jit_compile"],full_chain_xla_diagnostic_ready=config["jit_compile"])
            targets.append((f"chart-w{width}",transformed,loaded.transport.inverse_theta_to_z_batch(starts)))
        for name,target,state in targets:
            for length in sorted({min(config["tuning"]["l_grid"]),max(config["tuning"]["l_grid"])}):
                count = config["execution"]["pricing_transitions"]
                runner=ReusableFullChainHMCRunner(target,state,FullChainHMCConfig(
                    num_results=count,num_burnin_steps=0,step_size=config["tuning"]["initial_epsilon"],
                    num_leapfrog_steps=length,use_xla=config["jit_compile"],
                    seed=scoped_seed(config,"price-hmc",beta,name,length),target_scope=adapter.target_scope,
                    target_status_trace_policy="per_chain_step",capture_candidate_health=True))
                times=[]
                for index in range(config["budget"]["pricing_updates"]):
                    begin=time.monotonic()
                    result=runner.run(current_state=state,seed=scoped_seed(config,"price-hmc",beta,name,length,index))
                    result.samples.numpy()
                    times.append(time.monotonic()-begin)
                    check_full_chain_health(result)
                rows.append({"beta":beta,"kind":name,"L":length,"first_seconds":times[0],
                    "steady_seconds":max(times[1:] or times)/count,"transitions_per_chain":count,
                    "runner":"public_batched_chain_with_proposal_telemetry"})
                write_json(root / f"hmc-{beta}-{name}-{length}.json",rows[-1])
        # Last target is the widest actual transport; use its measured primitive
        # inside the real full replica-exchange graph for an initial cost quote.
        selected=targets[-1][1]
        transport=selected.transport
        primitive=build_fixed_transport_one_step_transition(selected,state_shape=shape,
            step_size=config["tuning"]["initial_epsilon"],num_leapfrog_steps=max(config["tuning"]["l_grid"]),
            use_xla=config["jit_compile"],capture_health=True)
        def make_physical(transport,primitive):
            @tf.function(input_signature=(tf.TensorSpec(shape,tf.float64),tf.TensorSpec([2],tf.int32)),
                         jit_compile=config["jit_compile"])
            def physical(state,seed):
                z=transport.inverse_theta_to_z_batch(state)
                result=primitive(z,seed)
                valid=result[5]
                return tf.where(valid,transport.forward_batch(result[0]),tf.fill(shape,tf.constant(float("nan"),tf.float64)))
            return physical
        exchange_bindings.append(BoundWithinTemperatureKernel(beta=beta,bridge_signature=bridge.signature,
            kernel_signature=digest(["pricing-chart",beta,selected.adapter_signature()]),
            kernel=make_physical(transport,primitive),mechanics_role="pricing_only_unqualified_chart"))
        exchange_initial.append(starts)
        begin=time.monotonic()
        cfg=HMCKernelTuningConfig.serious(target_scope=adapter.target_scope,use_xla=config["jit_compile"],
            chain_execution_mode="tf_function",target_status_trace_policy="per_chain_step",
            metric_update_requirement="require_operational_update", public_timeout_budget_s=config["tuning"]["max_wall_seconds"])
        prepare_operational_windowed_mass_handoff(adapter=adapter,initial_position=starts[0],config=cfg,
            parameter_scales=tf.fill([bridge.parameter_dim],tf.constant(4.,tf.float64)))
        rows.append({"beta":beta,"kind":"classical_preparation","wall_seconds":time.monotonic()-begin})
    # Batch reference work measured directly; no optimizer-time surrogate.
    batch=config["reference"]["batch_size"]
    points=tf.broadcast_to(bridge.prior_center,[batch,bridge.parameter_dim])
    @tf.function(input_signature=(tf.TensorSpec([batch,bridge.parameter_dim],tf.float64),),
                 jit_compile=config["jit_compile"],reduce_retracing=False)
    def reference_batch(points):
        return bridge.component_terms(points)[0]
    times=[]
    for _ in range(config["budget"]["pricing_updates"]):
        begin=time.monotonic()
        likelihood=reference_batch(points)
        likelihood.numpy()
        times.append(time.monotonic()-begin)
    from bayesfilter.inference.q20_production_config import scoped_seed
    exchange=ProperReplicaExchangeTransitionProgram(ProperBridgeReplicaExchange(bridge,config["training"]["betas"]),
        exchange_bindings,jit_compile=config["jit_compile"])
    initial=exchange.initial_state(tf.stack(exchange_initial))
    exchange_times=[]
    count=config["execution"]["pricing_transitions"]
    for i in range(config["budget"]["pricing_updates"]):
        begin=time.monotonic()
        output=exchange(initial,num_results=count,seed=scoped_seed(config,"pricing-exchange",i),stage="warmup")
        # Materialize actual cold stream before stopping the timer.
        for item in tf.nest.flatten(output):
            if tf.is_tensor(item):
                item.numpy()
        exchange_times.append((time.monotonic()-begin)/count)
    from bayesfilter.inference.q20_production_reference import importance_summary
    from bayesfilter.inference.q20_production_comparison import posterior_summary
    with tf.device("/CPU:0"):
        reference_fixture=tf.random.stateless_normal([config["reference"]["banks"],max(config["reference"]["rungs"]),bridge.parameter_dim],
            scoped_seed(config,"pricing-reference-analysis"),dtype=tf.float64)
        posterior_fixture=tf.random.stateless_normal([config["posterior"]["retained_max"],config["posterior"]["chains"],bridge.parameter_dim],
            scoped_seed(config,"pricing-posterior-analysis"),dtype=tf.float64)
    begin=time.monotonic()
    importance_summary(config,reference_fixture,tf.zeros(reference_fixture.shape[:2],tf.float64))
    reference_analysis=(time.monotonic()-begin)/(config["reference"]["banks"]*max(config["reference"]["rungs"]))
    begin=time.monotonic()
    posterior_summary(config,posterior_fixture,target_signature="pricing_fixture",label="pricing",sequential_passed=False)
    posterior_analysis=time.monotonic()-begin
    result={"status":"complete_stage_cost_measurements","config_hash":digest(config),
        "sources":source_snapshot(),"training":priced,"training_quote":quote,"hmc":rows,
        "reference_batch_first_seconds":times[0],"reference_batch_steady_seconds":max(times[1:] or times),
        "reference_batch_size":batch,"wall_seconds":time.monotonic()-started,
        "exchange_first_seconds":exchange_times[0],"exchange_steady_seconds":max(exchange_times[1:] or exchange_times),
        "reference_analysis_seconds_per_row":reference_analysis,"posterior_analysis_seconds":posterior_analysis,
        "forecast_role":"initial_observed_costs_with_unmeasured_tail_risk","production_qualified":False}
    write_json(root / "result.json",result)
    return result
