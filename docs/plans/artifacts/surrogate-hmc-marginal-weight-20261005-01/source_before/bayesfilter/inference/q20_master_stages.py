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

from bayesfilter.inference.q20_production_config import digest, validate_protocol, method_betas, write_json
from bayesfilter.inference.q20_gpu_runtime import GPUResourceUnavailable, select_worker_gpu, check_gpu_contention
from bayesfilter.inference.q20_stage_budget import StageBudgetPause


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
        else:
            manifest["launch_readiness"] = select_worker_gpu(requested=request.get("gpu", "auto"))
            manifest["cuda_visible_devices"] = os.environ["CUDA_VISIBLE_DEVICES"]
            write_json(output / "manifest.json", manifest, exclusive=False)
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
            bridge = attach_qualification(bridge, request["qualification_path"], config,
                                          betas=request.get("qualification_betas"))
        initialization_seconds=time.monotonic()-started
        request = {**request, "resource_readiness": manifest.get("launch_readiness")}
        if "cooperative_seconds" in request:
            request = {**request, "cooperative_seconds": max(0.,
                request["cooperative_seconds"]-initialization_seconds)}
        try:
            result = dispatch(config, bridge, output / "data", request, memory)
        except StageBudgetPause as error:
            (output / "data").mkdir(exist_ok=True)
            result = {"status": "budget_paused", "reason": str(error),
                      "checkpoint_directory": str(output / "data"), "production_qualified": False}
        if request["stage"] in {"price", "price-selected", "price-training", "price-downstream", "price-preparation"}:
            result["worker_initialization_seconds"] = initialization_seconds
            write_json(output / "data/result.json", result, exclusive=False)
        final = {"completed": True, "status": result.get("status", "computed"),
                 "result": result, "result_path": str(output / "data/result.json"),
                 "wall_seconds": time.monotonic()-started, "production_qualified": False}
        if not (output / "data/result.json").exists():
            write_json(output / "data/result.json", result)
        write_json(output / "worker-result.json", final)
        from bayesfilter.inference.q20_pricing import resource_observation
        observation = resource_observation(None if config["cpu_reference"] else
            lambda: check_gpu_contention(manifest["launch_readiness"]))
        final.update(numerical_completed=result.get("status") != "budget_paused", terminal_resource=observation)
        write_json(output / "worker-result.json", final, exclusive=False)
        manifest["terminal_capacity"] = observation
        manifest.update(status="completed", wall_seconds=final["wall_seconds"])
        write_json(output / "manifest.json", manifest, exclusive=False)
        return final
    except GPUResourceUnavailable as error:
        failure = {"completed": False, "status": "waiting_for_gpu", "resource_receipt": error.receipt,
                   "wall_seconds": time.monotonic()-started, "production_qualified": False}
        write_json(output / "worker-result.json", failure)
        manifest.update(status="waiting_for_gpu", wall_seconds=failure["wall_seconds"], failure=failure)
        write_json(output / "manifest.json", manifest, exclusive=False)
        return failure
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
    deadline = None if "cooperative_seconds" not in request else time.monotonic()+request["cooperative_seconds"]
    def remaining_seconds():
        return None if deadline is None else max(0., deadline-time.monotonic())
    if config["role"] != "smoke" and (stage in {"price-preparation", "replica_exchange", "compare"}
            or (stage == "tune" and request.get("method") != "neutra")):
        raise ValueError("active q20 estimation excludes classical preparation and method comparisons")
    if stage == "migrate-training":
        from bayesfilter.inference.q20_checkpoint_migration import migrate_training_checkpoint
        from bayesfilter.inference.q20_training_resume import checksum
        if checksum(request["training_checkpoint"]) != request["previous_sha256"]:
            raise ValueError("historical checkpoint changed after migration audit")
        return migrate_training_checkpoint(config, bridge, root,
            previous_config=request["previous_config"], checkpoint=request["training_checkpoint"],
            previous_root=request["previous_source_root"], memory_policy=memory,
            reference_bridge=bridge if request.get("fixture") else None)
    if stage == "qualify":
        from bayesfilter.inference.q20_hmc_qualification import qualify_bridge
        return qualify_bridge(config, bridge, root, betas=request.get("betas"))
    if stage == "repair-training":
        from bayesfilter.inference.q20_training_repair import run_repair_arm
        return run_repair_arm(config, bridge, root, request=request, memory_policy=memory)
    if stage == "price":
        return price_complete(config, bridge, root, memory,
            reservation_limit_seconds=request.get("reservation_limit_seconds"),
            checkpoint=request.get("training_checkpoint"), method=request.get("method", "neutra"),
            resume=request.get("pricing_resume"), max_seconds=remaining_seconds(),
            historical_pricing=request.get("historical_pricing"),
            resource_readiness=request.get("resource_readiness"),
            block_reserves=request.get("block_reserves", {}))
    if stage == "price-downstream":
        from bayesfilter.inference.q20_training_resume import read_training_checkpoint
        from bayesfilter.inference.q20_production_training import source_snapshot
        checkpoint = request.get("training_checkpoint")
        if checkpoint:
            read_training_checkpoint(checkpoint, config, sources=source_snapshot())
        return price_complete(config, bridge, root, memory,
            checkpoint=checkpoint, training_pricing=json.loads(Path(request["training_pricing"]).read_text()),
            skip_preparation=True, method=request.get("method", "neutra"))
    if stage == "price-preparation":
        return price_preparation(config, bridge, root, beta=request["beta"],
            max_seconds=request["max_seconds"], bootstrap_resume=request.get("bootstrap_resume"),
            bootstrap_transition_seconds=request.get("bootstrap_transition_seconds"),
            allow_deferred=request.get("allow_deferred", False))
    if stage == "price-training":
        from bayesfilter.inference.q20_production_training import price_training, source_snapshot, scope_for, training_config
        restored = []
        if request.get("training_checkpoint"):
            from bayesfilter.inference.q20_training_resume import read_training_checkpoint
            from bayesfilter.inference.neutra_training_protocol import TrainingSession
            from bayesfilter.inference.q20_production_config import scoped_seed
            state = read_training_checkpoint(request["training_checkpoint"], config, sources=source_snapshot())
            for name, item in state["cohort"].items():
                saved = item["session"]
                candidate = saved["scope"]["candidate"]
                scope = scope_for(config, bridge, candidate, sources=saved["scope"]["sources"], memory_policy=memory)
                session = TrainingSession.restore(saved, bridge=bridge, config=training_config(config, candidate),
                    expected_scope=scope, preflight_seed=scoped_seed(config, "preflight", name, saved["map"]["beta"]))
                checked = session.checkpoint()
                if any(checked[key] != saved[key] for key in ("optimizer", "rng_index", "iteration", "history", "map")):
                    raise ValueError("restored checkpoint differs: " + name)
                restored.append({"candidate": name, "updates": saved["level_updates"], "exact_state_restore": True})
        result = price_training(config, bridge, root, memory_policy=memory,
            reservation_limit_seconds=request.get("reservation_limit_seconds"), calibration_only=True,
            method=request.get("method"))
        result["restored_training_states"] = restored
        return result
    if stage == "train":
        from bayesfilter.inference.q20_production_training import run_training_cohort
        return run_training_cohort(config, bridge, root, memory_policy=memory,
            max_seconds=request["cooperative_seconds"], resume=request.get("resume_checkpoint"),
            calibration_only=request.get("calibration_only", False), method=request.get("method"),
            stop_after_rung=request.get("stop_after_rung"),
            stop_when_trial_ready=request.get("stop_when_trial_ready", False))
    if stage == "reference":
        from bayesfilter.inference.q20_production_reference import run_reference
        return run_reference(config, bridge, root, resume_chunks=request.get("resume_chunks"),
            max_seconds=request.get("cooperative_seconds"), chunk_reserve_seconds=request.get("chunk_reserve_seconds", 0.))
    if stage in {"tune", "reverify"}:
        from bayesfilter.inference.q20_production_hmc import tune_scope, reverify_member, draw_start_bank
        beta = request.get("beta", 1.)
        initial, _ = draw_start_bank(config, bridge, beta, request["start_label"])
        if stage == "reverify":
            return reverify_member(config, bridge, root, parent_member_path=request["member_path"],
                beta=beta, label=request["label"], initial_position=initial)
        return tune_scope(config, bridge, root, method=request["method"], beta=beta,
            training_export=request.get("training_export"), initial_position=initial,
            resume=request.get("resume_checkpoint"), max_seconds=remaining_seconds(),
            explicit_cohort=request.get("explicit_cohort"))
    if stage == "price-selected":
        from bayesfilter.inference.q20_production_hmc import sample_member, sample_ensemble
        common = dict(label=request["label"], pricing_only=True)
        if request["method"] == "neutra":
            return sample_member(config, bridge, root, member_path=request["member_path"], **common)
        from bayesfilter.inference.q20_production_hmc import draw_start_bank
        initial = {str(beta): draw_start_bank(config, bridge, beta, request["start_label"])[0]
                   for beta in config["training"]["betas"]}
        return sample_ensemble(config, bridge, root, members_by_beta=request["members_by_beta"],
                               initial_by_beta=initial, **common)
    if stage in {"sample", "ensemble", "replica_exchange"}:
        from bayesfilter.inference.q20_production_hmc import sample_member, sample_ensemble, draw_start_bank
        if stage == "sample":
            sample_member(config, bridge, root, member_path=request["member_path"],
                          label=request["label"], resume_chunks=request.get("resume_chunks"),
                          max_seconds=request.get("cooperative_seconds"), chunk_reserve_seconds=request.get("chunk_reserve_seconds", 0.))
        else:
            starts = {str(beta): draw_start_bank(config, bridge, beta, request["start_label"])[0]
                      for beta in config["training"]["betas"]}
            sample_ensemble(config, bridge, root, members_by_beta=request["members_by_beta"],
                label=request["label"], physical_baseline=stage=="replica_exchange",
                initial_by_beta=starts, resume_chunks=request.get("resume_chunks"),
                max_seconds=remaining_seconds(), chunk_reserve_seconds=request.get("chunk_reserve_seconds", 0.))
        return json.loads((root / "result.json").read_text())
    if stage == "assess":
        from bayesfilter.inference.q20_production_comparison import assess_estimate
        return assess_estimate(config, root, reference_path=request["reference_path"],
                              posterior_path=request["posterior_path"], method=request["method"])
    raise ValueError("unknown executable master stage: " + stage)


def price_preparation(config, bridge, root, *, beta, max_seconds, bootstrap_resume=None,
                      bootstrap_transition_seconds=None, allow_deferred=False):
    if config["role"] != "smoke":
        raise ValueError("classical preparation is outside the active NeuTra estimation objective")
    import tensorflow as tf
    from bayesfilter.inference.hmc_kernel_tuning import HMCKernelTuningConfig, prepare_operational_windowed_mass_handoff
    from bayesfilter.inference.hmc_preparation import HMCPreparationProgress, HMCPreparationBudgetExceeded
    from bayesfilter.inference.hmc_bootstrap_checkpoint import CheckpointedBootstrapRunner
    from bayesfilter.inference.q20_production_training import source_snapshot
    from bayesfilter.inference.q20_production_hmc import draw_start_bank
    root.mkdir(parents=True, exist_ok=False)
    starts, receipt = draw_start_bank(config, bridge, beta, "pricing")
    adapter = bridge.fixed_beta_adapter(beta)
    cfg = HMCKernelTuningConfig.serious(target_scope=adapter.target_scope,
        use_xla=config["jit_compile"], chain_execution_mode="tf_function",
        target_status_trace_policy="per_chain_step", metric_update_requirement="require_operational_update",
        public_timeout_budget_s=min(config["tuning"]["max_wall_seconds"], max_seconds))
    write_json(root / "preparation-policy.json", cfg.payload())
    from bayesfilter.inference.hmc_bootstrap_initialization import POLICY
    write_json(root / "bootstrap-initialization-policy.json", {"policy": POLICY, "enabled": True})
    write_json(root / "starts.json", {**receipt, "positions": starts.numpy().tolist()})
    began = time.monotonic()
    progress = HMCPreparationProgress(root, max_wall_time_seconds=max_seconds)
    execution = CheckpointedBootstrapRunner(root / "bootstrap-checkpoints", sources=source_snapshot(),
        progress=progress, seconds_per_transition=bootstrap_transition_seconds,
        safety_factor=config["budget"]["forecast_safety_factor"], resume_from=bootstrap_resume)
    try:
        with progress:
            prepare_operational_windowed_mass_handoff(adapter=adapter, initial_position=starts[0],
                config=cfg, parameter_scales=tf.fill([bridge.parameter_dim], tf.constant(4., tf.float64)),
                progress_callback=progress.phase, initialize_bootstrap=True, bootstrap_execution=execution)
    except HMCPreparationBudgetExceeded:
        if not allow_deferred:
            raise
        startup = execution.root / "bootstrap-result.json"
        return {"status": "preparation_deferred_by_cost", "beta": beta,
            "kind": "classical_preparation_deferred", "wall_seconds": time.monotonic()-began,
            "bootstrap_passed": startup.exists() and json.loads(startup.read_text())["passed"],
            "checkpoint_dir": str(execution.root), "bootstrap_result": str(startup) if startup.exists() else None,
            "seconds_per_transition": execution.seconds_per_transition,
            "preparation_progress_file": str(progress.path), "production_qualified": False,
            "mass_adaptation_completed": False}
    return {"status": "preparation_cost_measured", "beta": beta, "kind": "classical_preparation",
            "wall_seconds": time.monotonic()-began, "production_qualified": False,
            "preparation_progress_file": str(root / "preparation_progress.json")}


def price_complete(config, bridge, root, memory, *, reservation_limit_seconds=None,
                   checkpoint=None, training_pricing=None, skip_preparation=False, method="neutra",
                   resume=None, max_seconds=None, historical_pricing=None, resource_readiness=None,
                   block_reserves=None):
    """Price one estimation method, with no classical preparation dependency.

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
    from bayesfilter.inference.hmc import FullChainHMCConfig, ReusableFullChainHMCRunner
    from bayesfilter.inference.q20_hmc_qualification import check_full_chain_health
    from bayesfilter.inference.q20_production_config import scoped_seed
    root.mkdir(parents=True, exist_ok=False)
    positive_betas = method_betas(config, method)
    started = time.monotonic()
    from bayesfilter.inference.q20_pricing import PricingLedger, checked_historical_price
    from bayesfilter.inference.q20_training_resume import checksum
    history, historical_origin = (None, None) if historical_pricing is None else checked_historical_price(
        historical_pricing, config, Path(__file__).resolve().parents[2])
    if history is not None and history["method"] != method:
        raise ValueError("historical pricing belongs to another method")
    ledger = PricingLedger(root / "pricing-ledger", {"config": digest(config), "method": method,
        "sources": source_snapshot(), "bridge": bridge.signature, "memory_mode": memory.get("mode"),
        "device_class": "CPU_reference" if resource_readiness is None else next(
            g["name"] for g in resource_readiness["gpu_inventory"] if g["index"] == resource_readiness["selected_host_gpu"]),
        "checkpoint": None if checkpoint is None else checksum(checkpoint),
        "historical_receipt": None if historical_pricing is None else checksum(historical_pricing)},
        resume=resume, deadline=None if max_seconds is None else started+max_seconds,
        probe=None if resource_readiness is None else lambda: check_gpu_contention(resource_readiness))
    block_reserves = block_reserves or {}
    def measured(key, inputs, compute, old=None):
        return ledger.run(key, inputs, compute, reserve_seconds=block_reserves.get(key, 0.),
                          imported=None if old is None else (old, historical_origin))
    cohort = {}
    if checkpoint is not None:
        from bayesfilter.inference.q20_campaign_costs import training_reservation
        from bayesfilter.inference.q20_training_resume import read_training_checkpoint
        cohort = read_training_checkpoint(checkpoint, config, sources=source_snapshot())["cohort"]
    priced = training_pricing or measured("training", {"method": method},
        lambda: price_training(config, bridge, root / "training", memory_policy=memory,
                            reservation_limit_seconds=reservation_limit_seconds,
                            method=method, checkpoint=checkpoint), None if history is None else history["training"])
    if priced.get("method") != method:
        raise ValueError("training pricing belongs to a different estimation method")
    if priced["status"] == "training_reservation_exceeds_allowance":
        result = {"status": "unaffordable_under_declared_reservation", "config_hash": digest(config), "method": method,
            "sources": source_snapshot(), "training": priced, "training_quote": priced["reservation"],
            "reservation_limit_seconds": reservation_limit_seconds,
            "wall_seconds": time.monotonic()-started, "full_campaign_priced": False,
            "production_qualified": False}
        write_json(root / "result.json", result)
        return result
    from bayesfilter.inference.q20_campaign_costs import training_reservation
    quote = training_reservation(config, priced["rows"], checkpoint=checkpoint, method=method)
    if reservation_limit_seconds is not None and quote["minimum_cohort_seconds"] > reservation_limit_seconds:
        result = {"status": "unaffordable_under_declared_reservation", "config_hash": digest(config), "method": method,
            "sources": source_snapshot(), "training": priced, "training_quote": quote,
            "reservation_limit_seconds": reservation_limit_seconds,
            "wall_seconds": time.monotonic()-started, "full_campaign_priced": False, "production_qualified": False}
        write_json(root / "result.json", result)
        return result
    shape = (config["posterior"]["chains"], bridge.parameter_dim)
    rows = []
    map_prices = []
    exchange_bindings=[]
    exchange_initial=[]
    from bayesfilter.inference.tempered_transitions_tf import (
        BoundWithinTemperatureKernel,ProperBridgeReplicaExchange,ProperReplicaExchangeTransitionProgram,
        FixedChartKernelMixture)
    @tf.function(input_signature=(tf.TensorSpec(shape,tf.float64),tf.TensorSpec([2],tf.int32)),
                 jit_compile=config["jit_compile"])
    def prior_refresh(state,seed):
        return bridge.prior_center+tf.sqrt(tf.constant(bridge.prior_variance,tf.float64))*tf.random.stateless_normal(shape,seed,dtype=tf.float64)
    exchange_bindings.append(BoundWithinTemperatureKernel(beta=0.,bridge_signature=bridge.signature,
        kernel_signature=digest(["pricing-prior",bridge.signature]),kernel=prior_refresh,mechanics_role="pricing_only_exact_prior"))
    exchange_initial.append(tf.broadcast_to(bridge.prior_center,shape))
    for beta in positive_betas:
        if history is not None:
            for old in history["hmc"]:
                if old["beta"] == beta:
                    key = f"hmc-{beta}-{old['kind']}-{old['L']}"
                    rows.append(measured(key, {"beta": beta, "kind": old["kind"], "L": old["L"]},
                        None, {**old, "timing_origin": historical_origin, "role": "historical_cost_hypothesis"}))
            if method == "neutra":
                map_prices.extend(history["map_prices"])
                continue
        starts, _ = draw_start_bank(config, bridge, beta, "pricing")
        adapter = bridge.fixed_beta_adapter(beta)
        targets = []
        chart_targets = []
        template_width = min(config["training"]["widths"])
        for width in ([template_width] if history is not None else config["training"]["widths"]):
            width_targets = []
            for chart in range(config["ensemble"]["charts"] if method == "ensemble" else 1):
                chart_root = config["training"]["roots"][chart]
                saved = next((item["session"] for name,item in sorted(cohort.items())
                              if item["session"]["scope"]["candidate"]["width"] == width
                              and item["session"]["scope"]["candidate"]["root"] == chart_root
                              and item["session"]["scope"]["candidate"]["schedule"] ==
                                  ("direct" if method == "neutra" else "continuation")
                              and item["session"]["map"]["beta"] == beta), None)
                if saved is None:
                    candidate={"id":f"price-hmc-w{width}-r{chart_root}","width":width,
                        "learning_rate":config["training"]["learning_rates"][0],"root":chart_root}
                    session=new_session(config,bridge,candidate,scope=scope_for(config,bridge,candidate,sources=source_snapshot(),memory_policy=memory))
                    transport, state_hash = session.trainer.transport, session.checkpoint()["state_hash"]
                    map_origin = {"role": "initialization_map_pricing_only", "candidate": candidate["id"]}
                else:
                    from bayesfilter.inference.tempered_transport_ensemble_tf import restore_trainable_transport_checkpoint
                    transport = restore_trainable_transport_checkpoint(saved["map"], expected_context={
                        "bridge_signature": bridge.signature, "target_signature": bridge.target_signature})
                    state_hash = saved["state_hash"]
                    map_origin = {"role": "unqualified_saved_map_pricing_only", "candidate": saved["scope"]["candidate"]["id"],
                              "updates": saved["level_updates"], "state_hash": state_hash}
                payload,_=export_weighted_transport(transport, target_signature=adapter.adapter_signature(),
                    training_state_hash=state_hash, transport_id=f"pricing-beta{beta}-w{width}-r{chart_root}")
                map_prices.append({"beta": beta, "width": width, "root": chart_root, **map_origin})
                loaded=load_frozen_neutra_artifact(payload,expected_target_signature=adapter.adapter_signature())
                transformed=build_fixed_transport_value_score_adapter(base_adapter=adapter,fixed_transport=loaded.transport,
                    target_scope=adapter.target_scope,evidence_path=adapter.value_score_capability().evidence_path,
                    xla_hmc_ready=config["jit_compile"],full_chain_xla_diagnostic_ready=config["jit_compile"])
                width_targets.append(transformed)
                if chart == 0:
                    targets.append((f"chart-w{width}",transformed,loaded.transport.inverse_theta_to_z_batch(starts)))
            if width == template_width:
                chart_targets = width_targets
        if history is not None:
            targets = []
        for name,target,state in targets:
            for length in sorted({min(config["tuning"]["l_grid"]),max(config["tuning"]["l_grid"])}):
                count = config["execution"]["pricing_transitions"]
                def compute_hmc():
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
                    return {"beta":beta,"kind":name,"L":length,"first_seconds":times[0],
                        "steady_seconds":max(times[1:] or times)/count,"transitions_per_chain":count,
                        "runner":"public_batched_chain_with_proposal_telemetry"}
                rows.append(measured(f"hmc-{beta}-{name}-{length}", {"beta":beta,"kind":name,"L":length}, compute_hmc))
                write_json(root / f"hmc-{beta}-{name}-{length}.json",rows[-1])
        # Price the actual K-chart dispatch graph at every positive temperature.
        # Unqualified maps price mechanics only; frozen trained weights can alter cost.
        def make_physical(transport,primitive):
            @tf.function(input_signature=(tf.TensorSpec(shape,tf.float64),tf.TensorSpec([2],tf.int32)),
                         jit_compile=config["jit_compile"])
            def physical(state,seed):
                z=transport.inverse_theta_to_z_batch(state)
                result=primitive(z,seed)
                valid=result[5]
                return tf.where(valid,transport.forward_batch(result[0]),tf.fill(shape,tf.constant(float("nan"),tf.float64)))
            return physical
        if method == "ensemble":
            kernels, ids = [], []
            for selected in chart_targets:
                primitive=build_fixed_transport_one_step_transition(selected,state_shape=shape,
                    step_size=config["tuning"]["initial_epsilon"],num_leapfrog_steps=min(config["tuning"]["l_grid"]),
                    use_xla=config["jit_compile"],capture_health=True)
                kernels.append(make_physical(selected.transport,primitive))
                ids.append(selected.adapter_signature())
            mixture = FixedChartKernelMixture(kernels, gamma=[1./len(kernels)]*len(kernels), chart_ids=ids)
            exchange_bindings.append(BoundWithinTemperatureKernel(beta=beta,bridge_signature=bridge.signature,
                kernel_signature=mixture.selection.signature, kernel=mixture.transition_state,
                mechanics_role="pricing_only_unqualified_chart_mixture"))
            exchange_initial.append(starts)
    # Batch reference work measured directly; no optimizer-time surrogate.
    batch=config["reference"]["batch_size"]
    @tf.function(input_signature=(tf.TensorSpec([batch,bridge.parameter_dim],tf.float64),),
                 jit_compile=config["jit_compile"],reduce_retracing=False)
    def reference_batch(points):
        likelihood, score, prior, prior_score, status = bridge.component_terms(points)
        valid = (status["valid_pre_regularized_score"] & (status["status_code"] == 0)
                 & tf.reduce_all(tf.math.is_finite(score), axis=-1))
        return likelihood, valid
    def compute_reference():
        times=[]
        for repeat in range(config["budget"]["pricing_updates"]):
            with tf.device("/CPU:0"):
                z = tf.random.stateless_normal([batch, bridge.parameter_dim],
                    scoped_seed(config, "pricing-reference-target", repeat), dtype=tf.float64)
                points = bridge.prior_center + tf.sqrt(tf.constant(bridge.prior_variance, tf.float64))*z
            begin=time.monotonic()
            likelihood, valid=reference_batch(points)
            likelihood.numpy()
            tf.debugging.assert_equal(tf.reduce_all(valid), True)
            tf.debugging.assert_all_finite(likelihood, "reference price likelihood")
            times.append(time.monotonic()-begin)
        return {"first_seconds": times[0], "steady_seconds": max(times[1:] or times)}
    reference_price = measured("reference", {"batch": batch}, compute_reference,
        None if history is None else {"first_seconds": history["reference_batch_first_seconds"],
                                     "steady_seconds": history["reference_batch_steady_seconds"]})
    from bayesfilter.inference.q20_production_config import scoped_seed
    exchange_times=[]
    if method == "ensemble":
        exchange=ProperReplicaExchangeTransitionProgram(ProperBridgeReplicaExchange(bridge,config["training"]["betas"]),
            exchange_bindings,jit_compile=config["jit_compile"])
        initial=exchange.initial_state(tf.stack(exchange_initial))
        count=config["execution"]["pricing_transitions"]
        def compute_exchange():
            times=[]
            for i in range(config["budget"]["pricing_updates"]):
                begin=time.monotonic()
                output=exchange(initial,num_results=count,seed=scoped_seed(config,"pricing-exchange",i),stage="warmup")
                for item in tf.nest.flatten(output):
                    if tf.is_tensor(item):
                        item.numpy()
                if not output["health"]["passed"]:
                    raise ValueError("ensemble pricing transition health failed")
                times.append(time.monotonic()-begin)
            return {"first_seconds": times[0], "steady_seconds": max(times[1:] or times)/count}
        exchange_price = measured("exchange", {"L": min(config["tuning"]["l_grid"]),
            "width": template_width, "charts": config["ensemble"]["charts"],
            "transitions_per_call": count}, compute_exchange)
        exchange_times = [exchange_price["first_seconds"], exchange_price["steady_seconds"]]
    from bayesfilter.inference.q20_production_reference import importance_summary
    from bayesfilter.inference.q20_production_comparison import posterior_summary
    def compute_analysis():
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
        return {"reference": reference_analysis, "posterior": time.monotonic()-begin}
    analysis = measured("analysis", {"posterior_max": config["posterior"]["retained_max"]}, compute_analysis,
        None if history is None else {"reference": history["reference_analysis_seconds_per_row"],
                                     "posterior": history["posterior_analysis_seconds"]})
    result={"status":"estimation_method_cost_measurements","config_hash":digest(config), "method": method,
        "sources":source_snapshot(),"training":priced,"training_quote":quote,"hmc":rows,
        "map_prices": map_prices, "training_checkpoint": checkpoint,
        "missing_cost_categories": [],
        "exchange_price_role": "actual_multi_chart_mixture" if method == "ensemble" else "not_required",
        "exchange_charts_per_temperature": config["ensemble"]["charts"] if method == "ensemble" else 0,
        "reference_batch_first_seconds":reference_price["first_seconds"],"reference_batch_steady_seconds":reference_price["steady_seconds"],
        "reference_batch_size":batch,"wall_seconds":time.monotonic()-started,
        "exchange_first_seconds":exchange_times[0] if exchange_times else None,
        "exchange_steady_seconds":exchange_times[1] if exchange_times else None,
        "reference_analysis_seconds_per_row":analysis["reference"],"posterior_analysis_seconds":analysis["posterior"],
        "exchange_template": {"L": min(config["tuning"]["l_grid"]), "width": min(config["training"]["widths"]),
            "epsilon": config["tuning"]["initial_epsilon"], "role": "conditional_cost_only"} if method == "ensemble" else None,
        "pricing_ledger": str(ledger.root), "historical_timing_origin": historical_origin,
        "timing_units": {"first": "whole_call_seconds", "steady": "seconds_per_transition"},
        "forecast_role":"initial_observed_costs_with_unmeasured_tail_risk","production_qualified":False}
    write_json(root / "result.json",result)
    return result
