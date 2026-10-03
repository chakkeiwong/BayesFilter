#!/usr/bin/env python3
"""Bounded q20 clipping diagnostic and exact-state calibration continuation.

Uses a frozen numerical checkout and the existing Campaign process supervisor.
Shadow Adam updates are diagnostic only; the actual trainer retains its cap.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def graph_inventory(compiled):
    concrete = compiled.get_concrete_function()
    definition = concrete.graph.as_graph_def()
    nodes = list(definition.node) + [n for f in definition.library.function for n in f.node_def]
    callbacks = sorted({n.op for n in nodes if "PyFunc" in n.op or "HostCompute" in n.op})
    result = {"xla": bool(concrete.function_def.attr["_XlaMustCompile"].b),
              "traces": compiled.experimental_get_tracing_count(), "callbacks": callbacks}
    if callbacks or result["traces"] != 1:
        raise ValueError("callback or retracing in diagnostic graph")
    return result


def probe_step(session, *, rtol, atol, log_path=None):
    """One real unchanged update, checked against independent VJP/Adam copies."""
    import tensorflow as tf
    from bayesfilter.inference.neutra_training_protocol import _variables

    trainer = session.trainer
    cfg = trainer.config
    before = tuple(tf.identity(v) for v in trainer.variables)
    original_optimizer = tuple(tf.identity(v) for v in _variables(trainer.optimizer))
    copies = []
    for label in ("cap10", "double_cap_and_history", "raw_with_existing_history"):
        parameters = tuple(tf.Variable(v) for v in before)
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate,
            beta_1=cfg.beta1, beta_2=cfg.beta2, epsilon=cfg.epsilon)
        optimizer.build(parameters)
        for variable, value in zip(_variables(optimizer), original_optimizer, strict=True):
            variable.assign(value)
        if label == "double_cap_and_history":
            # These are the actual Keras Adam m/v slots, not a reconstructed
            # unclipped history. The perturbation has a declared scale of two.
            for value in optimizer._momentums:
                value.assign(2. * value)
            for value in optimizer._velocities:
                value.assign(4. * value)
        copies.append((label, parameters, optimizer))

    def value_gradient(seed):
        latent = trainer.fresh_latent_batch(seed)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(trainer.variables)
            physical, logdet = trainer.transport.forward_and_logdet(latent)
            target, score, status = trainer.target_bridge.value_score_status(
                physical, tf.constant(trainer.beta, tf.float64))
            # Independent VJP of the declared score. This does not differentiate
            # the filter value or reuse the trainer's custom-gradient wrapper.
            surrogate = tf.reduce_mean(-tf.reduce_sum(tf.stop_gradient(score) * physical, axis=1) - logdet)
        gradients = tuple(tape.gradient(surrogate, trainer.variables))
        finite = tf.reduce_all(status["bridge_valid"])
        finite &= tf.reduce_all(tf.math.is_finite(target)) & tf.reduce_all(tf.math.is_finite(score))
        finite &= tf.reduce_all(tf.math.is_finite(logdet))
        return gradients, tf.reduce_mean(-target-logdet), tf.linalg.global_norm(gradients), finite

    gradient_graph = tf.function(value_gradient, input_signature=(tf.TensorSpec([2], tf.int32),),
                                 jit_compile=cfg.jit_compile, reduce_retracing=False)

    def shadow_updates(*gradients):
        for label, parameters, optimizer in copies:
            if label == "raw_with_existing_history":
                supplied = gradients
            else:
                cap = cfg.gradient_clip_norm * (2. if label == "double_cap_and_history" else 1.)
                supplied, _ = tf.clip_by_global_norm(gradients, tf.constant(cap, tf.float64))
            optimizer.apply_gradients(zip(supplied, parameters))
        return tuple(tuple(tf.identity(v) for v in parameters) for _, parameters, _ in copies)

    shadow_graph = tf.function(shadow_updates,
        input_signature=tuple(tf.TensorSpec(v.shape, v.dtype) for v in trainer.variables),
        jit_compile=cfg.jit_compile, reduce_retracing=False)
    seed = tf.random.experimental.stateless_fold_in(tf.constant(session.root_seed, tf.int32), session.rng_index)
    gradients, loss, norm, finite = gradient_graph(seed)
    if not bool(finite.numpy()):
        raise ValueError("invalid diagnostic target or score")
    for g in gradients:
        tf.debugging.assert_all_finite(g, "diagnostic gradient")
    shadow = shadow_graph(*gradients)
    session.advance(1, log_path=log_path)
    step = session.history[-1]
    tf.debugging.assert_near(loss, tf.constant(step["loss"], tf.float64), rtol=rtol, atol=atol)
    tf.debugging.assert_near(norm, tf.constant(step["gradient_norm"], tf.float64), rtol=rtol, atol=atol)
    errors = []
    for actual, expected in zip(trainer.variables, shadow[0], strict=True):
        tf.debugging.assert_near(actual, expected, rtol=rtol, atol=atol)
        errors.append(tf.reduce_max(tf.abs(actual-expected)))
    for actual, expected in zip(_variables(trainer.optimizer), _variables(copies[0][2]), strict=True):
        if tf.as_dtype(actual.dtype).is_floating:
            tf.debugging.assert_near(tf.convert_to_tensor(actual), tf.convert_to_tensor(expected), rtol=rtol, atol=atol)
        else:
            tf.debugging.assert_equal(actual, expected)
    deltas = [tf.concat([tf.reshape(v-b, [-1]) for v, b in zip(values, before, strict=True)], axis=0)
              for values in shadow]
    base_norm = tf.linalg.norm(deltas[0])
    comparisons = {}
    for (label, _, _), delta in zip(copies, deltas, strict=True):
        tf.debugging.assert_all_finite(delta, "shadow update")
        comparisons[label] = {"update_norm": float(tf.linalg.norm(delta).numpy()),
            "relative_difference_from_baseline": float(tf.math.divide_no_nan(tf.linalg.norm(delta-deltas[0]), base_norm).numpy()),
            "cosine_with_baseline": float(tf.math.divide_no_nan(tf.reduce_sum(delta*deltas[0]),
                tf.linalg.norm(delta)*base_norm).numpy()),
            "zero_baseline_update": bool((base_norm == 0).numpy())}
    graphs = {"diagnostic_vjp": graph_inventory(gradient_graph),
              "shadow_adam": graph_inventory(shadow_graph),
              "actual_trainer": graph_inventory(trainer._compiled_train_step)}
    if cfg.jit_compile and not all(g["xla"] for g in graphs.values()):
        raise ValueError("non-XLA graph in GPU diagnostic")
    return {"passed": True, "seed": seed.numpy().tolist(), "loss": float(loss.numpy()),
        "raw_gradient_norm": float(norm.numpy()), "clipped_gradient_norm": step["clipped_gradient_norm"],
        "gradient_scale": min(1., cfg.gradient_clip_norm / float(norm.numpy())) if float(norm.numpy()) else 1.,
        "actual_shadow_max_absolute_difference": float(tf.reduce_max(errors).numpy()),
        "comparisons": comparisons, "graphs": graphs,
        "interpretation": "matched_next_step_sensitivity_only_not_a_cap_selection_or_unclipped_training_history"}


def load_cohort(path, config, source_root):
    from bayesfilter.inference.q20_campaign_runtime import source_snapshot
    from bayesfilter.inference.q20_production_config import digest
    state = json.loads(Path(path).read_text())
    expected = state.pop("checkpoint_hash")
    if digest(state) != expected or state["config_hash"] != digest(config):
        raise ValueError("checkpoint checksum or configuration differs")
    if state["sources"] != source_snapshot(source_root):
        raise ValueError("checkpoint numerical sources differ")
    return state


def worker(request, output):
    from bayesfilter.inference.q20_production_config import digest, write_json, scoped_seed
    from bayesfilter.inference.q20_gpu_runtime import GPUResourceUnavailable, select_worker_gpu, check_gpu_contention

    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {"schema": "bayesfilter.q20.training_continuation_diagnostic.v1", "status": "initializing",
        "command": [sys.executable, *sys.argv], "started_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable, "pid": os.getpid(), "request": request,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=request["source_root"], text=True).strip(),
        "plan_file": request["plan_file"], "result_file": str(output / "result.json"),
        "production_qualified": False, "posterior_qualified": False, "rows": []}

    def save():
        record["wall_seconds"] = time.monotonic()-started
        write_json(output / "result.json", record, exclusive=False)

    def deadline_ok():
        return time.monotonic()-started < request["cooperative_seconds"]

    def capacity():
        receipt = check_gpu_contention(record["launch_readiness"])
        record.setdefault("contention_checks", []).append({"wall_seconds": time.monotonic()-started, **receipt})
        if not deadline_ok():
            raise TimeoutError("cooperative phase deadline; complete checkpoints preserved")

    def interrupted(signum, _frame):
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM, interrupted)
    save()
    try:
        if sha256(__file__) != request["driver_sha256"]:
            raise ValueError("diagnostic driver changed")
        if sha256(request["checkpoint"]) != request["checkpoint_sha256"]:
            raise ValueError("input checkpoint changed")
        config = request["config"]
        state = load_cohort(request["checkpoint"], config, request["source_root"])
        record["sources"] = state["sources"]
        record["launch_readiness"] = select_worker_gpu()
        save()
        import tensorflow as tf
        import tensorflow_probability as tfp
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        record.update(memory_policy=memory, tensorflow=tf.__version__, tfp=tfp.__version__,
            cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
            tf32=tf.config.experimental.tensor_float_32_execution_enabled(), jit_compile=True,
            dtype="float64", sample_wise_target_fallback=False)
        from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge
        from bayesfilter.inference.neutra_training_protocol import TrainingSession
        from bayesfilter.inference.q20_production_training import scope_for, training_config, _evaluate_rung
        from bayesfilter.inference.q20_training_validation import FrozenLossCache
        from bayesfilter.inference.q20_campaign_runtime import source_snapshot
        bridge = make_q20_tempered_bridge(config["target"]["q"], jit_compile=True,
            principal_sqrt_backend=config["target"]["principal_sqrt_backend"])
        record.update(target_signature=bridge.target_signature, bridge_signature=bridge.signature,
                      seeds=config["seed"], status="running")
        data = output / "data"
        data.mkdir()
        cache_root = data / "validation-cache"
        shutil.copytree(Path(request["checkpoint"]).parent / "validation-cache", cache_root)
        original_elapsed = state.get("total_elapsed_seconds", 0.)

        def checkpoint(status):
            state.update(status=status, calibration_only=True, elapsed_seconds=time.monotonic()-started,
                previous_elapsed_seconds=original_elapsed,
                total_elapsed_seconds=original_elapsed+time.monotonic()-started,
                diagnostic_driver_sha256=request["driver_sha256"], parent_checkpoint=request["checkpoint"])
            path = data / f"cohort-{len(list(data.glob('cohort-*.json'))):05d}.json"
            write_json(path, {**state, "checkpoint_hash": digest(state)})
            record["checkpoint"] = str(path)
            save()

        checkpoint("diagnostic_initialized")
        for name, item in state["cohort"].items():
            capacity()
            if request["phase"] == "clipping" and name in state.get("clipping_diagnostics", {}):
                record["rows"].append(state["clipping_diagnostics"][name])
                continue
            if request["phase"] == "continuation":
                completed = [a for a in item["assessments"] if a["updates"] == request["target_updates"]
                             and a["beta"] == item["session"]["map"]["beta"]]
                if completed:
                    record["rows"].append({"candidate": name, "assessment": completed[-1]})
                    continue
            candidate = item["session"]["scope"]["candidate"]
            scope = scope_for(config, bridge, candidate, sources=state["sources"], memory_policy=memory)
            beta = item["session"]["map"]["beta"]
            session = TrainingSession.restore(item["session"], bridge=bridge,
                config=training_config(config, candidate), expected_scope=scope,
                preflight_seed=scoped_seed(config, "preflight", name, beta))
            if request["phase"] == "clipping":
                result = probe_step(session, rtol=config["validation"]["reliability_rtol"],
                    atol=config["validation"]["reliability_atol"], log_path=data / f"{name}.jsonl")
                item["session"] = session.checkpoint()
                row = {"candidate": name, "beta": beta, **result}
                state.setdefault("clipping_diagnostics", {})[name] = row
                record["rows"].append(row)
                checkpoint("clipping_member_checked")
            else:
                while session.level_updates < request["target_updates"]:
                    capacity()
                    count = min(config["training"]["checkpoint_every"], request["target_updates"]-session.level_updates)
                    status = session.advance(count, log_path=data / f"{name}.jsonl", budget_check=deadline_ok)
                    item["session"] = session.checkpoint()
                    checkpoint("training")
                    if status == "budget_exhausted":
                        raise TimeoutError("continuation budget exhausted")
                capacity()
                cache = FrozenLossCache(cache_root, bridge=bridge, scope=scope,
                    batch_size=config["training"]["batch_size"], jit_compile=True)
                assessment, payload = _evaluate_rung(config, bridge, candidate, session,
                    item["baseline"], item["previous"], item["plateaus"], cache=cache,
                    budget_check=deadline_ok, calibration_only=True)
                item["assessments"].append(assessment)
                item.update(previous=session.checkpoint(), session=session.checkpoint(),
                    plateaus=assessment["decision"]["plateaus"], status=assessment["decision"]["status"])
                export = data / f"{name}-beta{beta:g}-u{session.level_updates}.json"
                write_json(export, {"frozen_transport": payload, "assessment": assessment,
                    "candidate": candidate, "config_hash": digest(config), "role": "diagnostic_calibration",
                    "sources": state["sources"], "training_checkpoint_hash": session.checkpoint()["state_hash"]})
                item["exports"][str(beta)] = str(export)
                record["rows"].append({"candidate": name, "assessment": assessment})
                checkpoint("continued_member_assessed")
                if not assessment["map_reliability"]["passed"]:
                    raise ValueError("map reliability veto")
        capacity()
        if source_snapshot(request["source_root"]) != state["sources"]:
            raise ValueError("numerical source drift during worker")
        record["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
        checkpoint("clipping_checked" if request["phase"] == "clipping" else "calibration_extended")
        record["status"] = "completed"
        save()
        return 0
    except GPUResourceUnavailable as error:
        record.update(status="waiting_for_gpu", resource_receipt=error.receipt)
        save()
        return 3
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        save()
        raise


def coordinate(request, output):
    from bayesfilter.inference.q20_campaign_runtime import Campaign, atomic_json
    from bayesfilter.inference.q20_production_config import digest
    allowance = json.loads(Path(request["allowance"]).read_text())
    campaign = Campaign(output, repo=request["source_root"], config=request["config"], allowance=allowance)
    with campaign.locked():
        if campaign.state.get("diagnostic_request_hash", digest(request)) != digest(request):
            raise ValueError("diagnostic continuation request changed")
        campaign.state["diagnostic_request_hash"] = digest(request)
        campaign.save()
        current = {**request}
        for phase in ("clipping", "continuation"):
            if phase in campaign.state["stages"]:
                saved = campaign.state["stages"][phase]
                if sha256(saved["result_path"]) != saved["result_sha256"]:
                    raise ValueError("completed result changed")
                if sha256(saved["checkpoint"]) != saved["checkpoint_sha256"]:
                    raise ValueError("completed checkpoint changed")
                result = json.loads(Path(saved["result_path"]).read_text())
            else:
                for previous in reversed(campaign.state["attempts"]):
                    if previous["stage"] == phase:
                        checkpoints = sorted((Path(previous["directory"]) / "worker/data").glob("cohort-*.json"))
                        if checkpoints:
                            current.update(checkpoint=str(checkpoints[-1]), checkpoint_sha256=sha256(checkpoints[-1]))
                            break
                spent = sum(a.get("elapsed_seconds", 0.) for a in campaign.state["attempts"] if a["stage"] == phase)
                cap = min(request["caps"][phase]-spent, campaign.remaining(True))
                current.update(phase=phase, cooperative_seconds=cap-2*request["config"]["execution"]["termination_grace_seconds"])
                attempt = campaign.execute(phase, [sys.executable, str(Path(__file__).resolve()), "worker",
                    "--request", "{attempt}/request.json", "--output-dir", "{attempt}/worker"],
                    cap_seconds=cap, diagnostic=True, request=current, request_hash=digest(current),
                    environment={"TF_FORCE_GPU_ALLOW_GROWTH": "true", "BAYESFILTER_PRELOAD_CUSTOM_OP": "0",
                        "TF_NUM_INTRAOP_THREADS": "2", "TF_NUM_INTEROP_THREADS": "2", "PYTHONUNBUFFERED": "1"})
                path = Path(attempt["directory"]) / "worker/result.json"
                result = json.loads(path.read_text()) if path.exists() else {"status": attempt["status"]}
                if attempt["status"] != "completed" or result["status"] != "completed":
                    campaign.state["status"] = "WAITING_FOR_GPU" if result["status"] == "waiting_for_gpu" else "DIAGNOSTIC_REPAIR_REQUIRED"
                    campaign.save()
                    break
                campaign.state["stages"][phase] = {"result_path": str(path), "result_sha256": sha256(path),
                    "checkpoint": result["checkpoint"], "checkpoint_sha256": sha256(result["checkpoint"])}
                campaign.save()
            current.update(checkpoint=result["checkpoint"], checkpoint_sha256=sha256(result["checkpoint"]))
        else:
            campaign.state["status"] = "TRAINING_CONTINUATION_DIAGNOSTIC_COMPLETE"
            campaign.save()
        summary = {"status": campaign.state["status"], "stages": campaign.state["stages"],
            "remaining_campaign_seconds": campaign.remaining(), "remaining_diagnostic_seconds": campaign.remaining(True),
            "production_qualified": False, "posterior_qualified": False}
        atomic_json(output / "result.json", summary)
        atomic_json(output / "settled-allowance.json", {"schema": "bayesfilter.q20.recovered_allowance.v1",
            "source_ledger": str(campaign.path), "source_sha256": sha256(campaign.path),
            "campaign_remaining_seconds": campaign.remaining(), "diagnostic_remaining_seconds": campaign.remaining(True),
            "all_attempts_settled": True, "master_status": campaign.state["status"], "scope": "Diagnostics included in campaign; no renewal."})
        print(json.dumps(summary, indent=2))
        return 0 if campaign.state["status"] == "TRAINING_CONTINUATION_DIAGNOSTIC_COMPLETE" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "worker"))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    sys.path.insert(0, request["source_root"])
    from bayesfilter.inference.q20_production_config import validate_protocol
    validate_protocol(request["config"])
    if request["config"]["cpu_reference"] or not request["config"]["jit_compile"]:
        raise ValueError("actual q20 continuation requires GPU/XLA")
    if sha256(__file__) != request["driver_sha256"]:
        raise ValueError("driver checksum changed")
    return (worker if args.mode == "worker" else coordinate)(request, args.output_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
