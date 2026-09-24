"""Bounded plain NeuTra HMC assessment of an exactly frozen configured map.

Posterior sampling calls run_sequential_neutra_hmc through run_hmc_posterior.
Tuning, diagnostic pricing, warmup and retained estimation remain separate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nominate(report):
    """Select reproducible representatives without ranking descriptive metrics."""
    order = {name: index for index, name in enumerate(("naf16", "naf32", "iaf16", "legacy_control"))}
    def key(row):
        family_pass = report["families"][row["family"]]["replicated_local_improvement"]
        group = 0 if family_pass else (1 if row["fixed_checkpoint_training_improvement"] else 2)
        return group, order[row["family"]], row["root_seed"]
    return sorted(report["comparisons"], key=key)


def protocol():
    from bayesfilter.inference.q20_production_config import protocol_template
    original = protocol_template()
    result = {key: original[key] for key in ("role", "cpu_reference", "jit_compile",
        "target", "tuning", "posterior", "starts", "validation", "execution", "budget")}
    result.update(schema="bayesfilter.q20.configured_training_hmc_check.v1", seed=[20260924, 729],
        training={"betas": [0., 1.], "batch_size": 32},
        plan_file="docs/plans/bayesfilter-neutra-precision-training-plan-2026-09-24.md")
    result["target"]["principal_sqrt_backend"] = "tensorflow_eigh_strict_factor_cached"
    result["tuning"].update(l_grid=[3, 5, 9], total_budget_units=80, repair_reserve_units=20,
        max_candidates=30, max_repairs_per_family=8, max_wall_seconds=2500.)
    # These are price controls; the old campaign balances are not authority
    # for this run and must not appear to grant an additional allowance.
    result["budget"] = {key: original["budget"][key] for key in ("pricing_updates", "forecast_safety_factor")}
    return result


def curvature_warm_start(adapter, starts, *, jit_compile=True):
    """FP64 central-score differences nominate epsilon, never the HMC force."""
    import tensorflow as tf
    chains, dimension = starts.shape
    displacement = (2.**-52)**(1./3.)
    @tf.function(input_signature=[tf.TensorSpec([chains, dimension], tf.float64)],
                 jit_compile=jit_compile, autograph=False)
    def calculate(z):
        h = tf.constant(displacement, tf.float64)*(1.+tf.abs(z))
        offsets = tf.eye(dimension, dtype=tf.float64)[None, :, :]*h[:, :, None]
        probes = tf.concat((z[:, None, :]+offsets, z[:, None, :]-offsets), axis=1)
        value, score = adapter.log_prob_and_grad(tf.reshape(probes, [chains*2*dimension, dimension]))
        score = tf.reshape(score, [chains, 2*dimension, dimension])
        jacobian = tf.transpose((score[:, :dimension]-score[:, dimension:])/(2.*h[:, :, None]), [0, 2, 1])
        curvature = -.5*(jacobian+tf.transpose(jacobian, [0, 2, 1]))
        eigenvalues = tf.linalg.eigvalsh(curvature)
        rho = tf.reduce_max(tf.abs(eigenvalues))
        epsilon = .5/tf.sqrt(tf.maximum(tf.constant(1., tf.float64), rho))
        return epsilon, eigenvalues, tf.reduce_max(tf.abs(jacobian-tf.transpose(jacobian, [0, 2, 1]))), value, score
    epsilon, eigenvalues, asymmetry, value, score = calculate(starts)
    for tensor in (epsilon, eigenvalues, asymmetry, value, score):
        tf.debugging.assert_all_finite(tensor, "invalid local-curvature warm start")
    return {"epsilon": float(epsilon), "curvature_eigenvalues": eigenvalues.numpy().tolist(),
        "max_jacobian_asymmetry": float(asymmetry), "relative_displacement": displacement,
        "role": "local pilot hypothesis only; no global stability bound or mass adaptation"}


def tune_map(config, bridge, finalized, root, *, max_seconds):
    import tensorflow as tf
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
    from bayesfilter.inference.hmc_candidate_set_execution import _rebuild_geometry
    from bayesfilter.inference.q20_production_config import digest, frozen_scope_hash
    from bayesfilter.inference.q20_production_training import source_snapshot
    from bayesfilter.inference.q20_production_hmc import draw_start_bank, tuning_configs, _export_tuning_result
    from bayesfilter.inference import bind_hmc_candidate_set_execution, tune_fixed_transport_hmc_kernel
    from dataclasses import replace

    root.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    adapter = bridge.fixed_beta_adapter(1.)
    payload = finalized["frozen_transport"]
    loaded = load_frozen_neutra_artifact(payload, expected_target_signature=adapter.adapter_signature())
    label = "configured-neutra-"+payload["transport_hash"][:16]
    starts, start_receipt = draw_start_bank(config, bridge, 1., label)
    @tf.function(input_signature=[tf.TensorSpec([4, bridge.parameter_dim], tf.float64)],
                 jit_compile=config["jit_compile"], autograph=False)
    def inverse(points):
        z = loaded.transport.inverse_theta_to_z_batch(points)
        return z, loaded.transport.forward(z)
    latent, recovered = inverse(starts)
    error = float(tf.reduce_max(tf.abs(recovered-starts)/(1.+tf.abs(starts))))
    if not error <= 1.e-7:
        raise ValueError("candidate start inverse roundtrip failed")
    save(root/"starts.json", {**start_receipt, "positions": starts.numpy().tolist(),
        "active_positions": latent.numpy().tolist(), "scaled_inverse_error": error})
    transformed, _ = _rebuild_geometry(adapter, [{"kind": "frozen_transport", "artifact": payload}], adapter.target_scope)
    proposal = curvature_warm_start(transformed, latent, jit_compile=config["jit_compile"])
    lo, hi = config["tuning"]["epsilon_domain"]
    config["tuning"]["initial_epsilon"] = min(hi, max(lo, proposal["epsilon"]))
    save(root/"curvature.json", proposal)
    save(root/"protocol.json", config)
    execution, search = tuning_configs(config, label)
    paths = [ROOT/path for path in source_snapshot()] + [Path(__file__).resolve()]
    binding = bind_hmc_candidate_set_execution(adapter=adapter, initial_position=latent,
        target_scope=adapter.target_scope, target_lineage={"model": "ssl_lstm_q20", "data": bridge.target_signature,
            "prior": "gaussian_sd4", "beta": 1., "method": "neutra", "role": config["role"],
            "frozen_scope_hash": frozen_scope_hash(config), "configured_transport_hash": payload["transport_hash"]},
        config=execution, source_paths=paths, scope_id=label, search_id=digest([config, label])[:20],
        epsilon_domain=tuple(config["tuning"]["epsilon_domain"]),
        repair_factor=config["tuning"]["repair_factor"],
        max_repairs_per_family=config["tuning"]["max_repairs_per_family"],
        frozen_transport_payload=payload, start_coordinates="active")
    remaining = max_seconds-(time.monotonic()-began)
    if remaining <= 0:
        from bayesfilter.inference.q20_stage_budget import StageBudgetPause
        raise StageBudgetPause("configured-map tuning setup exhausted its allocation")
    search = replace(search, max_wall_time_seconds=min(search.max_wall_time_seconds, remaining))
    run = tune_fixed_transport_hmc_kernel(base_adapter=adapter, fixed_transport=binding.fixed_transport,
        initial_position=binding.initial_active_state, candidate_set_adapter=binding.typed_adapter,
        config=search, output_dir=root/"tuning")
    return _export_tuning_result(config, root, "neutra", 1., label, run, binding)


def assess_member(config, bridge, member_path, root, *, max_seconds, chunk_seconds):
    """Preserve every warmup chunk through the shared sequential controller."""
    from bayesfilter.inference import load_hmc_candidate_retained_runner, run_hmc_posterior
    from bayesfilter.inference.neutra_hmc import SequentialNeuTraHMCConfig
    from bayesfilter.inference.q20_production_hmc import (
        sequential_kwargs, posterior_quantities, _check_member_protocol, _write_posterior_result)
    from bayesfilter.inference.q20_production_config import PARAMETERS, digest
    from bayesfilter.inference.q20_stage_budget import BudgetedCheckpoint

    root.mkdir(parents=True, exist_ok=False)
    member = load_hmc_candidate_retained_runner(member_path, adapter=bridge.fixed_beta_adapter(1.))
    lineage = _check_member_protocol(member, config)
    if lineage["method"] != "neutra":
        raise ValueError("configured training assessment requires a NeuTra member")
    label = "configured-training-"+member.candidate_id
    kwargs = sequential_kwargs(config, label)
    policy = kwargs["assessment_policy"]
    controller = SequentialNeuTraHMCConfig(step_size=member.step_size, num_leapfrog_steps=member.num_leapfrog_steps,
        jit_compile=config["jit_compile"], energy_error_log_accept_threshold=config["posterior"]["energy_log_accept_alert"],
        **kwargs)
    with BudgetedCheckpoint(root/"chunks", {"member": member.member_hash, "config": digest(config),
            "label": label, "names": list(PARAMETERS), "quantities": policy.quantities_id},
            deadline=time.monotonic()+max_seconds, chunk_seconds=chunk_seconds,
            safety_factor=config["budget"]["forecast_safety_factor"]) as store:
        result = run_hmc_posterior(member=member, config=controller, parameter_names=PARAMETERS,
            quantities_fn=posterior_quantities, checkpoint_store=store)
    _write_posterior_result(root, result, config, bridge, label)
    return {"passed": result["passed"], "result": str(root/"result.json")}


def run(request, root, manifest, began):
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    manifest["memory_policy"] = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)
    from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge
    from bayesfilter.inference.q20_hmc_qualification import qualify_bridge, attach_qualification
    from bayesfilter.inference.q20_production_hmc import sample_member
    from bayesfilter.inference.q20_stage_budget import StageBudgetPause
    from bayesfilter.inference.q20_production_training import source_snapshot

    deadline = datetime.fromisoformat(request["deadline_utc"])
    def remaining():
        return min(request["worker_seconds"]-(time.monotonic()-began),
            (deadline-datetime.now(timezone.utc)).total_seconds())
    config = protocol()
    bridge = make_q20_tempered_bridge(20, jit_compile=True,
        principal_sqrt_backend="tensorflow_eigh_strict_factor_cached")
    if bridge.target_signature != request["target_signature"] or bridge.signature != request["bridge_signature"]:
        raise ValueError("downstream target differs from training")
    manifest.update(tensorflow=tf.__version__, target_signature=bridge.target_signature,
        bridge_signature=bridge.signature, dtype="float64", tf32=True, jit_compile=True,
        chain_mode="batched", chains=4, latent_mass="identity", source_sha256=source_snapshot(),
        native_op_sha256={p.name: sha(p) for p in (ROOT/"bayesfilter/ops").glob("*.so")},
        gpu_name=tf.config.experimental.get_device_details(tf.config.list_physical_devices("GPU")[0]).get("device_name"))
    save(root/"manifest.json", manifest)
    report = read(request["training_report"])
    if sha(request["training_report"]) != request["training_report_sha256"]:
        raise ValueError("training report checksum changed")
    candidates = nominate(report)
    save(root/"nominations.json", [{key: row[key] for key in ("family", "root_seed", "checkpoint", "finalized")}
        for row in candidates])
    qualify_bridge(config, bridge, root/"bridge-qualification", betas=[1.])
    attach_qualification(bridge, root/"bridge-qualification/result.json", config, betas=[1.])
    tuning_spent = 0.
    attempts = []
    result = {"status": "downstream_in_progress", "attempts": attempts,
        "posterior_qualified": False, "independent_reference_checked": False,
        "training_report": request["training_report"]}
    for index, candidate in enumerate(candidates):
        tuning_left = min(2500.-tuning_spent, remaining())
        if tuning_left <= 0.:
            break
        candidate_root = root/f"candidate-{index:02d}"
        candidate_root.mkdir()
        record = {key: candidate[key] for key in ("family", "root_seed", "checkpoint", "finalized")}
        attempts.append(record)
        save(root/"progress.json", result)
        start = time.monotonic()
        finalized = read(candidate["finalized"])
        if finalized["checkpoint"] != read(candidate["checkpoint"]):
            raise ValueError("candidate finalized checkpoint differs from paired report")
        candidate_config = json.loads(json.dumps(config))
        try:
            tuning = tune_map(candidate_config, bridge, finalized, candidate_root/"tuning", max_seconds=tuning_left)
        except StageBudgetPause as error:
            record.update(status="tuning_budget_incomplete", reason=str(error))
            break
        except ValueError as error:
            if not str(error).startswith("candidate start inverse roundtrip failed"):
                raise
            record.update(status="candidate_inverse_rejected", reason=str(error))
            continue
        finally:
            elapsed = time.monotonic()-start
            tuning_spent += elapsed
            record["tuning_seconds"] = elapsed
            save(root/"progress.json", result)
        record["tuning"] = str(candidate_root/"tuning/result.json")
        record["verified_members"] = tuning["verified_members"]
        if not tuning["verified_members"]:
            record["status"] = "no_verified_kernel_in_bounded_search"
            continue
        members = sorted(tuning["verified_members"],
            key=lambda cid: (tuning["verified_member_parameters"][cid]["L"], cid))
        record["member_order"] = members
        record["posterior_attempts"] = []
        for ordinal, member_id in enumerate(members):
            if remaining() <= 0.:
                break
            member_path = tuning["verified_members"][member_id]
            price = sample_member(candidate_config, bridge, candidate_root/f"price-{ordinal}",
                member_path=member_path, label="configured-training-price-"+member_id, pricing_only=True)
            chunk_seconds = price["steady_seconds"]*candidate_config["posterior"]["warmup_chunk"]
            attempt = {"member_id": member_id, "price_seconds_per_transition": price["steady_seconds"],
                "first_chunk_forecast_seconds": chunk_seconds}
            record["posterior_attempts"].append(attempt)
            save(root/"progress.json", result)
            try:
                outcome = assess_member(candidate_config, bridge, member_path, candidate_root/f"posterior-{ordinal}",
                    max_seconds=remaining(), chunk_seconds=chunk_seconds)
            except StageBudgetPause as error:
                attempt.update(status="posterior_budget_incomplete", reason=str(error))
                record["status"] = "posterior_budget_incomplete"
                result["status"] = "downstream_budget_incomplete"
                save(root/"progress.json", result)
                return result
            attempt.update(status="declared_posterior_checks_passed" if outcome["passed"] else "declared_posterior_checks_failed", **outcome)
            if outcome["passed"]:
                record["status"] = "declared_posterior_checks_passed"
                result.update(status="declared_posterior_checks_passed_reference_pending",
                    selected_member=member_path, posterior_result=outcome["result"])
                save(root/"progress.json", result)
                return result
        record.setdefault("status", "no_assessed_member_passed")
    result["status"] = "no_posterior_pass_within_declared_scope"
    save(root/"progress.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    if not args.gpu.isdecimal():
        parser.error("gpu must be a physical device index")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    if os.environ["TF_FORCE_GPU_ALLOW_GROWTH"].lower() != "true":
        raise ValueError("memory growth must be enabled before import")
    for key in ("TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS"):
        os.environ.setdefault(key, "2")
    args.output.mkdir(parents=True, exist_ok=False)
    request = read(args.request)
    began = time.monotonic()
    manifest = {"command": sys.argv, "python": sys.executable, "request": request,
        "runner_sha256": sha(__file__), "request_sha256": sha(args.request),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "cuda_visible_devices": args.gpu, "started_at": datetime.now(timezone.utc).isoformat(),
        "plan_file": request["plan_file"], "plan_sha256": sha(ROOT/request["plan_file"]),
        "result_file": str(args.output/"result.json"), "status": "starting"}
    save(args.output/"manifest.json", manifest)
    try:
        result = run(request, args.output, manifest, began)
        result["wall_seconds"] = time.monotonic()-began
        save(args.output/"result.json", result)
        manifest["status"] = result["status"]
    except BaseException as error:
        manifest["status"] = "failed"
        save(args.output/"failure.json", {"type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc()})
        raise
    finally:
        manifest["wall_seconds"] = time.monotonic()-began
        save(args.output/"manifest.json", manifest)


if __name__ == "__main__":
    main()
