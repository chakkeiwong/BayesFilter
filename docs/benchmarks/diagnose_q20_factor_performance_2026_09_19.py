"""Diagnostic-only paired GPU profile of strict and safe-factor cached q20.

Uses independent reference tests and public HMC mechanics; issues no tuning or
posterior authority. NumPy is confined to the imported independent test fixtures.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

BACKENDS = {"strict": "tensorflow_eigh_strict",
            "safe_factor_cache": "tensorflow_eigh_strict_factor_cached"}


class CandidateRejected(RuntimeError):
    pass


class ProfileBudgetReached(RuntimeError):
    pass


def parity_values(left, right, *, allow_positive_infinity=False):
    """Normalize only an explicitly allowed, matching positive-infinity sentinel."""
    import tensorflow as tf
    if allow_positive_infinity:
        left_inf = tf.math.is_inf(left) & (left > 0)
        right_inf = tf.math.is_inf(right) & (right > 0)
        tf.debugging.assert_equal(left_inf, right_inf, message="infinity sentinel mismatch")
        left = tf.where(left_inf, tf.zeros_like(left), left)
        right = tf.where(right_inf, tf.zeros_like(right), right)
    tf.debugging.assert_all_finite(left, "baseline parity values")
    tf.debugging.assert_all_finite(right, "candidate parity values")
    return left, right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from bayesfilter.inference.q20_campaign_runtime import atomic_json, source_snapshot
    from bayesfilter.inference.q20_production_config import validate_protocol, scoped_seed, digest
    from bayesfilter.inference.q20_gpu_runtime import select_worker_gpu, check_gpu_contention, GPUResourceUnavailable

    request = json.loads(args.request.read_text())
    config = validate_protocol(request["config"])
    if config["cpu_reference"] or not config["jit_compile"]:
        raise ValueError("this profile requires the declared GPU/XLA path")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    record = {"schema": "bayesfilter.q20.factor_performance.v1", "status": "initializing",
        "role": "engineering_parity_and_descriptive_cost_only", "candidate_parity_passed": False,
        "started_at": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, *sys.argv],
        "source_root": str(root), "sources": source_snapshot(root),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "environment": sys.executable, "plan_file": request["plan_file"],
        "result_file": str(output / "result.json"), "config": config, "config_hash": digest(config),
        "backends": BACKENDS, "measurements": {}, "checks": [], "graphs": {}, "seeds": [],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def save():
        record["wall_seconds"] = time.monotonic()-began
        atomic_json(output / "result.json", record)

    def admit(estimate=60.):
        if time.monotonic()-began + estimate >= request["max_seconds"]:
            raise ProfileBudgetReached("insufficient phase time for next measured call")

    def interrupt(signum, _frame):
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM, interrupt)
    save()
    try:
        record["readiness"] = select_worker_gpu()
        save()
        import tensorflow as tf
        import tensorflow_probability as tfp
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        record["memory_policy"] = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        record.update(tensorflow=tf.__version__, tfp=tfp.__version__,
            tf32=tf.config.experimental.tensor_float_32_execution_enabled(),
            cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"], target_dtype="float64",
            cpu_reference_role="saved_independent_reference_values_only")
        from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge
        from bayesfilter.inference.q20_hmc_qualification import _QualifiedAdapter
        from bayesfilter.inference.hmc_bootstrap_checkpoint import require_chunk_health
        from bayesfilter.inference.hmc import FullChainHMCConfig, ReusableFullChainHMCRunner
        from bayesfilter.inference.hmc_kernel_tuning import initialize_hmc_kernel_geometry
        from bayesfilter.inference.hmc_bootstrap import build_bootstrap_fixed_mass_adapter
        from bayesfilter.nonlinear.experimental_batched_svd_sigma_point_tf import (
            _checked_batched_principal_sqrt_factor_first_derivatives)

        def contention():
            record["latest_contention_check"] = check_gpu_contention(record["readiness"])

        def sync(value):
            for tensor in tf.nest.flatten(value):
                tensor.numpy()

        def graph(fn):
            concrete = fn.get_concrete_function()
            definition = concrete.graph.as_graph_def()
            nodes = list(definition.node)+[n for f in definition.library.function for n in f.node_def]
            counts = Counter(n.op for n in nodes)
            callbacks = [op for op in counts if "PyFunc" in op or "HostCompute" in op]
            xla = concrete.function_def.attr["_XlaMustCompile"].b
            traces = fn.experimental_get_tracing_count()
            if callbacks or not xla or traces != 1:
                raise ValueError("non-XLA, callback or retraced diagnostic graph")
            return {"xla": xla, "traces": traces, "callbacks": callbacks, "op_counts": dict(counts)}

        def compare(label, left, right, *, rtol=1e-9, atol=1e-10, allow_positive_infinity=False):
            tf.nest.assert_same_structure(left, right)
            error = 0.
            try:
                for a, b in zip(tf.nest.flatten(left), tf.nest.flatten(right)):
                    if a.dtype.is_floating:
                        a, b = parity_values(a, b, allow_positive_infinity=allow_positive_infinity)
                        tf.debugging.assert_near(a, b, rtol=rtol, atol=atol, message=label)
                        error = max(error, float(tf.reduce_max(tf.abs(a-b))))
                    else:
                        tf.debugging.assert_equal(a, b, message=label)
            except (tf.errors.InvalidArgumentError, AssertionError) as exc:
                record["checks"].append({"label": label, "passed": False, "error": str(exc)})
                raise CandidateRejected(label) from exc
            record["checks"].append({"label": label, "passed": True,
                                     "maximum_absolute_difference": error, "rtol": rtol, "atol": atol})

        def measure(label, arm, call, *, estimate=60.):
            key = label+":"+arm
            rows = record["measurements"].setdefault(key, [])
            admit(max(estimate, 2*max((row["seconds"] for row in rows), default=0.)))
            contention()
            record["active_measurement"] = key
            save()
            start = time.monotonic()
            result = call()
            sync(result)
            seconds = time.monotonic()-start
            contention()
            rows.append({"seconds": seconds, "role": "first_compile_execute" if not rows else "warm_execute"})
            save()
            print(key, len(rows), seconds, flush=True)
            return result

        def import_reference(name):
            path = root / "tests" / name
            record.setdefault("reference_files", {})[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            spec = importlib.util.spec_from_file_location("q20_profile_"+path.stem, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        # These fixture assertions retain their independent references and
        # tolerances. A baseline failure is distinct from candidate rejection.
        eigen = import_reference("test_principal_sqrt_eigen_refinement_tf.py")
        factors = import_reference("test_experimental_batched_svd_sigma_point_tf.py")
        import pytest
        checks = [
            ("eigenpairs_dim80", lambda: eigen.test_refined_eigh_preserves_eigenpairs_and_indefinite_rows(80)),
            ("factor_invalid_and_roundoff_parity", factors.test_cached_strict_factor_reuses_covariance_eigensystem_without_changing_rows),
            ("factor_recursive_fixture", factors.test_strict_factor_cached_score_matches_strict_on_batched_fixture),
        ]
        for label, call in checks:
            admit()
            contention()
            record["active_reference"] = label
            save()
            start = time.monotonic()
            call()
            record["checks"].append({"label": label, "passed": True, "seconds": time.monotonic()-start})
            save()
        for label, call in (("nonfinite_eigen_input", eigen.test_refined_eigh_does_not_send_nonfinite_rows_to_backend),
                            ("unconverged_eigen_veto", eigen.test_refined_eigh_fails_closed_when_sweeps_do_not_converge)):
            with pytest.MonkeyPatch.context() as patch:
                call(patch)
            record["checks"].append({"label": label, "passed": True})
        for arm, backend in BACKENDS.items():
            for label, call in (("phase9b_preflight", eigen.test_phase9b_fixed_preflight_row_converges_without_relaxing_status),
                                ("phase9b_endpoint", eigen.test_phase9b_false_indefiniteness_endpoint_matches_cpu_reference)):
                admit()
                contention()
                record["active_reference"] = label+":"+arm
                save()
                start = time.monotonic()
                try:
                    call(backend)
                except AssertionError as exc:
                    if arm == "safe_factor_cache":
                        raise CandidateRejected(label) from exc
                    raise
                record["checks"].append({"label": label+":"+arm, "passed": True,
                                         "seconds": time.monotonic()-start})
                save()
                print("reference", label, arm, "passed", flush=True)

        bridges = {arm: make_q20_tempered_bridge(config["target"]["q"], jit_compile=True,
                   principal_sqrt_backend=backend) for arm, backend in BACKENDS.items()}
        record["target_signatures"] = {arm: bridge.target_signature for arm, bridge in bridges.items()}
        record["bridge_signatures"] = {arm: bridge.signature for arm, bridge in bridges.items()}
        # This existing adapter capability is explicitly diagnostic-only. No
        # production qualification or tuning receipt is issued by this worker.
        record["runner_capability_role"] = "measured_full_chain_diagnostic_only"
        banks = {}
        for beta in config["training"]["betas"][1:]:
            receipt = request["starts"][str(beta)]
            raw = Path(receipt["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
                raise ValueError("predecessor start bank changed")
            banks[beta] = tf.constant(json.loads(raw)["positions"], tf.float64)
        record["starts"] = {str(beta): points.numpy().tolist() for beta, points in banks.items()}
        record["horizon"] = config["target"]["horizon"]
        record["status"] = "measuring"
        save()

        for beta, starts in banks.items():
            adapters = {arm: _QualifiedAdapter(bridge, beta=beta) for arm, bridge in bridges.items()}
            for size in (1, 4):
                label = f"target_beta{beta:g}_batch{size}"
                functions = {arm: tf.function(adapter.log_prob_and_grad_status,
                    input_signature=(tf.TensorSpec((size, 4), tf.float64),),
                    jit_compile=True, autograph=False) for arm, adapter in adapters.items()}
                for repeat in range(3):
                    values = {}
                    for arm in (tuple(BACKENDS) if repeat % 2 == 0 else tuple(reversed(BACKENDS))):
                        values[arm] = measure(label, arm, lambda: functions[arm](starts[:size]))
                        if not bool(tf.reduce_all(values[arm][2]["valid_pre_regularized_score"])):
                            if arm == "safe_factor_cache":
                                raise CandidateRejected(label+":invalid_target")
                            raise ValueError(label+":baseline_invalid_target")
                    compare(label+":value", values["strict"][0], values["safe_factor_cache"][0], rtol=1e-10)
                    compare(label+":score", values["strict"][1], values["safe_factor_cache"][1])
                    for key in values["strict"][2]:
                        compare(label+":status:"+key, values["strict"][2][key], values["safe_factor_cache"][2][key],
                                allow_positive_infinity=key == "min_innovation_eigen_gap")
                for arm, fn in functions.items():
                    record["graphs"][label+":"+arm] = graph(fn)

        starts = banks[1.]
        model, _ = bridges["strict"].component_target._batched_components(starts)
        covariance = tf.linalg.LinearOperatorBlockDiag([
            tf.linalg.LinearOperatorFullMatrix(model.initial_covariance),
            tf.linalg.LinearOperatorFullMatrix(model.innovation_covariance)]).to_dense()
        dimension = int(covariance.shape[-1])
        direction = tf.broadcast_to(tf.eye(dimension, dtype=tf.float64), (4, 4, dimension, dimension))
        functions = {}
        for arm, backend in BACKENDS.items():
            def factor(matrix, directions, backend=backend):
                result = _checked_batched_principal_sqrt_factor_first_derivatives(matrix, directions,
                    singular_floor=tf.constant(0., tf.float64), fixed_null_tolerance=tf.constant(1e-10, tf.float64),
                    factor_backend=backend, label="diagnostic_initial_covariance")
                return result.factor, result.d_factor, result.classified_invalid_count, result.roundoff_repair_count
            functions[arm] = tf.function(factor, input_signature=(
                tf.TensorSpec((4, dimension, dimension), tf.float64),
                tf.TensorSpec((4, 4, dimension, dimension), tf.float64)), jit_compile=True, autograph=False)
        for repeat in range(3):
            values = {}
            for arm in (tuple(BACKENDS) if repeat % 2 == 0 else tuple(reversed(BACKENDS))):
                values[arm] = measure("initial_factor_batch4", arm, lambda: functions[arm](covariance, direction))
            compare("initial_factor_batch4", values["strict"], values["safe_factor_cache"])
        record.update(placement_dimension=dimension, state_dimension=int(model.initial_mean.shape[-1]),
            innovation_dimension=int(model.innovation_covariance.shape[-1]),
            factor_directions="four identity directions; explanatory only")
        record["hmc_policy"] = {"epsilon": .011048543456039808, "L": 25, "transitions": 2,
            "coordinates": "existing_bootstrap_prior_scale", "parameter_scales": [4.]*4,
            "batch_starts": "four independent momenta at the same scalar bootstrap origin",
            "repeats": 3, "health": "all retained and proposed states and momenta"}

        for beta, starts in banks.items():
            latent = {}
            for arm, bridge in bridges.items():
                adapter = _QualifiedAdapter(bridge, beta=beta)
                geometry = initialize_hmc_kernel_geometry(adapter=adapter, initial_position=starts[0],
                    parameter_scales=tf.fill([4], tf.constant(4., tf.float64)))
                latent[arm] = build_bootstrap_fixed_mass_adapter(adapter=adapter,
                    mass_artifact=geometry.mass_artifact, mass_signature=geometry.mass_artifact_signature,
                    target_scope=adapter.target_scope)
            for size in (1, 4):
                initial = latent["strict"].initial_position()
                if size == 4:
                    initial = tf.broadcast_to(initial, (4, 4))
                label = f"hmc_beta{beta:g}_batch{size}_L25"
                runners = {}
                for arm, adapter in latent.items():
                    cfg = FullChainHMCConfig(num_results=2, num_burnin_steps=0,
                        step_size=.011048543456039808, num_leapfrog_steps=25, use_xla=True,
                        seed=scoped_seed(config, "factor-profile", beta, size), target_scope=adapter.target_scope,
                        target_status_trace_policy="per_chain_step", capture_candidate_health=True)
                    runners[arm] = ReusableFullChainHMCRunner(adapter, initial, cfg)
                for repeat in range(3):
                    seed = scoped_seed(config, "factor-profile-pair", beta, size, repeat)
                    record["seeds"].append({"label": label, "repeat": repeat, "seed": list(seed)})
                    values = {}
                    for arm in (tuple(BACKENDS) if repeat % 2 == 0 else tuple(reversed(BACKENDS))):
                        def run():
                            result = runners[arm].run(current_state=initial, seed=seed)
                            return result.samples, result.trace
                        values[arm] = measure(label, arm, run, estimate=180.)
                        try:
                            require_chunk_health(*values[arm])
                        except RuntimeError as exc:
                            if arm == "safe_factor_cache":
                                raise CandidateRejected(label+":health") from exc
                            raise
                    compare(label+f":repeat{repeat}", values["strict"], values["safe_factor_cache"])
                for arm, runner in runners.items():
                    record["graphs"][label+":"+arm] = graph(runner._runner)
                    if "GPU" not in values[arm][0].device:
                        raise ValueError("HMC endpoint did not execute on GPU")
                save()
        record["mass_forecasts"] = {}
        for beta in banks:
            for arm in BACKENDS:
                rows = record["measurements"][f"hmc_beta{beta:g}_batch1_L25:"+arm]
                per_transition = max(row["seconds"] for row in rows[1:])/2
                record["mass_forecasts"][f"beta{beta:g}:"+arm] = {
                    "seconds_per_transition": per_transition, "minimum_transitions": 1000,
                    "unmargined_seconds": per_transition*1000,
                    "with_existing_safety_factor_seconds": per_transition*1000*config["budget"]["forecast_safety_factor"],
                    "role": "fixed_start_fixed_metric_extrapolation_not_completed_adaptation_price"}
        record["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
        if source_snapshot(root) != record["sources"]:
            raise ValueError("source changed during profile")
        contention()
        record.update(status="completed", candidate_parity_passed=True)
    except CandidateRejected as exc:
        record.update(status="candidate_rejected", failure=str(exc), traceback=traceback.format_exc())
    except ProfileBudgetReached as exc:
        record.update(status="profile_budget_deferred", failure=str(exc))
    except GPUResourceUnavailable as exc:
        record.update(status="waiting_for_gpu", resource_receipt=exc.receipt)
    except BaseException as exc:
        record.update(status="failed", failure=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
