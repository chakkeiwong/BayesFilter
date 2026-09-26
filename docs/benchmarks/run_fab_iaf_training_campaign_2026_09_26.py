"""Paired FAB/reverse-KL training of the canonical IAF.

This is a bounded diagnostic campaign.  It trains one arm per process, with
identical map initialization, target, seed, dtype, batch and optimizer settings.
It does not run HMC or issue a posterior/default-readiness claim.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(host(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def host(value):
    if isinstance(value, dict):
        return {str(k): host(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [host(v) for v in value]
    if hasattr(value, "numpy"):
        return host(value.numpy())
    # TensorFlow scalar tensors become NumPy scalars at the boundary.  Use
    # ``item`` for scalars and ``tolist`` for arrays without importing NumPy
    # into this campaign's runtime path.  The latter also handles nested
    # arrays after recursively normalizing their elements.
    if hasattr(value, "item"):
        try:
            return host(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        return host(value.tolist())
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else None
    return value


def digest(seed: int, role: str) -> tuple[int, int]:
    raw = hashlib.sha256(f"fab-iaf-campaign-20260926:{seed}:{role}".encode()).digest()
    return tuple(int.from_bytes(raw[i:i + 4], "big") & 0x7fffffff for i in (0, 4))


class AnalyticBridge:
    """Small bridge adapter exposing the q20 method's diagnostic interface."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.parameter_dim = adapter.parameter_dim
        self.target_signature = adapter.target["target_signature"]
        self.signature = adapter.adapter_signature()

    def value_score_status(self, values, beta):
        del beta
        log_value, score, status = self.adapter.log_prob_and_grad_status(values)
        return log_value, score, {"bridge_valid": status["valid_pre_regularized_score"]}


def build_target(name: str):
    import tensorflow as tf

    if name == "q20":
        from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge

        bridge = make_q20_tempered_bridge(
            20, jit_compile=True,
            principal_sqrt_backend="tensorflow_eigh_strict_factor_cached",
        )
        adapter = bridge.fixed_beta_adapter(1.0)
        center = tuple(host(bridge.prior_center))
        scale_value = math.sqrt(float(bridge.prior_variance))
        scale = (scale_value,) * bridge.parameter_dim

        def target(values):
            value, score, status = bridge.value_score_status(
                values, tf.constant(1.0, tf.float64)
            )
            return value, score, status["bridge_valid"]

        return bridge, target, center, scale, adapter.adapter_signature(), None

    if name == "three_mode":
        from bayesfilter.testing.weighted_neutra_gaussian_mixture_hmc_tf import (
            AnalyticGaussianMixtureValueScoreAdapter,
            analytic_three_mode_target,
        )

        adapter = AnalyticGaussianMixtureValueScoreAdapter(analytic_three_mode_target())
        target_data = adapter.target
        probabilities = target_data["probabilities"]
        means = target_data["means"]
        covariances = target_data["covariances"]
        center_tensor = tf.reduce_sum(probabilities[:, None] * means, axis=0)
        centered = means - center_tensor[None, :]
        covariance = tf.reduce_sum(
            probabilities[:, None, None]
            * (covariances + centered[:, :, None] * centered[:, None, :]), axis=0
        )
        center = tuple(host(center_tensor))
        scale = tuple(host(tf.sqrt(tf.linalg.diag_part(covariance))))
        bridge = AnalyticBridge(adapter)

        def target(values):
            value, score, status = adapter.log_prob_and_grad_status(values)
            return value, score, status["valid_pre_regularized_score"]

        return bridge, target, center, scale, adapter.adapter_signature(), target_data

    raise ValueError(f"unknown target: {name}")


def target_log_score(bridge, values):
    import tensorflow as tf

    value, score, status = bridge.value_score_status(
        values, tf.constant(1.0, tf.float64)
    )
    return value, score, status["bridge_valid"]


def diagonal_gaussian_log_prob(points, center, scale):
    """Density of the actual independent-bank sampling law, in FP64."""
    import tensorflow as tf

    scale = tf.convert_to_tensor(scale, tf.float64)
    z = (points - tf.convert_to_tensor(center, tf.float64)) / scale
    return -0.5 * tf.reduce_sum(tf.square(z) + math.log(2.0 * math.pi), -1) - tf.reduce_sum(tf.math.log(scale))


def coverage_diagnostic(flow, initial_flow, bridge, center, scale, target_data, *, rows, seed):
    """Independent support checks; each importance denominator matches its draws."""
    import tensorflow as tf

    dimension, batch = bridge.parameter_dim, 32
    if rows % batch:
        raise ValueError("coverage rows must be whole native batches of 32")
    signature = [tf.TensorSpec([batch, dimension], tf.float64)]

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def evaluate(latent):
        mapped, ld = flow.forward_and_logdet(latent)
        reference = tf.constant(center, tf.float64) + tf.constant(scale, tf.float64) * latent
        points = tf.concat([mapped, reference], axis=0)
        log_p, score, valid = target_log_score(bridge, points)
        base_log = diagonal_gaussian_log_prob(latent, [0.] * dimension, [1.] * dimension)
        log_g = diagonal_gaussian_log_prob(reference, center, scale)
        reference_log_q = flow.log_prob(reference)
        initial_log_q = initial_flow.log_prob(reference)
        quantities = tf.stack([log_p[:batch], log_p[batch:], base_log - ld,
                               log_g, reference_log_q, initial_log_q])
        finite = tf.reduce_all(tf.math.is_finite(quantities)) & tf.reduce_all(tf.math.is_finite(score))
        return {"points": points, "log_weights": tf.concat([log_p[:batch] - (base_log - ld), log_p[batch:] - log_g], 0),
                "reference_log_p_over_q": log_p[batch:] - reference_log_q,
                "reference_log_p_over_initial_q": log_p[batch:] - initial_log_q,
                "valid": tf.reduce_all(valid) & finite & tf.reduce_all(tf.math.is_finite(points))}

    # Small stateless Gaussian banks are generated with TensorFlow's native CPU
    # kernel; process sharding would add overhead without expensive target work.
    with tf.device("/CPU:0"):
        latent = tf.random.stateless_normal([rows, dimension], seed, dtype=tf.float64)
    blocks = [evaluate(latent[i:i + batch]) for i in range(0, rows, batch)]
    if not all(bool(b["valid"]) for b in blocks):
        raise ValueError("independent coverage target/density evaluation was invalid")
    map_points = tf.concat([b["points"][:batch] for b in blocks], 0)
    ref_points = tf.concat([b["points"][batch:] for b in blocks], 0)
    map_log_weights = tf.concat([b["log_weights"][:batch] for b in blocks], 0)
    ref_log_weights = tf.concat([b["log_weights"][batch:] for b in blocks], 0)

    @tf.function(input_signature=[tf.TensorSpec([rows], tf.float64), tf.TensorSpec([rows, dimension], tf.float64)],
                 jit_compile=True, autograph=False)
    def summarize(log_weights, points):
        weights = tf.nn.softmax(log_weights)
        positive = tf.cast(points[:, 2] > 0., tf.float64)
        output = {
            "ess": 1.0 / tf.reduce_sum(tf.square(weights)),
            "ess_fraction": 1.0 / tf.reduce_sum(tf.square(weights)) / rows,
            "maximum_normalized_weight": tf.reduce_max(weights),
            "positive_unweighted_fraction": tf.reduce_mean(positive),
            "positive_weighted_fraction": tf.reduce_sum(weights * positive),
        }
        if target_data is not None:
            from bayesfilter.testing.importance_sampling_tf import gaussian_mixture_log_prob_responsibilities_score
            _, responsibilities, _ = gaussian_mixture_log_prob_responsibilities_score(
                points, target_data["probabilities"], target_data["means"], target_data["covariances"])
            labels = tf.argmin(tf.reduce_sum(tf.square(points[:, None, :] - target_data["means"][None, :, :]), -1), -1)
            output["nearest_mean_occupancy"] = tf.reduce_mean(tf.one_hot(labels, 3, dtype=tf.float64), 0)
            output["component_unweighted_mass"] = tf.reduce_mean(responsibilities, 0)
            output["component_weighted_mass"] = tf.reduce_sum(weights[:, None] * responsibilities, 0)
            output["raw_component_mass_absolute_error"] = tf.abs(output["component_unweighted_mass"] - target_data["probabilities"])
            output["weighted_component_mass_absolute_error"] = tf.abs(output["component_weighted_mass"] - target_data["probabilities"])
        return output

    coverage = {"rows": rows, "batch_size": batch, "seed": list(seed),
        "map_base": host(summarize(map_log_weights, map_points)),
        "independent_prior_bank": host(summarize(ref_log_weights, ref_points)),
        "reference_sampling_law": "diagonal_gaussian_center_scale_in_manifest",
        "reference_log_weight_definition": "log_p_minus_log_g_not_log_p_minus_log_q",
        "role": "descriptive_coverage_diagnostic_not_exhaustive_mode_discovery"}
    weights = tf.nn.softmax(ref_log_weights)
    for name in ("reference_log_p_over_q", "reference_log_p_over_initial_q"):
        values = tf.concat([b[name] for b in blocks], 0)
        coverage[name] = host({"target_weighted_mean": tf.reduce_sum(weights * values),
                              "maximum_on_bank": tf.reduce_max(values)})
    if target_data is not None:
        with tf.device("/CPU:0"):
            labels = tf.random.stateless_categorical(tf.math.log(target_data["probabilities"])[None, :], rows,
                                                    (seed[1], seed[0]))[0]
            normals = tf.random.stateless_normal([rows, dimension], (seed[0], seed[0]), dtype=tf.float64)
            exact = tf.gather(target_data["means"], labels) + tf.linalg.matvec(
                tf.gather(tf.linalg.cholesky(target_data["covariances"]), labels), normals)

        @tf.function(input_signature=signature, jit_compile=True, autograph=False)
        def exact_evaluate(points):
            log_p, _, valid = target_log_score(bridge, points)
            ratios = log_p - flow.log_prob(points)
            return ratios, tf.reduce_all(valid) & tf.reduce_all(tf.math.is_finite(ratios))

        exact_blocks = [exact_evaluate(exact[i:i + batch]) for i in range(0, rows, batch)]
        if not all(bool(b[1]) for b in exact_blocks):
            raise ValueError("exact mixture coverage reference was invalid")
        coverage["exact_component_probabilities"] = host(target_data["probabilities"])
        coverage["exact_target_draws"] = host(summarize(tf.zeros([rows], tf.float64), exact))
        coverage["exact_target_draws"]["forward_kl_monte_carlo"] = host(tf.reduce_mean(tf.concat([b[0] for b in exact_blocks], 0)))
    return coverage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("fab", "reverse_kl"), required=True)
    parser.add_argument("--target", choices=("q20", "three_mode"), default="q20")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", choices=("0", "1", "2"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--worker-seconds", type=float, default=2400.0)
    parser.add_argument("--deadline-utc", default="none")
    parser.add_argument("--replay-capacity", type=int, default=4096)
    parser.add_argument("--replay-min-size", type=int, default=1280)
    parser.add_argument("--updates-per-pass", type=int, default=4)
    parser.add_argument("--post-rows", type=int, default=1000)
    parser.add_argument("--coverage-rows", type=int, default=4096)
    args = parser.parse_args()
    if args.updates < 1 or args.batch_size < 2 or args.worker_seconds <= 0:
        raise ValueError("updates, batch size and worker time must be positive")
    if args.post_rows != 1000 or args.coverage_rows < 32 or args.coverage_rows % 32:
        raise ValueError("the campaign requires the standard 1,000-point probe")
    if args.arm == "fab" and args.updates % args.updates_per_pass:
        raise ValueError("requested updates must be divisible by updates per pass")
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise RuntimeError("TF_FORCE_GPU_ALLOW_GROWTH=true is required before TensorFlow import")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "1"
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def deadline_signal(signum, frame):
        del signum, frame
        raise TimeoutError("external worker wall budget expired")
    signal.signal(signal.SIGTERM, deadline_signal)
    manifest = {
        "schema": "bayesfilter.neutra.fab_iaf_campaign.manifest.v1",
        "command": sys.argv,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "plan": "docs/plans/bayesfilter-fab-iaf-training-campaign-2026-09-26.md",
        "arm": args.arm,
        "target": args.target,
        "seed": args.seed,
        "gpu": args.gpu,
        "worker_seconds": args.worker_seconds,
        "deadline_utc": args.deadline_utc,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "memory_growth_env": True,
        "tf32": False,
        "jit_compile": True,
        "batch_native_target": True,
        "sample_wise_fallback": False,
        "environment": sys.executable,
        "cpu_threads": {"intra": os.environ["TF_NUM_INTRAOP_THREADS"], "inter": os.environ["TF_NUM_INTEROP_THREADS"]},
        "result": str(args.output / "result.json"),
        "output": str(args.output),
    }
    save(args.output / "manifest.json", manifest)
    result = {
        "schema": "bayesfilter.neutra.fab_iaf_campaign.result.v1",
        "status": "started",
        "posterior_ready": False,
        "default_ready": False,
    }
    trainer = None
    try:
        import tensorflow as tf

        from bayesfilter.runtime.gpu_memory_policy import (
            configure_tensorflow_gpu_memory_growth,
        )

        memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        tf.config.set_soft_device_placement(False)
        from bayesfilter.inference.neutra_fab import FABConfig, FABTrainer
        from bayesfilter.inference.neutra_post_training import PostTrainingProbe
        from bayesfilter.inference.neutra_transport import (
            NeuTraOptimizerConfig,
            NeuTraTransport,
            NeuTraTransportConfig,
            NeuTraTransportTrainer,
        )

        bridge, target, center, scale, target_signature, target_data = build_target(args.target)
        dimension = bridge.parameter_dim
        map_config = replace(
            NeuTraTransportConfig.hoffman_author_iaf(
                dimension,
                conditional_scale_cap=2.0,
                seed=digest(args.seed, "map"),
                dtype="float32",
            ),
            hidden_layers=(16, 16),
            affine_center=center,
            affine_scale=scale,
        )
        flow = NeuTraTransport(map_config)
        initial_flow = flow.as_dtype("float64")
        if args.arm == "fab":
            fab_config = FABConfig(
                args.batch_size,
                10,
                1,
                1,
                0.3,
                1.0e-3,
                0.9,
                0.999,
                1.0e-8,
                args.replay_capacity,
                args.replay_min_size,
                args.updates_per_pass,
                10.0,
                None,
                True,
                0.65,
                1.02,
                jit_compile=True,
                transition_operator="metropolis",
            )
            trainer = FABTrainer(
                flow, target, fab_config,
                target_signature=target_signature,
                seed=digest(args.seed, "fab"),
            )
            optimizer_config = asdict(fab_config)
        else:
            optimizer_config_obj = NeuTraOptimizerConfig(
                args.batch_size, "standard", 1.0e-3, 0.9, 0.999,
                1.0e-8, None, jit_compile=True, target_dtype="float64",
            )
            trainer = NeuTraTransportTrainer(
                flow, target, optimizer_config_obj,
                target_signature=target_signature,
            )
            optimizer_config = asdict(optimizer_config_obj)
        manifest.update(
            memory_policy=memory,
            tensorflow=tf.__version__,
            target_signature=target_signature,
            bridge_signature=bridge.signature,
            transport_config=map_config.payload(),
            optimizer_config=optimizer_config,
            transport_dtype="float32",
            target_dtype="float64",
            data_version=bridge.signature,
            random_seeds={role: digest(args.seed, role) for role in ("map", "fab", "probe", "coverage")},
            optimizer_implementation="Optax_form_FABAdam" if args.arm == "fab" else "Keras_Adam_epsilon_placement_differs",
            source_sha256={
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (
                    ROOT / "bayesfilter/inference/neutra_fab.py",
                    ROOT / "bayesfilter/inference/neutra_transport.py",
                    ROOT / "bayesfilter/inference/neutra_transport_core.py",
                    Path(__file__),
                )
            },
        )
        save(args.output / "manifest.json", manifest)
        save(args.output / "initial-checkpoint.json", {
            "schema": "bayesfilter.neutra.fab_iaf_campaign.initial.v1",
            "target": args.target,
            "target_signature": target_signature,
            "transport_config": map_config.payload(),
            "parameters": flow.parameter_state(),
        })

        def elapsed() -> float:
            return time.monotonic() - started

        def available(reserve: float = 0.0) -> bool:
            if elapsed() + reserve >= args.worker_seconds:
                return False
            if args.deadline_utc != "none":
                deadline = datetime.fromisoformat(args.deadline_utc)
                if deadline.tzinfo is None or datetime.now(timezone.utc) >= deadline:
                    return False
            return True

        progress = []
        pass_count = 0
        update_count = 0
        diagnostic_started = time.monotonic()
        initial_probe_runner = PostTrainingProbe(initial_flow, bridge, 1.0, rows=args.post_rows,
                                                batch_size=20, jit_compile=True)
        probe_seed = digest(args.seed, "probe")
        initial_latent = initial_probe_runner.latent_bank(probe_seed)
        initial_blocks = [initial_probe_runner.batch(initial_latent[:20])]
        steady_started = time.monotonic()
        initial_blocks.extend(initial_probe_runner.batch(initial_latent[i:i + 20])
                              for i in range(20, args.post_rows, 20))
        steady_probe_seconds = (time.monotonic() - steady_started) * args.post_rows / (args.post_rows - 20)
        initial_probe = initial_probe_runner.summarize(initial_blocks, probe_seed)
        save(args.output / "initial-post-training-1000.json", initial_probe)
        if not (initial_probe["complete"] and initial_probe["finite"] and initial_probe["valid_rows"] == args.post_rows):
            raise ValueError("initial 1,000-point verification failed")
        initial_probe_seconds = time.monotonic() - diagnostic_started
        # Same target backend: final probe plus 2*coverage_rows evaluations.
        # 1.5 is a scheduling margin, not a scientific threshold; 40 seconds
        # reserves fresh diagnostic graphs. External timeout enforces the cap.
        diagnostic_reserve = steady_probe_seconds * (1.0 + 2.0 * args.coverage_rows / args.post_rows) * 1.5 + 40.
        result.update(initial_probe_seconds=initial_probe_seconds, steady_probe_seconds=steady_probe_seconds,
                      diagnostic_reserve_seconds=diagnostic_reserve)
        save(args.output / "timing.json", result)
        save(args.output / "checkpoint.json", trainer.checkpoint())
        while update_count < args.updates:
            pass_reserve = max([r["seconds"] for r in progress[-2:]] or [0.])
            if not available(diagnostic_reserve + pass_reserve):
                result["status"] = "partial_budget"
                break
            pass_started = time.monotonic()
            if args.arm == "fab":
                row = trainer.step(train=True)
                update_count = int(trainer.optimizer.iterations.numpy())
                pass_count = int(trainer.pass_index)
                updates = row.get("updates", ())
                finite = bool(row["valid"])
                row_out = {
                    "pass": pass_count,
                    "optimizer_updates": update_count,
                    "ais_ess": float(row["ess_fraction"].numpy()) * args.batch_size,
                    "ais_acceptance": float(tf.reduce_mean(row["acceptance"]).numpy()),
                    "ais_positive_count": int(tf.reduce_sum(tf.cast(row["x"][:, 2] > 0., tf.int32))),
                    "update_count_this_pass": len(updates),
                    "update_losses": [host(item["loss"]) for item in updates],
                    "gradient_norms": [host(item["gradient_norm"]) for item in updates],
                    "correction_clipped_fractions": [host(item["correction_clipped_fraction"]) for item in updates],
                    "maximum_ais_weight": host(row["max_weight"]),
                    "ais_movement": host(row["movement"]),
                    "ais_invalid_proposals": host(row["invalid_proposals"]),
                    "mutation_step_sizes": host(row["steps"]),
                    "replay_rows": int(trainer.replay.size.numpy()) if trainer.replay is not None else 0,
                    "valid": finite,
                }
            else:
                z = tf.random.stateless_normal(
                    [args.batch_size, dimension], digest(args.seed, f"reverse:{update_count}"),
                    dtype=tf.float32,
                )
                row = trainer.train_step(z)
                update_count = int(trainer.optimizer.iterations.numpy())
                pass_count = update_count
                finite = bool(row["valid"])
                row_out = {
                    "pass": pass_count,
                    "optimizer_updates": update_count,
                    "loss": host(row["loss"]),
                    "gradient_norm": host(row["gradient_norm"]),
                    "clipped_gradient_norm": host(row["clipped_gradient_norm"]),
                    "valid": finite,
                }
            row_out["seconds"] = time.monotonic() - pass_started
            row_out["elapsed_seconds"] = elapsed()
            progress.append(row_out)
            save(args.output / "progress.json", progress)
            checkpoint = trainer.checkpoint()
            save(args.output / "checkpoint.json", checkpoint)
            if update_count and update_count % 80 == 0:
                save(args.output / f"checkpoints/update-{update_count:06d}.json", checkpoint)
            if not finite:
                raise ValueError("nonfinite or invalid optimizer update")
        else:
            result["status"] = "updates_complete"
        result.update(
            passes=pass_count,
            optimizer_updates=update_count,
            requested_updates=args.updates,
            progress_rows=len(progress),
            training_complete=update_count == args.updates,
        )
        # Exercise the checkpoint validator on the exact state before export.
        trainer.restore(json.loads((args.output / "checkpoint.json").read_text()))

        # The probe evaluates the exact represented map in FP64.  It is a
        # standard post-training verification and remains geometry evidence.
        evaluation_map = flow.as_dtype("float64")
        probe = PostTrainingProbe(
            evaluation_map, bridge, 1.0,
            rows=args.post_rows,
            batch_size=20,
            jit_compile=True,
        )(digest(args.seed, "probe"))
        save(args.output / "post-training-1000.json", probe)
        if not (
            probe["complete"]
            and probe["finite"]
            and probe["valid_rows"] == args.post_rows
        ):
            raise ValueError("post-training 1,000-point verification failed")

        coverage = coverage_diagnostic(
            evaluation_map, initial_flow, bridge, center, scale, target_data,
            rows=args.coverage_rows, seed=digest(args.seed, "coverage"))
        save(args.output / "coverage.json", coverage)
        frozen = evaluation_map.frozen_payload(
            target_signature=target_signature,
            training_state_hash=trainer.checkpoint()["checkpoint_hash"],
        )
        save(args.output / "frozen-map.json", frozen)
        from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
        load_frozen_neutra_artifact(frozen, expected_target_signature=target_signature)
        result.update(
            status="complete" if result.get("status") == "updates_complete" else result["status"],
            post_training_complete=True,
            coverage_complete=True,
            training_quality_established=False,
            posterior_qualified=False,
            gpu_allocator=tf.config.experimental.get_memory_info("GPU:0"),
            artifact_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(args.output.glob("*.json")) if p.name not in {"manifest.json", "result.json"}},
        )
    except Exception as exc:
        result.update(
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        print(result["traceback"], flush=True)
        if trainer is not None:
            try:
                save(args.output / "failure-checkpoint.json", trainer.checkpoint())
            except Exception:
                pass
    finally:
        result["wall_seconds"] = time.monotonic() - started
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["wall_seconds"] = result["wall_seconds"]
        save(args.output / "manifest.json", manifest)
        save(args.output / "result.json", result)
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result["status"] in {"complete", "partial_budget"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
