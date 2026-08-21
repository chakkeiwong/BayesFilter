#!/usr/bin/env python3
"""Run generic five-stage or matched cold NeuTra on one known-law target."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = ROOT / "docs/plans/bayesfilter-neutra-generic-five-stage-training-plan-2026-08-15.md"
INTERVAL_LEVEL = 0.999
CRITICAL_VALUE = 3.2905267314919255
EXACT_TAIL_PROBABILITY = 0.02275013194817921


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--target", choices=("funnel", "gaussian", "banana", "mixture"), required=True
    )
    parser.add_argument("--route", choices=("staged", "cold"), required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rates", default="2e-4,5e-4,1e-3")
    parser.add_argument("--affine-updates", type=int, default=250)
    parser.add_argument("--simple-updates", type=int, default=2000)
    parser.add_argument("--progressive-updates", type=int, default=500)
    parser.add_argument("--joint-updates", type=int, default=1000)
    parser.add_argument("--cold-updates", type=int, default=5000)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--proposal-audit-count", type=int, default=131072)
    return parser.parse_args()


def _rates(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value.strip()) for value in str(raw).split(","))
    except ValueError as error:
        raise ValueError("learning-rates must be comma-separated numbers") from error
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("learning-rates must be finite and positive")
    return values


def _ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_ready(item) for item in value]
    if hasattr(value, "numpy"):
        return _ready(value.numpy().tolist())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_ready(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _ready(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _scheduled_learning_rate(peak: float, update: int, total_updates: int) -> float:
    fraction = float(update) / float(total_updates)
    multiplier = 1.0 if fraction < 0.60 else 0.1 if fraction < 0.85 else 0.01
    return float(peak) * multiplier


def _mean_interval(tf: Any, values: Any, exact: Any) -> Mapping[str, Any]:
    rows = tf.convert_to_tensor(values, tf.float64)
    truth = tf.convert_to_tensor(exact, tf.float64)
    count = tf.cast(tf.shape(rows)[0], tf.float64)
    estimate = tf.reduce_mean(rows, axis=0)
    centered = rows - estimate
    variance = tf.reduce_sum(tf.square(centered), axis=0) / (count - 1.0)
    standard_error = tf.sqrt(variance / count)
    radius = tf.constant(CRITICAL_VALUE, tf.float64) * standard_error
    passed = tf.logical_and(truth >= estimate - radius, truth <= estimate + radius)
    return {
        "estimate": estimate,
        "exact": truth,
        "standard_error": standard_error,
        "lower": estimate - radius,
        "upper": estimate + radius,
        "passed": passed,
        "all_passed": tf.reduce_all(passed),
    }


def _wilson_interval(tf: Any, successes: Any, count: int, exact: float) -> Mapping[str, Any]:
    n = tf.constant(float(count), tf.float64)
    estimate = tf.cast(successes, tf.float64) / n
    z = tf.constant(CRITICAL_VALUE, tf.float64)
    z2 = tf.square(z)
    denominator = 1.0 + z2 / n
    center = (estimate + z2 / (2.0 * n)) / denominator
    radius = z * tf.sqrt(estimate * (1.0 - estimate) / n + z2 / (4.0 * n * n)) / denominator
    exact_tensor = tf.constant(float(exact), tf.float64)
    return {
        "estimate": estimate,
        "exact": exact_tensor,
        "lower": center - radius,
        "upper": center + radius,
        "passed": tf.logical_and(exact_tensor >= center - radius, exact_tensor <= center + radius),
    }


def _model(tf: Any, name: str) -> Mapping[str, Any]:
    if name == "funnel":
        from bayesfilter.inference.neutra_paper_d100_target import (
            make_paper_funnel_spec,
            paper_d100_log_prob_batch,
        )

        spec = make_paper_funnel_spec()
        return {
            "name": name,
            "dimension": 100,
            "hidden_width": 100,
            "stages": 3,
            "stage_caps": (4.0, 0.5, 0.5),
            "target_log_prob": lambda rows: paper_d100_log_prob_batch(spec, rows),
            "exact_latent": lambda rows: tf.concat(
                (rows[:, :1], rows[:, 1:] * tf.exp(-rows[:, :1])), axis=1
            ),
            "manifest": spec.manifest_payload(),
        }
    if name == "gaussian":
        dimension = 16
        mean = tf.linspace(tf.constant(-0.75, tf.float64), tf.constant(0.75, tf.float64), dimension)
        diagonal = tf.linspace(tf.constant(0.6, tf.float64), tf.constant(1.8, tf.float64), dimension)
        factor = tf.linalg.diag(diagonal)
        factor = factor + tf.linalg.diag(tf.fill((dimension - 1,), tf.constant(0.25, tf.float64)), k=-1)
        factor = factor + tf.linalg.diag(tf.fill((dimension - 2,), tf.constant(0.10, tf.float64)), k=-2)

        def exact_latent(rows: Any) -> Any:
            return tf.transpose(
                tf.linalg.triangular_solve(factor, tf.transpose(rows - mean), lower=True)
            )

        return {
            "name": name,
            "dimension": dimension,
            "hidden_width": 32,
            "stages": 3,
            "stage_caps": (4.0, 0.5, 0.5),
            "target_log_prob": lambda rows: -0.5
            * tf.reduce_sum(tf.square(exact_latent(rows)), axis=1),
            "exact_latent": exact_latent,
            "manifest": {
                "schema": "bayesfilter.neutra.generic_control.gaussian.v1",
                "dimension": dimension,
                "mean": mean,
                "factor": factor,
            },
        }
    if name == "banana":
        dimension = 16
        curvature = tf.constant(0.35, tf.float64)

        def exact_latent(rows: Any) -> Any:
            return tf.concat(
                (
                    rows[:, :1],
                    rows[:, 1:2] - curvature * (tf.square(rows[:, :1]) - 1.0),
                    rows[:, 2:],
                ),
                axis=1,
            )

        return {
            "name": name,
            "dimension": dimension,
            "hidden_width": 32,
            "stages": 3,
            "stage_caps": (4.0, 0.5, 0.5),
            "target_log_prob": lambda rows: -0.5
            * tf.reduce_sum(tf.square(exact_latent(rows)), axis=1),
            "exact_latent": exact_latent,
            "manifest": {
                "schema": "bayesfilter.neutra.generic_control.banana.v1",
                "dimension": dimension,
                "curvature": curvature,
                "unit_jacobian": True,
            },
        }
    from bayesfilter.testing.weighted_neutra_gaussian_mixture_hmc_tf import (
        analytic_three_mode_target,
    )
    from bayesfilter.testing.importance_sampling_tf import gaussian_mixture_log_prob

    target = analytic_three_mode_target()
    return {
        "name": name,
        "dimension": 4,
        "hidden_width": 32,
        "stages": 3,
        "stage_caps": (4.0, 0.5, 0.5),
        "target_log_prob": lambda rows: gaussian_mixture_log_prob(
            rows,
            target["probabilities"],
            target["means"],
            target["covariances"],
        ),
        "mixture": target,
        "manifest": target["signature_payload"],
    }


def _selection_loss(tf: Any, transport: Any, target_log_prob: Any, latent: Any) -> Any:
    physical, logdet = transport.forward_and_logdet(latent)
    rows = -target_log_prob(physical) - logdet
    tf.debugging.assert_all_finite(rows, "held-out reverse KL")
    return tf.reduce_mean(rows)


def _proposal_audit(
    tf: Any,
    transport: Any,
    model: Mapping[str, Any],
    *,
    sample_count: int,
    seed: tuple[int, int],
) -> Mapping[str, Any]:
    latent = tf.random.stateless_normal(
        (int(sample_count), int(model["dimension"])), seed=seed, dtype=tf.float64
    )
    physical, logdet = transport.forward_and_logdet(latent)
    target = model["target_log_prob"](physical)
    log_base = -0.5 * (
        tf.reduce_sum(tf.square(latent), axis=1)
        + tf.cast(model["dimension"], tf.float64)
        * tf.math.log(tf.constant(2.0 * math.pi, tf.float64))
    )
    log_ratio = target + logdet - log_base
    normalized = tf.nn.softmax(log_ratio)
    common = {
        "sample_count": int(sample_count),
        "interval_level": INTERVAL_LEVEL,
        "importance_ess_fraction": tf.math.reciprocal(tf.reduce_sum(tf.square(normalized)))
        / tf.cast(sample_count, tf.float64),
        "maximum_normalized_importance_weight": tf.reduce_max(normalized),
        "log_target_to_proposal_ratio_stddev": tf.math.reduce_std(log_ratio),
    }
    if model["name"] == "funnel":
        y = physical[:, 0]
        residual = physical[:, 1:] * tf.exp(-y[:, tf.newaxis])
        screens = {
            "y_mean": _mean_interval(tf, y, 0.0),
            "y_second_moment": _mean_interval(tf, tf.square(y), 1.0),
            "residual_mean": _mean_interval(tf, tf.reduce_mean(residual, axis=1), 0.0),
            "residual_second_moment": _mean_interval(
                tf, tf.reduce_mean(tf.square(residual), axis=1), 1.0
            ),
        }
        tails = {
            "below_minus2": _wilson_interval(
                tf,
                tf.reduce_sum(tf.cast(y < -2.0, tf.int64)),
                sample_count,
                EXACT_TAIL_PROBABILITY,
            ),
            "above_plus2": _wilson_interval(
                tf,
                tf.reduce_sum(tf.cast(y > 2.0, tf.int64)),
                sample_count,
                EXACT_TAIL_PROBABILITY,
            ),
        }
        passed = tf.reduce_all(
            tf.stack(
                [
                    *(screen["all_passed"] for screen in screens.values()),
                    *(screen["passed"] for screen in tails.values()),
                ]
            )
        )
        return {**common, "screens": screens, "tails": tails, "passed": passed}
    if model["name"] in {"gaussian", "banana"}:
        exact_latent = model["exact_latent"](physical)
        cross = exact_latent[:, :-1] * exact_latent[:, 1:]
        screens = {
            "coordinate_mean": _mean_interval(
                tf, exact_latent, tf.zeros((int(model["dimension"]),), tf.float64)
            ),
            "coordinate_second_moment": _mean_interval(
                tf, tf.square(exact_latent), tf.ones((int(model["dimension"]),), tf.float64)
            ),
            "adjacent_cross_moment": _mean_interval(
                tf, cross, tf.zeros((int(model["dimension"]) - 1,), tf.float64)
            ),
        }
        passed = tf.reduce_all(
            tf.stack([screen["all_passed"] for screen in screens.values()])
        )
        return {**common, "screens": screens, "passed": passed}

    from bayesfilter.testing.gaussian_mixture_diagnostics_tf import gaussian_mixture_moments
    from bayesfilter.testing.importance_sampling_tf import (
        gaussian_mixture_log_prob_responsibilities_score,
    )

    mixture = model["mixture"]
    _value, responsibilities, _score = gaussian_mixture_log_prob_responsibilities_score(
        physical,
        mixture["probabilities"],
        mixture["means"],
        mixture["covariances"],
    )
    moments = gaussian_mixture_moments(
        mixture["probabilities"], mixture["means"], mixture["covariances"]
    )
    exact_second = tf.linalg.diag_part(moments["covariance"]) + tf.square(moments["mean"])
    screens = {
        "component_responsibility_mass": _mean_interval(
            tf, responsibilities, mixture["probabilities"]
        ),
        "coordinate_mean": _mean_interval(tf, physical, moments["mean"]),
        "coordinate_second_moment": _mean_interval(
            tf, tf.square(physical), exact_second
        ),
    }
    passed = tf.reduce_all(tf.stack([screen["all_passed"] for screen in screens.values()]))
    return {**common, "screens": screens, "passed": passed}


def _transport(tf: Any, model: Mapping[str, Any], seed_index: int) -> Any:
    from bayesfilter.inference.neutra_weighted_training import (
        WeightedDenseIAFTransport,
        WeightedNeuTraConfig,
    )

    stages = int(model["stages"])
    config = WeightedNeuTraConfig(
        dimension=int(model["dimension"]),
        hidden_layers=(int(model["hidden_width"]), int(model["hidden_width"])),
        stages=stages,
        activation="elu",
        s_max=1.0,
        stage_s_max=tuple(model["stage_caps"]),
        stage_unbounded_scale_linear=(True,) + (False,) * (stages - 1),
        permutation_policy="full_reverse",
        initialization_scale=0.02,
        initialization_seed=(20260815, 40001 + 100 * int(seed_index)),
        learning_rate=1.0e-3,
        gradient_clip_norm=10.0,
        jit_compile=True,
    )
    return WeightedDenseIAFTransport(config)


def _stage_summary(result: Any) -> list[Mapping[str, Any]]:
    return [
        {
            "name": stage.name,
            "stage": stage.stage,
            "active_groups": stage.active_groups,
            "trainable_variables": stage.trainable_variables,
            "incoming_loss": stage.incoming_loss,
            "selected_learning_rate": stage.selected_learning_rate,
            "selected_update": stage.selected_update,
            "selected_loss": stage.selected_loss,
            "candidates": [
                {
                    "learning_rate": candidate.learning_rate,
                    "selected_update": candidate.selected_update,
                    "selected_loss": candidate.selected_loss,
                    "terminal_loss": candidate.terminal_loss,
                    "clipped_updates": candidate.clipped_updates,
                    "gradient_norm": candidate.gradient_norm,
                }
                for candidate in stage.candidates
            ],
        }
        for stage in result.stages
    ]


def _cold_train(
    tf: Any,
    model: Mapping[str, Any],
    rates: tuple[float, ...],
    *,
    updates: int,
    checkpoint_every: int,
    batch_size: int,
    seed_index: int,
    selection_latent: Any,
) -> tuple[Any, Mapping[str, Any]]:
    from bayesfilter.inference.neutra_weighted_training import MatchedReverseKLNeuTraTrainer

    candidates = []
    selected_trainer = None
    selected_loss = float("inf")
    selected_state = None
    for rate_index, rate in enumerate(rates):
        transport = _transport(tf, model, seed_index)
        trainer = MatchedReverseKLNeuTraTrainer(transport.config, model["target_log_prob"])
        best_loss = float(
            _selection_loss(tf, trainer.transport, model["target_log_prob"], selection_latent).numpy()
        )
        best_update = 0
        best_state = [tf.identity(variable) for variable in trainer.variables]
        clipped_updates = 0
        terminal_loss = best_loss
        for update in range(1, int(updates) + 1):
            trainer.optimizer.learning_rate.assign(
                _scheduled_learning_rate(rate, update, int(updates))
            )
            latent = tf.random.stateless_normal(
                (int(batch_size), int(model["dimension"])),
                seed=(20260815 + int(seed_index), 41000 + update),
                dtype=tf.float64,
            )
            step = trainer.train_step(latent)
            clipped_updates += int(bool(step.clipping_applied.numpy()))
            if update % int(checkpoint_every) == 0 or update == int(updates):
                terminal_loss = float(
                    _selection_loss(
                        tf, trainer.transport, model["target_log_prob"], selection_latent
                    ).numpy()
                )
                if terminal_loss < best_loss:
                    best_loss = terminal_loss
                    best_update = update
                    best_state = [tf.identity(variable) for variable in trainer.variables]
        candidates.append(
            {
                "learning_rate": rate,
                "selected_update": best_update,
                "selected_loss": best_loss,
                "terminal_loss": terminal_loss,
                "clipped_updates": clipped_updates,
            }
        )
        if best_loss < selected_loss:
            selected_loss = best_loss
            selected_trainer = trainer
            selected_state = best_state
    if selected_trainer is None or selected_state is None:
        raise RuntimeError("cold route produced no candidate")
    for variable, value in zip(selected_trainer.variables, selected_state, strict=True):
        variable.assign(value)
    selected = min(candidates, key=lambda row: (row["selected_loss"], row["learning_rate"]))
    return selected_trainer.transport, {"selected": selected, "candidates": candidates}


def main() -> int:
    args = _args()
    rates = _rates(args.learning_rates)
    if int(args.batch_size) <= 1 or int(args.proposal_audit_count) < 32768:
        raise ValueError("batch size must exceed one and audit count must be at least 32768")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"output root must be fresh: {output}")
    if not PLAN.is_file():
        raise FileNotFoundError("reviewed plan is missing")
    output.mkdir(parents=True)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    started = time.perf_counter()

    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical_gpus) != 1:
        raise RuntimeError(f"expected one visible GPU, found {logical_gpus}")

    model = _model(tf, str(args.target))
    selection_latent = tf.random.stateless_normal(
        (65536, int(model["dimension"])), seed=(20260815, 42001), dtype=tf.float64
    )
    if args.route == "staged":
        from bayesfilter.inference.neutra_staged_training import (
            dense_iaf_five_stage_spec,
            dense_iaf_five_stage_variable_groups,
            train_neutra_five_stage,
        )

        transport = _transport(tf, model, int(args.seed_index))
        stage_spec = dense_iaf_five_stage_spec(
            stages=int(model["stages"]),
            learning_rates=rates,
            affine_updates=int(args.affine_updates),
            simple_updates=int(args.simple_updates),
            progressive_updates=int(args.progressive_updates),
            joint_updates=int(args.joint_updates),
            checkpoint_every=int(args.checkpoint_every),
        )
        phase_index = {
            phase.name: index for index, phase in enumerate(stage_spec.optimizer_phases())
        }

        def latent_batch(phase: str, update: int, _candidate: int) -> Any:
            return tf.random.stateless_normal(
                (int(args.batch_size), int(model["dimension"])),
                seed=(20260815 + int(args.seed_index), 43000 + 10000 * phase_index[phase] + update),
                dtype=tf.float64,
            )

        staged = train_neutra_five_stage(
            transport=transport,
            target_log_prob_fn=model["target_log_prob"],
            variable_groups=dense_iaf_five_stage_variable_groups(transport),
            spec=stage_spec,
            latent_batch_fn=latent_batch,
            selection_loss_fn=lambda active: _selection_loss(
                tf, active, model["target_log_prob"], selection_latent
            ),
            validation_fn=lambda active: _proposal_audit(
                tf,
                active,
                model,
                sample_count=int(args.proposal_audit_count),
                seed=(20260815, 44001 + int(args.seed_index)),
            ),
            gradient_clip_norm=10.0,
            jit_compile=True,
        )
        validation = staged.validation
        training = {
            "route": "generic_five_stage",
            "stages": _stage_summary(staged),
            "selected_path_updates": staged.selected_path_updates,
            "tuning_optimizer_updates": staged.tuning_optimizer_updates,
            "nonclaims": staged.nonclaims,
        }
    else:
        transport, training = _cold_train(
            tf,
            model,
            rates,
            updates=int(args.cold_updates),
            checkpoint_every=int(args.checkpoint_every),
            batch_size=int(args.batch_size),
            seed_index=int(args.seed_index),
            selection_latent=selection_latent,
        )
        validation = _proposal_audit(
            tf,
            transport,
            model,
            sample_count=int(args.proposal_audit_count),
            seed=(20260815, 44001 + int(args.seed_index)),
        )

    state = {
        "schema": "bayesfilter.neutra.generic_five_stage_state.v1",
        "target": model["manifest"],
        "route": str(args.route),
        "config": transport.config.manifest_payload(),
        "variables": [variable.numpy().tolist() for variable in transport.trainable_variables],
    }
    state["state_hash"] = _stable_hash(state)
    allocator = tf.config.experimental.get_memory_info("GPU:0")
    manifest = {
        "schema": "bayesfilter.neutra.generic_five_stage_manifest.v1",
        "plan": PLAN.as_posix(),
        "target": model["manifest"],
        "target_name": str(args.target),
        "route": str(args.route),
        "config": transport.config.manifest_payload(),
        "learning_rates": rates,
        "batch_size": int(args.batch_size),
        "seed_index": int(args.seed_index),
        "updates": {
            "affine": int(args.affine_updates),
            "simple": int(args.simple_updates),
            "progressive_each": int(args.progressive_updates),
            "joint": int(args.joint_updates),
            "cold": int(args.cold_updates),
        },
        "proposal_audit_count": int(args.proposal_audit_count),
        "jit_compile": True,
        "dtype": "float64",
        "tf32_enabled": False,
        "sample_wise_loop_or_scalar_fallback": False,
        "memory_policy": memory_policy,
        "allocator_bytes": allocator,
        "gpu": str(logical_gpus[0]),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "tensorflow_version": tf.__version__,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "command": " ".join(sys.argv),
    }
    passed = bool(tf.convert_to_tensor(validation["passed"]).numpy())
    result = {
        "schema": "bayesfilter.neutra.generic_five_stage_result.v1",
        "manifest": manifest,
        "training": training,
        "validation": validation,
        "decision": {
            "status": "known_law_gate_passed" if passed else "known_law_gate_failed",
            "known_law_gate_passed": passed,
            "promotion": False,
            "no_hmc": True,
            "nonclaims": [
                "no universal training-procedure claim",
                "no SSL-LSTM transfer claim",
                "no HMC or posterior-correctness claim",
            ],
        },
        "wall_seconds": time.perf_counter() - started,
    }
    _write(output / "trainer_state.json", state)
    _write(output / "run_manifest.json", manifest)
    _write(output / "result.json", result)
    _write(
        output / "artifact_hashes.json",
        {
            "schema": "bayesfilter.neutra.generic_five_stage_hashes.v1",
            "artifacts": {
                path.relative_to(output).as_posix(): _sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "artifact_hashes.json"
            },
        },
    )
    print(
        json.dumps(
            {
                "output_root": output.as_posix(),
                "target": str(args.target),
                "route": str(args.route),
                "passed": passed,
                "wall_seconds": result["wall_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
