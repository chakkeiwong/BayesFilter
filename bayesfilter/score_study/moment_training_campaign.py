"""GPU/XLA SIR moment training through the shared full LEDH executor.

This is a scoped research experiment, not a canonical tuning artifact issuer.
"""

import argparse
from functools import lru_cache
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth


def serializable(value):
    if tf.is_tensor(value):
        return serializable(value.numpy().tolist())
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def save(path, value):
    path.write_text(json.dumps(serializable(value), indent=2, allow_nan=False) + "\n")


def append(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(serializable(value), allow_nan=False) + "\n")


def evaluation_diagnostics(outputs):
    return [{"valid": output["valid"], "value": output["value"],
             "score": output["score"], "checks": output["checks"],
             "diagnostics": output["diagnostics"]} for output in outputs]


class Campaign:
    def __init__(self, arguments):
        self.arguments = arguments
        self.started = time.monotonic()
        self.output = arguments.output
        self.output.mkdir(parents=True, exist_ok=False)
        self.calls = 0
        self.memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        from bayesfilter.score_study.moment_training_tf import (
            calibrate_moments, make_guide_adapters, make_moment_endpoint,
            make_moment_objective, random_direction_descent,
        )
        from bayesfilter.score_study.supervised_guidance_tf import make_sir_supervision_programs
        from bayesfilter.highdim.ledh_backward_guide_tf import build_backward_guided_ledh_kernel
        from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
        from bayesfilter.highdim.transport_chunk_policy import select_transport_chunk_size
        from bayesfilter.score_study.sir_backward_guidance_diagnostic_tf import SIRGuideFitter

        self.spec = NonlinearSQMCSpec("sir_d18")
        self.theta = self.spec.default_theta(tf.float64)
        self.horizon, self.particles, self.components = 5, 64, 32
        if select_transport_chunk_size(self.particles) != self.particles:
            raise ValueError("Shared dense executor requires the one-block chunk scope")
        self.controls = dict(flow_substeps=24, reset_policy="contract_e",
            reset_epsilon=1., reset_sinkhorn_steps=5, reset_balance_steps=32,
            reset_ridge=1e-5, reset_denominator_regularization=1e-7,
            correction_steps=1, correction_strength=.2, correction_lm_damping=.01,
            correction_lm_scale_floor=.0001, correction_trust_radius=.5, pairwise_steps=1,
            pairwise_strength=.02, pairwise_rms_cap=2., coordinate_cap=.94,
            coordinate_cap_power=8, moment_safety=False)
        self.simulate, self.label = make_sir_supervision_programs(self.horizon, 1)
        calibrate, _ = make_sir_supervision_programs(self.horizon, 256)
        paths, observations = calibrate(self.theta, tf.constant(262020001))
        self.calibration = calibrate_moments(paths[:, -1])
        self.objective = make_moment_objective(self.particles, 18, *self.calibration)
        means = tf.sort(tf.reduce_mean(observations[:, -1], axis=-1))
        self.regime_threshold = float((means[127] + means[128]) / 2)
        _, labels = make_sir_supervision_programs(self.horizon, 256)
        score_labels = labels(self.theta, paths, observations)["score"]
        self.score_scale = tf.maximum(tf.sqrt(tf.reduce_mean(score_labels ** 2, axis=0)), 1.)
        save(self.output / "calibration.json", dict(center=self.calibration[0],
            state_scale=self.calibration[1], feature_scale=self.calibration[2],
            regime_threshold=self.regime_threshold, score_scale=self.score_scale,
            scale_floor=1e-6, samples=256, seed=262020001))
        self.partitions = dict(training=list(range(262021001, 262021065)),
            validation=list(range(262022001, 262022017)), test=list(range(262023001, 262023065)))
        design = tf.random.stateless_normal([self.particles, 18], [262029901, 0], dtype=tf.float64)
        common = dict(horizon=self.horizon, particle_count=self.particles,
            state_dimension=18, observation_dimension=9, parameter_dimension=3,
            dtype=tf.float64, jit_compile=True, return_trace=True, reset_design=design,
            **self.controls)
        factory = lambda parameter, direction: self.spec.model(parameter, direction)[0]
        self.kernels = dict(csmc=build_backward_guided_ledh_kernel(
            factory, None, dynamic_quadratic=True, **common),
            kernel=build_backward_guided_ledh_kernel(factory, None, mixture_components=self.components,
                mixture_conditioning="mixture_product_moments", **common))
        self.endpoints = {name: make_moment_endpoint(kernel, self.objective, 18)
                          for name, kernel in self.kernels.items()}
        self.adapters = {capacity: make_guide_adapters(self.horizon, 18, self.components,
            capacity == "time_specific") for capacity in ("pooled", "time_specific")}
        self.fitter = SIRGuideFitter(self.horizon, components=self.components)
        self.datasets = {}
        dependencies = [Path(inspect.getsourcefile(item)) for item in
            (Campaign, random_direction_descent, make_sir_supervision_programs,
             build_backward_guided_ledh_kernel, SIRGuideFitter, NonlinearSQMCSpec)]
        dependencies += [Path("bayesfilter/highdim") / name for name in
            ("ledh_canonical_score_tf.py", "ledh_canonical_models_tf.py",
             "ledh_canonical_reset_score_tf.py", "ledh_unified_correction_tf.py")]
        self.manifest = dict(command=sys.argv, environment=sys.executable,
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            branch=subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
            dirty=subprocess.check_output(["git", "status", "--short", "--untracked-files=no"], text=True),
            tensorflow=tf.__version__, memory_policy=self.memory,
            gpu_devices=[device.name for device in tf.config.list_logical_devices("GPU")],
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            dtype="float64", tf32=False, jit_compile=True, cpu_only=False,
            exception="scoped FP64 diagnostic; no production/default promotion",
            plan="docs/plans/ledh-simulated-moment-training-20261020.md",
            result=str(self.output / "result.json"), source_hashes={str(path):
                hashlib.sha256(path.read_bytes()).hexdigest() for path in dependencies},
            horizon=self.horizon, particles=self.particles, chunk=self.particles,
            controls=self.controls, partitions=self.partitions, calibration_seed=262020001,
            particle_namespaces=dict(training=272000000, validation=273000000, test=274000000),
            guide_seed_offset=1000000, optimizer_seeds=[262025001, 262025002],
            budget_seconds=arguments.budget_seconds, smoke=arguments.smoke,
            target="terminal filtering centered first/raw-second moments",
            training_objective="moment_loss_only", data_version=self.spec.target_id,
            original_neural_kdm_training=False)
        save(self.output / "manifest.json", self.manifest)

    def check_budget(self):
        if time.monotonic() - self.started > self.arguments.budget_seconds:
            raise TimeoutError("Moment-training process budget exhausted")

    def progress(self, stage, **details):
        self.check_budget()
        row = dict(stage=stage, wall_seconds=time.monotonic() - self.started,
                   calls=self.calls, **details)
        save(self.output / "progress.json", row)
        print(json.dumps(serializable(row), allow_nan=False), flush=True)

    def dataset(self, seed):
        if seed not in self.datasets:
            self.check_budget()
            paths, observations = self.simulate(self.theta, tf.constant(seed))
            labels = self.label(self.theta, paths, observations)
            if not bool(labels["valid"]):
                raise ValueError("Invalid simulated label")
            quadratic, mixture, diagnostics = self.fitter.fit(observations[0], seed + 1000000)
            self.datasets[seed] = dict(observations=observations[0], label=paths[0, -1],
                score_label=labels["score"][0], csmc=(quadratic.precision, quadratic.linear),
                kernel=(mixture.logits, mixture.centers, mixture.covariances, mixture.active))
            save(self.output / f"dataset-{seed}.json", self.datasets[seed])
            diagnostic_scalars = {name: value for name, value in diagnostics.items()
                                  if tf.is_tensor(value) and value.shape.rank <= 1
                                  and bool(tf.reduce_all(tf.math.is_finite(tf.cast(value, tf.float64))))}
            save(self.output / f"fit-{seed}.json", diagnostic_scalars)
        return self.datasets[seed]

    @lru_cache(maxsize=32)
    def particles_for(self, seed):
        normals = tf.random.stateless_normal([self.particles, 18], [seed, 0], dtype=tf.float64)
        states, covariances, d_states, d_covariances = self.spec.initial_cloud(self.theta, normals)
        noises = tf.random.stateless_normal([self.horizon, self.particles, 18], [seed, 1], dtype=tf.float64)
        return states, covariances, noises, d_states, d_covariances

    def evaluate(self, family, capacity, eta, seed, particle_seed, *, direction=None,
                 multiplier=1., theta=None):
        self.check_budget()
        data = self.dataset(seed)
        coefficients = data[family]
        adapter = self.adapters[capacity][0 if family == "csmc" else 1]
        coefficients = adapter(eta, *coefficients)
        if multiplier != 1.:
            if family != "csmc":
                raise ValueError("Shrinkage comparator is quadratic")
            coefficients = tuple(multiplier * item for item in coefficients)
        states, covariances, noises, d_states, d_covariances = self.particles_for(particle_seed)
        direction = tf.zeros([3], tf.float64) if direction is None else direction
        result = self.endpoints[family](self.theta if theta is None else theta, direction,
            states, covariances, noises, data["observations"], d_states, d_covariances,
            *coefficients, data["label"])
        self.calls += 1
        return result

    def validation(self, family, capacity, eta):
        values = []
        for seed in self.partitions["validation"]:
            for replicate in range(2):
                result = self.evaluate(family, capacity, eta, seed, 273000000 + 2 * (seed % 1000) + replicate)
                if not bool(result["valid"]):
                    return math.inf
                values.append(float(result["loss"]))
        return sum(values) / len(values)

    def train_one(self, family, capacity, seed, *, repair=False):
        run_id = f"{family}-{capacity}-{seed}" + ("-repair" if repair else "")
        parameter_rows = 4 if capacity == "time_specific" else 1
        initial = tf.zeros([parameter_rows, 3], tf.float64)
        checkpoints = []
        batches = {}

        def objective(eta, update):
            if update not in batches:
                random = tf.random.stateless_uniform([64], [seed, update + 1000], dtype=tf.float64)
                indices = tf.argsort(random)[:4].numpy().tolist()
                batches[update] = [self.partitions["training"][index] for index in indices]
            losses = []
            for index, dataset_seed in enumerate(batches[update]):
                particle_seed = 272000000 + (seed % 100) * 10000 + update * 4 + index
                result = self.evaluate(family, capacity, eta, dataset_seed, particle_seed)
                if not bool(result["valid"]):
                    return math.inf
                losses.append(float(result["loss"]))
            return sum(losses) / len(losses)

        def record(row):
            append(self.output / "training.jsonl", dict(run=run_id, **row))
            self.progress("training", run=run_id, update=row["update"],
                          accepted=row["accepted_count"], loss=row["loss"])

        def checkpoint(update, eta):
            loss = self.validation(family, capacity, eta)
            row = dict(run=run_id, family=family, capacity=capacity, optimizer_seed=seed,
                       repair=repair, update=update, eta=serializable(eta),
                       validation_loss=loss if math.isfinite(loss) else None)
            checkpoints.append(row)
            append(self.output / "validation.jsonl", row)
            self.progress("validation", run=run_id, update=update, loss=row["validation_loss"])

        from bayesfilter.score_study.moment_training_tf import random_direction_descent
        _, accepted = random_direction_descent(objective, initial, seed=seed,
            updates=96 if repair else 48, perturbation=.025 if repair else .1,
            step_size=.0625 if repair else .25, record=record, checkpoint=checkpoint)
        finite = [row for row in checkpoints if row["validation_loss"] is not None]
        if not finite:
            raise ValueError("No valid validation candidate")
        best = min(finite, key=lambda row: row["validation_loss"])
        needs_repair = accepted < 2 or best["validation_loss"] >= checkpoints[0]["validation_loss"]
        save(self.output / f"training-{run_id}.json", dict(best=best,
            accepted=accepted, needs_repair=needs_repair, batches=batches))
        return best, needs_repair

    def smoke(self):
        seed = 262000111
        zero = tf.zeros([1, 3], tf.float64)
        results = {}
        for family in ("csmc", "kernel"):
            data = self.dataset(seed)
            start = time.monotonic()
            adapted = self.evaluate(family, "pooled", zero, seed, 272999001)
            if not bool(adapted["valid"]):
                raise ValueError(f"Invalid zero-adaptation {family} smoke: {serializable(adapted['checks'])}")
            states, covariances, noises, d_states, d_covariances = self.particles_for(272999001)
            value, derivative, trace = self.kernels[family](self.theta, tf.zeros([3], tf.float64),
                states, covariances, noises, data["observations"], d_states, d_covariances, *data[family])
            direct = self.objective(trace[-1]["states_after_reset"], data["label"])
            parity = float(tf.abs(direct["loss"] - adapted["loss"]))
            if parity > 1e-10 * (1 + abs(float(direct["loss"]))):
                raise ValueError("Moment endpoint does not match shared executor")
            eta = tf.constant([[.1, .1, -.1]], tf.float64)
            direction = tf.constant([1., .25, -.5], tf.float64)
            analytic = self.evaluate(family, "pooled", eta, seed, 272999001, direction=direction)
            increment = tf.constant(1e-5, tf.float64)
            plus = self.evaluate(family, "pooled", eta, seed, 272999001,
                theta=self.theta + increment * direction)
            minus = self.evaluate(family, "pooled", eta, seed, 272999001,
                theta=self.theta - increment * direction)
            numerical = (plus["value"] - minus["value"]) / (2 * increment)
            error = float(tf.abs(analytic["score"] - numerical))
            if error > 1e-4 * (1 + abs(float(numerical))):
                raise ValueError("Adapted analytical score fails finite-difference parity")
            results[family] = dict(parity_error=parity, score_directional_error=error,
                analytical=analytic["score"], numerical=numerical,
                elapsed_seconds=time.monotonic() - start, zero=adapted, changed=analytic)
            self.progress("smoke_family_complete", family=family)
        save(self.output / "smoke.json", results)

    def run(self):
        if self.arguments.smoke:
            self.smoke()
            return
        timing = {}
        for family in ("csmc", "kernel"):
            durations = []
            for repeat in range(4):
                started = time.monotonic()
                result = self.evaluate(family, "pooled", tf.zeros([1, 3], tf.float64),
                                       262000112, 272999002)
                if not bool(result["valid"]):
                    raise ValueError("Invalid throughput preflight")
                if repeat:
                    durations.append(time.monotonic() - started)
            timing[family] = dict(seconds_per_call=sum(durations) / len(durations),
                                  samples=durations)
        timing["worst_case_executor_calls"] = 32768
        timing["projected_executor_seconds"] = 32768 * max(
            timing[family]["seconds_per_call"] for family in ("csmc", "kernel"))
        save(self.output / "timing.json", timing)
        self.progress("throughput_checked", timing=timing)
        candidates = []
        for family in ("csmc", "kernel"):
            for capacity in ("pooled", "time_specific"):
                for seed in (262025001, 262025002):
                    best, repair = self.train_one(family, capacity, seed)
                    candidates.append(best)
                    if repair:
                        repaired, _ = self.train_one(family, capacity, seed, repair=True)
                        candidates.append(repaired)
        selected = {family: min([row for row in candidates if row["family"] == family],
            key=lambda row: row["validation_loss"]) for family in ("csmc", "kernel")}
        save(self.output / "selection.json", dict(selected=selected,
            frozen_before_test=True, criterion="terminal_moment_loss", candidates=candidates,
            source_hashes=self.manifest["source_hashes"]))
        self.progress("selection_frozen", selected=selected)
        zero = tf.zeros([1, 3], tf.float64)
        arms = [("unguided", "csmc", "pooled", zero, 0.),
                ("csmc_half", "csmc", "pooled", zero, .5),
                ("csmc_original", "csmc", "pooled", zero, 1.),
                ("kernel_original", "kernel", "pooled", zero, 1.)]
        arms += [(family + "_trained", family, row["capacity"],
                  tf.constant(row["eta"], tf.float64), 1.) for family, row in selected.items()]
        for seed in self.partitions["test"]:
            data = self.dataset(seed)
            regime = "high" if float(tf.reduce_mean(data["observations"][-1])) >= self.regime_threshold else "low"
            for replicate in range(2):
                particle_seed = 274000000 + 2 * (seed % 1000) + replicate
                for name, family, capacity, eta, multiplier in arms:
                    scores, outputs = [], []
                    for coordinate in range(3):
                        result = self.evaluate(family, capacity, eta, seed, particle_seed,
                            direction=tf.one_hot(coordinate, 3, dtype=tf.float64), multiplier=multiplier)
                        scores.append(result["score"])
                        outputs.append(result)
                    valid = all(bool(output["valid"]) for output in outputs)
                    score = tf.stack(scores)
                    row = dict(dataset_seed=seed, particle_seed=particle_seed, method=name,
                        regime=regime, valid=valid,
                        directional_diagnostics=evaluation_diagnostics(outputs))
                    if valid:
                        row.update(loss=result["loss"], blocks=result["blocks"],
                            raw_blocks=result["raw_blocks"], score=score,
                            score_loss=tf.square((score - data["score_label"]) / self.score_scale),
                            checks=result["checks"], diagnostics=result["diagnostics"])
                    append(self.output / "test.jsonl", row)
                    if name == "unguided" and not valid:
                        raise ValueError("Invalid unguided baseline: continuation veto")
            self.progress("test_dataset_complete", dataset_seed=seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=21000)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    campaign = None
    started = time.monotonic()
    try:
        campaign = Campaign(arguments)
        campaign.run()
        save(arguments.output / "completion.json", dict(status="complete",
            wall_seconds=time.monotonic() - started, calls=campaign.calls,
            allocator=tf.config.experimental.get_memory_info("GPU:0")))
    except Exception as error:
        if arguments.output.exists():
            save(arguments.output / "failure.json", dict(error_type=type(error).__name__,
                error=str(error), wall_seconds=time.monotonic() - started,
                calls=campaign.calls if campaign is not None else None))
        raise


if __name__ == "__main__":
    main()
