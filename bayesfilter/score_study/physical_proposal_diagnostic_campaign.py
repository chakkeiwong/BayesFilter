"""Bounded SIR physical-proposal diagnostic; no canonical or default admission."""

import argparse
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
from bayesfilter.score_study.moment_training_campaign import Campaign, append, save
from bayesfilter.score_study.physical_campaign_evidence import (
    HEURISTIC_ARMS, PARTICLE_REPLICATIONS, TEST_PARTICLE_SEED_BASE, validate_holdout_rows,
)
from bayesfilter.score_study.run_source_snapshot import capture_source_snapshot, verify_source_snapshot


class PhysicalCampaign(Campaign):
    def __init__(self, arguments):
        self.arguments = arguments
        self.started = time.monotonic()
        self.output = arguments.output
        self.output.mkdir(parents=True, exist_ok=False)
        self.calls = 0
        self.repository = Path(__file__).resolve().parents[2]
        self.source_snapshot = capture_source_snapshot(self.repository, self.output / "source")
        self.memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        from bayesfilter.highdim.ledh_backward_guide_tf import build_backward_guided_ledh_kernel
        from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
        from bayesfilter.highdim.transport_chunk_policy import select_transport_chunk_size
        from bayesfilter.score_study.moment_training_tf import (
            calibrate_moments, make_guide_adapters, make_moment_endpoint, make_moment_objective)
        from bayesfilter.score_study.supervised_guidance_tf import make_sir_supervision_programs
        from bayesfilter.score_study.sir_backward_guidance_diagnostic_tf import SIRGuideFitter

        self.spec = NonlinearSQMCSpec("sir_d18")
        self.theta = self.spec.default_theta(tf.float64)
        self.horizon, self.particles, self.components = 5, 64, 32
        assert select_transport_chunk_size(self.particles) == self.particles
        self.controls = dict(flow_substeps=24, reset_policy="contract_e",
            reset_epsilon=1., reset_sinkhorn_steps=5, reset_balance_steps=32,
            reset_ridge=1e-5, reset_denominator_regularization=1e-7,
            correction_steps=1, correction_strength=.2, correction_lm_damping=.01,
            correction_lm_scale_floor=.0001, correction_trust_radius=.5, pairwise_steps=1,
            pairwise_strength=.02, pairwise_rms_cap=2., coordinate_cap=.94,
            coordinate_cap_power=8, moment_safety=False)
        self.simulate, self.label = make_sir_supervision_programs(self.horizon, 1)
        calibrate, _ = make_sir_supervision_programs(self.horizon, 256)
        paths, observations = calibrate(self.theta, tf.constant(263020001))
        self.calibration = calibrate_moments(paths[:, -1])
        self.objective = make_moment_objective(self.particles, 18, *self.calibration)
        ordered = tf.sort(tf.reduce_mean(observations[:, -1], axis=-1))
        self.regime_threshold = float((ordered[127] + ordered[128]) / 2.)
        save(self.output / "calibration.json", dict(center=self.calibration[0],
            scale=self.calibration[1], feature_scale=self.calibration[2],
            regime_threshold=self.regime_threshold, seed=263020001, samples=256))
        self.partitions = dict(training=list(range(263021001, 263021033)),
            validation=list(range(263022001, 263022009)), test=list(range(263023001, 263023033)))
        design = tf.random.stateless_normal([64, 18], [262029901, 0], dtype=tf.float64)
        common = dict(horizon=5, particle_count=64, state_dimension=18,
            observation_dimension=9, parameter_dimension=3, dtype=tf.float64,
            jit_compile=True, return_trace=True, reset_design=design, **self.controls)
        factory = lambda parameter, direction: self.spec.model(parameter, direction)[0]
        self.kernels = {
            "csmc": build_backward_guided_ledh_kernel(factory, None,
                dynamic_quadratic=True, physical_mixture_weight=.5, **common),
            "kernel": build_backward_guided_ledh_kernel(factory, None,
                mixture_components=32, mixture_conditioning="mixture_product_moments",
                physical_mixture_weight=.5, **common)}
        for name, beta in (("plain", 0.), ("conditional", 1.), ("half", .5)):
            self.kernels[name] = build_backward_guided_ledh_kernel(
                factory, None, physical_mixture_weight=beta, **common)
        self.endpoints = {name: make_moment_endpoint(kernel, self.objective, 18)
                          for name, kernel in self.kernels.items()}
        self.adapters = {capacity: make_guide_adapters(5, 18, 32, capacity == "time_specific")
                         for capacity in ("pooled", "time_specific")}
        self.fitter = SIRGuideFitter(5, components=32)
        self.datasets = {}
        self.manifest = dict(command=sys.argv, environment=sys.executable,
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            branch=subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
            memory_policy=self.memory, tensorflow=tf.__version__,
            gpu_devices=[device.name for device in tf.config.list_logical_devices("GPU")],
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            dtype="float64", tf32=False, jit_compile=True, cpu_only=False,
            exception="scoped diagnostic FP64 candidate; no canonical or default promotion",
            plan="docs/plans/ledh-flow-aware-proposal-repair-20261020.md",
            result=str(self.output / "result.json"), mode=arguments.mode,
            source_hashes=self.source_snapshot["source_hashes"],
            source_snapshot=str(self.output / "source"),
            source_scope=self.source_snapshot["scope"],
            partitions=self.partitions, calibration_seed=263020001,
            particle_namespaces=dict(training=275000000, validation=276000000, test=276100000),
            reset_design_seed=262029901, controls=self.controls, chunk=64,
            optimizer_seeds=[263025001, 263025002], stepsizes=[.05, .15],
            data_version=self.spec.target_id, target="terminal first/raw-second filtering moments",
            training_objective="simulated moment loss only", beta=.5,
            uniform_stream="stateless uniform [particle_seed,2], separate from normal streams 0,1",
            budget_seconds=arguments.budget_seconds)
        save(self.output / "manifest.json", self.manifest)

    def data(self, seed, need_guide=False):
        self.check_budget()
        if seed not in self.datasets:
            paths, observations = self.simulate(self.theta, tf.constant(seed))
            self.datasets[seed] = dict(observations=observations[0], label=paths[0, -1])
        data = self.datasets[seed]
        if need_guide and "csmc" not in data:
            quadratic, mixture, diagnostics = self.fitter.fit(data["observations"], seed + 1000000)
            data.update(csmc=(quadratic.precision, quadratic.linear),
                        kernel=(mixture.logits, mixture.centers, mixture.covariances, mixture.active))
            save(self.output / f"guide-fit-{seed}.json", diagnostics)
        save(self.output / f"dataset-{seed}.json", data)
        return data

    def evaluate(self, family, capacity, eta, seed, particle_seed, *, direction=None, theta=None):
        self.check_budget()
        data = self.data(seed, family in ("csmc", "kernel"))
        coefficients = ()
        if family in ("csmc", "kernel"):
            coefficients = self.adapters[capacity][0 if family == "csmc" else 1](eta, *data[family])
        states, covariances, noises, d_states, d_covariances = self.particles_for(particle_seed)
        uniforms = tf.random.stateless_uniform([5, 64], [particle_seed, 2], dtype=tf.float64)
        direction = tf.zeros([3], tf.float64) if direction is None else direction
        result = self.endpoints[family](self.theta if theta is None else theta, direction,
            states, covariances, noises, data["observations"], d_states, d_covariances,
            uniforms, *coefficients, data["label"])
        self.calls += 1
        return result

    def integration(self):
        rows = []
        eta = tf.zeros([1, 3], tf.float64)
        prior = Path("docs/plans/artifacts/ledh-moment-collapse-repair-20261020/replay-01")
        for seed, particle_seed in ((262023017, 274000034), (262023001, 274000002)):
            payload = json.loads((prior / f"dataset-{seed}.json").read_text())
            self.datasets[seed] = dict(observations=tf.constant(payload["observations"], tf.float64),
                                      label=tf.constant(payload["label"], tf.float64))
            self.datasets[seed]["csmc"] = tuple(tf.constant(item, tf.float64) for item in payload["csmc"])
            self.datasets[seed]["kernel"] = tuple(tf.constant(item,
                tf.bool if index == 3 else tf.float64) for index, item in enumerate(payload["kernel"]))
            for family in ("plain", "conditional", "half", "csmc", "kernel"):
                begin = time.monotonic()
                output = self.evaluate(family, "pooled", eta, seed, particle_seed)
                row = dict(dataset=seed, particle_seed=particle_seed, arm=family,
                           valid=output["valid"], value=output["value"], loss=output["loss"],
                           checks=output["checks"], diagnostics=output["diagnostics"],
                           seconds=time.monotonic() - begin)
                rows.append(row)
                append(self.output / "integration.jsonl", row)
                print(f"integration {seed} {family} valid={bool(output['valid'])}", flush=True)
        finite_differences = []
        for family in ("conditional", "half", "csmc", "kernel"):
            for direction in tf.eye(3, dtype=tf.float64):
                center = self.evaluate(family, "pooled", eta, 262023001, 274000002, direction=direction)
                delta = tf.constant(2e-5, tf.float64)
                upper = self.evaluate(family, "pooled", eta, 262023001, 274000002,
                                      theta=self.theta + delta * direction)
                lower = self.evaluate(family, "pooled", eta, 262023001, 274000002,
                                      theta=self.theta - delta * direction)
                numeric = (upper["value"] - lower["value"]) / (2 * delta)
                relative = tf.abs(center["score"] - numeric) / tf.maximum(1., tf.abs(numeric))
                valid = bool(center["valid"] & upper["valid"] & lower["valid"])
                finite_differences.append(dict(arm=family, direction=direction, valid=valid,
                    analytical=center["score"], finite_difference=numeric, relative_error=relative,
                    passed=valid and float(relative) <= 1e-3))
        return dict(status="integration_complete", rows=rows, finite_differences=finite_differences,
                    scientific_ranking="not evaluated; regression fixtures only")

    def validation(self, family, capacity, eta):
        losses = []
        for index, seed in enumerate(self.partitions["validation"]):
            for replicate in range(2):
                output = self.evaluate(family, capacity, eta, seed, 276000000 + 2 * index + replicate)
                if not bool(output["valid"]):
                    return math.inf
                losses.append(float(output["loss"]))
        return sum(losses) / len(losses)

    def fit_adapter(self, family, capacity, seed, step_size, updates, initial=None):
        from bayesfilter.score_study.moment_training_tf import random_direction_descent
        rows = 4 if capacity == "time_specific" else 1
        initial = tf.zeros([rows, 3], tf.float64) if initial is None else initial
        run_id = f"{family}-{capacity}-{seed}-{step_size}-{updates}"
        candidates = []
        batches = {}

        def objective(eta, update):
            if update not in batches:
                random = tf.random.stateless_uniform([32], [seed, update + 1000 + updates], dtype=tf.float64)
                indices = tf.argsort(random)[:2].numpy().tolist()
                batches[update] = [self.partitions["training"][index] for index in indices]
            losses = []
            for index, data_seed in enumerate(batches[update]):
                particle_seed = 275000000 + (seed % 100) * 10000 + updates * 100 + update * 2 + index
                output = self.evaluate(family, capacity, eta, data_seed, particle_seed)
                if not bool(output["valid"]):
                    return math.inf
                losses.append(float(output["loss"]))
            return sum(losses) / len(losses)

        def checkpoint(update, eta):
            loss = self.validation(family, capacity, eta)
            candidate = dict(run=run_id, family=family, capacity=capacity,
                optimizer_seed=seed, step_size=step_size, updates=updates, checkpoint=update,
                eta=eta, validation_loss=loss if math.isfinite(loss) else None)
            candidates.append(candidate)
            append(self.output / "validation.jsonl", candidate)
            print(f"validation {run_id} update={update} loss={loss:.6g}", flush=True)

        def record(row):
            append(self.output / "training.jsonl", dict(run=run_id, **row))

        _, accepted = random_direction_descent(objective, initial, seed=seed,
            updates=updates, perturbation=.1, step_size=step_size, record=record,
            checkpoint=checkpoint)
        finite = [candidate for candidate in candidates if candidate["validation_loss"] is not None]
        result = dict(run=run_id, accepted=accepted, candidates=candidates,
            selected=min(finite, key=lambda candidate: candidate["validation_loss"]) if finite else None)
        save(self.output / f"fit-{run_id}.json", result)
        return result["selected"]

    def make_heuristics(self, return_blocks=False):
        from bayesfilter.highdim.ledh_physical_proposal_tf import conditional_moments
        from bayesfilter.score_study.supervised_guidance_tf import moment_features
        from dataclasses import replace
        model = self.spec.model(self.theta, tf.zeros_like(self.theta))[0]
        center, scale, feature_scale = self.calibration

        @tf.function(input_signature=[tf.TensorSpec([5, 9], tf.float64),
                                      tf.TensorSpec([18], tf.float64)],
                     jit_compile=True, autograph=False)
        def evaluate(observations, label):
            estimates = []
            validity = []
            blocks = []
            for mode in ("prediction", "observation_only", "recursive_gaussian"):
                mean = self.spec.initial_mean(tf.float64)
                covariance = tf.eye(18, dtype=tf.float64)
                valid = tf.constant(True)
                for time_index in range(5):
                    factor = tf.linalg.cholesky(covariance)
                    offsets = tf.sqrt(tf.constant(18., tf.float64)) * tf.transpose(factor)
                    points = mean + tf.concat([offsets, -offsets], axis=0)
                    propagated = model.transition_mean_fn(self.theta, points)
                    mean = tf.reduce_mean(propagated, axis=0)
                    centered = propagated - mean
                    covariance = tf.transpose(centered) @ centered / 36. + model.process_covariance
                    if mode == "recursive_gaussian" or (mode == "observation_only" and time_index == 4):
                        local = replace(model, process_covariance=covariance)
                        updated = conditional_moments(local, mean[None], tf.zeros([1, 18], tf.float64),
                            observations[time_index], tf.zeros_like(covariance),
                            tf.zeros_like(model.observation_covariance))
                        mean, covariance = updated[0][0], updated[1][0]
                        valid &= updated[4]
                normalized = (mean - center) / scale
                normalized_covariance = covariance / (scale[:, None] * scale[None, :])
                indices = [row * 18 + column for row in range(18) for column in range(row, 18)]
                estimate = moment_features(normalized) + tf.concat([
                    tf.zeros([18], tf.float64), tf.gather(tf.reshape(normalized_covariance, [-1]), indices)], 0)
                squared = tf.square((estimate - moment_features((label - center) / scale)) / feature_scale)
                loss = .5 * (tf.reduce_mean(squared[:18]) + tf.reduce_mean(squared[18:]))
                estimates.append(loss)
                blocks.append(tf.stack([tf.reduce_mean(squared[:18]), tf.reduce_mean(squared[18:])]))
                validity.append(valid & tf.math.is_finite(loss))
            if return_blocks:
                return tf.stack(estimates), tf.stack(validity), tf.stack(blocks)
            return tf.stack(estimates), tf.stack(validity)

        return evaluate

    def train(self):
        selected = {}
        for family in ("csmc", "kernel"):
            candidates = []
            for capacity in ("pooled", "time_specific"):
                for seed in (263025001, 263025002):
                    for step_size in (.05, .15):
                        candidate = self.fit_adapter(family, capacity, seed, step_size, 8)
                        if candidate is not None:
                            candidates.append(candidate)
            if not candidates:
                return dict(status="no_valid_training_candidate", family=family,
                            holdout_opened=False, scientific_ranking="none")
            nominee = min(candidates, key=lambda candidate: candidate["validation_loss"])
            extension = self.fit_adapter(family, nominee["capacity"], nominee["optimizer_seed"],
                                         nominee["step_size"], 24, nominee["eta"])
            if extension is not None:
                candidates.append(extension)
            selected[family] = min(candidates, key=lambda candidate: candidate["validation_loss"])
        save(self.output / "frozen-selection.json", dict(selected=selected,
            training_protocol_complete=True, holdout_opened=False,
            elapsed_seconds=time.monotonic() - self.started))
        heuristics = self.make_heuristics()
        all_rows = []
        for index, data_seed in enumerate(self.partitions["test"]):
            data = self.data(data_seed)
            regime = "high" if float(tf.reduce_mean(data["observations"][-1])) >= self.regime_threshold else "low"
            heuristic_loss, heuristic_valid = heuristics(data["observations"], data["label"])
            for offset, name in enumerate(HEURISTIC_ARMS):
                row = dict(dataset=data_seed, regime=regime, arm=name, replicate=None,
                    loss=float(heuristic_loss[offset]), valid=bool(heuristic_valid[offset]))
                all_rows.append(row)
                append(self.output / "holdout.jsonl", row)
            for replicate in range(PARTICLE_REPLICATIONS):
                particle_seed = TEST_PARTICLE_SEED_BASE + PARTICLE_REPLICATIONS * index + replicate
                arms = [(name, name, "pooled", tf.zeros([1, 3], tf.float64))
                        for name in ("plain", "conditional", "half", "csmc", "kernel")]
                arms += [(family + "_trained", family, selected[family]["capacity"], selected[family]["eta"])
                         for family in ("csmc", "kernel")]
                for name, family, capacity, eta in arms:
                    output = self.evaluate(family, capacity, eta, data_seed, particle_seed)
                    row = dict(dataset=data_seed, regime=regime, arm=name, replicate=replicate,
                        particle_seed=particle_seed, loss=float(output["loss"]), valid=bool(output["valid"]),
                        blocks=output["blocks"], raw_blocks=output["raw_blocks"],
                        checks=output["checks"], diagnostics=output["diagnostics"])
                    all_rows.append(row)
                    append(self.output / "holdout.jsonl", row)
            print(f"holdout dataset={data_seed} completed={index+1}/32", flush=True)
        summary = self.summarize(all_rows)
        return dict(status="completed", training_protocol_complete=True,
                    holdout_complete=True, selected=selected, **summary)

    def summarize(self, rows):
        integrity = validate_holdout_rows(rows, self.partitions["test"])
        arms = sorted({row["arm"] for row in rows})
        means = {}
        invalid = {}
        for arm in arms:
            arm_rows = [row for row in rows if row["arm"] == arm]
            invalid[arm] = sum(not row["valid"] for row in arm_rows)
            means[arm] = {}
            for seed in self.partitions["test"]:
                values = [row["loss"] for row in arm_rows if row["dataset"] == seed]
                means[arm][seed] = sum(values) / len(values)
        comparisons = []
        regimes = {row["dataset"]: row["regime"] for row in rows}
        for family in ("csmc", "kernel"):
            candidate = family + "_trained"
            for comparator in ("plain", "conditional", "half", family,
                               "prediction", "observation_only", "recursive_gaussian"):
                for regime in ("all", "low", "high"):
                    seeds = [seed for seed in self.partitions["test"] if regime == "all" or regimes[seed] == regime]
                    valid = all(row["valid"] for row in rows
                                if row["arm"] in (candidate, comparator) and row["dataset"] in seeds)
                    difference = [means[candidate][seed] - means[comparator][seed] for seed in seeds]
                    if not seeds or not valid or not all(math.isfinite(value) for value in difference):
                        comparisons.append(dict(candidate=candidate, comparator=comparator,
                            regime=regime, count=len(seeds), valid=False, inference="veto; no ranking"))
                        continue
                    values = tf.constant(difference, tf.float64)
                    indices = tf.random.stateless_uniform([2000, len(seeds)],
                        [263026001, len(seeds)], maxval=len(seeds), dtype=tf.int32)
                    bootstraps = tf.sort(tf.reduce_mean(tf.gather(values, indices), axis=1))
                    lower, upper = float(bootstraps[49]), float(bootstraps[1949])
                    comparisons.append(dict(candidate=candidate, comparator=comparator, regime=regime,
                        count=len(seeds), valid=True, mean_difference=float(tf.reduce_mean(values)),
                        pointwise_ci95=[lower, upper], inference="candidate_lower" if upper < 0 else
                        "candidate_higher" if lower > 0 else "not_resolved"))
        heuristic_names = ("prediction", "observation_only", "recursive_gaussian")
        verdict = {}
        for family in ("csmc", "kernel"):
            candidate = family + "_trained"
            evidence = [row for row in comparisons if row["candidate"] == candidate
                        and row["comparator"] in heuristic_names]
            losses = [row for row in evidence if row.get("mean_difference", 0.) > 0.]
            verdict[candidate] = dict(passed=not invalid[candidate] and not losses
                and all(row["valid"] for row in evidence), underperformance=losses,
                interpretation="descriptive conditional underperformance veto; never a tuning objective")
        return dict(integrity=integrity, invalid_counts=invalid, per_dataset_mean_loss=means,
            comparisons=comparisons, heuristic_dominance=verdict,
            default_readiness=False, score_accuracy="not evaluated",
            uncertainty="paired dataset-cluster percentile bootstrap; pointwise, not simultaneous",
            scientific_ranking="See predeclared primary comparisons and vetoes; no blanket superiority")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("integration", "train"), default="integration")
    parser.add_argument("--budget-seconds", type=float, default=900.)
    arguments = parser.parse_args()
    campaign = PhysicalCampaign(arguments)
    try:
        result = campaign.integration() if arguments.mode == "integration" else campaign.train()
    except Exception as error:
        result = dict(status="incomplete", error_type=type(error).__name__, error=str(error),
                      scientific_ranking="none", holdout_eligible=False)
        raise
    finally:
        elapsed = time.monotonic() - campaign.started
        source_check = verify_source_snapshot(campaign.repository, campaign.source_snapshot)
        result["source_verification"] = source_check
        if not source_check["unchanged"]:
            result.update(status="incomplete", scientific_ranking="none", holdout_eligible=False)
        result.update(seconds=elapsed, calls=campaign.calls)
        save(campaign.output / "result.json", result)
        campaign.manifest.update(wall_seconds=elapsed, calls=campaign.calls,
            allocator=tf.config.experimental.get_memory_info("GPU:0"))
        save(campaign.output / "manifest.json", campaign.manifest)


if __name__ == "__main__":
    main()
