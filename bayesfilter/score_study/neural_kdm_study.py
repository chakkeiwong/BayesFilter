"""Scoped neural KDM training and paired moment diagnostics, not score admission."""

import argparse
import json
from pathlib import Path
import time

import tensorflow as tf

from bayesfilter.score_study.neural_kdm_diagnostic_campaign import NeuralKDMCampaign
from bayesfilter.score_study.moment_training_campaign import save, append, serializable
from bayesfilter.score_study.run_source_snapshot import verify_source_snapshot


class NeuralKDMStudy(NeuralKDMCampaign):
    def __init__(self, arguments):
        super().__init__(arguments)
        self.completed_updates = 0
        self.row_inputs = tf.function(self._row_inputs,
            input_signature=[tf.TensorSpec([], tf.int32), tf.TensorSpec([], tf.int32)],
            jit_compile=True, autograph=False)
        self.neural_evaluate = tf.function(self._neural_evaluate,
            input_signature=list(self.row_gradient.input_signature),
            jit_compile=True, autograph=False)
        if arguments.resume is not None:
            self.restore(arguments.resume)

    def _row_inputs(self, dataset_seed, particle_seed):
        paths, observations = self.simulate_one(self.theta, dataset_seed)
        normals = tf.random.stateless_normal([64, 18], tf.stack([particle_seed, 0]),
                                             dtype=tf.float64)
        states, covariances, d_states, d_covariances = self.spec.initial_cloud(self.theta, normals)
        noises = tf.random.stateless_normal([5, 64, 18], tf.stack([particle_seed, 1]),
                                            dtype=tf.float64)
        uniforms = tf.random.stateless_uniform([5, 64], tf.stack([particle_seed, 2]),
                                               dtype=tf.float64)
        return (self.theta, tf.zeros([3], tf.float64), states, covariances, noises,
                observations[0], d_states, d_covariances, uniforms, paths[0, -1])

    def _neural_evaluate(self, *arguments):
        coefficients = self.network.predict.python_function(arguments[5])
        result = self.endpoint.python_function(*arguments[:-1], *coefficients, arguments[-1])
        return dict(result, maximum_absolute_offset=tf.reduce_max(tf.abs(coefficients[0])),
                    minimum_log_scale=tf.reduce_min(coefficients[1]),
                    maximum_log_scale=tf.reduce_max(coefficients[1]))

    def configuration(self):
        return {name: getattr(self.arguments, name) for name in
                ("components", "width", "beta", "learning_rate", "seed")}

    def restore(self, path):
        checkpoint = json.loads(path.read_text())
        if checkpoint["configuration"] != self.configuration():
            raise ValueError("Checkpoint configuration does not match this training scope")
        for group, variables in (("parameters", self.network.parameters),
                                 ("optimizer", self.optimizer.variables)):
            if len(checkpoint[group]) != len(variables):
                raise ValueError("Incomplete neural/Adam checkpoint")
            for variable, value in zip(variables, checkpoint[group]):
                tensor = tf.convert_to_tensor(value, dtype=variable.dtype)
                if tensor.shape != variable.shape or not bool(tf.reduce_all(tf.math.is_finite(
                        tf.cast(tensor, tf.float64)))):
                    raise ValueError("Invalid checkpoint tensor")
                variable.assign(tensor)
        self.completed_updates = checkpoint["updates"]
        if int(self.optimizer.iterations) != self.completed_updates:
            raise ValueError("Adam iteration and training-data index disagree")
        if checkpoint["source_hashes"] != self.source["source_hashes"]:
            raise ValueError("Training checkpoint source differs from the current implementation")

    def validation(self):
        losses = []
        valid = True
        for dataset_index in range(32):
            for replicate in range(2):
                self.check_budget()
                dataset = 263042001 + dataset_index
                particle_seed = 283020000 + dataset_index * 2 + replicate
                inputs = self.row_inputs(tf.constant(dataset), tf.constant(particle_seed))
                result = self.neural_evaluate(*inputs)
                row = dict(dataset=dataset, replicate=replicate, particle_seed=particle_seed,
                           updates=self.completed_updates, **result)
                append(self.output / "validation.jsonl", row)
                valid &= bool(result["valid"])
                losses.append(float(result["loss"]))
        return dict(loss=sum(losses) / len(losses) if valid else None, valid=valid,
                    datasets=32, replicates=2)

    def checkpoint(self, validation):
        path = self.output / f"checkpoint-{self.completed_updates:04d}.json"
        save(path, dict(configuration=self.configuration(), updates=self.completed_updates,
            parameters=[tf.convert_to_tensor(variable) for variable in self.network.parameters],
            optimizer=[tf.convert_to_tensor(variable) for variable in self.optimizer.variables],
            validation=validation, calibration_seed=263040001, reset_seed=263040002,
            source_hashes=self.source["source_hashes"],
            model_score_accuracy_evaluated=False))
        return str(path)

    def fit(self):
        if self.arguments.updates <= self.completed_updates:
            raise ValueError("The requested rung must extend the checkpoint")
        checkpoints = []
        if self.completed_updates == 0:
            initial_validation = self.validation()
            checkpoints.append(self.checkpoint(initial_validation))
            if not initial_validation["valid"]:
                return dict(status="invalid_initial_candidate", checkpoints=checkpoints,
                            holdout_opened=False)
        rows = []
        for update_index in range(self.completed_updates, self.arguments.updates):
            self.check_budget()
            inputs = self.make_batch(tf.constant(263041000 + update_index),
                                     tf.constant(283010000 + update_index))
            started = time.monotonic()
            result = self.update(*inputs)
            result["loss"].numpy()
            row = dict(update=update_index, **result, wall_seconds=time.monotonic() - started)
            append(self.output / "training.jsonl", row)
            rows.append(row)
            if not bool(result["valid"]) or not bool(result["applied"]):
                return dict(status="invalid_training_candidate", failed_update=update_index,
                            checkpoints=checkpoints, holdout_opened=False)
            self.completed_updates = update_index + 1
            if self.completed_updates % 8 == 0:
                print(f"updates={self.completed_updates} loss={float(result['loss']):.6g}", flush=True)
            save(self.output / "progress.json", dict(updates=self.completed_updates,
                wall_seconds=time.monotonic() - self.started, loss=result["loss"]))
        validation = self.validation()
        checkpoints.append(self.checkpoint(validation))
        return dict(status="fit_complete", checkpoints=checkpoints, validation=validation,
            updates=self.completed_updates, holdout_opened=False,
            warm_update_seconds=[row["wall_seconds"] for row in rows[1:]],
            model_score_accuracy_evaluated=False)

    def probe(self):
        rows = []
        for index in range(3):
            inputs = self.row_inputs(tf.constant(263041900 + index), tf.constant(283019900 + index))
            started = time.monotonic()
            result = self.neural_evaluate(*inputs)
            result["loss"].numpy()
            rows.append(dict(index=index, **result, wall_seconds=time.monotonic() - started))
        save(self.output / "probe.json", rows)
        if not all(bool(row["valid"]) for row in rows):
            raise ValueError("Invalid actual-SIR evaluation endpoint")
        return dict(status="probe_complete", rows=rows, holdout_opened=False)

    def family(self):
        from bayesfilter.highdim.ledh_neural_kdm_tf import ObservationKDMNetwork
        root = self.output
        results = []
        for seed in (263044001, 263044002):
            for rate in (1e-4, 1e-3):
                self.check_budget()
                self.arguments.seed, self.arguments.learning_rate = seed, rate
                initial = ObservationKDMNetwork(5, 18, self.arguments.components,
                    self.arguments.width, self.network.observation_center,
                    self.network.observation_scale, seed=seed)
                for parameter, value in zip(self.network.parameters, initial.parameters):
                    parameter.assign(value)
                for variable in self.optimizer.variables:
                    variable.assign(tf.zeros_like(variable))
                self.optimizer.learning_rate.assign(rate)
                self.completed_updates = 0
                self.output = root / f"seed-{seed}-lr-{rate}"
                self.output.mkdir()
                save(self.output / "manifest.json", dict(self.manifest,
                    **self.configuration(), output=str(self.output), family_root=str(root)))
                begin = time.monotonic()
                result = self.fit()
                result.update(configuration=self.configuration(), wall_seconds=time.monotonic() - begin)
                save(self.output / "result.json", result)
                results.append(result)
        self.output = root
        return dict(status="family_complete", candidates=results, holdout_opened=False)

    def holdout(self):
        from bayesfilter.highdim.ledh_backward_guide_tf import build_backward_guided_ledh_kernel
        from bayesfilter.score_study.moment_training_tf import make_moment_endpoint
        from bayesfilter.score_study.neural_moment_evidence import analyze_holdout
        from bayesfilter.score_study.physical_proposal_diagnostic_campaign import PhysicalCampaign
        if self.arguments.resume is None or self.completed_updates == 0:
            raise ValueError("Final evaluation requires a frozen trained checkpoint")
        trained = [tf.identity(parameter) for parameter in self.network.parameters]
        factory = lambda parameter, direction: self.spec.model(parameter, direction)[0]
        baselines = {}
        for name, beta in (("plain", 0.), ("conditional", 1.), ("physical", self.arguments.beta)):
            kernel = build_backward_guided_ledh_kernel(factory, None,
                physical_mixture_weight=beta, **self.common)
            baselines[name] = make_moment_endpoint(kernel, self.objective, 18)
        heuristics = PhysicalCampaign.make_heuristics(self, return_blocks=True)
        from bayesfilter.score_study.supervised_guidance_tf import make_sir_supervision_programs
        calibrate, _ = make_sir_supervision_programs(5, 256)
        paths, _ = calibrate(self.theta, tf.constant(263040001))
        boundary_threshold = float(tf.sort(tf.reduce_min(paths[:, -1], axis=-1))[25])
        frozen = dict(configuration=self.configuration(), checkpoint=str(self.arguments.resume),
            updates=self.completed_updates, boundary_threshold=boundary_threshold,
            observation_threshold=self.regime_threshold,
            missing_comparators=["freshly refitted cSMC restricted adapter", "freshly refitted RBF restricted adapter"],
            score_accuracy_evaluated=False, holdout_opened=False)
        save(self.output / "frozen-selection.json", frozen)
        rows = []
        for dataset_index in range(64):
            dataset = 263043001 + dataset_index
            inputs = self.row_inputs(tf.constant(dataset), tf.constant(283030000 + dataset_index * 4))
            observations, label = inputs[5], inputs[-1]
            regimes = dict(observation_regime="high" if float(tf.reduce_mean(observations[-1]))
                           >= self.regime_threshold else "low",
                           compartment_regime="nearer_boundary" if float(tf.reduce_min(label))
                           <= boundary_threshold else "interior")
            save(self.output / f"dataset-{dataset}.json", dict(observations=observations, label=label,
                                                             **regimes))
            heuristic_losses, heuristic_valid, heuristic_blocks = heuristics(observations, label)
            for index, name in enumerate(("prediction", "observation_only", "recursive_gaussian")):
                row = dict(dataset=dataset, arm=name, replicate=None, particle_seed=None,
                           loss=float(heuristic_losses[index]), valid=bool(heuristic_valid[index]),
                           blocks=heuristic_blocks[index],
                           **regimes)
                append(self.output / "holdout.jsonl", row)
                rows.append(row)
            for replicate in range(4):
                self.check_budget()
                particle_seed = 283030000 + dataset_index * 4 + replicate
                inputs = self.row_inputs(tf.constant(dataset), tf.constant(particle_seed))
                for arm in ("trained", "untrained", "plain", "conditional", "physical"):
                    if arm in ("trained", "untrained"):
                        parameters = trained if arm == "trained" else self.initial
                        for parameter, value in zip(self.network.parameters, parameters):
                            parameter.assign(value)
                        result = self.neural_evaluate(*inputs)
                    else:
                        result = baselines[arm](*inputs)
                    row = dict(dataset=dataset, arm=arm, replicate=replicate,
                               particle_seed=particle_seed, **regimes, **result)
                    append(self.output / "holdout.jsonl", row)
                    rows.append(row)
            print(f"holdout_dataset={dataset_index + 1}/64", flush=True)
        return analyze_holdout(serializable(rows), list(range(263043001, 263043065)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["probe", "fit", "family", "holdout"], required=True)
    parser.add_argument("--components", type=int, default=2)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--beta", type=float, default=.5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=263044001)
    parser.add_argument("--updates", type=int, default=32)
    parser.add_argument("--budget-seconds", type=float, default=1500.)
    parser.add_argument("--resume", type=Path)
    arguments = parser.parse_args()
    if arguments.updates < 1 or arguments.budget_seconds <= 0:
        parser.error("Positive updates and process budget are required")
    if arguments.mode == "family" and arguments.resume is not None:
        parser.error("A family screen cannot resume a single checkpoint")
    study = NeuralKDMStudy(arguments)
    try:
        result = getattr(study, arguments.mode)()
        result.update(wall_seconds=time.monotonic() - study.started,
            source_check=verify_source_snapshot(study.repository, study.source),
            memory=tf.config.experimental.get_memory_info("GPU:0"))
        if not result["source_check"]["unchanged"]:
            raise ValueError("Source changed during the run")
        save(study.output / "result.json", result)
    except Exception as error:
        save(study.output / "failure.json", dict(error=repr(error),
            wall_seconds=time.monotonic() - study.started,
            source_check=verify_source_snapshot(study.repository, study.source)))
        raise


if __name__ == "__main__":
    main()
