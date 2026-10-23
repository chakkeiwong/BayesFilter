"""Scoped GPU/XLA SIR neural KDM adaptation; no default or model-score claim."""

import argparse
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
from bayesfilter.score_study.moment_training_campaign import save, append
from bayesfilter.score_study.run_source_snapshot import capture_source_snapshot, verify_source_snapshot


class NeuralKDMCampaign:
    def __init__(self, arguments):
        self.arguments = arguments
        self.started = time.monotonic()
        self.output = arguments.output
        self.output.mkdir(parents=True, exist_ok=False)
        self.repository = Path(__file__).resolve().parents[2]
        self.source = capture_source_snapshot(self.repository, self.output / "source")
        self.memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
        from bayesfilter.highdim.ledh_backward_guide_tf import build_backward_guided_ledh_kernel
        from bayesfilter.highdim.ledh_neural_kdm_tf import ObservationKDMNetwork
        from bayesfilter.highdim.transport_chunk_policy import select_transport_chunk_size
        from bayesfilter.score_study.supervised_guidance_tf import make_sir_supervision_programs
        from bayesfilter.score_study.moment_training_tf import (
            calibrate_moments, make_moment_objective, make_moment_endpoint,
        )
        from bayesfilter.score_study.neural_moment_training_tf import make_neural_moment_trainer

        self.horizon, self.particles, self.batch_size = 5, 64, 4
        self.spec = NonlinearSQMCSpec("sir_d18")
        self.theta = self.spec.default_theta(tf.float64)
        self.controls = dict(flow_substeps=24, reset_policy="contract_e",
            reset_epsilon=1., reset_sinkhorn_steps=5, reset_balance_steps=32,
            reset_ridge=1e-5, reset_denominator_regularization=1e-7,
            correction_steps=1, correction_strength=.2, correction_lm_damping=.01,
            correction_lm_scale_floor=.0001, correction_trust_radius=.5, pairwise_steps=1,
            pairwise_strength=.02, pairwise_rms_cap=2., coordinate_cap=.94,
            coordinate_cap_power=8, moment_safety=False)
        calibrate, _ = make_sir_supervision_programs(5, 256)
        paths, observations = calibrate(self.theta, tf.constant(263040001))
        self.calibration = calibrate_moments(paths[:, -1])
        observation_center = tf.reduce_mean(observations, axis=0)
        observation_scale = tf.math.reduce_std(observations, axis=0)
        if not bool(tf.reduce_all(observation_scale > 0.)):
            raise ValueError("Degenerate observation calibration")
        self.network = ObservationKDMNetwork(5, 18, arguments.components, arguments.width,
            observation_center, observation_scale, seed=arguments.seed)
        self.objective = make_moment_objective(64, 18, *self.calibration)
        self.common = dict(horizon=5, particle_count=64, state_dimension=18,
            observation_dimension=9, parameter_dimension=3, dtype=tf.float64,
            jit_compile=True, return_trace=True,
            reset_design=tf.random.stateless_normal([64, 18], [263040002, 0], dtype=tf.float64),
            **self.controls)
        factory = lambda parameter, direction: self.spec.model(parameter, direction)[0]
        self.kernel = build_backward_guided_ledh_kernel(factory, None,
            neural_mixture_components=arguments.components,
            physical_mixture_weight=arguments.beta, **self.common)
        self.endpoint = make_moment_endpoint(self.kernel, self.objective, 18)
        self.update, self.row_gradient, self.optimizer = make_neural_moment_trainer(
            self.network, self.endpoint, batch_size=4, learning_rate=arguments.learning_rate)
        self.simulate_batch, _ = make_sir_supervision_programs(5, 4)
        self.simulate_one, _ = make_sir_supervision_programs(5, 1)
        self.datasets = {}
        self.make_batch = tf.function(self._make_batch,
            input_signature=[tf.TensorSpec([], tf.int32), tf.TensorSpec([], tf.int32)],
            jit_compile=True, autograph=False)
        self.initial = [tf.identity(parameter) for parameter in self.network.parameters]
        ordered = tf.sort(tf.reduce_mean(observations[:, -1], axis=-1))
        self.regime_threshold = float((ordered[127] + ordered[128]) / 2.)
        self.manifest = dict(command=sys.argv, environment=sys.executable,
            git_commit=self.source["commit"], tensorflow=tf.__version__,
            source_snapshot=str(self.output / "source"), memory_policy=self.memory,
            gpu_devices=[device.name for device in tf.config.list_logical_devices("GPU")],
            visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            gpu_details=[tf.config.experimental.get_device_details(device)
                         for device in tf.config.list_physical_devices("GPU")],
            dtype="float64", tf32=False, jit_compile=True, cpu_only=False,
            xla_flags=os.environ.get("XLA_FLAGS", ""),
            plan="docs/plans/ledh-sir-neural-kdm-adaptation-20261020.md",
            result=str(self.output / "result.json"), data_version=self.spec.target_id,
            controls=self.controls, chunk=select_transport_chunk_size(64),
            horizon=5, particles=64, batch_size=4, components=arguments.components,
            width=arguments.width, beta=arguments.beta, seed=arguments.seed,
            learning_rate=arguments.learning_rate, updates=arguments.updates,
            parameter_names=[parameter.name for parameter in self.network.parameters],
            parameter_count=sum(parameter.shape.num_elements() for parameter in self.network.parameters),
            calibration_seed=263040001, reset_seed=263040002,
            training_namespace=263041000, particle_namespace=283010000,
            training_target="actual corrected terminal first/raw-second moments",
            model_score_backend="shared analytical recursion",
            exception="diagnostic adaptation; no canonical/default/model-score-accuracy claim",
            budget_seconds=arguments.budget_seconds)
        save(self.output / "manifest.json", self.manifest)
        save(self.output / "calibration.json", dict(moment=self.calibration,
            observation_center=observation_center, observation_scale=observation_scale,
            regime_threshold=self.regime_threshold))

    def check_budget(self):
        if time.monotonic() - self.started > self.arguments.budget_seconds:
            raise TimeoutError("Neural KDM campaign process budget exhausted")

    def _make_batch(self, dataset_seed, particle_seed):
        paths, observations = self.simulate_batch(self.theta, dataset_seed)
        normals = tf.random.stateless_normal([4 * 64, 18],
            tf.stack([particle_seed, 0]), dtype=tf.float64)
        states, covariances, d_states, d_covariances = self.spec.initial_cloud(self.theta, normals)
        noises = tf.random.stateless_normal([4, 5, 64, 18],
            tf.stack([particle_seed, 1]), dtype=tf.float64)
        uniforms = tf.random.stateless_uniform([4, 5, 64],
            tf.stack([particle_seed, 2]), dtype=tf.float64)
        return (tf.broadcast_to(self.theta, [4, 3]), tf.zeros([4, 3], tf.float64),
            tf.reshape(states, [4, 64, 18]), tf.reshape(covariances, [4, 64, 18, 18]),
            noises, observations, tf.reshape(d_states, [4, 64, 18]),
            tf.reshape(d_covariances, [4, 64, 18, 18]), uniforms, paths[:, -1])

    def run(self):
        rows = []
        for update in range(self.arguments.updates):
            self.check_budget()
            arguments = self.make_batch(tf.constant(263041000 + update),
                                        tf.constant(283010000 + update))
            profile = update == self.arguments.profile_update
            if profile:
                tf.profiler.experimental.start(str(self.output / "profile"))
            started = time.monotonic()
            result = self.update(*arguments)
            result["loss"].numpy()
            if profile:
                tf.profiler.experimental.stop()
            row = dict(update=update, **result, wall_seconds=time.monotonic() - started)
            row["update_trace_count"] = self.update.experimental_get_tracing_count()
            row["row_gradient_trace_count"] = self.row_gradient.experimental_get_tracing_count()
            append(self.output / "training.jsonl", row)
            rows.append(row)
            if not bool(result["valid"]) or not bool(result["applied"]):
                raise ValueError("Invalid neural loss/gradient; no optimizer update applied")
            save(self.output / "progress.json", dict(update=update + 1,
                wall_seconds=time.monotonic() - self.started, loss=result["loss"]))
            print(f"update={update + 1} loss={float(result['loss']):.6g} seconds={row['wall_seconds']:.3f}",
                  flush=True)
        changes = [tf.linalg.norm(parameter - initial)
                   for parameter, initial in zip(self.network.parameters, self.initial)]
        save(self.output / "network.json", dict(parameters=self.network.parameters,
            names=[parameter.name for parameter in self.network.parameters]))
        source_check = verify_source_snapshot(self.repository, self.source)
        result = dict(complete=True, mode=self.arguments.mode,
            wall_seconds=time.monotonic() - self.started,
            parameter_changes=changes, all_tensors_updated=all(float(change) > 0 for change in changes),
            first_update=rows[0], last_update=rows[-1], source_check=source_check,
            memory=tf.config.experimental.get_memory_info("GPU:0"),
            efficacy_evaluated=False, model_score_accuracy_evaluated=False)
        result["complete"] &= source_check["unchanged"]
        save(self.output / "result.json", result)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "train"], default="smoke")
    parser.add_argument("--components", type=int, default=2)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--beta", type=float, default=.5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=263044001)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--budget-seconds", type=float, default=900.)
    parser.add_argument("--profile-update", type=int, default=-1)
    arguments = parser.parse_args()
    if arguments.updates < 1 or arguments.budget_seconds <= 0.:
        parser.error("Positive update count and process budget required")
    campaign = NeuralKDMCampaign(arguments)
    try:
        campaign.run()
    except Exception as error:
        save(campaign.output / "failure.json", dict(error=repr(error),
            wall_seconds=time.monotonic() - campaign.started,
            source_check=verify_source_snapshot(campaign.repository, campaign.source)))
        raise


if __name__ == "__main__":
    main()
