"""Bounded target-specific GPU learning curves; no map or posterior promotion."""
import argparse
import datetime as dt
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=1580)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from bayesfilter.testing.inference_validation.designs import ScenarioSpec, ValidationDesign, seed_for
    from bayesfilter.testing.inference_validation.execution import configure_worker, source_state
    from bayesfilter.testing.inference_validation.storage import write_json
    plan = 'docs/plans/bayesfilter-hmc-remaining-gap-execution-2026-09-24.md'
    device_design = ValidationDesign(design_id='d-pricing', engine='accuracy',
        scenario=ScenarioSpec('banana', 'fixed_transport'), replications=1, draws=64,
        seed=2026092421, budget_seconds=args.seconds, device='gpu',
        purpose='target-specific learning-curve cost and gradient diagnostics only', numerical_provenance=plan)
    runtime = configure_worker(device_design)
    import tensorflow as tf
    from bayesfilter.inference.neutra_training import NeuTraReverseKLTrainer, NeuTraTrainerConfig
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
    from bayesfilter.testing.inference_validation.targets import ValidationTarget

    manifest = {'command': sys.argv, 'source': source_state(), 'runtime': runtime,
                'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                'plan_file': plan, 'result_file': str(args.output/'result.json'),
                'python': sys.executable, 'data_version': 'analytic banana b=.5; mixture a=5 w=.3',
                'batch_size': 64, 'validation_size': 1024, 'looks': [0, 32, 128, 512],
                'seeds': [2026092421, 2026092422], 'jit_compile': True,
                'widths': [8, 16], 'learning_rates': [1e-4, 1e-3], 'planned_arms': 16,
                'batch_native_target': 'ValidationTarget.log_prob_and_grad',
                'scalar_fallback': False, 'role': 'development_pricing_not_quality',
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(args.output/'manifest.json', manifest)
    start = time.monotonic()

    @tf.function(input_signature=[tf.TensorSpec((2,), tf.int32)], autograph=False, jit_compile=True)
    def train_noise(seed):
        return tf.random.stateless_normal((64, 2), seed, dtype=tf.float64)

    @tf.function(input_signature=[tf.TensorSpec((2,), tf.int32)], autograph=False, jit_compile=True)
    def validation_noise(seed):
        return tf.random.stateless_normal((1024, 2), seed, dtype=tf.float64)

    rows = []
    arms = [(name, params, seed, width, lr)
        for name, params in [('banana', {'bend': .5}), ('mixture', {'separation': 5., 'weight': .3})]
        for seed in (2026092421, 2026092422) for width in (8, 16) for lr in (1e-4, 1e-3)]
    for index, (name, params, seed, width, lr) in enumerate(arms):
        if time.monotonic()-start >= args.seconds:
            break
        arm_start = time.monotonic()
        target = ValidationTarget(name, params, jit_compile=True)
        observed = []

        class BatchTarget:
            def batch_value_and_score(self, theta):
                if theta.shape.rank != 2 or theta.shape[0] not in (64, 1024):
                    raise ValueError('fixed native batch dimension required')
                observed.append(tuple(theta.shape))
                return target.log_prob_and_grad(theta)

        config = NeuTraTrainerConfig(dimension=2, family='dense_iaf', hidden_layers=(width, width),
            initialization_seed=seed_for(seed, name, width, 'initialization'),
            learning_rate=lr, activation='tanh', s_max=1., initialization_scale=.02,
            learning_rate_schedule='constant', beta1=.9, beta2=.999, epsilon=1e-8,
            gradient_clip_norm=10., gradient_clip_mode='global',
            kernel_initialization='legacy_zero_output', scale_transform='bounded_tanh', jit_compile=True)
        trainer = NeuTraReverseKLTrainer(BatchTarget(), config)
        z = train_noise(tf.constant(seed_for(seed, name, 'derivative'), tf.int32))
        heldout = validation_noise(tf.constant(seed_for(seed, name, 'validation'), tf.int32))
        # One-off diagnostic API, outside the repeated compiled update path.
        _, gradients = trainer.loss_and_gradients(z)
        norm = tf.linalg.global_norm(gradients)
        tf.debugging.assert_positive(norm)
        variables = trainer.transport.trainable_variables
        originals = [tf.identity(v) for v in variables]
        directions = [g/norm for g in gradients]
        differences = []
        for h in (1e-5, 5e-6):
            losses = []
            for sign in (1., -1.):
                for variable, original, direction in zip(variables, originals, directions, strict=True):
                    variable.assign(original + sign*h*direction)
                losses.append(float(tf.reduce_mean(trainer.validation_batch(z).per_sample_loss)))
            differences.append((losses[0]-losses[1])/(2*h))
        for variable, original in zip(variables, originals, strict=True):
            variable.assign(original)
        if any(abs(d-float(norm)) > 1e-5*max(1., float(norm)) for d in differences):
            raise ValueError('scalar objective and supplied-score gradient disagree')

        row = {'target': name, 'parameters': params, 'seed': seed, 'config': asdict(config),
               'gradient_norm': float(norm), 'directional_differences': differences, 'curve': []}
        clips = 0
        update_seconds = 0.
        for update in range(513):
            if time.monotonic()-start >= args.seconds:
                raise TimeoutError('declared learning-curve budget exhausted')
            if update in (0, 32, 128, 512):
                result = trainer.validation_batch(heldout)
                losses = result.per_sample_loss
                tf.debugging.assert_all_finite(losses, 'nonfinite validation objective')
                row['curve'].append({'updates':update, 'validation_loss':float(tf.reduce_mean(losses)),
                    'validation_loss_iid_se':float(tf.sqrt(tf.math.reduce_variance(losses)/1023.)),
                    'clipped_update_count':clips, 'cumulative_update_seconds':update_seconds})
            if update == 512:
                break
            z = train_noise(tf.constant(seed_for(seed, name, update, 'training'), tf.int32))
            tick = time.monotonic()
            step = trainer.train_step(z)
            update_seconds += time.monotonic()-tick
            clips += int(step.clipping_applied)
        payload = trainer.frozen_transport_payload(transport_id=f'd-pricing-{index:02d}',
                                                  target_signature=target.adapter_signature())
        restored = load_frozen_neutra_artifact(payload, expected_target_signature=target.adapter_signature()).transport
        mapped, logdet = trainer.forward_and_logdet(heldout)
        tf.debugging.assert_equal(restored.forward_batch(heldout), mapped)
        tf.debugging.assert_equal(restored.log_abs_det_jacobian_batch(heldout), logdet)
        tf.debugging.assert_near(restored.inverse_theta_to_z_batch(mapped), heldout, atol=1e-10)
        traces = [program.experimental_get_tracing_count() for program in trainer._compiled_train_step.programs.values()]
        if not traces or any(count != 1 for count in traces) or not observed:
            raise ValueError('unbounded update tracing or unobserved batch target')
        row.update(elapsed_seconds=time.monotonic()-arm_start, graph_trace_counts=traces,
                   observed_batch_shapes=sorted(set(observed)), frozen_reload_passed=True,
                   map_file=str(args.output/f'map-{index:02d}.json'))
        write_json(args.output/f'map-{index:02d}.json', payload)
        rows.append(row)
        write_json(args.output/'progress.json', {'planned':len(arms), 'completed':len(rows), 'rows':rows})
    write_json(args.output/'result.json', {'planned':len(arms), 'completed':len(rows), 'rows':rows,
        'elapsed_seconds':time.monotonic()-start, 'posterior_quality_established':False,
        'allocator':tf.config.experimental.get_memory_info('GPU:0'),
        'ranking_supported':False, 'default_promoted':False,
        'scope':'learning-curve and cost hypotheses; every map still requires fresh tuning and posterior assessment'})


if __name__ == '__main__':
    main()
