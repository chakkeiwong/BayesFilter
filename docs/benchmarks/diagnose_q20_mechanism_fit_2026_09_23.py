"""Bounded q20 correction-fit diagnostic. Never issues a production/HMC artifact."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback


class CandidateNumericalFailure(ValueError):
    """Reject the affected correction, without rejecting the research direction."""


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def seed(role, index=0):
    raw = hashlib.sha256(f'q20-short-fit-canary-2026-09-23:{role}:{index}'.encode()).digest()
    return [int.from_bytes(raw[i:i+4], 'big') & 0x7fffffff for i in (0, 4)]


def host(value):
    if isinstance(value, dict):
        return {k: host(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [host(v) for v in value]
    if hasattr(value, 'numpy'):
        return value.numpy().tolist()
    return value


def paired(after, before):
    differences = [a-b for a, b in zip(after, before, strict=True)]
    mean = statistics.mean(differences)
    se = statistics.stdev(differences) / math.sqrt(len(differences))
    critical = statistics.NormalDist().inv_cdf(1-.05/(2*4))
    return {'mean_difference': mean, 'standard_error': se,
            'simultaneous_approx_95_interval': [mean-critical*se, mean+critical*se],
            'normal_critical': critical, 'rows': len(differences)}


def flatten(blocks, key):
    return [v for b in blocks for v in b[key]]


def geometry(blocks):
    residual = flatten(blocks, 'residual')
    norm2 = [sum(x*x for x in row) for row in residual]
    r = flatten(blocks, 'log_ratio')
    return {'rows': len(residual), 'valid_rows': sum(flatten(blocks, 'status')),
            'mean_loss': statistics.mean(flatten(blocks, 'loss_rows')),
            'vector_residual_rms': math.sqrt(statistics.mean(norm2)),
            'coordinate_2_residual_rms': math.sqrt(statistics.mean(row[1]**2 for row in residual)),
            'r_range': max(r)-min(r), 'r_centered_rms': statistics.pstdev(r)}


def conditional(blocks):
    latent, loss, residual = (flatten(blocks, k) for k in ('latent', 'loss_rows', 'residual'))
    result = {}
    for name, predicate in [('left', lambda x: x < -1),
                            ('center', lambda x: -1 <= x <= 1),
                            ('right', lambda x: x > 1)]:
        indices = [i for i, row in enumerate(latent) if predicate(row[1])]
        result[name] = {'rows': len(indices),
            'mean_loss': statistics.mean(loss[i] for i in indices) if indices else None,
            'residual_rms': math.sqrt(statistics.mean(
                sum(x*x for x in residual[i]) for i in indices)) if indices else None}
    return result


def run_worker(request, root):
    began = time.monotonic()
    root.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'initializing', 'command': sys.argv, 'python': sys.executable,
        'started_at': datetime.now(timezone.utc).isoformat(), 'git_commit': request['git_commit'],
        'plan_file': request['plan_file'], 'result_file': str(root/'result.json'),
        'request': request, 'runner_sha256': sha(__file__),
        'data_version': 'original source-bound q20/T30 UKF approximate posterior',
        'hmc_admitted': False, 'cpu_reference': False,
        'tf_force_gpu_allow_growth': os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH'),
        'seeds': {role: [seed(role, i) for i in range(count)] for role, count in
                  ([('confirmation', 16)] if request.get('confirmation_from') else
                   [('heldout', 4), ('gradient', 16), ('calibration', 1), ('training', 64)])}}
    save(root/'manifest.json', manifest)
    try:
        return numerical_worker(request, root, manifest, began)
    except BaseException as error:
        save(root/'failure.json', {'type': type(error).__name__, 'message': str(error),
                                  'traceback': traceback.format_exc(),
                                  'wall_seconds': time.monotonic()-began})
        manifest.update(status='failed', wall_seconds=time.monotonic()-began)
        save(root/'manifest.json', manifest)
        raise


def numerical_worker(request, root, manifest, began):
    if os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH', '').lower() != 'true':
        raise ValueError('memory growth must be set before framework import')
    if sha(request['mechanism_module']) != request['mechanism_sha256']:
        raise ValueError('mechanism source changed')
    from bayesfilter.inference.q20_campaign_runtime import source_snapshot
    if source_snapshot(request['source_root']) != request['source_hashes']:
        raise ValueError('original numerical source changed')
    from bayesfilter.inference.q20_gpu_runtime import select_worker_gpu
    manifest['readiness'] = select_worker_gpu(requested='auto')
    save(root/'manifest.json', manifest)
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge
    from bayesfilter.inference.tempered_transport_ensemble_tf import restore_trainable_transport_checkpoint
    from bayesfilter.inference.q20_production_config import digest
    spec = importlib.util.spec_from_file_location('fit_canary_mechanisms', request['mechanism_module'])
    mechanisms = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mechanisms)
    if sha(request['checkpoint']) != request['checkpoint_sha256']:
        raise ValueError('saved checkpoint changed')
    cohort = read(request['checkpoint'])
    if digest({k:v for k,v in cohort.items() if k != 'checkpoint_hash'}) != cohort['checkpoint_hash']:
        raise ValueError('cohort checksum differs')
    state = cohort['cohort'][request['candidate']]['session']
    if digest({k:v for k,v in state.items() if k != 'state_hash'}) != state['state_hash']:
        raise ValueError('checkpoint state checksum differs')
    bridge = make_q20_tempered_bridge(20, jit_compile=True,
        principal_sqrt_backend='tensorflow_eigh_strict_factor_cached')
    base = restore_trainable_transport_checkpoint(state['map'], expected_context={
        'bridge_signature': bridge.signature, 'target_signature': bridge.target_signature})
    if state['map']['beta'] != 1. or bridge.parameter_dim != 4:
        raise ValueError('unexpected target scope')
    old_base = [tf.identity(v) for v in base.trainable_variables]
    manifest.update(status='running', tensorflow=tf.__version__, memory_policy=memory,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        tf32=tf.config.experimental.tensor_float_32_execution_enabled(), jit_compile=True,
        target_signature=bridge.target_signature, bridge_signature=bridge.signature,
        checkpoint_state_hash=state['state_hash'], base_device=base.trainable_variables[0].device,
        source_hashes=request['source_hashes'], batch_size=32,
        batch_native_target=True, sample_wise_target_fallback=False,
        parameter_dimension=4, baseline_lifetime_updates=state['iteration'])
    if 'GPU' not in base.trainable_variables[0].device.upper():
        raise ValueError('restored transport not placed on GPU')
    save(root/'manifest.json', manifest)

    def budget():
        if time.monotonic()-began >= request['cooperative_seconds']:
            raise TimeoutError('shared canary wall-time limit reached')

    @tf.function(input_signature=[tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
    def noise(key):
        return tf.random.stateless_normal([32, 4], key, dtype=tf.float64)

    def target(physical):
        value, score, status = bridge.value_score_status(physical, tf.constant(1., tf.float64))
        return value, score, status['bridge_valid']

    def compose(parts):
        correction = parts[0]
        for part in parts[1:]:
            correction = mechanisms.ComposedMechanism(correction, part,
                                                       train_first=True, train_second=True)
        return mechanisms.ComposedMechanism(correction, base, train_first=True, train_second=False)

    def make_measure(flow):
        @tf.function(input_signature=[tf.TensorSpec([32, 4], tf.float64)],
                     jit_compile=True, autograph=False)
        def measure(z):
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
                tape.watch(z)
                physical, ld = flow.forward_and_logdet(z)
            value, score, status = target(physical)
            residual = (tape.gradient(physical, z, output_gradients=score)
                        + tape.gradient(ld, z, unconnected_gradients=tf.UnconnectedGradients.ZERO) + z)
            valid = tf.reduce_all(status) & tf.reduce_all(tf.stack([
                tf.reduce_all(tf.math.is_finite(x)) for x in (physical, ld, value, score, residual)]))
            return {'latent': z, 'physical': physical, 'target_value': value, 'target_score': score,
                    'status': status, 'valid': valid, 'loss_rows': -value-ld,
                    'log_ratio': value+ld+.5*tf.reduce_sum(z*z, axis=1), 'residual': residual}
        return measure

    def checked_measure(measure, z):
        budget()
        result = measure(z)
        if not bool(result['valid'].numpy()):
            raise CandidateNumericalFailure('invalid target/geometry batch')
        return result

    def bank(measure, destination):
        role, batches = ('confirmation', 16) if request.get('confirmation_from') else ('heldout', 4)
        rows = [host(checked_measure(measure, noise(tf.constant(seed(role, i))))) for i in range(batches)]
        save(destination, rows)
        return rows

    def cached_gradient(flow, estimator):
        signature = [tf.TensorSpec([32, 4], tf.float64), tf.TensorSpec([32, 4], tf.float64),
                     tf.TensorSpec([32], tf.float64), tf.TensorSpec([32, 4], tf.float64)]
        @tf.function(input_signature=signature, jit_compile=True, autograph=False)
        def evaluate(z, expected_physical, values, score):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(flow.trainable_variables)
                if estimator == 'path':
                    physical, ld, qscore = mechanisms._forward_proposal_score(flow, z)
                    carrier = tf.reduce_mean(tf.reduce_sum(
                        physical*tf.stop_gradient(qscore-score), axis=1))
                else:
                    physical, ld = flow.forward_and_logdet(z)
                    attached = tf.stop_gradient(values) + tf.reduce_sum(
                        (physical-tf.stop_gradient(physical))*tf.stop_gradient(score), axis=1)
                    carrier = tf.reduce_mean(-attached-ld)
            gradients = tape.gradient(carrier, flow.trainable_variables)
            flat = tf.concat([tf.reshape(g, [-1]) for g in gradients], axis=0)
            discrepancy = tf.reduce_max(tf.abs(physical-expected_physical))
            valid = (tf.reduce_all(tf.math.is_finite(flat)) &
                     (discrepancy <= 1e-12*(1+tf.reduce_max(tf.abs(expected_physical)))))
            return flat, valid
        return evaluate

    baseline_flow = compose([mechanisms.FreeDiagonalAffine([0.]*4, [0.]*4)])
    baseline_measure = make_measure(baseline_flow)
    baseline = bank(baseline_measure, root/'baseline-heldout.json')
    results = {'baseline': geometry(baseline), 'conditional_baseline': conditional(baseline), 'arms': {}}
    print(json.dumps({'event': 'baseline', **results['baseline'], 'seconds': time.monotonic()-began}), flush=True)

    if request.get('confirmation_from'):
        prior_path = Path(request['confirmation_from'])
        if sha(prior_path) != request['confirmation_sha256']:
            raise ValueError('frozen canary result changed')
        prior = read(prior_path)
        parts = [mechanisms.mechanism_from_state(s) for s in prior['arms']['scalar']['final_parameters']]
        flow = compose(parts)
        before = [tf.identity(v) for v in flow.trainable_variables]
        final = bank(make_measure(flow), root/'scalar-heldout.json')
        differences = [a-b for a,b in zip(flatten(final,'loss_rows'), flatten(baseline,'loss_rows'), strict=True)]
        mean = statistics.mean(differences)
        se = statistics.stdev(differences)/math.sqrt(len(differences))
        critical = statistics.NormalDist().inv_cdf(.975)
        interval = [mean-critical*se, mean+critical*se]
        unchanged = all(bool(tf.reduce_all(v == old).numpy()) for v,old in
                        zip(flow.trainable_variables,before)) and all(
                        bool(tf.reduce_all(v == old).numpy()) for v,old in zip(base.trainable_variables,old_base))
        if not unchanged or source_snapshot(request['source_root']) != request['source_hashes']:
            raise ValueError('frozen confirmation map/source changed')
        results.update(status='complete', role='single_fresh_confirmation_of_frozen_scalar_checkpoint',
            scalar=geometry(final), conditional_scalar=conditional(final),
            loss_difference={'mean':mean, 'standard_error':se, 'approximate_95_interval':interval,
                             'rows':512, 'single_predeclared_primary_contrast':True},
            primary_fit_screen_passed=interval[1] < 0, parameters_unchanged=unchanged,
            new_optimizer_updates=0, pooled_with_pilot=False, wall_seconds=time.monotonic()-began,
            posterior_qualified=False, production_qualified=False, full_1000_point_verification=False)
        save(root/'result.json',results)
        manifest.update(status='complete',wall_seconds=results['wall_seconds'],
                        allocator=tf.config.experimental.get_memory_info('GPU:0'))
        save(root/'manifest.json',manifest)
        return

    for arm in ('affine', 'scalar'):
        budget()
        arm_began = time.monotonic()
        folder = root/arm
        folder.mkdir()
        affine = mechanisms.FreeDiagonalAffine([0.]*4, [0.]*4)
        parts = [affine]
        if arm == 'scalar':
            scalar = mechanisms.ScalarSigmoidMixture(dimension=4, coordinate=1,
                log_slopes=[0.]*3, offsets=[-1., 0., 1.], weight_logits=[0.]*3,
                inverse_atol=1e-10, inverse_rtol=1e-10, inverse_max_iterations=100)
            parts.insert(0, scalar)
        flow = compose(parts)
        measure = make_measure(flow)
        initial_states = [p.parameter_state() for p in parts]
        try:
            initial = bank(measure, folder/'initial-heldout.json')
        except CandidateNumericalFailure as error:
            results['arms'][arm] = {'status': 'initialization_rejected', 'reason': str(error)}
            save(root/'progress.json', results)
            continue
        std = cached_gradient(flow, 'standard')
        path = cached_gradient(flow, 'path') if arm == 'scalar' else None
        probe_rows = []
        for i in range(16):
            z = noise(tf.constant(seed('gradient', i)))
            start = time.monotonic()
            block = checked_measure(measure, z)
            target_seconds = time.monotonic()-start
            args = (z, block['physical'], block['target_value'], block['target_score'])
            start = time.monotonic()
            gradient, valid = std(*args)
            if not bool(valid.numpy()):
                raise ValueError('invalid standard gradient probe')
            standard_seconds = time.monotonic()-start
            row = {'index': i, 'standard': host(gradient),
                   'target_and_geometry_seconds': target_seconds, 'standard_seconds': standard_seconds}
            if path is not None:
                start = time.monotonic()
                gradient, valid = path(*args)
                if not bool(valid.numpy()):
                    raise ValueError('invalid path gradient probe')
                row.update(path=host(gradient), path_seconds=time.monotonic()-start)
            probe_rows.append(row)
        save(folder/'gradient-probe.json', probe_rows)
        clip_norm = 10.*max(math.sqrt(sum(x*x for x in r['standard'])) for r in probe_rows)
        trainer = mechanisms.AdamMechanismCanary(flow, target, batch_size=32,
            estimator='standard', learning_rate=.01, beta1=.9, beta2=.999, epsilon=1e-7,
            gradient_clip_norm=clip_norm, jit_compile=True)
        state_variables = trainer.variables + tuple(trainer.optimizer.variables)
        saved = [tf.identity(v) for v in state_variables]
        calibration_z = noise(tf.constant(seed('calibration')))
        cal_before = checked_measure(measure, calibration_z)
        before_loss = float(tf.reduce_mean(cal_before['loss_rows']).numpy())
        # Check the diagnostic cached derivative against the actual training call chain.
        evaluated = trainer.evaluate(calibration_z)
        cached, _ = std(calibration_z, cal_before['physical'], cal_before['target_value'], cal_before['target_score'])
        actual = tf.concat([tf.reshape(g, [-1]) for g in evaluated['gradients']], 0)
        tf.debugging.assert_near(actual, cached, atol=1e-10, rtol=1e-10)
        calibration = []
        selected = None
        for rate in (.01, .001, .0001):
            budget()
            for variable, old in zip(state_variables, saved):
                variable.assign(old)
            trainer.optimizer.learning_rate.assign(rate)
            step = trainer.train_step(calibration_z)
            accepted = bool(step['valid'].numpy())
            after_loss = None
            if accepted:
                try:
                    candidate = checked_measure(measure, calibration_z)
                    after_loss = float(tf.reduce_mean(candidate['loss_rows']).numpy())
                except CandidateNumericalFailure:
                    accepted = False
            predicted = float(tf.add_n([tf.reduce_sum(g*(v-old)) for g,v,old in
                zip(evaluated['gradients'], trainer.variables, saved)]).numpy())
            calibration.append({'rate': rate, 'valid': accepted, 'before_loss': before_loss,
                                'after_loss': after_loss, 'linear_predicted_change': predicted})
            if accepted and after_loss < before_loss:
                selected = rate
                break
        for variable, old in zip(state_variables, saved):
            variable.assign(old)
        save(folder/'calibration.json', {'trials': calibration, 'selected_rate': selected,
            'gradient_clip_norm': clip_norm, 'heldout_used': False})
        if selected is None:
            results['arms'][arm] = {'status': 'calibration_failed', 'initial': geometry(initial)}
            save(root/'progress.json', results)
            continue
        trainer.optimizer.learning_rate.assign(selected)

        @tf.function(input_signature=[tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
        def train(key):
            z = tf.random.stateless_normal([32, 4], key, dtype=tf.float64)
            out = trainer._train_step(z)
            return {k: out[k] for k in ('loss', 'valid', 'iteration', 'gradient_norm',
                                        'clipped', 'parameter_update_norm')}

        history = []
        training_rejected = False
        for i in range(64):
            budget()
            start = time.monotonic()
            entry = host(train(tf.constant(seed('training', i))))
            entry['wall_seconds'] = time.monotonic()-start
            if not entry['valid']:
                save(folder/'rejected-update.json', {'step': i, **entry})
                training_rejected = True
                break
            history.append(entry)
            if (i+1) % 16 == 0:
                save(folder/'history.json', history)
                save(folder/f'checkpoint-{i+1:03d}.json', {'parts': [p.parameter_state() for p in parts],
                    'optimizer': [{'name': v.name, 'value': host(v)} for v in trainer.optimizer.variables],
                    'next_training_index': i+1, 'hmc_admitted': False})
                print(json.dumps({'event': 'training', 'arm': arm, 'updates': i+1,
                                  'seconds': time.monotonic()-began}), flush=True)
        if training_rejected:
            results['arms'][arm] = {'status': 'update_rejected', 'updates': len(history)}
            save(root/'progress.json', results)
            continue
        try:
            final = bank(measure, folder/'final-heldout.json')
        except CandidateNumericalFailure as error:
            results['arms'][arm] = {'status': 'heldout_rejected', 'reason': str(error), 'updates': 64}
            save(root/'progress.json', results)
            continue
        reconstructed = compose([mechanisms.mechanism_from_state(p.parameter_state()) for p in parts])
        @tf.function(input_signature=[tf.TensorSpec([32, 4], tf.float64)], jit_compile=True, autograph=False)
        def roundtrip(z):
            a, la = flow.forward_and_logdet(z)
            b, lb = reconstructed.forward_and_logdet(z)
            back, _ = reconstructed.inverse_and_forward_logdet(b)
            return tf.reduce_max(tf.abs(a-b)), tf.reduce_max(tf.abs(la-lb)), tf.reduce_max(tf.abs(back-z))
        errors = host(roundtrip(calibration_z))
        if not all(math.isfinite(v) and v < 1e-8 for v in errors):
            raise ValueError('correction reconstruction or inverse failure')
        unchanged = all(bool(tf.reduce_all(v == old).numpy()) for v, old in zip(base.trainable_variables, old_base))
        if not unchanged:
            raise ValueError('frozen base changed')
        final_loss, initial_loss = (flatten(x, 'loss_rows') for x in (final, initial))
        vs_initial = paired(final_loss, initial_loss)
        vs_baseline = paired(final_loss, flatten(baseline, 'loss_rows'))
        result = {'status': 'complete', 'updates': 64, 'training_rows': 64*32,
            'initial': geometry(initial), 'final': geometry(final),
            'loss_vs_initial': vs_initial, 'loss_vs_baseline': vs_baseline,
            'fit_screen_passed': all(x['simultaneous_approx_95_interval'][1] < 0 for x in (vs_initial, vs_baseline)),
            'initial_parameters': initial_states, 'final_parameters': [p.parameter_state() for p in parts],
            'frozen_base_unchanged': unchanged, 'reconstruction_max_errors': errors,
            'conditional_initial': conditional(initial), 'conditional_final': conditional(final),
            'calibrated_learning_rate': selected, 'calibrated_clip_norm': clip_norm,
            'clipped_updates': sum(row['clipped'] for row in history),
            'training_seconds': sum(row['wall_seconds'] for row in history),
            'steady_state_update_seconds': statistics.mean(row['wall_seconds'] for row in history[1:]),
            'measure_traces': measure.experimental_get_tracing_count(),
            'train_traces': train.experimental_get_tracing_count(),
            'wall_seconds': time.monotonic()-arm_began, 'hmc_admitted': False}
        if arm == 'scalar':
            def variance_trace(key):
                rows = [r[key] for r in probe_rows]
                return sum(statistics.variance(column) for column in zip(*rows))
            a, b = variance_trace('standard'), variance_trace('path')
            result['path_probe'] = {'minibatches': 16, 'batch_size': 32,
                'standard_covariance_trace': a, 'path_covariance_trace': b,
                'path_over_standard_variance': b/a,
                'standard_gradient_seconds': statistics.mean(r['standard_seconds'] for r in probe_rows[1:]),
                'path_gradient_seconds': statistics.mean(r['path_seconds'] for r in probe_rows[1:]),
                'shared_target_geometry_seconds': statistics.mean(r['target_and_geometry_seconds'] for r in probe_rows[1:]),
                'inference': 'descriptive_only_fixed_initial_checkpoint_not_training_ranking'}
        results['arms'][arm] = result
        save(folder/'result.json', result)
        save(root/'progress.json', results)
        print(json.dumps({'event': 'arm_complete', 'arm': arm, 'fit_screen_passed': result['fit_screen_passed'],
                          'final': result['final'], 'seconds': time.monotonic()-began}), flush=True)
    if source_snapshot(request['source_root']) != request['source_hashes']:
        raise ValueError('frozen source changed during run')
    results.update(status='complete', wall_seconds=time.monotonic()-began,
        posterior_qualified=False, production_qualified=False,
        full_1000_point_verification=False, repeated_training_ranking_established=False)
    save(root/'result.json', results)
    manifest.update(status='complete', wall_seconds=results['wall_seconds'],
                    allocator=tf.config.experimental.get_memory_info('GPU:0'))
    save(root/'manifest.json', manifest)


def supervise(request, root):
    from bayesfilter.inference.q20_campaign_runtime import Campaign
    campaign = Campaign(request['campaign_root'], repo=request['source_root'], config=request['config'])
    with campaign.locked():
        original_stage = 'short-mechanism-fit-canary'
        confirmation = bool(request.get('confirmation_from'))
        stage = 'short-mechanism-fit-confirmation' if confirmation else original_stage
        allowance = campaign.state.setdefault('owner_requested_canaries', {})
        if original_stage not in allowance and confirmation:
            raise ValueError('confirmation requires the original bounded canary allocation')
        if original_stage not in allowance:
            allowance[stage] = {'instruction': 'I agree. do the test', 'plan': request['plan_file'],
                'diagnostic_seconds': 1800., 'within_existing_total_campaign': True}
            campaign.state['diagnostic_limit'] += 1800.
        campaign.state.setdefault('stage_limits', {})[stage] = 360. if confirmation else 1800.
        previous = [a for a in campaign.state['attempts'] if a['stage'] == stage]
        if any(a['status'] == 'completed' for a in previous):
            raise ValueError('canary already ran; inspect its artifacts')
        if len(previous) >= 2:
            raise ValueError('two-attempt cap exhausted')
        aggregate_spent = sum(a.get('elapsed_seconds',0.) for a in campaign.state['attempts']
                              if a['stage'] in (original_stage,'short-mechanism-fit-confirmation'))
        cap = min(campaign.remaining(True), campaign.stage_remaining(stage), 1800.-aggregate_spent)
        destination = root/f'worker-{len(previous)+1:02d}'
        worker_request = {**request, 'cooperative_seconds': cap-15.}
        request_path = root/f'request-{len(previous)+1:02d}.json'
        save(request_path, worker_request)
        attempt = campaign.execute(stage, [sys.executable, str(Path(__file__).resolve()),
            'worker', '--request', str(request_path), '--output', str(destination)],
            cap_seconds=cap, diagnostic=True, environment={'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
                'TF_NUM_INTRAOP_THREADS': '2', 'TF_NUM_INTEROP_THREADS': '2', 'OMP_NUM_THREADS': '2'},
            request=worker_request)
        accounting = {'attempt': attempt, 'remaining_campaign_seconds': campaign.remaining(),
            'remaining_diagnostic_seconds': campaign.remaining(True),
            'remaining_stage_seconds': campaign.stage_remaining(stage),
            'remaining_this_canary_seconds': 1800.-sum(
                a.get('elapsed_seconds',0.) for a in campaign.state['attempts']
                if a['stage'] in (original_stage,'short-mechanism-fit-confirmation'))}
        save(root/f'accounting-{len(previous)+1:02d}.json', accounting)
        campaign.state['status'] = 'SHORT_FIT_CANARY_'+attempt['status'].upper()
        campaign.save()
        print(json.dumps({k:v for k,v in accounting.items() if k != 'attempt'}), flush=True)
        if attempt['status'] != 'completed':
            raise RuntimeError('canary worker '+attempt['status'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('supervise', 'worker'))
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    request = read(args.request)
    sys.path.insert(0, request['source_root'])
    (supervise if args.mode == 'supervise' else run_worker)(request, Path(args.output))
