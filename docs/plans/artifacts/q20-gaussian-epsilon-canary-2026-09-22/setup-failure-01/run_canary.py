"""Bounded diagnostic of Gaussian-derived epsilon proposals on the frozen q20 map.

See docs/plans/bayesfilter-q20-gaussian-epsilon-canary-plan-2026-09-22.md.
This uses the preserved public batched GPU/XLA runner; it issues no tuning or
posterior artifact. Numerical snapshots and production settings are unchanged.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

REPO = Path('/home/ubuntu/python/BayesFilter')
SOURCE = Path('/tmp/BayesFilter-q20-recovery-20260922-r2')
RECOVERY = REPO / 'docs/plans/artifacts/q20-recovery-and-affordability-2026-09-22'
CAMPAIGN = RECOVERY / 'campaign-05'
HERE = Path(__file__).resolve().parent
PLAN = REPO / 'docs/plans/bayesfilter-q20-gaussian-epsilon-canary-plan-2026-09-22.md'
MAP = RECOVERY / 'campaign-03/attempts/00003-train-neutra/worker/data/direct-w16-lr0.0005-r0-beta1-u512.json'
STARTS = RECOVERY / 'campaign-04/attempts/00005-tune-neutra-beta1/worker/data/starts.json'
QUALIFICATION = CAMPAIGN / 'attempts/00001-qualify-beta1/worker/data/result.json'
ROOTS = REPO / 'docs/plans/artifacts/q20-gaussian-initial-epsilon-2026-09-22/result.json'
STAGE = 'gaussian-epsilon-canary'
CAP = 1200.0
sys.path.insert(0, str(SOURCE))


def read(path):
    return json.loads(Path(path).read_text())


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if hasattr(value, 'numpy'):
        return clean(value.numpy().tolist())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': str(value)}
    return value


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(clean(value), indent=2, sort_keys=True,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def supervise():
    from bayesfilter.inference.q20_campaign_runtime import Campaign
    config = read(CAMPAIGN / 'campaign.json')['config']
    campaign = Campaign(CAMPAIGN, repo=SOURCE, config=config)
    with campaign.locked():
        original_status = campaign.state['status']
        if original_status != 'ESTIMATION_BUDGET_PAUSED':
            raise RuntimeError('Canary expects the inspected paused campaign')
        if campaign.remaining(True) < CAP:
            raise RuntimeError('Canary cap exceeds remaining diagnostics')
        campaign.state.setdefault('stage_limits', {})[STAGE] = CAP
        campaign.save()
        request = {'plan': str(PLAN), 'stage': STAGE, 'cap_seconds': CAP,
                   'source_root': str(SOURCE), 'map': str(MAP),
                   'config': config, 'root_table': str(ROOTS),
                   'script_sha256': checksum(__file__),
                   'role': 'diagnostic_canary_no_tuning_authority'}
        before = {'campaign_seconds': campaign.remaining(),
                  'diagnostic_seconds': campaign.remaining(True)}
        write(HERE / 'launch.json', {'request': request, 'budget_before': before,
              'expected_attempt': str(CAMPAIGN / 'attempts' /
                 f'{len(campaign.state["attempts"]):05d}-{STAGE}')})
        attempt = campaign.execute(STAGE,
            [sys.executable, str(Path(__file__).resolve()), '--worker', '{attempt}/worker'],
            cap_seconds=CAP, diagnostic=True, request=request,
            request_hash=hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
            environment={'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
                         'TF_NUM_INTRAOP_THREADS': '2', 'TF_NUM_INTEROP_THREADS': '2',
                         'OPENBLAS_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1',
                         'BAYESFILTER_PRELOAD_CUSTOM_OP': '0'})
        receipt = {'attempt': attempt, 'budget_before': before,
                   'budget_after': {'campaign_seconds': campaign.remaining(),
                                    'diagnostic_seconds': campaign.remaining(True)},
                   'result': str(Path(attempt['directory']) / 'worker/result.json')}
        campaign.state.setdefault('diagnostic_results', {})[STAGE] = receipt
        campaign.state['status'] = original_status
        campaign.save()
        write(HERE / 'budget-receipt.json', receipt)
        # Preserve prior terminal snapshots; refresh only remaining balances.
        for name in ('result.json', 'next-phase.json'):
            path = CAMPAIGN / name
            old = read(path)
            write(HERE / ('before-' + name), old)
            if name == 'result.json':
                old['remaining_campaign_seconds'] = campaign.remaining()
                old['remaining_diagnostic_seconds'] = campaign.remaining(True)
            else:
                old['campaign_remaining_seconds'] = campaign.remaining()
                old['diagnostic_remaining_seconds'] = campaign.remaining(True)
            old['latest_diagnostic'] = receipt['result']
            write(path, old)
        state_path = RECOVERY / 'status.json'
        status = read(state_path)
        write(HERE / 'before-status.json', status)
        status['result'] = read(CAMPAIGN / 'result.json')
        status['observed_at_utc'] = datetime.now(timezone.utc).isoformat()
        write(state_path, status)
        print(json.dumps(receipt, indent=2), flush=True)


def worker(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {'role': 'diagnostic_canary_no_tuning_authority', 'plan': str(PLAN),
                'command': sys.argv, 'python': sys.version, 'pid': os.getpid(),
                'source_root': str(SOURCE), 'script_sha256': checksum(__file__),
                'started_at_utc': datetime.now(timezone.utc).isoformat(),
                'status': 'initializing', 'result_file': str(output / 'result.json')}
    write(output / 'manifest.json', manifest)
    try:
        if os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH') != 'true':
            raise ValueError('Memory growth must precede framework import')
        from bayesfilter.inference.q20_gpu_runtime import select_worker_gpu, check_gpu_contention
        readiness = select_worker_gpu()
        manifest.update(launch_readiness=readiness,
                        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
        write(output / 'manifest.json', manifest)
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        from bayesfilter.inference.q20_production_config import scoped_seed
        from bayesfilter.inference.q20_production_training import source_snapshot
        from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge
        from bayesfilter.inference.q20_hmc_qualification import attach_qualification, check_full_chain_health
        from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
        from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import build_fixed_transport_value_score_adapter
        from bayesfilter.inference.hmc import FullChainHMCConfig, ReusableFullChainHMCRunner
        from bayesfilter.inference.hmc_verification import target_status_telemetry_has_failure

        state = read(CAMPAIGN / 'campaign.json')
        config = state['config']
        sources = source_snapshot()
        if sources != state['sources']:
            raise ValueError('Numerical snapshot differs from campaign')
        export = read(MAP)
        differences = sorted(k for k in set(sources) | set(export['sources'])
                             if sources.get(k) != export['sources'].get(k))
        if differences != ['bayesfilter/inference/q20_master_program.py',
                           'bayesfilter/inference/q20_master_stages.py']:
            raise ValueError('Map differs beyond the previously audited host-only repair')
        bridge = make_q20_tempered_bridge(config['target']['q'], jit_compile=True,
                    principal_sqrt_backend=config['target']['principal_sqrt_backend'])
        attach_qualification(bridge, QUALIFICATION, config, betas=[1.0])
        base = bridge.fixed_beta_adapter(1.0)
        loaded = load_frozen_neutra_artifact(export['frozen_transport'],
                    expected_target_signature=base.adapter_signature())
        adapter = build_fixed_transport_value_score_adapter(base_adapter=base,
            fixed_transport=loaded.transport, target_scope=base.target_scope,
            evidence_path=str(QUALIFICATION), xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True)
        physical = tf.constant(read(STARTS)['positions'], tf.float64)
        original = loaded.transport.inverse_theta_to_z_batch(physical)
        tf.debugging.assert_near(loaded.transport.forward_batch(original), physical,
            rtol=config['validation']['reliability_rtol'],
            atol=config['validation']['reliability_atol'])
        proposal_seed = scoped_seed(config, STAGE, 'map-proposal-bank')
        with tf.device('/CPU:0'):
            generated = tf.random.stateless_normal([4, 4], proposal_seed, dtype=tf.float64)
        banks = {'original_prior_starts': original, 'map_proposal_starts': generated}
        manifest.update(status='running', sources=sources, git_commit=state['git_commit'],
            config=config, tensorflow=tf.__version__, memory_policy=memory,
            tf32=tf.config.experimental.tensor_float_32_execution_enabled(),
            jit_compile=True, dtype='float64', batch_native_chains=4,
            transitions_per_chain=4, mass='identity_in_frozen_latent_coordinates',
            map_path=str(MAP), map_sha256=checksum(MAP),
            map_transport_hash=export['frozen_transport']['transport_hash'],
            map_source_differences=differences, qualification=str(QUALIFICATION),
            qualification_sha256=checksum(QUALIFICATION), start_bank_sha256=checksum(STARTS),
            gaussian_root_table_sha256=checksum(ROOTS), proposal_seed=proposal_seed,
            target_signature=base.adapter_signature(), bridge_signature=bridge.signature)
        write(output / 'manifest.json', manifest)

        def save_tensors(directory, name, tree):
            if isinstance(tree, dict):
                for key, value in tree.items():
                    save_tensors(directory, name + '-' + key, value)
            else:
                tf.io.write_file(str(directory / (name + '.tensor')),
                                 tf.io.serialize_tensor(tf.convert_to_tensor(tree)))

        bank_notes = {}
        for name, bank in banks.items():
            value, score, status = adapter.log_prob_and_grad_status(bank)
            valid = (bool(tf.reduce_all(tf.math.is_finite(value))) and
                     bool(tf.reduce_all(tf.math.is_finite(score))) and
                     not target_status_telemetry_has_failure(status, expected_shape=(4,)))
            bank_notes[name] = {'valid': valid, 'latent_positions': bank,
                'physical_positions': loaded.transport.forward_batch(bank),
                'target_value': value, 'score': score, 'status': status,
                'score_plus_z_norm_by_chain': tf.norm(score + bank, axis=-1),
                'role': 'diagnostic_starts_not_posterior_samples'}
        write(output / 'start-banks.json', bank_notes)
        runner = ReusableFullChainHMCRunner(adapter, original,
            FullChainHMCConfig(num_results=4, num_burnin_steps=0,
                step_size=.062002709114199195, num_leapfrog_steps=3,
                seed=scoped_seed(config, STAGE, 'runner'), use_xla=True,
                target_scope=base.target_scope, target_status_trace_policy='per_chain_step',
                capture_candidate_health=True), dynamic_num_leapfrog_steps=True)
        pairs = [{'L': 3, 'epsilon': .062002709114199195, 'role': 'small_step_control'}]
        pairs += [{'L': row['L'], 'epsilon': row['epsilon'], 'role': 'gaussian_root_canary'}
                  for row in read(ROOTS)['rows']]
        rows = []
        for pair in pairs:
            for name, bank in banks.items():
                if not bank_notes[name]['valid']:
                    rows.append({**pair, 'bank': name, 'status': 'invalid_initial_bank_not_run'})
                    continue
                check_gpu_contention(readiness)
                seed = scoped_seed(config, STAGE, 'paired-transitions', name)
                label = f'{len(rows):02d}-{name}-L{pair["L"]}-{pair["role"]}'
                folder = output / label
                folder.mkdir()
                write(output / 'progress.json', {'current': label, 'completed': len(rows),
                                                'elapsed_seconds': time.monotonic()-started})
                print('Starting ' + label, flush=True)
                call_started = time.monotonic()
                result = runner.run(current_state=bank, seed=seed,
                    step_size=pair['epsilon'], num_leapfrog_steps=pair['L'])
                samples, trace = result.samples, result.trace
                save_tensors(folder, 'samples', samples)
                save_tensors(folder, 'trace', trace)
                health_error = None
                try:
                    check_full_chain_health(result)
                except (ValueError, tf.errors.InvalidArgumentError) as error:
                    health_error = str(error)
                retained_valid = (bool(tf.reduce_all(tf.math.is_finite(samples))) and
                    bool(tf.reduce_all(tf.math.is_finite(trace['target_log_prob']))) and
                    not target_status_telemetry_has_failure(trace['target_status_telemetry'],
                                                           expected_shape=(4, 4)))
                log_accept = trace['log_accept_ratio']
                probability = tf.exp(tf.minimum(log_accept, tf.zeros_like(log_accept)))
                previous = tf.concat([bank[None], samples[:-1]], axis=0)
                movement = tf.reduce_sum(tf.cast(tf.reduce_any(samples != previous, axis=-1), tf.int32), axis=0)
                row = {**pair, 'bank': name, 'seed': seed, 'trace_path': str(folder),
                    'status': ('failed_numerical_health' if health_error else
                               'no_movement_in_at_least_one_chain' if bool(tf.reduce_any(movement == 0)) else
                               'not_rejected_by_tiny_canary'),
                    'health_error': health_error, 'retained_valid': retained_valid,
                    'mean_acceptance_probability': tf.reduce_mean(probability),
                    'acceptance_probability_by_chain': tf.reduce_mean(probability, axis=0),
                    'acceptance_probabilities': probability,
                    'accepted_count_by_chain': tf.reduce_sum(tf.cast(trace['is_accepted'], tf.int32), axis=0),
                    'movement_count_by_chain': movement,
                    'log_accept_ratio': log_accept,
                    'proposal_displacement_norm': tf.norm(trace['proposed_state']-previous, axis=-1),
                    'retained_displacement_norm': tf.norm(samples-previous, axis=-1),
                    'score_finite': trace['target_score_finite'],
                    'proposed_status': trace['proposed_target_status_telemetry'],
                    'native_divergence': trace.get('has_divergence', 'not_exposed'),
                    'device': samples.device,
                    'wall_seconds': time.monotonic()-call_started,
                    'runner_trace_count': runner._runner.experimental_get_tracing_count()}
                write(folder / 'summary.json', row)
                rows.append(clean(row))
                write(output / 'partial-result.json', {'rows': rows, 'promotion_qualified': False})
                print(json.dumps({k: clean(row[k]) for k in ('L', 'epsilon', 'bank', 'status',
                    'acceptance_probability_by_chain', 'movement_count_by_chain', 'wall_seconds')}), flush=True)
                if not retained_valid:
                    raise RuntimeError('Invalid retained state/target stops the shared diagnostic')
                if pair['role'] == 'small_step_control' and health_error:
                    raise RuntimeError('Small-step control failed numerical health; inspect harness/starts')
        final = {'status': 'canary_completed', 'rows': rows,
                 'wall_seconds': time.monotonic()-started,
                 'promotion_qualified': False, 'posterior_estimate': False,
                 'terminal_capacity': check_gpu_contention(readiness),
                 'allocator': tf.config.experimental.get_memory_info('GPU:0')}
        write(output / 'result.json', final)
        manifest.update(status='completed', wall_seconds=final['wall_seconds'],
                        runner_trace_count=runner._runner.experimental_get_tracing_count())
        write(output / 'manifest.json', manifest)
    except BaseException as error:
        failure = {'status': 'canary_incomplete', 'type': type(error).__name__,
                   'error': str(error), 'traceback': traceback.format_exc(),
                   'wall_seconds': time.monotonic()-started, 'promotion_qualified': False}
        write(output / 'failure.json', failure)
        manifest.update(status='failed', failure=failure)
        write(output / 'manifest.json', manifest)
        raise


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(sys.argv[2])
    elif len(sys.argv) == 1:
        supervise()
    else:
        raise SystemExit('usage: run_canary.py [--worker OUTPUT]')
