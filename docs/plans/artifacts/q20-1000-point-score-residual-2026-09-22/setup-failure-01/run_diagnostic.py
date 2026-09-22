"""Run the bounded 1,000-point q20 latent score-residual diagnostic."""
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
PLAN = REPO / 'docs/plans/bayesfilter-q20-1000-point-score-residual-plan-2026-09-22.md'
MAP = RECOVERY / 'campaign-03/attempts/00003-train-neutra/worker/data/direct-w16-lr0.0005-r0-beta1-u512.json'
QUALIFICATION = CAMPAIGN / 'attempts/00001-qualify-beta1/worker/data/result.json'
STAGE = 'gaussian-score-residual-1000'
CAP = 900.0
N = 1000
D = 4
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
    state = read(CAMPAIGN / 'campaign.json')
    config = state['config']
    campaign = Campaign(CAMPAIGN, repo=SOURCE, config=config)
    with campaign.locked():
        original_status = campaign.state['status']
        if original_status != 'ESTIMATION_BUDGET_PAUSED':
            raise RuntimeError('diagnostic expects the inspected paused campaign')
        if campaign.remaining(True) < CAP:
            raise RuntimeError('diagnostic cap exceeds remaining diagnostic budget')
        campaign.state.setdefault('stage_limits', {})[STAGE] = CAP
        campaign.save()
        request = {
            'plan': str(PLAN), 'stage': STAGE, 'cap_seconds': CAP,
            'source_root': str(SOURCE), 'map': str(MAP), 'config': config,
            'script_sha256': checksum(__file__),
            'role': 'diagnostic_score_residual_no_tuning_authority',
            'point_count': N, 'point_distribution': 'iid_standard_normal_latent',
        }
        before = {'campaign_seconds': campaign.remaining(),
                  'diagnostic_seconds': campaign.remaining(True)}
        write(HERE / 'launch.json', {
            'request': request, 'budget_before': before,
            'expected_attempt': str(CAMPAIGN / 'attempts' /
                                    f'{len(campaign.state["attempts"]):05d}-{STAGE}'),
        })
        attempt = campaign.execute(
            STAGE,
            [sys.executable, str(Path(__file__).resolve()), '--worker', '{attempt}/worker'],
            cap_seconds=CAP, diagnostic=True, request=request,
            request_hash=hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
            environment={
                'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
                'TF_NUM_INTRAOP_THREADS': '2', 'TF_NUM_INTEROP_THREADS': '2',
                'OPENBLAS_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1',
                'BAYESFILTER_PRELOAD_CUSTOM_OP': '0',
            },
        )
        receipt = {
            'attempt': attempt, 'budget_before': before,
            'budget_after': {'campaign_seconds': campaign.remaining(),
                             'diagnostic_seconds': campaign.remaining(True)},
            'result': str(Path(attempt['directory']) / 'worker/result.json'),
        }
        campaign.state.setdefault('diagnostic_results', {})[STAGE] = receipt
        campaign.state['status'] = original_status
        campaign.save()
        write(HERE / 'budget-receipt.json', receipt)
        # Update only balance pointers, preserving prior terminal evidence.
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
        status_path = RECOVERY / 'status.json'
        status = read(status_path)
        write(HERE / 'before-status.json', status)
        status['result'] = read(CAMPAIGN / 'result.json')
        status['observed_at_utc'] = datetime.now(timezone.utc).isoformat()
        write(status_path, status)
        print(json.dumps(receipt, indent=2), flush=True)


def _nearest_lower_quantiles(values, probabilities):
    """Return deterministic empirical nearest-lower quantiles without NumPy."""
    import tensorflow as tf
    sorted_values = tf.sort(tf.convert_to_tensor(values, tf.float64))
    indices = tf.cast(tf.floor(tf.constant(probabilities, tf.float64) * (N - 1)), tf.int32)
    return tf.gather(sorted_values, indices)


def worker(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        'role': 'diagnostic_score_residual_no_tuning_authority',
        'plan': str(PLAN), 'command': sys.argv, 'python': sys.version,
        'pid': os.getpid(), 'source_root': str(SOURCE),
        'script_sha256': checksum(__file__),
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'initializing', 'result_file': str(output / 'result.json'),
        'point_count': N, 'point_distribution': 'iid_standard_normal_latent',
    }
    write(output / 'manifest.json', manifest)
    try:
        if os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH') != 'true':
            raise ValueError('memory growth must precede framework import')
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
        from bayesfilter.inference.q20_hmc_qualification import attach_qualification
        from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
        from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import build_fixed_transport_value_score_adapter
        from bayesfilter.inference.hmc_verification import target_status_telemetry_has_failure

        state = read(CAMPAIGN / 'campaign.json')
        config = state['config']
        sources = source_snapshot()
        if sources != state['sources']:
            raise ValueError('numerical snapshot differs from campaign')
        export = read(MAP)
        differences = sorted(k for k in set(sources) | set(export['sources'])
                             if sources.get(k) != export['sources'].get(k))
        allowed = ['bayesfilter/inference/q20_master_program.py',
                   'bayesfilter/inference/q20_master_stages.py']
        if differences != allowed:
            raise ValueError('map differs beyond the previously audited host-only repair')
        bridge = make_q20_tempered_bridge(
            config['target']['q'], jit_compile=True,
            principal_sqrt_backend=config['target']['principal_sqrt_backend'])
        attach_qualification(bridge, QUALIFICATION, config, betas=[1.0])
        base = bridge.fixed_beta_adapter(1.0)
        loaded = load_frozen_neutra_artifact(
            export['frozen_transport'], expected_target_signature=base.adapter_signature())
        adapter = build_fixed_transport_value_score_adapter(
            base_adapter=base, fixed_transport=loaded.transport,
            target_scope=base.target_scope, evidence_path=str(QUALIFICATION),
            xla_hmc_ready=True, full_chain_xla_diagnostic_ready=True)
        seed = scoped_seed(config, STAGE, 'standard-normal-points')
        with tf.device('/CPU:0'):
            points = tf.random.stateless_normal([N, D], seed, dtype=tf.float64)
        check_gpu_contention(readiness)
        manifest.update(
            status='running', sources=sources, git_commit=state['git_commit'],
            config=config, tensorflow=tf.__version__, memory_policy=memory,
            tf32=tf.config.experimental.tensor_float_32_execution_enabled(),
            jit_compile=True, dtype='float64', device=points.device,
            map_path=str(MAP), map_sha256=checksum(MAP),
            map_transport_hash=export['frozen_transport']['transport_hash'],
            map_source_differences=differences, qualification=str(QUALIFICATION),
            qualification_sha256=checksum(QUALIFICATION), point_seed=seed,
            target_signature=base.adapter_signature(), bridge_signature=bridge.signature,
        )
        write(output / 'manifest.json', manifest)

        # The q20 target compiles a fixed leading batch dimension. Evaluate in
        # bounded chunks to avoid an untested 1,000-row device allocation while
        # retaining the batch-native target path in every call.
        batch_size = 32
        if N % batch_size:
            raise RuntimeError('diagnostic point count must divide batch size')
        values, scores, statuses = [], [], []
        eval_started = time.monotonic()
        for offset in range(0, N, batch_size):
            batch = points[offset:offset + batch_size]
            batch_value, batch_score, _batch_status = adapter.log_prob_and_grad_status(batch)
            normalized = adapter.target_status_telemetry(batch)
            target_status_telemetry_has_failure(normalized, expected_shape=(batch_size,))
            values.append(batch_value)
            scores.append(batch_score)
            statuses.append(normalized)
        value = tf.concat(values, axis=0)
        score = tf.concat(scores, axis=0)
        status = {
            key: tf.concat([row[key] for row in statuses], axis=0)
            for key in statuses[0]
            if tf.is_tensor(statuses[0][key]) and statuses[0][key].shape.rank == 1
        }
        eval_seconds = time.monotonic() - eval_started
        telemetry_failure = tf.logical_or(
            status['status_code'] != 0,
            tf.logical_not(status['valid_pre_regularized_score']),
        )
        value_finite = tf.math.is_finite(value)
        score_finite = tf.reduce_all(tf.math.is_finite(score), axis=-1)
        residual = tf.norm(score + points, axis=-1)
        z_norm = tf.norm(points, axis=-1)
        r_value = value + 0.5 * tf.reduce_sum(tf.square(points), axis=-1)
        finite = tf.logical_and(tf.logical_and(value_finite, score_finite),
                                tf.math.is_finite(residual))
        finite = tf.logical_and(finite, tf.math.is_finite(r_value))
        finite_count = int(tf.reduce_sum(tf.cast(finite, tf.int32)).numpy())
        status_failure_count = int(tf.reduce_sum(tf.cast(telemetry_failure, tf.int32)).numpy())
        if finite_count != N or status_failure_count:
            raise RuntimeError(
                f'invalid diagnostic rows: finite={finite_count}/{N}, '
                f'status_failures={status_failure_count}/{N}')

        probabilities = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
        residual_q = _nearest_lower_quantiles(residual, probabilities)
        z_q = _nearest_lower_quantiles(z_norm, probabilities)
        r_q = _nearest_lower_quantiles(r_value, probabilities)
        thresholds = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        exceedance = {
            str(t): float(tf.reduce_mean(tf.cast(residual > t, tf.float64)).numpy())
            for t in thresholds
        }
        r_range = float((tf.reduce_max(r_value) - tf.reduce_min(r_value)).numpy())
        density_ratio = math.exp(r_range) if r_range < math.log(float.fromhex('0x1.fffffffffffffp1023')) else float('inf')
        point_rows = {
            'z': points, 'target_log_prob': value, 'score': score,
            'score_plus_z_norm': residual, 'z_norm': z_norm,
            'r_log_target_over_gaussian_up_to_constant': r_value,
            'target_status': status,
        }
        write(output / 'points.json', point_rows)
        summary = {
            'schema': 'bayesfilter.q20.latent_score_residual_diagnostic.v1',
            'status': 'completed', 'role': 'pointwise_gaussianity_diagnostic_only',
            'point_count': N, 'dimension': D,
            'point_distribution': 'iid_standard_normal_latent', 'seed': seed,
            'quantile_definition': 'nearest_lower_index_floor(p*(N-1))',
            'quantile_probabilities': probabilities,
            'score_residual_norm': {
                'min': float(tf.reduce_min(residual).numpy()),
                'mean': float(tf.reduce_mean(residual).numpy()),
                'std': float(tf.math.reduce_std(residual).numpy()),
                'quantiles': residual_q.numpy().tolist(),
                'max': float(tf.reduce_max(residual).numpy()),
                'exceedance_fraction': exceedance,
            },
            'latent_norm': {'quantiles': z_q.numpy().tolist(),
                            'min': float(tf.reduce_min(z_norm).numpy()),
                            'max': float(tf.reduce_max(z_norm).numpy())},
            'r_log_target_over_gaussian_up_to_constant': {
                'quantiles': r_q.numpy().tolist(),
                'min': float(tf.reduce_min(r_value).numpy()),
                'max': float(tf.reduce_max(r_value).numpy()),
                'range': r_range,
                'implied_density_ratio_range': density_ratio,
            },
            'validity': {
                'finite_value_rows': finite_count, 'finite_score_rows': finite_count,
                'status_failure_rows': status_failure_count,
                'target_status_failure_role': 'hard_diagnostic_validity_veto',
            },
            'timing': {'target_eval_seconds': eval_seconds,
                       'wall_seconds': time.monotonic() - started},
            'promotion_qualified': False, 'posterior_estimate': False,
            'nonclaims': [
                'global posterior Gaussianity', 'posterior covariance',
                'HMC acceptance or convergence', 'map qualification',
                'production default readiness',
            ],
        }
        write(output / 'result.json', summary)
        manifest.update(status='completed', wall_seconds=summary['timing']['wall_seconds'],
                        target_eval_seconds=eval_seconds, finite_rows=finite_count,
                        status_failure_rows=status_failure_count)
        write(output / 'manifest.json', manifest)
    except BaseException as error:
        failure = {'status': 'diagnostic_incomplete', 'type': type(error).__name__,
                   'error': str(error), 'traceback': traceback.format_exc(),
                   'wall_seconds': time.monotonic() - started,
                   'promotion_qualified': False}
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
        raise SystemExit('usage: run_diagnostic.py [--worker OUTPUT]')
