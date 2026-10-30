"""Independent saved-draw diagnostic; never a runtime estimator or tuning route."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PLAN = 'docs/plans/bayesfilter-modern-hmc-c4-raw-precision-audit-2026-10-30.md'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + chr(10))
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
            label + ': shape or finite-value mismatch')
    require(np.allclose(a, b, rtol=2e-9, atol=1e-12), label + ': arithmetic mismatch')


def payload_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def positions_from_active(active, execution):
    """Replay the issued affine map, never refactor its covariance."""
    layers = execution['layers']
    require(len(layers) == 1 and layers[0]['kind'] == 'fixed_mass',
            'audit supports exactly one frozen affine layer')
    artifact = layers[0]['artifact']
    require(artifact['factor_orientation'] == 'row_right_transpose',
            'unsupported affine orientation')
    z = np.asarray(active, dtype=float)
    center = np.asarray(artifact['position'], dtype=float)
    factor = np.asarray(artifact['factor'], dtype=float)
    require(z.ndim == 3 and center.ndim == 1 and z.shape[-1] == center.size
            and factor.shape == (center.size, center.size)
            and all(np.isfinite(v).all() for v in (z, center, factor)),
            'invalid frozen affine dimensions or values')
    return center + z @ factor.T


def batch_report(values, batch, *, min_batches=20):
    x = np.asarray(values, dtype=float)
    require(x.ndim == 3 and x.shape[1] >= 2 and np.isfinite(x).all(),
            'finite [draw,chain,quantity] array required')
    require(type(batch) is int and batch >= 1, 'positive integral batch required')
    n, m, _ = x.shape
    small = batch // 3
    if small < 1 or n // batch < min_batches or n // small < min_batches:
        return dict(available=False, batch=batch, reason='insufficient_complete_batches')

    def lrv(b):
        count = n // b
        means = x[:count * b].reshape(count, b, m, x.shape[2]).mean(axis=1)
        return b * ((means - means.mean(axis=0)) ** 2).sum(axis=0) / (count - 1)

    plain = lrv(batch)
    extrapolated = 2 * plain - lrv(small)
    valid = np.all(extrapolated > 0, axis=0)
    variance = extrapolated.sum(axis=0) / (m * m * n)
    mcse = np.sqrt(np.where(valid, variance, np.nan))
    return dict(available=True, batch=batch, small_batch=small, batch_count=n // batch,
                unused=n % batch, plain_lrv=plain, lugsail_lrv=extrapolated,
                variance=variance, mcse=mcse, valid=valid)


def reference_decomposition(estimate, mcse, reference):
    sd = np.asarray(reference['sd'], dtype=float)
    truth = np.asarray(reference['mean'], dtype=float)
    uncertainty = np.asarray(reference['screen']['uncertainty'], dtype=float)
    require(sd.shape == truth.shape == uncertainty.shape == np.shape(estimate) == np.shape(mcse),
            'reference dimensions differ')
    require(np.isfinite(sd).all() and np.all(sd > 0) and
            np.isfinite(truth).all() and np.isfinite(uncertainty).all() and
            np.all(uncertainty >= 0), 'invalid reference')
    error = np.abs(estimate - truth)
    terms = dict(error=error / sd, uncertainty_margin=3 * mcse / sd,
                 reference_uncertainty=uncertainty / sd)
    terms['screen'] = terms['error'] + terms['uncertainty_margin'] + terms['reference_uncertainty']
    terms['passed'] = bool(np.all(terms['screen'] <= .1))
    return terms


def native(value):
    if isinstance(value, dict):
        return {k: native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, np.generic):
        return native(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def audit(args):
    start = time.monotonic()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '-1', 'explicit CPU-only diagnostic required')
    require(os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH') == 'true', 'pre-import memory policy required')
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = args.campaign / 'posterior-01/evidence'
    prior_manifest = read(evidence / 'manifest.json')
    cohort = read(evidence / 'posterior/result.json')
    require(cohort['status'] == 'completed' and len(cohort['rows']) == 7, 'complete seven-member cohort required')
    require([r['candidate_id'] for r in cohort['rows']] == cohort['verified_candidate_ids'], 'candidate identity mismatch')
    local = ROOT / 'bayesfilter/inference/hmc_precision.py'
    original = [(p, h) for p, h in prior_manifest['source_sha256'].items() if p.endswith('/hmc_precision.py')]
    require(len(original) == 1 and digest(local) == original[0][1], 'estimator differs from frozen C3 source')
    reference = read(args.reference)
    manifest = dict(plan=PLAN, command=[sys.executable, *sys.argv],
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        started_utc=datetime.now(timezone.utc).isoformat(), seeds=None,
        seed_reason='no new draws; all seven preserved C3 sequences',
        environment={k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'TF_FORCE_GPU_ALLOW_GROWTH',
            'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'OPENBLAS_NUM_THREADS')},
        cpu_exception='independent saved-data reduction audit; GPU intentionally hidden',
        input_sha256={str(evidence / 'manifest.json'): digest(evidence / 'manifest.json'),
            str(args.reference): digest(args.reference), str(local): digest(local),
            str(Path(__file__).resolve()): digest(__file__)}, wall_seconds=args.wall_seconds)
    write(args.output / 'manifest.json', manifest)
    report = dict(status='running', rows=[], scientific_ranking=None, reliability_established=False,
                  historical_outcomes_changed=False, release_qualified=False)

    def budget():
        require(time.monotonic() - start < args.wall_seconds, 'audit budget exhausted; reprice continuation')

    try:
        for path, expected in prior_manifest['source_sha256'].items():
            budget()
            require(digest(path) == expected, 'original source/input changed: ' + path)
        require(str(args.reference) in prior_manifest['source_sha256'], 'reference is not original C3 input')
        inputs_path = next(Path(p) for p in prior_manifest['source_sha256']
                           if p.endswith('/public-qualification-20261029-01/inputs.json'))
        qualification = inputs_path.parent / 'qualification-01/evidence'
        geometry = read(args.geometry)
        original_execution = read(qualification / 'manifest.json')
        require(payload_digest(geometry['execution']) == geometry['binding_hash']
                == original_execution['binding_hash'], 'frozen execution identity differs')
        artifact = geometry['execution']['layers'][0]['artifact']
        inputs = read(inputs_path)
        require(artifact['position'] == inputs['initial_position'] and
                artifact['covariance'] == inputs['initial_covariance'],
                'geometry differs from original hash-bound inputs')
        close(np.asarray(artifact['factor']) @ np.asarray(artifact['factor']).T,
              artifact['covariance'], 'issued factor reconstruction')
        members = read(qualification / 'members.json')
        for p in (args.geometry, qualification / 'manifest.json', qualification / 'members.json'):
            manifest['input_sha256'][str(p)] = digest(p)
        manifest['binding_hash'] = geometry['binding_hash']
        manifest['checkpoint_coordinates'] = 'active whitened coordinates; issued affine map replayed'
        import tensorflow as tf
        sys.path.insert(0, str(ROOT))
        from bayesfilter.inference.hmc_precision import mean_precision
        from bayesfilter.testing.inference_validation.references.modern_macrofinance import quantities_standardized
        manifest.update(tensorflow=tf.__version__, jit_compile=True, source_hashes_verified=len(prior_manifest['source_sha256']))
        for index, row in enumerate(cohort['rows']):
            budget()
            folder = evidence / 'posterior' / f'member-{index:03d}'
            path = folder / 'retained_raw.tensor'
            result_path = folder / 'result.json'
            manifest['input_sha256'].update({str(path): digest(path), str(result_path): digest(result_path)})
            result = read(result_path)
            member_path = Path(members[row['candidate_id']])
            member = read(member_path)
            member_hash = member.pop('content_hash')
            require(payload_digest(member) == member_hash and
                    member['candidate_id'] == row['candidate_id'], 'member identity differs')
            identity_path = folder / 'chunks/identity.json'
            require(read(identity_path)['member_hash'] == member['member_hash'],
                    'checkpoint member differs')
            for p in (member_path, identity_path):
                manifest['input_sha256'][str(p)] = digest(p)
            tensor = tf.io.parse_tensor(tf.io.read_file(str(path)), tf.float64)
            x = tensor.numpy()
            require(x.shape == (result['retained_results_per_chain'], 4, 31) and np.isfinite(x).all(), 'wrong retained tensor')
            checkpoints = sorted((folder / 'chunks/committed').glob('chunk-*'))
            warmup_chunks = result['warmup_check_count']
            require(len(checkpoints) == warmup_chunks + result['retained_check_count'], 'checkpoint count differs')
            positions = []
            for checkpoint in checkpoints[warmup_chunks:]:
                bundle_path = checkpoint / 'bundle.json'
                require(digest(bundle_path) == (checkpoint / 'bundle.sha256').read_text().strip(), 'bundle checksum mismatch')
                payload = read(bundle_path)['tree']['sequence'][0]['tensor']
                raw = checkpoint / payload['file']
                require(digest(raw) == payload['sha256'], 'committed tensor checksum mismatch')
                positions.append(tf.io.parse_tensor(tf.io.read_file(str(raw)), tf.float64).numpy())
            reconstructed = positions_from_active(np.concatenate(positions), geometry['execution'])
            close(x[:, :, :9], reconstructed, 'retained positions from committed active chunks')
            flat = x[:, :, :9].reshape(-1, 9)
            independent = np.concatenate([quantities_standardized(flat[j:j+1024] / .35)
                                           for j in range(0, len(flat), 1024)]).reshape(x.shape)
            close(independent, x, 'independent quantity transform')
            final = result['retained_checks'][-1]['modern_rhat']['precision']
            targets = final['targets']
            require([t['name'] for t in targets] == reference['quantities'], 'quantity ordering differs')
            n, m, _ = x.shape
            mean, sd = x.mean(axis=(0, 1)), x.reshape(-1, 31).std(axis=0, ddof=1)
            batch = math.isqrt(n)
            baseline = batch_report(x, batch)
            close(mean, [t['estimate'] for t in targets], 'stored mean')
            close(baseline['mcse'], [t['mcse'] for t in targets], 'stored MCSE')
            close(baseline['mcse'] / sd, [t['mcse_sd_ratio'] for t in targets], 'stored MCSE/SD')
            close(baseline['lugsail_lrv'], final['batch_metadata']['per_chain_long_run_variance'], 'stored per-chain LRV')
            public = mean_precision(tensor, method='lugsail', jit_compile=True)
            close(baseline['mcse'], public['mcse'].numpy(), 'merged TensorFlow/XLA MCSE')
            close(sd, public['posterior_sd'].numpy(), 'merged posterior SD')
            require([bool(v) for v in (baseline['mcse'] / sd <= .02)] == [t['passed'] for t in targets], 'precision decision differs')
            decomposition = reference_decomposition(mean, baseline['mcse'], reference)
            close(decomposition['screen'], row['reference']['normalized_error_screen'], 'reference screen')
            require(decomposition['passed'] == row['reference']['passed'], 'reference decision differs')
            means = x.mean(axis=0)
            contributions = (means - means.mean(axis=0)) ** 2
            total = contributions.sum(axis=0)
            shares = np.divide(contributions, total, out=np.full_like(contributions, np.nan), where=total > 0)
            cfg = result.get('config', result.get('configuration'))
            report['rows'].append(native(dict(candidate_id=row['candidate_id'], L=cfg['num_leapfrog_steps'],
                epsilon=cfg['step_size'], draws=n, original_delivered=row['delivered'],
                quantity_parity_max=float(np.max(np.abs(independent - x))),
                affine_parity_max=float(np.max(np.abs(reconstructed - x[:, :, :9]))),
                checks=dict(serialization=True, quantities=True, original_arithmetic=True, public_xla=True, decisions=True),
                baseline=baseline, batch_sensitivity=[batch_report(x, b) for b in sorted({batch // 2, batch, 2 * batch})],
                quantities=reference['quantities'], mean=mean, posterior_sd=sd,
                marginal_sd_reference_ratio=sd / np.asarray(reference['sd']), reference=decomposition,
                error_over_mcse=(mean - reference['mean']) / baseline['mcse'],
                chain_means=means, half_means=[x[:n//2].mean(axis=(0, 1)), x[n//2:].mean(axis=(0, 1))],
                between_chain_mean_mcse=np.sqrt(total / (m * (m - 1))),
                largest_between_chain_variance_share=np.max(shares, axis=0),
                interpretation='all alternative scales descriptive only; no outcome replacement')) )
            write(args.output / 'result.json', report)
            write(args.output / 'progress.json', dict(completed_members=index+1, elapsed_seconds=time.monotonic()-start))
        for path, expected in manifest['input_sha256'].items():
            require(digest(path) == expected, 'audit input changed: ' + path)
        report.update(status='completed', arithmetic_passed=True, assessed_members=7)
    except Exception as exc:
        report.update(status='error', error=repr(exc))
    report.update(elapsed_seconds=time.monotonic()-start, finished_utc=datetime.now(timezone.utc).isoformat())
    write(args.output / 'manifest.json', manifest)
    write(args.output / 'result.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}))
    return 0 if report['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign', 'reference', 'output', 'geometry'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--wall-seconds', type=float, default=900.)
    args = parser.parse_args()
    require(math.isfinite(args.wall_seconds) and 0 < args.wall_seconds <= 900, 'bounded diagnostic allowance required')
    for name in ('campaign', 'reference', 'output', 'geometry'):
        setattr(args, name, getattr(args, name).resolve())
    return audit(args)


if __name__ == '__main__':
    raise SystemExit(main())
