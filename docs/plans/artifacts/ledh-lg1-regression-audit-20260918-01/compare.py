"""CPU-only post-run inspection of fixed-input parity evidence."""
import json
import math
from pathlib import Path
import sys
import time
started = time.monotonic()
import tensorflow as tf

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
inputs = json.loads((source / 'inputs.json').read_text())
reference = inputs['original_record']
results = {name: json.loads((source / f'{name}.json').read_text())
           for name in ('baseline', 'current', 'consumer-fixed')}
assert len({r['input_sha256'] for r in results.values()}) == 1
summary = dict(classification='fixed_input_revision_parity_diagnostic_only',
               cpu_only_gpu_intentionally_hidden=True, threshold=1e-4,
               input_sha256=results['baseline']['input_sha256'], comparisons={}, errors={})

for key in ('children', 'posterior_weights', 'states_after_reset', 'predicted_covariances'):
    before = tf.io.parse_tensor((source / f'baseline-{key}.tensor').read_bytes(), out_type=tf.float32)
    after = tf.io.parse_tensor((source / f'current-{key}.tensor').read_bytes(), out_type=tf.float32)
    tf.debugging.assert_all_finite(before, key)
    tf.debugging.assert_all_finite(after, key)
    error = tf.abs(after-before)
    summary['comparisons'][key] = dict(max_abs=float(tf.reduce_max(error)),
        max_scaled=float(tf.reduce_max(error/(1+tf.abs(before)))),
        pass_=bool(tf.reduce_all(error/(1+tf.abs(before)) <= 1e-4)))

for name, record in results.items():
    score = record['score'][0] if isinstance(record['score'], list) else record['score']
    baseline_score = results['baseline']['score'][0]
    scalars = dict(log_likelihood=record['log_likelihood'], score=score)
    for key in ('filtering_mean','filtering_covariance','filtering_ess','post_reset_mean','post_reset_covariance'):
        a, b = record[key], results['baseline'][key]
        scalars[key] = dict(max_abs=max(abs(x-y) for x,y in zip(a,b)),
                            max_scaled=max(abs(x-y)/(1+abs(y)) for x,y in zip(a,b)))
    scalars['value_difference'] = record['log_likelihood']-results['baseline']['log_likelihood']
    scalars['score_difference'] = score-baseline_score
    scalars['value_scaled_difference'] = abs(scalars['value_difference'])/(1+abs(results['baseline']['log_likelihood']))
    scalars['score_scaled_difference'] = abs(scalars['score_difference'])/(1+abs(baseline_score))
    summary['comparisons'][name] = scalars
    summary['errors'][name] = dict(
        max_standardized_mean_error=max(abs(a-b)/math.sqrt(v) for a,b,v in zip(record['filtering_mean'],reference['reference_mean'],reference['reference_covariance'])),
        max_relative_covariance_error=max(abs(a/b-1) for a,b in zip(record['filtering_covariance'],reference['reference_covariance'])),
        log_likelihood_error=record['log_likelihood']-reference['exact_log_likelihood'],
        covariance_gate_pass=max(abs(a/b-1) for a,b in zip(record['filtering_covariance'],reference['reference_covariance'])) <= .05)

# Exact scalar Kalman innovation check on the represented parameter and data.
# This is diagnostic/reference arithmetic, not an executable filter candidate.
phi = float(tf.constant(inputs['phi'], tf.float32))
mean, cov = 0., 1/(1-phi*phi)
kalman = dict(innovations=[], means=[], covariances=[], log_likelihood=0.)
for observation in inputs['observations']:
    mean = phi*mean
    cov = phi*phi*cov+1
    variance = cov+.5
    residual = observation[0]-mean
    kalman['innovations'].append(residual/math.sqrt(variance))
    kalman['log_likelihood'] -= .5*(math.log(2*math.pi*variance)+residual*residual/variance)
    gain = cov/variance
    mean += gain*residual
    cov *= 1-gain
    kalman['means'].append(mean)
    kalman['covariances'].append(cov)
kalman['stored_reference_max_mean_difference'] = max(abs(a-b) for a,b in zip(kalman['means'],reference['reference_mean']))
kalman['stored_reference_max_covariance_difference'] = max(abs(a-b) for a,b in zip(kalman['covariances'],reference['reference_covariance']))
kalman['stored_reference_likelihood_difference'] = kalman['log_likelihood']-reference['exact_log_likelihood']
summary['kalman_diagnostic'] = kalman
summary['all_snapshot_comparisons_pass'] = all(summary['comparisons'][key]['pass_'] for key in ('children','posterior_weights','states_after_reset','predicted_covariances'))
summary['all_summary_comparisons_pass'] = all(
    max(record['value_scaled_difference'],record['score_scaled_difference'],
        *(record[k]['max_scaled'] for k in ('filtering_mean','filtering_covariance','filtering_ess','post_reset_mean','post_reset_covariance'))) <= 1e-4
    for key,record in summary['comparisons'].items() if key in results)
summary['seconds'] = time.monotonic()-started
destination.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(snapshot_pass=summary['all_snapshot_comparisons_pass'],
                     summary_pass=summary['all_summary_comparisons_pass'],
                     errors=summary['errors'],seconds=summary['seconds'])))
