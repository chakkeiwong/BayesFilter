"""CPU-only post-run diagnostics; no tuning, admission, or sampling authority."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT))
START = time.monotonic()
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bayesfilter.testing import surrogate_hmc_m13_cumulative as cumulative

SOURCE = ROOT / 'docs/plans/artifacts/surrogate-hmc-m13-reclassified-20261002-02/report.json'
PLAN = ROOT / 'docs/plans/surrogate-hmc-m13-saved-diagnostics-2026-10-02.md'
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def serial(x):
    if isinstance(x, dict):
        return {k: serial(v) for k, v in x.items()}
    return x.numpy().tolist() if tf.is_tensor(x) else x

report = read(SOURCE)
sample_path = Path(report['sample_archive']['path'])
assert sha(sample_path) == report['sample_archive']['sha256']
source_master = Path(report['input_report'])
assert sha(source_master) == report['input_report_sha256']
assert sha(report['tuning_manifest']) == report['tuning_manifest_sha256']
history = cumulative.load_history(report['tuning_manifest'],
    binding_hash=report['binding_hash'],
    resume_report=source_master.parent / 'worker/posterior/report.json')
samples = tf.io.parse_tensor(tf.io.read_file(str(sample_path)), tf.float64)
accepts = tf.concat(history.accepts, 0)
tf.debugging.assert_equal(samples, tf.concat(history.samples, 0))
tf.debugging.assert_equal(tf.shape(samples), [12750, 4, 2])
tf.debugging.assert_all_finite(samples, 'nonfinite sample archive')

@tf.function(input_signature=[tf.TensorSpec([None, 4, 2], tf.float64),
                             tf.TensorSpec([None, 4], tf.bool)],
             autograph=False, jit_compile=False)
def summarize_segment(x, accepted):
    return dict(mean=tf.reduce_mean(x, 0), sd=tf.math.reduce_std(x, 0),
                minimum=tf.reduce_min(x, 0), maximum=tf.reduce_max(x, 0),
                acceptance=tf.reduce_mean(tf.cast(accepted, tf.float64), 0))

@tf.function(input_signature=[tf.TensorSpec([None, 4, 2], tf.float64)],
             autograph=False, jit_compile=False)
def autocorrelation(x):
    return tfp.stats.auto_correlation(x, axis=0, max_lags=1000,
                                     center=True, normalize=True)

metrics = {}
acfs = {}
for label, n in [('first_9000', 9000), ('all_12750', 12750)]:
    expected = report['historical_prefix' if n == 9000 else 'diagnostics']
    actual = cumulative.diagnostics(samples[:n], accepts[:n])
    for category, fields in [('rhat', ['bulk', 'folded', 'maximum']),
                             ('ess', ['bulk', 'tail']),
                             ('means', ['pooled_mean', 'mean_mcse', 'posterior_sd'])]:
        for field in fields:
            tf.debugging.assert_near(tf.constant(actual[category][field], tf.float64),
                tf.constant(expected[category][field], tf.float64), rtol=1e-10, atol=1e-10)
    tf.debugging.assert_near(tf.constant(actual['acceptance_per_chain'], tf.float64),
        tf.constant(expected['acceptance_per_chain'], tf.float64), rtol=1e-12, atol=1e-12)
    acfs[label] = autocorrelation(samples[:n])
    actual['acf_lags'] = [1, 10, 100, 500, 1000]
    actual['acf_lag_chain_parameter'] = serial(tf.gather(acfs[label], actual['acf_lags']))
    metrics[label] = actual

segments = {}
for name, a, b in [('first_9000', 0, 9000), ('additional_3750', 9000, 12750),
                   ('all_12750', 0, 12750)]:
    summary = serial(summarize_segment(samples[a:b], accepts[a:b]))
    summary.update(start_inclusive=a, end_exclusive=b, draws_per_chain=b-a)
    longest = []
    for row in tf.transpose(accepts[a:b]).numpy().tolist():
        run = maximum = 0
        for accepted in row:
            run = 0 if accepted else run + 1
            maximum = max(maximum, run)
        longest.append(maximum)
    summary['longest_consecutive_rejections_per_chain'] = longest
    segments[name] = summary

blocks = []
for a in range(0, 12750, 1000):
    b = min(a+1000, 12750)
    blocks.append(dict(start_inclusive=a, end_exclusive=b,
                       **serial(summarize_segment(samples[a:b], accepts[a:b]))))

tail_bounds = tfp.stats.percentile(samples, [5., 95.], axis=[0, 1], interpolation='linear')
tail_occupancy = dict(bounds=serial(tail_bounds),
    lower_fraction_per_chain=serial(tf.reduce_mean(tf.cast(samples <= tail_bounds[0], tf.float64), 0)),
    upper_fraction_per_chain=serial(tf.reduce_mean(tf.cast(samples >= tail_bounds[1], tf.float64), 0)))

result = dict(schema='bayesfilter.m13.saved_diagnostic_inspection.v1',
    role='explanatory_post_run_diagnostics_only', new_hmc_transitions=0,
    stopping_rule_changed=False, independent_validation_required=False,
    gpu_intentionally_hidden=True, jit_compile=False,
    backend='TensorFlow/TFP CPU; non-XLA post-run reporting exception',
    target_scope=report['target_scope'], parameter_order=report['parameter_order'],
    metrics=metrics, segments=segments, blocks=blocks, tail_occupancy=tail_occupancy,
    archive_health_passed=True,
    maximum_absolute_proposal_energy_error=report['maximum_absolute_energy_error'],
    declared_energy_error_limit=history.payload['config']['energy_error_max'],
    ebfmi_status='not_available: archives retain delta_H, not total Hamiltonian energy',
    ranking_supported=False,
    stationarity_test_status='descriptive traces and segment summaries; no formal stationarity test',
    interpretation_limits=['Overlapping histories are not independent replicates.',
        'Within-chain centered ACF cannot detect between-chain mean separation.',
        'MCSE and ESS are estimates and may be unreliable under poor exploration.',
        'No diagnostic proves global posterior exploration or exact-likelihood equality.'])

colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
labels = ['log process scale', 'log observation scale']
fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
xs = tf.range(1, 12751, dtype=tf.float64)
running = tf.cumsum(samples, axis=0) / xs[:, None, None]
for p in range(2):
    for c in range(4):
        label = f'Chain {c+1}'
        axes[0, p].plot(xs.numpy(), samples[:, c, p].numpy(), color=colors[c],
                        lw=.45, alpha=.65, label=label)
        axes[1, p].plot(xs[499:].numpy(), running[499:, c, p].numpy(), color=colors[c], lw=1.3)
        axes[2, p].plot(range(1001), acfs['all_12750'][:, c, p].numpy(),
                        color=colors[c], lw=1.3)
    axes[0, p].set_title(labels[p])
    for row in [0, 1]:
        axes[row, p].axvline(9000, color='black', ls='--', lw=1)
        axes[row, p].set_xlabel('Post-warmup draw per chain')
    axes[0, p].set_ylabel('Saved value')
    axes[1, p].set_ylabel('Cumulative mean, separately by chain')
    axes[2, p].set_ylabel('Within-chain autocorrelation')
    axes[2, p].set_xlabel('Lag')
    axes[2, p].axhline(0, color='black', lw=.5)
    for ax in axes[:, p]:
        ax.grid(alpha=.15)
axes[0, 0].legend(ncol=4, fontsize=9)
fig.suptitle('M13: all saved posterior draws; dashed line marks the 9,000-draw stopping point')
fig.savefig(OUT/'trace-running-mean-acf.png', dpi=155)
fig.savefig(OUT/'trace-running-mean-acf.pdf')
plt.close(fig)

(OUT/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
manifest = dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
    command=[sys.executable, str(Path(__file__).resolve())], environment=sys.executable,
    tensorflow=tf.__version__, tensorflow_probability=tfp.__version__,
    cpu_gpu='CPU only; CUDA_VISIBLE_DEVICES=-1 before TensorFlow import',
    source_report=str(SOURCE), source_report_sha256=sha(SOURCE),
    sample_archive=str(sample_path), sample_archive_sha256=sha(sample_path),
    analysis_source_sha256=sha(__file__), diagnostic_module_sha256=sha(ROOT/'bayesfilter/inference/hmc_posterior_diagnostics.py'),
    history_loader_sha256=sha(ROOT/'bayesfilter/testing/surrogate_hmc_m13_cumulative.py'),
    data_seed=report['data_seed'], observation_checksum=report['observation_checksum'],
    original_seed=report['seed'], new_random_draws=0, wall_seconds=time.monotonic()-START,
    plan=str(PLAN), result=str(OUT/'result.md'), report=str(OUT/'report.json'),
    plots=[str(OUT/'trace-running-mean-acf.png'), str(OUT/'trace-running-mean-acf.pdf')])
(OUT/'run-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(dict(status='completed', wall_seconds=manifest['wall_seconds'],
                     report=str(OUT/'report.json'), new_hmc_transitions=0)))
