"""CPU-only independent diagnostic replay; never imported by runtime paths."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
started = time.monotonic()
import tensorflow as tf
from bayesfilter.highdim import observation_guided_tt_tf as lib

attempt = ROOT/'docs/benchmarks/artifacts/observation_tt_warm_improvement_20260916/attempt-confirmation-02'
data = json.loads((attempt/'d4-s04/data.json').read_text())
guide = json.loads((attempt/'d4-s04/guide.json').read_text())
manifest = json.loads((attempt/'run_manifest.json').read_text())
source = Path(lib.__file__).resolve()
digest = hashlib.sha256(source.read_bytes()).hexdigest()
assert digest == manifest['source_hashes'][str(source)], 'guide source changed'
D = tf.float64
tensor = lambda x: tf.constant(x, D)
model = lib.SVModel(tensor(data['A']), tensor(data['P0']), data['beta'], data['sigma'])
previous = next(r for r in guide[7]['rules'] if r['level'] == guide[7]['selected_level'])
previous_chart = lib.Chart.from_moments(tensor(previous['mean']), tensor(previous['covariance']))
pred_cov = model.transition @ previous_chart.factor @ tf.transpose(previous_chart.factor) @ tf.transpose(model.transition) + model.sigma**2*tf.eye(4, dtype=D)
predictive = lib.Chart.from_moments(tf.linalg.matvec(model.transition, previous_chart.mean), pred_cov)
cloud = lib.tf_fixed_sgqf_cloud(4, 2)
x = predictive.forward(cloud.points)
logg = model.observation_log_prob(x, tensor(data['observations'][8]))
signed = cloud.weights*tf.exp(logg-tf.reduce_max(logg))
weights = signed/tf.reduce_sum(signed)
mean, covariance = lib.gaussian_moments(x, weights)
selected = next(r for r in guide[8]['rules'] if r['level'] == guide[8]['selected_level'])
saved_cov = tensor(selected['covariance'])
sym = (saved_cov+tf.transpose(saved_cov))*.5
eig = tf.linalg.eigvalsh(sym)
left = tf.linalg.triangular_solve(predictive.factor, sym)
whitened = tf.transpose(tf.linalg.triangular_solve(predictive.factor, tf.transpose(left)))
relative_eig = tf.linalg.eigvalsh((whitened+tf.transpose(whitened))*.5)
ordered = sorted(range(int(weights.shape[0])), key=lambda i: abs(float(weights[i])), reverse=True)
maximum = ordered[0]
other_sum = tf.reduce_sum(tf.gather(tf.abs(weights), ordered[1:]))
to_list = lambda t: t.numpy().tolist()
result = dict(
    status='PASS_DEBUG_DIAGNOSIS', cpu_only=True, gpu_intentionally_hidden=True,
    jit_compile=False, exception='bounded eager CPU reconstruction of saved failure',
    source_sha256=digest, source_matches_execution=True,
    git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
    source_attempt=str(attempt.relative_to(ROOT)), step=8, dimension=4,
    observation_over_beta=to_list(tensor(data['observations'][8])/data['beta']),
    predictive_covariance_eigenvalues=to_list(tf.linalg.eigvalsh(pred_cov)),
    saved_covariance_eigenvalues=to_list(eig),
    saved_covariance_spd_guard_bound=float(64*lib.EPS*4*tf.reduce_max(tf.abs(eig))),
    saved_guard_passed=bool(float(eig[0])>float(64*lib.EPS*4*tf.reduce_max(tf.abs(eig)))),
    saved_posterior_relative_to_predictive_eigenvalues=to_list(relative_eig),
    normalized_node_weights=to_list(weights),
    dominant_node_index=maximum, dominant_node=to_list(x[maximum]),
    dominant_node_weight=float(weights[maximum]), sum_other_absolute_weights=float(other_sum),
    log_likelihood_gaps=to_list(tf.reduce_max(logg)-logg),
    reconstructed_mean=to_list(mean), reconstructed_covariance=to_list(covariance),
    saved_standardized_fourth=selected['standardized_fourth'],
    selected_level=guide[8]['selected_level'],
    higher_levels=[dict(level=r['level'],status=r['status'],reason=r.get('reason')) for r in guide[8]['rules'][1:]],
    wall_seconds=time.monotonic()-started,
)
out = Path(__file__).with_suffix('.json')
out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({k:result[k] for k in ('status','saved_guard_passed','dominant_node_weight','sum_other_absolute_weights','saved_covariance_spd_guard_bound','saved_posterior_relative_to_predictive_eigenvalues','wall_seconds')},allow_nan=False))
