"""Diagnostic only: inspect preserved source and TF clouds and fitted policies.

No filtering or fitting runs here. NumPy performs independent post-run analysis;
TensorFlow is used solely to decode its saved tensor format on the CPU.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--case', type=int, default=3)
    args = parser.parse_args()
    started = time.monotonic()
    root = Path(__file__).resolve().parents[2]
    campaign = root / 'docs/plans/artifacts/controlled-smc-kdm-replication-20260928'
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import tensorflow as tf

    source = campaign / 'attempt147_p0_forward_capture'
    forward = campaign / 'attempt148_p0_forward_compare'
    conditional = campaign / 'attempt149_p0_fitting_forward'
    autonomous = campaign / 'attempt154_p0_perturb_tf'
    fixture = json.loads((source / 'fixture.json').read_text())
    manifests = {p: json.loads((p / 'manifest.json').read_text())
                 for p in (source, forward, conditional, autonomous)}
    input_hashes = {}
    prefix = f'case{args.case}_'

    def verify(path, expected):
        actual = sha(path)
        assert actual == expected, str(path)
        input_hashes[str(path.relative_to(root))] = actual

    verify(source / 'fixture.json', manifests[source]['output_sha256']['fixture.json'])

    def raw(stage, name):
        item = fixture['arrays'][f'{prefix}stage{stage}_{name}']
        path = source / item['file']
        verify(path, item['sha256'])
        value = np.fromfile(path, '<f8').reshape(item['shape'])
        assert np.isfinite(value).all()
        return value

    def saved(directory, name, shape, dtype=tf.float64):
        path = directory / (name + '.tensor')
        verify(path, manifests[directory]['output_sha256'][path.name])
        with tf.device('/CPU:0'):
            value = tf.io.parse_tensor(tf.io.read_file(str(path)), dtype).numpy()
        assert np.isfinite(value).all()
        return value.reshape(shape)

    def difference(a, b):
        delta = a - b
        axes = tuple(range(1, delta.ndim))
        maximum = np.max(np.abs(delta), axis=axes) if axes else np.abs(delta)
        rms = np.sqrt(np.mean(delta**2, axis=axes)) if axes else np.abs(delta)
        return dict(max_absolute=float(np.max(maximum)),
                    max_time=int(np.argmax(maximum)),
                    rms=float(np.sqrt(np.mean(delta**2))),
                    per_time_max=maximum.tolist(), per_time_rms=rms.tolist())

    x = raw(0, 'x')
    lw = raw(0, 'lw')
    base = [raw(0, k) for k in ('A', 'b', 'c')]
    policy = [raw(1, k) for k in ('A', 'b', 'c')]
    source_ancestors = raw(0, 'ancestors')
    tf_x = saved(forward, prefix + 'stage0_x', x.shape)
    tf_lw = saved(forward, prefix + 'stage0_log_incremental_weight', lw.shape)
    tf_ancestors = saved(forward, prefix + 'stage0_ancestors', source_ancestors.shape, tf.int32)
    controls = dict(cloud=difference(tf_x, x), weights=difference(tf_lw, lw),
                    ancestry_equal=bool(np.array_equal(tf_ancestors, source_ancestors)),
                    ancestry_mismatches=int(np.count_nonzero(tf_ancestors != source_ancestors)))

    def polynomial(p):
        a, b, c = p
        return np.einsum('tni,tij,tnj->tn', x, a, x) + np.einsum('tni,ti->tn', x, b) + c[:, None]

    source_increment = [p-b for p, b in zip(policy, base)]
    source_polynomial = polynomial(source_increment)
    pairs = [(i, j) for i in range(8) for j in range(i+1, 8)]
    design = np.concatenate((np.ones((*x.shape[:2], 1)), x,
                            np.stack([x[:, :, i]*x[:, :, j] for i, j in pairs], -1), x*x), -1)
    singular_values = np.linalg.svd(design, compute_uv=False)
    source_terminal_residual = source_polynomial[-1] + lw[-1]
    rows = []
    for solver in ('source_normal_equations', 'centered_qr'):
        fitted = {}
        for kind, directory, stem in (
            ('conditional', conditional, prefix + 'stage1_' + solver + '_fit_'),
            ('autonomous', autonomous, prefix + solver + '_zero_')):
            values = [saved(directory, stem+k, ref.shape) for k, ref in zip(('A', 'b', 'c'), policy)]
            fitted[kind] = values
            increment = [v-b for v, b in zip(values, base)]
            prediction = polynomial(increment)
            residual = prediction[-1] + lw[-1]
            row = dict(solver=solver, fitting_input=kind,
                       coefficient_differences={k: difference(v, ref) for k, v, ref in zip(('A', 'b', 'c'), values, policy)},
                       polynomial_difference=difference(prediction, source_polynomial),
                       terminal_residual_rms=float(np.sqrt(np.mean(residual**2))))
            # Intercepts can be a harmless gauge shift; report centered effects too.
            delta = prediction - source_polynomial
            row['polynomial_difference_without_constant'] = difference(delta-delta.mean(-1, keepdims=True), np.zeros_like(delta))
            rows.append(row)
        rows.append(dict(solver=solver, fitting_input='autonomous_minus_conditional',
                         coefficient_differences={k: difference(a, c) for k, a, c in zip(('A', 'b', 'c'), fitted['autonomous'], fitted['conditional'])}))
    case = next(c for c in fixture['cases'] if c['case'] == args.case)
    summary = dict(status='PASS', scope='SAVED_POLICY_LOCALIZATION_ONLY', replication='INCOMPLETE',
                   case=case, stage0_control=controls, comparisons=rows,
                   design_condition=(singular_values[:, 0]/singular_values[:, -1]).tolist(),
                   source_terminal_residual_rms=float(np.sqrt(np.mean(source_terminal_residual**2))),
                   input_sha256=input_hashes, cpu_only=True, gpu_intentionally_hidden=True,
                   numpy_role='independent post-run diagnostics only')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    manifest = dict(status='PASS', command=[sys.executable, *sys.argv], cwd=str(root),
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                    plan='docs/plans/controlled-smc-offset-localization-20261006.md',
                    result_file='docs/plans/controlled-smc-offset-localization-result-20261006.md',
                    cpu_only=True, gpu_intentionally_hidden=True, jit_compile=False,
                    role='saved-data diagnostic, no numerical runtime',
                    seeds=[case['seed']], tensorflow=tf.__version__, numpy=np.__version__,
                    source_sha256={str(Path(__file__).relative_to(root)): sha(Path(__file__))},
                    wall_seconds=time.monotonic()-started, ended_utc=datetime.now(timezone.utc).isoformat(),
                    output_sha256={'summary.json': sha(args.output/'summary.json')})
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(status='PASS', output=str(args.output), wall_seconds=manifest['wall_seconds'],
                          stage0_cloud_max=controls['cloud']['max_absolute'],
                          stage0_weights_max=controls['weights']['max_absolute'],
                          stage0_ancestry_equal=controls['ancestry_equal'],
                          terminal_source_residual_rms=summary['source_terminal_residual_rms'],
                          comparisons=[dict(solver=r['solver'], kind=r['fitting_input'],
                                            max_A=r['coefficient_differences']['A']['max_absolute'],
                                            max_c=r['coefficient_differences']['c']['max_absolute'],
                                            terminal_residual_rms=r.get('terminal_residual_rms')) for r in rows])))


if __name__ == '__main__':
    main()
