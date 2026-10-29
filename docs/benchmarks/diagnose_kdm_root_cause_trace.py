"""Bounded diagnostic/reference tracing; never an admitted training backend."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path('/home/chakwong/BayesFilter')
BASE = ROOT / 'docs/plans/artifacts/controlled-smc-kdm-replication-20260928'
PRIOR = BASE / 'coupled-trajectory-20261012/attempt03/diagnostic'
CAL = BASE / 'paired-calibration-20261011/calibration-01'
SOURCE = BASE / 'kdm-source-variability-20261008-01'
PLAN = ROOT / 'docs/plans/controlled-smc-kdm-root-cause-trace-20261012.md'
AUTHOR_PY = '/home/chakwong/anaconda3/envs/tf-gpu2/bin/python'
PREFIXES = ('weight_model.', 'observation_encoder.', 'particle_encoder_for_weights_model.')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_noise(out):
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP'] = '0'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
    sys.path.insert(0, str(ROOT))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.set_soft_device_placement(False)

    @tf.function(input_signature=[tf.TensorSpec([2], tf.int64),
                 tf.TensorSpec([], tf.int32)], jit_compile=True, autograph=False)
    def key(root, t):
        fold = tf.random.experimental.stateless_fold_in
        return fold(fold(root, 1), 1000 + t)

    @tf.function(input_signature=[tf.TensorSpec([2], tf.int32)],
                 jit_compile=True, autograph=False)
    def draw(seed):
        return tf.random.stateless_normal([64, 25, 4], seed, dtype=tf.float32, alg='philox')

    with tf.device('/GPU:0'):
        k = key(tf.constant([202610122, 0], tf.int64), tf.constant(1, tf.int32))
        noise = draw(k)
    flat = tf.reshape(noise, [-1])
    assert int(tf.argmin(tf.abs(flat)).numpy()) == 607
    assert float(flat[607].numpy()) == 0.00026871569571085274
    write(out / 'noise.json', dict(noise=noise.numpy().tolist(), key=k.numpy().tolist(),
          root_seed=[202610122, 0], time_index=1, shape=[64, 25, 4], coordinate=607))
    write(out / 'environment.json', dict(tensorflow=tf.__version__, memory_growth=memory,
          device=noise.device, jit_compile=True, tf32=False,
          key_trace_count=key.experimental_get_tracing_count(),
          draw_trace_count=draw.experimental_get_tracing_count(),
          allocator=tf.config.experimental.get_memory_info('GPU:0')))


def source_trace(out):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['OMP_NUM_THREADS'] = '1'
    spec = importlib.util.spec_from_file_location('author_reference', ROOT / 'docs/benchmarks/reference_kdm_neural_exact_replay.py')
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    torch = ref.torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for name in ('src/models/internal_models/proposal_models.py',
                 'src/models/internal_models/particle_encoder_models.py',
                 'src/models/particle_transformer.py',
                 'src/models/kde_particle_filter/kde_particle_filter.py'):
        ref.verify(ref.AUTHOR / name)
    inputs = {}

    def tracked(path):
        inputs[str(path)] = sha(path)
        return read(path)

    def checkpoint(path):
        inputs[str(path)] = sha(path)
        return torch.load(path, map_location='cpu', weights_only=True)

    def tensor(value):
        return torch.tensor(value, dtype=torch.float32)

    def distance(a, b):
        assert a.shape == b.shape
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        d = (a.double() - b.double()).abs()
        return dict(changed=int((a != b).sum()), count=a.numel(), max_abs=float(d.max()),
                    rms=float(torch.sqrt(torch.mean(d.square()))))

    starts = []
    for unit in ('unit01', 'unit02'):
        src = SOURCE / (unit + '-full') / 'diagnostic'
        port = CAL / unit / 'training/diagnostic'
        constructor = tracked(src / 'mdpf-constructor.json')['parameters']
        source_state = checkpoint(src / 'mdpf/models/full_dpf_model_untrained.pt')
        source_best = {}
        for prefix in PREFIXES:
            module = checkpoint(src / 'weights/models' / (prefix[:-1] + '_best.pt'))
            source_best.update({prefix + name: value for name, value in module.items()})
        port_state = tracked(port / 'mdpf-starting-state.json')['parameters']
        port_best = tracked(port / 'weights-best.json')['model']['parameters']
        weights_constructor = tracked(src / 'weights-constructor.json')['parameters']
        port_weights_start = tracked(port / 'weights-starting-state.json')['parameters']
        assert set(port_state) == set(constructor)
        assert set(port_weights_start) == set(weights_constructor)
        assert all(torch.equal(tensor(v), tensor(port_weights_start[n])) for n, v in weights_constructor.items())
        checks = dict(common_pretraining_constructor=True, source_transfer=True, tf_transfer=True,
                      source_nontransferred_constructor=True, tf_nontransferred_constructor=True)
        modules = {}
        for group in PREFIXES + ('nontransferred',):
            names = [n for n in constructor if (n.startswith(group) if group != 'nontransferred' else not n.startswith(PREFIXES))]
            a = torch.cat([source_state[n].reshape(-1) for n in names])
            b = torch.cat([tensor(port_state[n]).reshape(-1) for n in names])
            modules[group] = distance(a, b)
            for n in names:
                if group != 'nontransferred':
                    checks['source_transfer'] &= torch.equal(source_state[n], source_best[n])
                    checks['tf_transfer'] &= torch.equal(tensor(port_state[n]), tensor(port_best[n]))
                else:
                    checks['source_nontransferred_constructor'] &= torch.equal(source_state[n], tensor(constructor[n]))
                    checks['tf_nontransferred_constructor'] &= torch.equal(tensor(port_state[n]), tensor(constructor[n]))
        assert all(checks.values()), checks
        starts.append(dict(unit=unit, checks=checks, modules=modules))

    cfg = next(x for x in tracked(SOURCE / 'unit01-full/diagnostic/configurations.json')['executed_stages'] if x['name'] == 'mdpf')
    noise = tensor(tracked(out / 'noise.json')['noise'])
    traces = []
    for role in ('starting', 'best'):
        p = CAL / 'unit01/training/diagnostic' / ('mdpf-starting-state.json' if role == 'starting' else 'mdpf-best.json')
        state = tracked(p)
        state = state if role == 'starting' else state['model']
        saved = tracked(PRIOR / (role + '-tensors00.json'))['source32']['captured']
        model = ref.KDEParticleFilter(cfg['configuration']['model_params'], cfg['architecture']).float()
        with torch.no_grad():
            for name, value in model.named_parameters():
                value.copy_(tensor(state['parameters'][name]))
        model.train()
        cloud = tensor(saved['initial']['particles_downscaled'])
        cloud = model.particle_transformer.apply_norm(cloud)
        internal = model.particle_transformer.backward_tranform(cloud)
        encoded = model.particle_encoder_for_particles_model(internal)
        proposal = model.proposal_model

        def run(z):
            observed = {}
            hooks = []
            for name, layer in proposal.proposal_model.named_children():
                key = 'layer_' + name + '_' + type(layer).__name__
                hooks.append(layer.register_forward_hook(lambda m, a, v, key=key: observed.__setitem__(key, v.detach().clone())))
            try:
                with torch.no_grad():
                    raw, _ = proposal(internal, encoded, None, z)
            finally:
                for hook in hooks:
                    hook.remove()
            observed = {'noise': z, 'concatenated_input': torch.cat([encoded, z], dim=-1), **observed}
            final_layer = list(proposal.proposal_model.named_children())[-1][0]
            logits = observed['layer_' + final_layer + '_Linear'].reshape(64, 25, 4)
            observed['sigmoid'] = torch.sigmoid(logits)
            observed['twice_sigmoid'] = observed['sigmoid'] * 2.0
            observed['centered_residual'] = observed['twice_sigmoid'] - 1.0
            observed['scaled_residual'] = observed['centered_residual'] * proposal.residual_scale_factor
            observed['added_to_particle'] = internal + observed['scaled_residual']
            assert torch.equal(raw, observed['added_to_particle'])
            observed['normalized_particle'] = model.particle_transformer.apply_norm(raw)
            observed['output_particle'] = model.particle_transformer.forward_tranform(observed['normalized_particle'])
            return observed

        baseline = run(noise)
        assert torch.equal(baseline['added_to_particle'], tensor(saved['raw'][0])), role
        repeat = run(noise.clone())
        assert all(torch.equal(baseline[k], repeat[k]) for k in baseline)
        cases = []
        for epsilon in (1e-8, 1e-7, 1e-6, 1e-5):
            perturbed = noise.clone()
            perturbed.reshape(-1)[607] += epsilon
            actual = float((perturbed.reshape(-1)[607] - noise.reshape(-1)[607]).double())
            assert actual > 0
            outputs = run(perturbed)
            comparisons = {k: distance(baseline[k], outputs[k]) for k in baseline}
            first_zero = next((k for k, v in comparisons.items() if v['changed'] == 0), None)
            cases.append(dict(requested_epsilon=epsilon, actual_epsilon=actual,
                              first_identical_stage=first_zero, stages=comparisons))
        traces.append(dict(role=role, exact_saved_raw_replay=True, identical_repeat=True,
                           coordinate=607, cases=cases))
    write(out / 'result.json', dict(schema='kdm_root_cause_trace_v1', status='PASS',
          torch=torch.__version__, cpu_only=True, gpu_intentionally_hidden=True,
          source_pin=ref.PIN, source_hashes=ref.HASHES, inputs=inputs,
          starting_checkpoints=starts, layer_traces=traces,
          inference='Diagnostic localization only; final trained-quality cause remains unresolved.'))


def supervise(out):
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    (out / 'driver.py').write_bytes(Path(__file__).read_bytes())
    (out / 'plan.md').write_bytes(PLAN.read_bytes())
    manifest = dict(schema='kdm_root_cause_trace_manifest_v1', plan=str(PLAN),
          result=str(out / 'result.json'), command=[sys.executable, '-I', '-B', __file__, '--output', str(out)],
          git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          git_branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
          file_hashes={str(Path(__file__).resolve()): sha(Path(__file__)), str(PLAN): sha(PLAN)},
          environment=sys.executable, source_environment=AUTHOR_PY, seed=[202610122, 0],
          data_version='Existing coupled attempt03 unit01 update0 initial cloud and checkpoints; hashes in result',
          hardware='GPU/XLA noise regeneration; CPU-only pinned author reference',
          worker_commands=[], timeout_seconds=600, status='RUNNING')
    write(out / 'manifest.json', manifest)
    try:
        with (out / 'nvidia-smi.log').open('w') as log:
            subprocess.run(['nvidia-smi'], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=20)
        for worker, python in (('gpu', sys.executable), ('source', AUTHOR_PY)):
            command = [python, '-I', '-B', str(Path(__file__).resolve()), '--worker', worker, '--output', str(out)]
            manifest['worker_commands'].append(command)
            write(out / 'manifest.json', manifest)
            env = dict(os.environ)
            if worker == 'source':
                env['CUDA_VISIBLE_DEVICES'] = '-1'
                env['LD_LIBRARY_PATH'] = str(Path(AUTHOR_PY).parent.parent / 'lib')
            with (out / (worker + '.log')).open('w') as log:
                remaining = 600 - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError('Diagnostic wall cap exhausted')
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=remaining)
        assert read(out / 'result.json')['status'] == 'PASS'
        manifest['status'] = 'PASS'
    except Exception:
        manifest['status'] = 'FAIL'
        manifest['error'] = traceback.format_exc()
        raise
    finally:
        manifest['wall_seconds'] = time.monotonic() - started
        manifest['conservative_cpu_charge_seconds'] = manifest['wall_seconds']
        manifest['conservative_gpu_charge_seconds'] = manifest['wall_seconds']
        write(out / 'manifest.json', manifest)
        print(json.dumps({k: manifest[k] for k in ('status', 'wall_seconds', 'result')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', choices=('gpu', 'source'))
    args = parser.parse_args()
    output = args.output.resolve()
    if args.worker == 'gpu':
        gpu_noise(output)
    elif args.worker == 'source':
        source_trace(output)
    else:
        supervise(output)
