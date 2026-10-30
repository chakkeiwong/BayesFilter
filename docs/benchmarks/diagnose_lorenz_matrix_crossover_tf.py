"""GPU/XLA matrix crossover diagnostic through the shared regression endpoints.

Host loops enumerate fixed diagnostic arms and write artifacts. No NumPy path.
Reporting reductions and binary I/O are explicit noncompiled diagnostics.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP'] = '0'
    sys.path.insert(0, str(Path.cwd()))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.set_soft_device_placement(False)
    source = Path(__file__).with_name('multivariate_controlled_smc_tf.py')
    spec = importlib.util.spec_from_file_location('lorenz_crossover_core', source)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    fixture_path = args.reference/'fixture.json'
    fixture = json.loads(fixture_path.read_text())
    assert fixture['schema'] == 'lorenz_matrix_crossover_v1' and fixture['stage'] == 'form'
    checks, comparisons = [], []
    index = dict(fixture, stage='tensorflow', reference_root=str(args.reference),
                 input_fixture_sha256=sha(fixture_path), arrays={})

    def flag(name, passed):
        checks.append(dict(name=name, passed=bool(passed)))
        if not passed:
            raise RuntimeError(name)

    def load(name):
        item = fixture['arrays'][name]
        path = args.reference/item['file']
        flag('hash_'+name, sha(path) == item['sha256'])
        with tf.device('/CPU:0'):
            value = tf.reshape(tf.io.decode_raw(path.read_bytes(), tf.float64, little_endian=True), item['shape'])
        with tf.device('/GPU:0'):
            return tf.identity(value)

    def save(name, value):
        with tf.device('/CPU:0'):
            values = tf.reshape(value, [-1]).numpy().tolist()
        path = args.output/(name+'.bin')
        with path.open('wb') as stream:
            for begin in range(0, len(values), 65536):
                block = values[begin:begin+65536]
                stream.write(struct.pack('<%dd' % len(block), *block))
        index['arrays'][name] = dict(file=path.name, shape=value.shape.as_list(), sha256=sha(path))

    def compare(name, actual, expected, exact=False):
        with tf.device('/GPU:0'):
            delta = tf.abs(actual-expected)
            finite = bool(tf.reduce_all(tf.math.is_finite(actual)).numpy())
            passed = bool(tf.reduce_all(delta == 0 if exact else delta <= 1e-7+1e-9*tf.abs(expected)).numpy())
            maximum = float(tf.reduce_max(delta).numpy())
        comparisons.append(dict(name=name, passed=finite and passed, finite=finite,
                                maximum_absolute=maximum if finite else None))

    # Copy byte-identical native artifacts for the final CPU cross-solve.
    for name, item in fixture['arrays'].items():
        path = args.reference/item['file']
        flag('copy_hash_'+name, sha(path) == item['sha256'])
        (args.output/item['file']).write_bytes(path.read_bytes())
        index['arrays'][name] = dict(item)
    with tf.device('/GPU:0'):
        x, target = load('x'), load('target')
        original_gram, original_rhs = load('native_gram'), load('native_rhs')
        batch, particles, dimension = x.shape.as_list()
        flag('frozen_scope', (batch, particles, dimension) == (32, 512, 8))
        identity = [[float(i == j) for j in range(dimension)] for i in range(dimension)]

        def engine(solver):
            return core.MultivariateGaussianControlledSMC(
                observations=1, particles=particles, replicas=batch, dimension=dimension,
                observation_dimension=dimension, initial_mean=[0.]*dimension,
                initial_covariance=identity,
                transition_covariance=[[.01*v for v in row] for row in identity],
                transition_mean=lambda value: value,
                observation_log_density=lambda state, observation: tf.zeros(tf.shape(state)[:-1], tf.float64),
                solver=solver, jit_compile=True)

        raw, qr = engine('source_normal_equations'), engine('centered_qr')
        width = raw.width

        @tf.function(input_signature=[tf.TensorSpec([batch,particles,dimension],tf.float64),
                                      tf.TensorSpec([batch,particles],tf.float64)],
                     jit_compile=True, autograph=False)
        def form(value, response):
            design = raw._design(value)
            return (design, tf.linalg.matmul(design,design,transpose_a=True),
                    tf.squeeze(tf.linalg.matmul(design,response[...,None],transpose_a=True),-1))

        @tf.function(input_signature=[tf.TensorSpec([batch,width,width],tf.float64),
                                      tf.TensorSpec([batch,width],tf.float64)],
                     jit_compile=True, autograph=False)
        def solve(gram, rhs):
            return tf.squeeze(tf.linalg.solve(gram,rhs[...,None]),-1)

        def beta(result):
            cross = tf.stack([2*result['A'][:,i,j] for i,j in raw.pairs],axis=-1)
            return tf.concat([result['c'][:,None],result['b'],cross,
                              tf.linalg.diag_part(result['A'])],axis=-1)

        design, gram, rhs = form(x,target)
        raw_result, qr_result = raw.regress(x,target), qr.regress(x,target)
        raw_beta, qr_beta = beta(raw_result), beta(qr_result)
        cross_beta = tf.stack([solve(g,r) for g,r in (
            (original_gram,original_rhs),(original_gram,rhs),
            (gram,original_rhs),(gram,rhs))])
        compare('full_design_exact', design, load('native_design'), exact=True)
        compare('extracted_tf_solve_vs_actual_endpoint', cross_beta[3], raw_beta)
        compare('raw_endpoint_vs_native', raw_beta, load('native_beta'))
        compare('qr_vs_independent_centered_qr', qr_beta, load('centered_qr_beta'))
        finite_status = {name:bool(tf.reduce_all(tf.math.is_finite(value)).numpy()) for name,value in
                        [('gram',gram),('rhs',rhs),('raw_endpoint',raw_beta),
                         ('qr_endpoint',qr_beta),('crossed_solves',cross_beta)]}
        for name,value in [('tf_design',design),('tf_gram',gram),('tf_rhs',rhs),
                           ('tf_raw_beta',raw_beta),('tf_qr_beta',qr_beta),('tf_cross_beta',cross_beta)]:
            save(name,value)
        for prefix,result in [('raw',raw_result),('qr',qr_result)]:
            for name,value in result.items():
                if value.dtype == tf.float64:
                    save('diagnostic_'+prefix+'_'+name,value)
        device = dict(tensor=x.device, details=tf.config.experimental.get_device_details(
                      tf.config.list_physical_devices('GPU')[0]))
    traces = {name:fn.experimental_get_tracing_count() for name,fn in
              [('form',form),('solve',solve),('raw_regress',raw.regress),('qr_regress',qr.regress)]}
    flag('single_trace_all_endpoints', all(value == 1 for value in traces.values()))
    flag('actual_gpu_tensors', 'GPU:0' in x.device and 'GPU:0' in cross_beta.device)
    (args.output/'fixture.json').write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')
    summary = dict(status='PASS',scope='LORENZ_MATRIX_CROSSOVER',replication='INCOMPLETE',
        scientific_verdict='TF_CROSSED_SOLVES_COMPLETE_PENDING_NATIVE_CROSSCHECK',
        tensorflow=tf.__version__,jit_compile=True,tf32=False,dtype='float64',
        gpu_memory_policy=memory,device=device,trace_counts=traces,checks=checks,
        comparisons=comparisons,finite_status=finite_status,
        raw_valid=bool(raw_result['valid'].numpy()),qr_valid=bool(qr_result['valid'].numpy()),
        gpu_allocator=tf.config.experimental.get_memory_info('GPU:0'),
        implementation_sha256=sha(source),input_fixture_sha256=sha(fixture_path),
        endpoints=['MultivariateGaussianControlledSMC.regress -> _regress',
                   'MultivariateGaussianControlledSMC._design -> crossed matrix diagnostic'],
        constructor_note='Only regress is consumed; model callbacks/covariances are dormant shape scaffolding.',
        algorithm_seconds=time.monotonic()-started)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='PASS',traces=traces,comparisons=comparisons)))


if __name__ == '__main__':
    main()
