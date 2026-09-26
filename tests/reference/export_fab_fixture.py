"""CPU-only diagnostic fixture export; never imported by a runtime path."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import tensorflow as tf
from bayesfilter.inference.neutra_transport import NeuTraTransport, NeuTraTransportConfig
from bayesfilter.inference.neutra_fab import fab_weighted_loss

def export(path, dimension, width):
    start = time.monotonic()
    cfg = replace(NeuTraTransportConfig.hoffman_author_iaf(
        dimension, conditional_scale_cap=2., seed=(12, 31)), hidden_layers=(width, width),
        affine_center=(.2, -.3, .1, -.2)[:dimension], affine_scale=(1.1, .9, 1.2, .8)[:dimension])
    flow = NeuTraTransport(cfg)
    # Nonzero biases and appreciable nonlinear conditioners expose inverse and
    # score errors that an identity map would hide. This is not a trained map.
    for i, v in enumerate(flow.trainable_variables):
        v.assign(v * 2. + tf.cast((i % 3 - 1) * .03, v.dtype))
    x = tf.reshape(tf.linspace(tf.constant(-1.7, tf.float64), tf.constant(1.8, tf.float64), 8*dimension), [8, dimension])
    w = tf.constant([-8., -3., -1., -.2, .3, 1.2, 3., 6.], tf.float64)
    with tf.GradientTape() as tape:
        loss = fab_weighted_loss(flow.log_prob(x), w)
    grads = tape.gradient(loss, flow.trainable_variables)
    fixture = {
        'schema': 'fab_author_equivalence_fixture.v1', 'config': cfg.payload(),
        'params': [v.numpy().tolist() for v in flow.trainable_variables],
        'masks': [[m.numpy().tolist() for m in s.masks] for s in flow.stages],
        'x': x.numpy().tolist(), 'log_w': w.numpy().tolist(),
        'baseline_fresh_loss': float(loss),
        'baseline_fresh_gradients': [v.numpy().tolist() for v in grads],
        'baseline_source_sha256': hashlib.sha256((ROOT/'bayesfilter/inference/neutra_fab.py').read_bytes()).hexdigest(),
        'environment': sys.executable, 'gpu_intentionally_hidden': True,
        'wall_seconds': time.monotonic()-start,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as out:
        json.dump(fixture, out, indent=2, allow_nan=False)
        out.write('\n')
    print(json.dumps({'fixture': str(path), 'baseline_fresh_loss': float(loss), 'wall_seconds': fixture['wall_seconds']}))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dimension', type=int, choices=[2, 4], default=2)
    p.add_argument('--width', type=int, default=4)
    a = p.parse_args()
    export(a.output, a.dimension, a.width)
