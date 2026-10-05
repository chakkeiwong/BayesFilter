"""Measure an exact batched acceptance-summary diagnostic on saved traces only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def worker(args):
    sys.path.insert(0, str(args.source)); os.chdir(args.source)
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    from bayesfilter.inference.hmc_candidate_set_execution import _trace_from_payload
    from bayesfilter.inference.hmc_acceptance_protocol import HMCReplicatedAcceptancePolicy
    from bayesfilter.inference.hmc_acceptance_statistics import complete_trial_scores

    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    record = json.loads(args.record.read_text())
    traces = []
    policy = HMCReplicatedAcceptancePolicy(
        trial_num_results=65, discarded_prefix=3, base_repetitions=32,
        max_repetitions=256, max_candidates=100,
        search_family_alpha=.05, verification_family_alpha=.05,
        diagnostic_family_alpha=.05, temporal_tolerance=.1)
    for trial in record["trials"]:
        trace = _trace_from_payload(trial["trace"])
        log_accept = tf.identity(trace["log_accept_ratio"][3:])
        traces.append(log_accept)
    data = tf.stack(traces, axis=0)

    @tf.function(input_signature=[tf.TensorSpec([65, 4], tf.float64)],
                 autograph=False, jit_compile=False)
    def summarize_one(row):
        probability = tf.exp(tf.minimum(row, 0.0))
        means = tf.reduce_mean(probability, axis=0)
        windows = tf.stack([
            tf.reduce_mean(probability[65*j//4:65*(j+1)//4], axis=0)
            for j in range(4)
        ], axis=1)
        return means, windows

    expected_means = tf.stack([summarize_one(row)[0] for row in traces], axis=0)
    expected_windows = tf.stack([summarize_one(row)[1] for row in traces], axis=0)

    @tf.function(input_signature=[tf.TensorSpec([None, 65, 4], tf.float64)],
                 autograph=False, jit_compile=False)
    def summarize(rows):
        count = tf.shape(rows)[0]
        ta = tf.TensorArray(tf.float64, size=count, element_shape=[4])
        windows = tf.TensorArray(tf.float64, size=count, element_shape=[4, 4])

        def body(i, ta, windows):
            probability = tf.exp(tf.minimum(rows[i], 0.0))
            means = tf.reduce_mean(probability, axis=0)
            block = tf.stack([
                tf.reduce_mean(probability[65*j//4:65*(j+1)//4], axis=0)
                for j in range(4)
            ], axis=1)
            return i + 1, ta.write(i, means), windows.write(i, block)

        _, ta, windows = tf.while_loop(lambda i, *_: i < count, body,
                                       (tf.constant(0), ta, windows))
        return ta.stack(), windows.stack()

    result_rows = []
    for batch in (1, 8, 32):
        for phase in ("cold", "warm1", "warm2"):
            # The first call of each batch shape includes its one graph trace.
            before = time.monotonic()
            for start in range(0, 64, batch):
                actual_means, actual_windows = summarize(data[start:start+batch])
                tf.debugging.assert_near(actual_means, expected_means[start:start+batch],
                                         rtol=1e-14, atol=1e-15)
                tf.debugging.assert_near(actual_windows, expected_windows[start:start+batch],
                                         rtol=1e-14, atol=1e-15)
            elapsed = time.monotonic() - before
            result_rows.append(dict(batch_size=batch, phase=phase,
                elapsed_seconds=elapsed, exact_parity=True))
    write(args.output / "result.json", dict(
        status="exact_batched_summary_parity_passed", trials=64,
        rows=result_rows, memory_policy=memory,
        device=data.device, tensorflow_version=tf.__version__, jit_compile=False,
        native_sampling=False, release_ready=False,
        timing_scope="descriptive saved-trace diagnostic; no target calls or admission"))
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--record', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', required=True)
    p.add_argument('--worker', action='store_true')
    args = p.parse_args()
    for name in ('source', 'record', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, TF_FORCE_GPU_ALLOW_GROWTH='true',
        TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='1',
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', TF_CPP_MIN_LOG_LEVEL='2',
        BAYESFILTER_PRELOAD_CUSTOM_OP='0')
    os.sched_setaffinity(0, {24, 25, 26, 27})
    if args.worker:
        return worker(args)
    args.output.mkdir(parents=True, exist_ok=False)
    runner = args.output / 'runner.py'; runner.write_bytes(Path(__file__).read_bytes())
    command = [sys.executable, str(runner), *sys.argv[1:], '--worker']
    started = time.monotonic()
    manifest = dict(command=command, started_utc=datetime.now(timezone.utc).isoformat(),
        resource='gpu', gpu_uuid=args.gpu, native_sampling=False,
        runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
        plan='docs/plans/bayesfilter-hmc-v7-release-plan-2026-10-02.md')
    write(args.output/'manifest.json', manifest)
    with (args.output/'run.log').open('x') as log:
        try:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=120).returncode
        except subprocess.TimeoutExpired:
            code = 124
    write(args.output/'receipt.json', {**manifest, 'exit_code': code,
        'wall_seconds': time.monotonic()-started})
    print(json.dumps({'exit_code': code, 'wall_seconds': time.monotonic()-started}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
