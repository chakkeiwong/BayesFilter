#!/usr/bin/env python
"""Focused CPU/XLA diagnostic for Kalman QR analytical batch scaling."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import tensorflow as tf

from scripts.benchmark_kalman_qr_parameter_count_scaling import (
    _make_parameter_batch,
    _synchronize_outputs,
    build_batch_native_analytic_fn,
    make_fixture,
)


def _timed_call(fn, parameters):
    started = time.perf_counter()
    outputs = fn(parameters)
    _synchronize_outputs(outputs)
    return time.perf_counter() - started, outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intra", type=int, required=True)
    parser.add_argument("--inter", type=int, required=True)
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--parameter-count", type=int, default=50)
    parser.add_argument("--timesteps", type=int, default=120)
    args = parser.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(args.intra)
    tf.config.threading.set_inter_op_parallelism_threads(args.inter)

    result = {
        "affinity": sorted(os.sched_getaffinity(0)),
        "intra": args.intra,
        "inter": args.inter,
        "cells": [],
    }
    for batch_size in (1, 16):
        fixture = make_fixture(
            args.dimension,
            args.parameter_count,
            args.timesteps,
            dtype=tf.float32,
        )
        parameters = _make_parameter_batch(fixture, batch_size)
        function = build_batch_native_analytic_fn(
            fixture, batch_size=batch_size, jit_compile=True
        ).get_concrete_function(parameters)
        _timed_call(function, parameters)
        durations = [_timed_call(function, parameters)[0] for _ in range(2)]
        _, (value, score) = _timed_call(function, parameters)
        result["cells"].append(
            {
                "batch_size": batch_size,
                "durations": durations,
                "median": statistics.median(durations),
                "all_finite": bool(
                    tf.reduce_all(tf.math.is_finite(value)).numpy()
                    and tf.reduce_all(tf.math.is_finite(score)).numpy()
                ),
            }
        )
    medians = {row["batch_size"]: row["median"] for row in result["cells"]}
    result["batch_native_b16_over_b1"] = medians[16] / medians[1]
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
