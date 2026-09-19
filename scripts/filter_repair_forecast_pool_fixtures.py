"""Independent before/after diagnostics for external CPU forecast generation.

The isolated legacy arm deliberately exercises its original scalar host loop.
Graph/XLA rejection is retained as evidence and can use the public arm for
numerical parity only. Process orchestration is measured separately.
"""

FIXTURES = ("cpu_forecast_shard", "cpu_forecast_pool")


def _worker_source_snapshot():
    """Inspect every initialized child after timing, using the startup barrier."""
    import os
    import resource
    from pathlib import Path

    from filter_repair_benchmark_worker import imported_sources

    from bayesfilter.inference import cpu_forecast_pool

    cpu_forecast_pool._WORKER_BARRIER.wait(timeout=30.)
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    source = Path(os.environ["FILTER_REPAIR_FORECAST_SOURCE_ROOT"]).resolve()
    endpoint = getattr(cpu_forecast_pool._WORKER_FORECAST, "__self__", None)
    programs = getattr(endpoint, "_shard_programs", {})
    return {"pid": os.getpid(), "source_root": str(source),
        "numerical_ru_maxrss_bytes": peak,
        "shard_signatures": [list(key) for key in programs],
        "shard_trace_counts": [program.experimental_get_tracing_count() for program in programs.values()],
        "imported_source_sha256": imported_sources(source)}


def fixture(tf, name, size, jit, *, public_boundary=False):
    if name not in FIXTURES:
        raise ValueError(f"Unregistered CPU forecast fixture: {name}")
    count = 2 if size == 1 else 5
    rows = tf.constant([[.35, -.08, .65, .05]], tf.float64) + .001 * tf.reshape(
        tf.cast(tf.range(4 * count), tf.float64), [count, 4])
    seeds = tf.stack((tf.fill([count], 20260719), tf.range(70001, 70001 + count)), axis=1)
    dimensions = {"rows": count, "q": 1, "parameters": 4, "filter_horizon": 30,
        "forecast_horizon": 10, "replications": 2, "seed_policy": "original_independent_row_philox",
        "role": "external_cpu_sample_generation", "neutra_training_eligible": False}

    if name == "cpu_forecast_pool":
        if not public_boundary:
            raise ValueError("The process-pool fixture requires its public host boundary")
        import resource

        from bayesfilter.inference.cpu_forecast_pool import (
            CPUForecastPool,
            CPUForecastPoolConfig,
        )

        pool = CPUForecastPool(CPUForecastPoolConfig(
            worker_factory_path="bayesfilter.nonlinear.ssl_lstm_complexity_predictive_tf:complexity_forecast_worker_factory",
            worker_config={"q": 1}, worker_count=2, cores_per_worker=1, timeout_seconds=120.))
        pool.__enter__()
        calls = []
        worker_sources = []
        final_memory = {}

        def evaluate(values, roots):
            means, variances, observations, metadata = pool.evaluate(values, roots,
                request_id="filter_repair_identical_seeded_forecast")
            calls.append(metadata)
            return (tf.convert_to_tensor(means, tf.float64),
                tf.convert_to_tensor(variances, tf.float64),
                tf.convert_to_tensor(observations, tf.float64))

        def collect_provenance():
            # Each process is sampled even if it did not receive the last shard.
            # Capture before hashing so diagnostic provenance work is excluded.
            parent_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
            futures = [pool._executor.submit(_worker_source_snapshot) for _ in range(2)]
            worker_sources.extend(future.result(timeout=35.) for future in futures)
            if {row["pid"] for row in worker_sources} != set(calls[0]["startup_worker_pids"]):
                raise RuntimeError("Missing source provenance for a forecast worker")
            child_peak_sum = sum(row["numerical_ru_maxrss_bytes"] for row in worker_sources)
            final_memory.update(parent_ru_maxrss_bytes=parent_peak,
                worker_ru_maxrss_sum_bytes=child_peak_sum,
                aggregate_parent_worker_ru_maxrss_bytes=parent_peak + child_peak_sum)

        evaluate.close = pool.close
        evaluate.collect_provenance = collect_provenance
        evaluate.worker_sources = worker_sources
        evaluate.final_memory = final_memory
        evaluate.pool_calls = calls
        evaluate.timing_scope = "complete_public_cpu_forecast_pool_with_ipc_validation_and_output_conversion"
        evaluate.numerical_execution = "owned_worker_defaults_with_cpu_xla; parent_is_host_orchestration"
        dimensions.update(worker_count=2, cores_per_worker=1)
        return evaluate, (rows, seeds), dimensions

    from bayesfilter.nonlinear.ssl_lstm_complexity_predictive_tf import (
        ComplexityForecastWorker,
    )

    worker = ComplexityForecastWorker(1)
    if hasattr(worker, "make_shard_program") and not public_boundary:
        evaluate = worker.make_shard_program(count, jit_compile=jit).python_function
    else:
        def evaluate(values, roots):
            if hasattr(worker, "evaluate_batch"):
                means, variances, observations = worker.evaluate_batch(values, roots)
            else:
                # Diagnostic execution of the pinned, unchanged legacy path.
                outputs = [worker.evaluate(values[index], roots[index]) for index in range(count)]
                means, variances, observations = (
                    tf.stack([output[index] for output in outputs]) for index in range(3))
            return means, variances, observations, tf.ones([count], tf.bool)

    evaluate.timing_scope = ("public_seeded_forecast_shard_with_host_validity" if public_boundary
        else "complete_seeded_forecast_shard_numerics_and_status")
    evaluate.numerical_execution = ("owned_public_defaults" if public_boundary
        else "enclosing_xla" if jit else "explicit_non_xla_graph_reference")
    return evaluate, (rows, seeds), dimensions
