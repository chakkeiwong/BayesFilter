"""CPU-only shard and process-pool diagnostics using the common timing engine."""

import hashlib
import os
import sys
from pathlib import Path

import filter_repair_benchmark_worker as worker
from filter_repair_forecast_pool_fixtures import fixture

# The common diagnostic import prepends its own checkout. Spawn re-executes
# this module before unpickling the measured pool initializer, so restore the
# parent's selected source immediately, before any BayesFilter import.
_SOURCE_ENV = "FILTER_REPAIR_FORECAST_SOURCE_ROOT"
if os.environ.get(_SOURCE_ENV):
    sys.path.insert(0, os.environ[_SOURCE_ENV])


def main():
    measure = worker.measure

    def measure_forecast(args, result):
        if args.device != "CPU":
            raise ValueError("External CPU forecast fixtures require --device CPU")
        for name in ("filter_repair_forecast_pool_worker.py", "filter_repair_forecast_pool_fixtures.py"):
            result["harness_sha256"][name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        calls = []
        previous_source = os.environ.get(_SOURCE_ENV)
        os.environ[_SOURCE_ENV] = str(args.source_root.resolve())

        def forecast_fixture(tf, name, size, jit):
            call, inputs, dimensions = fixture(tf, name, size, jit, public_boundary=args.jit == "eager")
            calls.append(call)
            result["timing_scope"] = call.timing_scope
            result["numerical_execution"] = call.numerical_execution
            if hasattr(call, "pool_calls"):
                result["pool_calls"] = call.pool_calls
                result["worker_sources"] = call.worker_sources
                result["pool_final_memory"] = call.final_memory
                result["pool_memory_scope"] = "sum_of_parent_and_worker_rss_high_water; not_simultaneous_live_memory"
            return call, inputs, dimensions

        worker.fixture = forecast_fixture
        try:
            measure(args, result)
            result["phase"] = "worker_source_verification"
            for call in calls:
                if hasattr(call, "collect_provenance"):
                    call.collect_provenance()
            result["phase"] = "complete"
        except Exception:
            result["status"] = "failed"
            raise
        finally:
            try:
                for call in calls:
                    if hasattr(call, "close"):
                        call.close()
            except Exception:
                result["status"] = "failed"
                raise
            finally:
                if previous_source is None:
                    os.environ.pop(_SOURCE_ENV, None)
                else:
                    os.environ[_SOURCE_ENV] = previous_source

    worker.measure = measure_forecast
    worker.main()


if __name__ == "__main__":
    main()
