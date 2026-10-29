"""Isolated source-route comparisons using the common measurement engine."""

import hashlib
from pathlib import Path

import filter_repair_benchmark_worker as worker
from filter_repair_source_fixtures import fixture


def main():
    measure = worker.measure

    def measure_source(args, result):
        for name in ("filter_repair_source_worker.py", "filter_repair_source_fixtures.py"):
            result["harness_sha256"][name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()

        def source_fixture(tf, name, size, jit):
            call, inputs, dimensions = fixture(tf, name, size, jit, public_boundary=args.jit == "eager")
            result["timing_scope"] = getattr(call, "timing_scope", "registered_numerical_fixture")
            result["numerical_execution"] = getattr(call, "numerical_execution", "registered_numerical_fixture")
            return call, inputs, dimensions

        worker.fixture = source_fixture
        measure(args, result)

    worker.fixture = fixture
    worker.measure = measure_source
    worker.main()


if __name__ == "__main__":
    main()
