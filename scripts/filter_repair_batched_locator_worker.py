"""Isolated batched-locator diagnostics with the shared memory/timing engine."""

import hashlib
from pathlib import Path

import filter_repair_benchmark_worker as worker
from filter_repair_batched_locator_fixtures import fixture


def main():
    measure = worker.measure

    def measure_locator(args, result):
        for name in ("filter_repair_batched_locator_worker.py", "filter_repair_batched_locator_fixtures.py"):
            result["harness_sha256"][name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()

        def scoped_fixture(tf, name, size, jit):
            evaluate, inputs, dimensions = fixture(tf, name, size, jit, public_boundary=args.jit == "eager")
            result["timing_scope"] = evaluate.timing_scope
            return evaluate, inputs, dimensions

        worker.fixture = scoped_fixture
        measure(args, result)

    worker.fixture = fixture
    worker.measure = measure_locator
    worker.main()


if __name__ == "__main__":
    main()
