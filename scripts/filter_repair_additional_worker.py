"""Use the unchanged measurement engine with separately versioned fixtures."""

import hashlib
from pathlib import Path

import filter_repair_benchmark_worker as worker
from filter_repair_additional_fixtures import fixture


def main():
    measure = worker.measure

    def measure_additional(args, result):
        for name in ("filter_repair_additional_worker.py", "filter_repair_additional_fixtures.py"):
            result["harness_sha256"][name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        return measure(args, result)

    worker.fixture = fixture
    worker.measure = measure_additional
    worker.main()


if __name__ == "__main__":
    main()
