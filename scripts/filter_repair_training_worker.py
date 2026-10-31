"""Isolated preparation and finite-loss measurements using the common engine."""

import hashlib
from pathlib import Path

import filter_repair_benchmark_worker as worker
from filter_repair_training_fixtures import fixture


def main():
    measure = worker.measure

    def measure_training(args, result):
        for name in ("filter_repair_training_worker.py", "filter_repair_training_fixtures.py",
                     "filter_repair_centered_fixtures.py"):
            result["harness_sha256"][name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        measure(args, result)

    worker.fixture = fixture
    worker.measure = measure_training
    worker.main()


if __name__ == "__main__":
    main()
