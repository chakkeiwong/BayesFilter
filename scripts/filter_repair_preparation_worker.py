"""Use the unchanged measurement engine with isolated preparation fixtures."""

import hashlib
from pathlib import Path

import filter_repair_benchmark_worker as worker
from filter_repair_preparation_fixtures import fixture


def main():
    measure = worker.measure

    def measure_preparation(args, result):
        for name in ("filter_repair_preparation_worker.py", "filter_repair_preparation_fixtures.py"):
            result["harness_sha256"][name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        measure(args, result)
        # tf.nest sorts mapping keys; the teacher's scalar validity is last.
        # Equal invalid fits cannot qualify this execution comparison.
        if args.fixture == "moment_teacher" and result["values"][-1] is not True:
            result["status"], result["phase"] = "failed", "validity"
            raise RuntimeError("The frozen moment-teacher fixture failed its numerical validity gate")

    worker.fixture = fixture
    worker.measure = measure_preparation
    worker.main()


if __name__ == "__main__":
    main()
