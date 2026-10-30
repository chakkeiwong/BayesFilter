"""Use the unchanged measurement engine with isolated forecast fixtures."""

import hashlib
from pathlib import Path

import filter_repair_benchmark_worker as worker
from filter_repair_forecast_fixtures import fixture


def main():
    measure = worker.measure

    def measure_forecast(args, result):
        for name in ("filter_repair_forecast_worker.py", "filter_repair_forecast_fixtures.py"):
            result["harness_sha256"][name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        return measure(args, result)

    worker.fixture = fixture
    worker.measure = measure_forecast
    worker.main()


if __name__ == "__main__":
    main()
