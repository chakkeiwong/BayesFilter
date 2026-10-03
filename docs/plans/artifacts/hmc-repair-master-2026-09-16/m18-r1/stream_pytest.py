"""Preserve each test report even if the outer diagnostic budget expires."""
import json
import os
from pathlib import Path


def pytest_runtest_logreport(report):
    path = Path(os.environ["HMC_TEST_REPORTS"])
    row = {"nodeid": report.nodeid, "when": report.when, "outcome": report.outcome,
           "duration": report.duration}
    if report.failed:
        row["failure"] = str(report.longrepr)
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def pytest_collection_modifyitems(config, items):
    excluded = set(json.loads(Path(os.environ["HMC_TEST_COMPLETED"]).read_text()))
    selected, deselected = [], []
    for item in items:
        (deselected if item.nodeid in excluded else selected).append(item)
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)
