"""Opt-in diagnostic of executable release between completed test modules."""

import gc
import json
import os
from pathlib import Path

import pytest

from scripts.filter_repair_process_memory import memory_snapshot


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    yield
    if nextitem is None or nextitem.path == item.path:
        return
    from bayesfilter.inference import factor_correlation_geometry as factor
    from bayesfilter.inference import fixed_center_fitting_tf as fitting
    from bayesfilter.inference import fixed_center_selection_tf as selection
    from bayesfilter.inference import fixed_center_stability_tf as stability

    before = memory_snapshot(os.getpid())
    factories = (
        factor._make_factor_program,
        fitting.fit_program,
        selection.mean_error_program,
        selection.selection_program,
        stability.stability_program,
    )
    counts = {factory.__qualname__: factory.cache_info()._asdict() for factory in factories}
    for factory in factories:
        factory.cache_clear()
    collected = gc.collect()
    after = memory_snapshot(os.getpid())
    report = {"completed_module": str(item.path), "next_module": str(nextitem.path),
        "cache_counts": counts, "collected": collected, "before": before, "after": after}
    directory = Path(item.config.getoption("xmlpath")).parent
    with (directory / "factor-cache-release.jsonl").open("a") as output:
        output.write(json.dumps(report) + "\n")
    print("FACTOR_CACHE_RELEASE " + json.dumps(report), flush=True)
