"""Budget continuation tests: standard-library orchestration, no GPU kernels."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


extension = load("budget_extension_test", "extend_surrogate_hmc_marginal_budget.py")
master = load("budget_extension_original_master_test", "run_surrogate_hmc_marginal_expanded.py")


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    output = tmp_path / "campaign"
    output.mkdir()
    record = output / "budget-extension"
    record.mkdir()
    args = argparse.Namespace(output=output, prior_campaign=None, budget_seconds=86400.,
                              gpu="test-no-device", scopes="lg1-t10,lg3-t10",
                              particle_ladder="1024", stage="all")
    attempts = []
    for kind, scope, n in (("engineering", None, 32), ("precision", "m13-t20", 4096),
                           ("pilot", "lg20-t240", 1024), ("scope", "lg1-t10", 1024),
                           ("scope", "lg3-t10", 1024)):
        status = "FAILED" if scope == "lg3-t10" else "COMPLETE"
        attempts.append(dict(key=f"{kind}-{scope or 'all'}-n{n}", kind=kind, scope=scope,
                             particles=n, status=status, wall_seconds=100.,
                             attempt=str(output / "attempts" / str(len(attempts)))))
    original = dict(configuration=master.configuration(args), source_sha256={"test": "sha"},
                    status="INCOMPLETE", attempts=attempts, prior_consumed_seconds=900.)
    (output / "campaign.json").write_text(json.dumps(original))
    cache = output / "cached-row.json"
    cache.write_text('{"complete":true,"saved_scientific_value":1.234}')
    monkeypatch.setattr(extension, "source_identity", lambda: {"test": "sha"})
    monkeypatch.setattr(extension, "process_matches", lambda *a: False)
    return output, record, original, args, cache


def test_extension_keeps_consumed_time_cache_source_and_prior_attempts(fixture):
    output, record, original, args, cache = fixture
    before = cache.read_bytes()
    updated = extension.extend(output, record, original, 86400.)
    assert updated["configuration"]["budget_seconds"] == 172800.
    assert original["configuration"]["budget_seconds"] == 86400.
    assert updated["prior_consumed_seconds"] == 900.
    assert updated["attempts"] == original["attempts"]
    assert updated["source_sha256"] == original["source_sha256"]
    assert cache.read_bytes() == before
    assert json.loads((record / "campaign-before-extension.json").read_text()) == original
    assert updated["budget_extension_history"][0]["additional_seconds"] == 86400.


def test_unchanged_master_skips_completed_jobs_and_creates_fresh_attempt(fixture, monkeypatch):
    output, record, original, args, cache = fixture
    extension.extend(output, record, original, 86400.)
    args.budget_seconds = 172800.
    launched = []
    monkeypatch.setattr(master, "identity", lambda: {"test": "sha"})

    class Process:
        pid = 987654321

        def __init__(self, command, **kwargs):
            launched.append(command)
            attempt = Path(command[command.index("--attempt") + 1])
            (attempt / "result.json").write_text('{"status":"COMPLETE"}')

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(master.subprocess, "Popen", Process)
    master.master(args)
    updated = json.loads((output / "campaign.json").read_text())
    assert len(launched) == 1
    assert launched[0][launched[0].index("--scope") + 1] == "lg3-t10"
    assert Path(launched[0][launched[0].index("--attempt") + 1]).name == "005-scope-lg3-t10-n1024"
    assert updated["attempts"][:-1] == original["attempts"]
    assert updated["prior_consumed_seconds"] == 900.
    assert updated["status"] == "DEFINED_GRID_COMPLETE_DIMENSION_EXTENSION_PENDING"


@pytest.mark.parametrize("problem", ["RUNNING", "ENGINEERING_REPAIR_REQUIRED",
                                    "PRECISION_REPAIR_REQUIRED", "SOURCE_CHANGED_REPAIR_REQUIRED",
                                    "configuration", "source", "worker"])
def test_extension_refuses_changed_or_unsafe_boundary(fixture, monkeypatch, problem):
    output, record, original, args, cache = fixture
    current = copy.deepcopy(original)
    if problem in ("RUNNING", "ENGINEERING_REPAIR_REQUIRED", "PRECISION_REPAIR_REQUIRED",
                   "SOURCE_CHANGED_REPAIR_REQUIRED"):
        current["status"] = problem
    elif problem == "configuration":
        current["configuration"]["gpu"] = "different"
    elif problem == "source":
        monkeypatch.setattr(extension, "source_identity", lambda: {"test": "changed"})
    else:
        current["attempts"][-1]["status"] = "RUNNING"
    (output / "campaign.json").write_text(json.dumps(current))
    before = (output / "campaign.json").read_bytes()
    with pytest.raises(ValueError):
        extension.extend(output, record, original, 86400.)
    assert (output / "campaign.json").read_bytes() == before
    assert not (record / "campaign-before-extension.json").exists()
