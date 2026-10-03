"""Mechanics checks for inherited budgets and profile result disposition."""
import json
from pathlib import Path
import ast
import importlib

import pytest

from bayesfilter.inference.q20_master_profile import (
    checked_profile_allowance, check_profile_sources, profile_disposition, prior_profile_charges,
    qualification_profile_budget, SETUP_SECONDS,
)
from bayesfilter.inference.q20_campaign_runtime import source_snapshot
from bayesfilter.inference.q20_training_resume import checksum


def verification(tmp_path):
    path = tmp_path / "verified.json"
    path.write_text(json.dumps({"status": "passed", "sources": {"kernel.py": "abc"}, "wall_seconds": 12.}))
    settled = {"source_sha256": "previous", "campaign_remaining_seconds": 8000.,
               "diagnostic_remaining_seconds": 4000.}
    allowance = {"predecessor_sha256": "previous", "verification": {"path": str(path), "sha256": checksum(path)},
                 "campaign_remaining_seconds": 8000.-12.-SETUP_SECONDS,
                 "diagnostic_remaining_seconds": 4000.-12.-SETUP_SECONDS}
    return allowance, settled


def test_profile_debits_both_balances(tmp_path):
    allowance, settled = verification(tmp_path)
    assert checked_profile_allowance(allowance, settled, {"kernel.py": "abc"}) == 12.
    allowance["diagnostic_remaining_seconds"] += 12.
    with pytest.raises(ValueError, match="without renewing budget"):
        checked_profile_allowance(allowance, settled, {"kernel.py": "abc"})


def test_profile_rejects_stale_verification(tmp_path):
    allowance, settled = verification(tmp_path)
    with pytest.raises(ValueError, match="stale"):
        checked_profile_allowance(allowance, settled, {"kernel.py": "changed"})
    Path(allowance["verification"]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="receipt changed"):
        checked_profile_allowance(allowance, settled, {"kernel.py": "abc"})


@pytest.mark.parametrize("result,status", [
    ({"completed": True, "result": {"candidate_parity_passed": True}}, "MASTER_PROFILE_PARITY_PASSED"),
    ({"completed": True, "result": {"candidate_parity_passed": False}}, "MASTER_PROFILE_CANDIDATE_REJECTED"),
    ({"completed": False, "status": "timed_out"}, "MASTER_PROFILE_INCOMPLETE"),
])
def test_profile_candidate_rejection_is_not_parity_or_campaign_success(result, status):
    assert profile_disposition(result) == status


def test_profile_source_overlay_rejects_numerical_change(tmp_path):
    for name in ("old", "new"):
        root = tmp_path / name
        (root / "bayesfilter").mkdir(parents=True)
        (root / "bayesfilter/kernel.py").write_text("value = 1\n")
        (root / "docs/benchmarks").mkdir(parents=True)
        for script in ("run_ssl_lstm_q20_production_2026_09_15.py", "diagnose_q20_hmc_status_reuse_2026_09_16.py"):
            (root / "docs/benchmarks" / script).write_text("# fixture\n")
    old, new = tmp_path / "old", tmp_path / "new"
    snapshot = source_snapshot(old)
    assert check_profile_sources(snapshot, old, new) == []
    (new / "bayesfilter/kernel.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="unreviewed profile source"):
        check_profile_sources(snapshot, old, new)


def test_diagnostic_bindings_import_in_the_executed_source():
    script = Path(__file__).resolve().parents[1] / "docs/benchmarks/diagnose_q20_factor_performance_2026_09_19.py"
    for node in ast.walk(ast.parse(script.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module.startswith("bayesfilter."):
            module = importlib.import_module(node.module)
            for name in node.names:
                assert getattr(module, name.name) is not None


def test_retry_shares_original_worker_cap_and_debits_failure(tmp_path):
    allowance, settled = verification(tmp_path)
    ledger = tmp_path / "failed.json"
    ledger.write_text(json.dumps({"profile_predecessor": {"sha256": "previous"},
        "attempts": [{"stage": "factor-profile", "status": "failed", "elapsed_seconds": 5.}],
        "spent_seconds": 5.}))
    allowance["prior_profile_ledgers"] = [{"path": str(ledger), "sha256": checksum(ledger)}]
    assert prior_profile_charges(allowance, settled) == (5., 1)
    with pytest.raises(ValueError, match="without renewing budget"):
        checked_profile_allowance(allowance, settled, {"kernel.py": "abc"})
    allowance["campaign_remaining_seconds"] -= 5.
    allowance["diagnostic_remaining_seconds"] -= 5.
    assert checked_profile_allowance(allowance, settled, {"kernel.py": "abc"}) == 12.
    allowance["prior_profile_ledgers"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        prior_profile_charges(allowance, settled)


def test_only_matching_declared_infinity_sentinel_is_allowed():
    import importlib.util
    import tensorflow as tf
    path = Path(__file__).resolve().parents[1] / "docs/benchmarks/diagnose_q20_factor_performance_2026_09_19.py"
    spec = importlib.util.spec_from_file_location("factor_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inf = tf.constant([float("inf")], tf.float64)
    a, b = module.parity_values(inf, inf, allow_positive_infinity=True)
    tf.debugging.assert_equal(a, b)
    for left, right, allow in ((inf, inf, False), (inf, tf.constant([1.], tf.float64), True),
                                (-inf, -inf, True), (inf*0., inf*0., True)):
        with pytest.raises(tf.errors.InvalidArgumentError):
            module.parity_values(left, right, allow_positive_infinity=allow)


def test_cache_protocol_is_optional_and_keeps_the_same_other_target_fields():
    from bayesfilter.inference.q20_production_config import protocol_template, validate_protocol, digest
    strict = protocol_template()
    assert strict["target"]["principal_sqrt_backend"] == "tensorflow_eigh_strict"
    candidate = json.loads(json.dumps(strict))
    candidate["target"]["principal_sqrt_backend"] = "tensorflow_eigh_strict_factor_cached"
    validate_protocol(candidate)
    assert digest(candidate) != digest(strict)
    candidate["target"]["q"] = 21
    with pytest.raises(ValueError, match="target changes"):
        validate_protocol(candidate)
    candidate["target"]["q"] = 20
    candidate["target"]["principal_sqrt_backend"] = "tensorflow_eigh_strict_cached"
    with pytest.raises(ValueError, match="safe-factor"):
        validate_protocol(candidate)


def test_qualification_cannot_renew_the_profile_allocation():
    old = {"spent_seconds": 1700., "profile_predecessor": {"prior_profile_seconds": 82.}}
    assert qualification_profile_budget(old) == 1218.
    old["spent_seconds"] = 3000.
    assert qualification_profile_budget(old) == 0.
