"""Read-only price observer tests; no TensorFlow or GPU is required."""
import json
from scripts import monitor_modern_hmc_prices as monitor_pricing


def test_active_price_snapshot_preserves_partial_root_denominators(tmp_path, monkeypatch):
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"status": "running", "rows": [
        {"method": "fixed_hmc", "status": "running", "roots": [
            {"calls": [{"status": "returned"}, {"status": "running"},
                       {"status": "not_started"}]},
            {"calls": [{"status": "not_started"}]},
        ]},
        {"method": "mams", "status": "not_started", "roots": [
            {"calls": [{"status": "not_started"}]},
        ]},
    ]}))
    monkeypatch.setattr(monitor_pricing, "service_state", lambda _: {
        "ActiveState": "active", "SubState": "running", "ExecMainStatus": "0"})

    snapshot = monitor_pricing.snapshot(tmp_path, "pricing.service")

    assert snapshot["status"] == "monitoring"
    assert [row["root_count"] for row in snapshot["methods"]] == [2, 1]
    assert snapshot["methods"][0]["returned_calls"] == 1
    assert snapshot["methods"][0]["running_calls"] == 1
    assert snapshot["methods"][0]["failed_calls"] == 0
    assert snapshot["methods"][1]["returned_calls"] == 0
    assert snapshot["numerical_launch_authority"] is False
    assert snapshot["numerical_stop_authority"] is False
    assert snapshot["settlement_authority"] is False


def test_inactive_service_without_supervisor_requires_terminal_inspection(tmp_path, monkeypatch):
    (tmp_path / "result.json").write_text(json.dumps({"status": "running", "rows": []}))
    monkeypatch.setattr(monitor_pricing, "service_state", lambda _: {
        "ActiveState": "inactive", "SubState": "dead", "ExecMainStatus": "1"})

    snapshot = monitor_pricing.snapshot(tmp_path, "pricing.service")

    assert snapshot["status"] == "terminal_requires_inspection"
    assert snapshot["settlement_authority"] is False


def test_terminal_supervisor_receipt_is_observed_not_settled(tmp_path, monkeypatch):
    (tmp_path / "result.json").write_text(json.dumps({"status": "completed",
        "supervisor_elapsed_seconds": 12., "rows": []}))
    monkeypatch.setattr(monitor_pricing, "service_state", lambda _: {
        "ActiveState": "inactive", "SubState": "dead", "ExecMainStatus": "0"})

    snapshot = monitor_pricing.snapshot(tmp_path, "pricing.service")

    assert snapshot["status"] == "terminal_observed"
    assert snapshot["supervisor_elapsed_seconds"] == 12.
    assert snapshot["settlement_authority"] is False
