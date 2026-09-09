"""CPU-only synthetic placement tests; these never inspect real GPUs."""

from __future__ import annotations

import sys

import pytest

from bayesfilter.runtime import display_gpu_policy as policy


def _row(index: int, *, display: bool = False, load: float = 0.0, free: float = 20000.0):
    return {
        "index": index, "uuid": f"GPU-test-{index}", "name": "synthetic",
        "pci_bus_id": f"0000:{index:02d}:00.0", "is_display": display,
        "utilization_gpu_pct": load, "memory_free_mib": free,
        "memory_used_mib": 32768.0 - free, "memory_total_mib": 32768.0,
        "processes": [],
    }


def _snapshot(*rows):
    return {"trust_basis": "trusted_escalated_gpu_execution", "gpus": list(rows)}


def test_non_display_preference_does_not_depend_on_index():
    selection = policy.select_gpu(_snapshot(
        _row(0, display=True), _row(1, load=41.0), _row(2, load=40.0, free=5120.0),
    ))
    assert selection["selected"]["uuid"] == "GPU-test-2"
    assert selection["reason"] == "eligible_non_display_preferred"
    assert selection["minimum_headroom_mib"] == 5120.0


@pytest.mark.parametrize("load,free,eligible", (
    (40.0, 5120.0, True), (40.01, 5120.0, False), (40.0, 5119.0, False),
    (0.0, 0.0, False), (100.0, 32768.0, False),
))
def test_inclusive_load_and_memory_thresholds(load, free, eligible):
    selection = policy.select_gpu(_snapshot(_row(0, load=load, free=free)))
    assert (selection["selected"] is not None) is eligible


def test_display_fallback_only_when_every_non_display_device_fails():
    snapshot = _snapshot(_row(0, load=41.0), _row(1, free=5119.0), _row(2, display=True))
    selection = policy.select_gpu(snapshot)
    assert selection["selected"]["index"] == 2
    assert selection["reason"] == "display_fallback_no_eligible_non_display"
    snapshot["gpus"][2]["utilization_gpu_pct"] = 41.0
    assert policy.select_gpu(snapshot)["selected"] is None


def test_tie_break_uses_load_then_free_memory():
    selection = policy.select_gpu(_snapshot(
        _row(0, load=2.0, free=32000), _row(1, load=1.0, free=30000),
        _row(2, load=1.0, free=31000),
    ))
    assert selection["selected"]["index"] == 2


def test_multi_gpu_selection_queues_excess_work_instead_of_using_display():
    selection = policy.select_gpus(
        _snapshot(
            _row(0, display=True, load=0.0),
            _row(1, display=False, load=2.0),
            _row(2, display=False, load=3.0),
        ),
        3,
    )
    assert selection["selected_uuids"] == ["GPU-test-1", "GPU-test-2"]
    assert selection["reason"] == "eligible_non_display_preferred"


def test_multi_gpu_selection_degrades_to_one_non_display_worker():
    selection = policy.select_gpus(
        _snapshot(_row(0), _row(1, load=41.0)),
        2,
    )
    assert selection["selected_uuids"] == ["GPU-test-0"]


def test_multi_gpu_display_fallback_requires_no_eligible_non_display():
    selection = policy.select_gpus(
        _snapshot(_row(0, display=True), _row(1, load=41.0), _row(2, free=9000)),
        3, estimated_peak_mib=4096,
    )
    assert selection["selected_uuids"] == ["GPU-test-0"]
    assert selection["reason"] == "display_fallback_no_eligible_non_display"


@pytest.mark.parametrize("free,expected", ((9215, []), (9216, ["GPU-test-1"])))
def test_multi_gpu_admission_preserves_headroom_after_expected_worker_peak(free, expected):
    selection = policy.select_gpus(_snapshot(_row(1, free=free)), 2, estimated_peak_mib=4096)
    assert selection["selected_uuids"] == expected


@pytest.mark.parametrize("count", (0, -1, True, 1.5))
def test_multi_gpu_count_is_a_positive_integer(count):
    with pytest.raises(policy.GPUPlacementError, match="count"):
        policy.select_gpus(_snapshot(_row(1)), count)


@pytest.mark.parametrize("field,value", (
    ("is_display", None), ("utilization_gpu_pct", "N/A"),
    ("memory_free_mib", float("nan")), ("memory_total_mib", -1),
))
def test_unknown_or_invalid_telemetry_cannot_select_device(field, value):
    row = _row(1)
    row[field] = value
    assert policy.select_gpu(_snapshot(row))["selected"] is None


def test_untrusted_or_duplicate_inventory_is_rejected():
    with pytest.raises(policy.GPUPlacementError, match="trusted"):
        policy.select_gpu({"gpus": [_row(0)]})
    with pytest.raises(policy.GPUPlacementError, match="duplicate"):
        policy.select_gpu(_snapshot(_row(1), _row(1)))


def test_xml_join_uses_uuid_not_minor_number_and_does_not_confuse_xorg_context():
    csv = "1, GPU-test-1, synthetic, 0000:01:00.0, Disabled, 0, 32768, 18, 32750\n"
    xml = """<nvidia_smi_log><gpu><uuid>GPU-test-1</uuid><minor_number>2</minor_number>
    <display_attached>No</display_attached><processes><process_info><pid>123</pid>
    <type>G</type><process_name>/usr/lib/xorg/Xorg</process_name><used_memory>4 MiB</used_memory>
    </process_info></processes></gpu></nvidia_smi_log>"""
    row = policy.parse_inventory(csv, xml)[0]
    assert row["index"] == 1
    assert row["is_display"] is False
    assert row["processes"][0]["type"] == "G"
    assert policy.parse_inventory(csv, xml.replace("<display_attached>No", "<display_attached>Yes"))[0]["is_display"] is True
    with pytest.raises(policy.GPUPlacementError, match="identities disagree"):
        policy.parse_inventory(csv, xml.replace("GPU-test-1", "GPU-test-2"))


def test_unknown_display_attachment_is_not_treated_as_non_display():
    csv = "1, GPU-test-1, synthetic, 0000:01:00.0, Disabled, 0, 32768, 18, 32750\n"
    xml = "<nvidia_smi_log><gpu><uuid>GPU-test-1</uuid></gpu></nvidia_smi_log>"
    row = policy.parse_inventory(csv, xml)[0]
    assert row["is_display"] is None
    assert policy.select_gpu(_snapshot(row))["selected"] is None


def test_selection_pins_uuid_and_preserves_inventory(monkeypatch):
    for framework in ("tensorflow", "tensorflow_probability", "jax", "torch"):
        monkeypatch.delitem(sys.modules, framework, raising=False)
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("BAYESFILTER_SELECTED_GPU_UUID", "")
    monkeypatch.setenv("BAYESFILTER_GPU_SELECTION_POLICY_ID", "")
    snapshot = _snapshot(_row(0, display=True), _row(1))
    monkeypatch.setattr(policy, "probe_inventory", lambda: snapshot)
    selection = policy.select_and_pin_gpu()
    assert policy.os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-test-1"
    assert policy.os.environ["BAYESFILTER_GPU_SELECTION_POLICY_ID"] == policy.POLICY_ID
    assert selection["snapshot"] == snapshot
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH", "false")
    with pytest.raises(policy.GPUPlacementError, match="ALLOW_GROWTH"):
        policy.select_and_pin_gpu()


def test_selection_after_framework_import_fails_before_probe(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorflow", object())
    monkeypatch.setattr(policy, "probe_inventory", lambda: pytest.fail("must not probe"))
    with pytest.raises(policy.GPUPlacementError, match="precede"):
        policy.select_and_pin_gpu()


def test_live_headroom_check_does_not_reject_our_own_utilization(monkeypatch):
    selection = policy.select_gpu(_snapshot(_row(1)))
    snapshot = {**_snapshot(_row(1, load=100, free=5120)), "captured_at_utc": "synthetic"}
    monkeypatch.setattr(policy, "probe_inventory", lambda: snapshot)
    assert policy.check_runtime_headroom(selection, 1024)["status"] == "pass"
    snapshot["gpus"][0]["memory_free_mib"] = 5119.0
    with pytest.raises(policy.GPUPlacementError, match="5 GiB"):
        policy.check_runtime_headroom(selection, 1024)
