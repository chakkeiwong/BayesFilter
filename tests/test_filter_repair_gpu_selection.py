"""Host scheduler checks; no real device or numerical worker is used."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import filter_repair_gpu_selection as selection


def snapshot(*, desktop=(0, 1), utilization=(5, 40, 0, 0), free=(47000,) * 4):
    return [{"index": index, "uuid": f"GPU-test-{index}", "name": "RTX 4090",
             "desktop": index in desktop, "desktop_reasons": ["desktop"] if index in desktop else [],
             "utilization_percent": utilization[index], "free_mib": free[index], "memory_mib": 49000-free[index]}
            for index in range(4)]


def test_inventory_protects_display_and_remote_encoder_not_incidental_xorg(monkeypatch):
    commands = []
    output = iter((
        ("0, GPU-a, RTX 4090, 47000, 1000, 5, Disabled\n"
        "1, GPU-b, RTX 4090, 47000, 1000, 40, Enabled\n"
        "2, GPU-c, RTX 4090, 48000, 18, 0, Disabled\n"
        "3, GPU-d, RTX 4090, 48000, 18, 0, Disabled\n"),
        "GPU-a, /usr/libexec/gnome-remote-desktop-daemon\nGPU-c, /usr/lib/xorg/Xorg\n"))
    monkeypatch.setattr(selection.subprocess, "check_output",
        lambda command, **_: commands.append(command) or next(output))
    rows = selection.gpu_snapshot()
    assert [row["desktop"] for row in rows] == [True, True, False, False]
    assert rows[0]["desktop_reasons"] == ["/usr/libexec/gnome-remote-desktop-daemon"]
    assert rows[1]["desktop_reasons"] == ["active_display"]
    assert len(commands) == 2


@pytest.mark.parametrize("bad_row", (
    "0, GPU-a, RTX 4090, 47000, 1000, 5, Unknown",
    "0, GPU-a, RTX 4090, 47000, 1000, 5, Disabled",
))
def test_unknown_display_or_missing_inventory_fails_closed(monkeypatch, bad_row):
    output = iter((bad_row, ""))
    monkeypatch.setattr(selection.subprocess, "check_output", lambda *_, **__: next(output))
    with pytest.raises(ValueError):
        selection.gpu_snapshot()


@pytest.mark.parametrize("index", (0, 1, 2, 3))
def test_every_physical_gpu_can_be_selected_when_not_desktop(monkeypatch, index):
    rows = snapshot(desktop=())
    monkeypatch.setattr(selection.time, "sleep", lambda _: None)
    samples = selection.check_gpu_available(index, snapshot_fn=lambda: rows)
    assert len(samples) == 2
    assert all(row["selected_gpu_index"] == index for row in samples)
    assert not samples[-1]["desktop_fallback"]


def test_auto_selection_prefers_utilization_then_free_memory_then_index(monkeypatch):
    monkeypatch.setattr(selection.time, "sleep", lambda _: None)
    for rows, expected in (
        (snapshot(), 2),
        (snapshot(utilization=(5, 40, 1, 0)), 3),
        (snapshot(free=(47000, 47000, 46000, 48000)), 3),
    ):
        samples = selection.check_gpu_available(snapshot_fn=lambda rows=rows: rows)
        assert samples[-1]["selected_gpu_index"] == expected


@pytest.mark.parametrize("desktop", ((0,), (0, 1)))
def test_desktop_fallback_requires_both_load_and_memory_pressure_on_every_other_gpu(desktop):
    rows = snapshot(desktop=desktop, utilization=(0,) * 4, free=(47000,) * 4)
    for row in rows:
        if not row["desktop"]:
            row.update(utilization_percent=51, free_mib=8191)
    assert {row["index"] for row in selection.eligible_devices(rows)} == set(desktop)
    for row in rows:
        if row["desktop"]:
            continue
        # High utilization alone, or low free memory alone, cannot invoke fallback.
        for changed in ({"free_mib": 8192}, {"utilization_percent": 50}):
            trial = [{**other, **(changed if other is row else {})} for other in rows]
            assert not any(other["desktop"] for other in selection.eligible_devices(trial))


@pytest.mark.parametrize("utilization,free", ((51, 47000), (0, 8191)))
def test_desktop_itself_needs_headroom(utilization, free):
    rows = snapshot(desktop=(0,), utilization=(utilization, 90, 90, 90), free=(free, 0, 0, 0))
    assert selection.eligible_devices(rows) == []


def test_reserved_memory_does_not_block_an_available_compute_gpu(monkeypatch):
    monkeypatch.setattr(selection.time, "sleep", lambda _: None)
    rows = snapshot(utilization=(0, 0, 50, 90), free=(47000, 47000, 8192, 0))
    samples = selection.check_gpu_available(snapshot_fn=lambda: rows)
    assert samples[-1]["selected_gpu_index"] == 2
    assert not samples[-1]["performance_preflight_uncontended"]


def test_idle_shared_gpu_can_check_correctness_but_cannot_qualify_timings(monkeypatch):
    monkeypatch.setattr(selection.time, "sleep", lambda _: None)
    rows = snapshot()
    rows[2]["compute_processes"] = ["other-worker"]
    samples = selection.check_gpu_available(2, snapshot_fn=lambda: rows)
    assert samples[-1]["selected_gpu_index"] == 2
    assert not samples[-1]["performance_preflight_uncontended"]


def test_confirmation_requires_same_uuid_and_records_reselection(monkeypatch):
    readings = iter((snapshot(), snapshot(utilization=(0, 0, 90, 0)), snapshot()))
    sleeps = []
    monkeypatch.setattr(selection.time, "sleep", sleeps.append)
    samples = selection.check_gpu_available(snapshot_fn=lambda: next(readings))
    assert [sample["selected_gpu_index"] for sample in samples] == [2, 3, 3]
    assert sleeps == [2, 2]
    assert all(sample["sampled_utc"] and len(sample["devices"]) == 4 for sample in samples)


@pytest.mark.parametrize("requested", (None, 0))
def test_unavailable_selection_rechecks_only_six_times(monkeypatch, requested):
    # Plenty of memory means the owner's conjunctive fallback condition is false.
    rows = snapshot(utilization=(0, 0, 90, 90))
    sleeps = []
    monkeypatch.setattr(selection.time, "sleep", sleeps.append)
    with pytest.raises(RuntimeError, match="contention veto after bounded recheck"):
        selection.check_gpu_available(requested, snapshot_fn=lambda: rows)
    assert sleeps == [2] * 5
