"""Master wiring, bounded nomination, and M13 resolution provenance guards."""
import pytest
from types import MappingProxyType
from bayesfilter.testing import surrogate_hmc_m13_particle_recovery as recovery


def test_resolution_guard_preserves_model_controls_data_and_source():
    old = dict(fixture="same", settings="same", source_sha256={"kernel": "hash"},
               particle_count=256, chunk_policy="old", input_sha256={"observations": "obs", "cloud": "old"})
    new = old | dict(particle_count=1024, chunk_policy="new", input_sha256={"observations": "obs", "cloud": "new"})
    recovery.verify_resolution_binding(MappingProxyType(new), old, 1024)
    for key in ("fixture", "settings", "source_sha256"):
        with pytest.raises(ValueError, match="unrelated"):
            recovery.verify_resolution_binding(new | {key: "changed"}, old, 1024)
    with pytest.raises(ValueError, match="observations"):
        recovery.verify_resolution_binding(new | {"input_sha256": {"observations": "other"}}, old, 1024)
    with pytest.raises(ValueError, match="resolution"):
        recovery.verify_resolution_binding(new, old, 4096)


@pytest.mark.parametrize("healthy,counts,nominee", [(True, [1024], 1024), (False, [1024, 4096], None)])
def test_particle_ladder_stops_at_first_valid_candidate(healthy, counts, nominee):
    seen = []
    def evaluate(n):
        seen.append(n)
        return dict(all_healthy=healthy, all_failures_flagged=True)
    arms, nominated = recovery.bounded_ladder(evaluate)
    assert seen == counts and len(arms) == len(counts) and nominated == nominee


def test_unflagged_failure_stops_ladder():
    with pytest.raises(ValueError, match="unflagged"):
        recovery.bounded_ladder(lambda n: dict(all_healthy=False, all_failures_flagged=False))


def test_particle_master_dispatch_budget_and_snapshot(tmp_path, monkeypatch):
    from bayesfilter.inference import surrogate_hmc_closure_program as program
    from bayesfilter.testing import surrogate_hmc_closure_diagnostics as worker
    jobs = program.jobs("m13-particle-recovery")
    assert [j["name"] for j in jobs] == ["review", "m13-particle-recovery"]
    assert sum(j["cpu"] for j in jobs) <= 1200
    assert jobs[1]["plan"] == recovery.PLAN
    assert program.remaining_pass_reservation([], "m13-force-recovery")["cpu_worker_seconds"] == 1200
    seen = []
    monkeypatch.setattr(recovery, "particle_recovery_stage", lambda output:
        seen.append(output) or dict(status="WIRING_ONLY", qualified=False))
    output = tmp_path/"report.json"
    worker.main(["--stage", jobs[1]["name"], "--device", "cpu", "--output", str(output)])
    assert seen == [output]
