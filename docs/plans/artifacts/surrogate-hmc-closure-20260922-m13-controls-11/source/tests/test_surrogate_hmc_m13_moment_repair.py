"""Weighted-moment overflow repair, total JVP, and shared-route regressions."""
import pytest
import tensorflow as tf
from types import SimpleNamespace

from bayesfilter.testing import surrogate_hmc_m13_moment_repair as repair


@pytest.mark.parametrize("extreme,zero", [(False, False), (True, False), (True, True)])
def test_weighted_moments_and_jvp_match_independent_fp64_reference(extreme, zero):
    inputs = repair.primitive_inputs(extreme, zero)
    actual, reference = repair.moment_primitive(*inputs), repair.direct_reference(*inputs)
    for a, b in zip(actual, reference):
        assert bool(tf.reduce_all(tf.math.is_finite(a)))
        assert repair.relative_error(a, b) < 2e-6
    if extreme:
        z, w, _, _ = inputs
        assert not bool(tf.reduce_all(tf.math.is_finite(tf.reduce_sum(w[:, None]*z**4, axis=0))))


def test_analytical_jvp_matches_centered_perturbation_in_both_weights_and_positions():
    inputs = tuple(tf.cast(x, tf.float64) for x in repair.primitive_inputs(True))
    z, w, dz, dw = inputs
    analytic = repair.moment_primitive(*inputs)
    eps = 1e-4
    plus = repair.direct_reference(z+eps*dz[..., 0], w+eps*dw[..., 0], dz, dw)
    minus = repair.direct_reference(z-eps*dz[..., 0], w-eps*dw[..., 0], dz, dw)
    for value, tangent in ((0, 2), (1, 3), (4, 6), (5, 7)):
        fd = (plus[value]-minus[value])/(2*eps)
        assert repair.relative_error(analytic[tangent][..., 0], fd) < 1e-7


def test_declared_repair_does_not_accept_other_source_or_input_changes():
    fields = ("fixture", "horizon", "particle_count", "dtype", "tf32", "jit_compile",
              "settings", "design_partition", "seeds", "input_sha256")
    saved = {k: "original" for k in fields}
    saved["source_sha256"] = {repair.CHANGED_SOURCE: "before", "other.py": "fixed"}
    changed = saved | {"source_sha256": {repair.CHANGED_SOURCE: "after", "other.py": "fixed"}}
    assert repair.bind_repaired_source(SimpleNamespace(manifest=changed), saved)
    with pytest.raises(ValueError, match="undeclared source"):
        repair.bind_repaired_source(SimpleNamespace(manifest=changed | {"source_sha256":
            {repair.CHANGED_SOURCE: "after", "other.py": "changed"}}), saved)
    with pytest.raises(ValueError, match="input_sha256"):
        repair.bind_repaired_source(SimpleNamespace(manifest=changed | {"input_sha256": "changed"}), saved)


def test_master_dispatch_and_budget(tmp_path, monkeypatch):
    from bayesfilter.inference import surrogate_hmc_closure_program as program
    from bayesfilter.testing import surrogate_hmc_closure_diagnostics as worker
    jobs = program.jobs("m13-moment-repair")
    assert [j["name"] for j in jobs] == ["review", "m13-moment-repair"]
    assert sum(j["cpu"] for j in jobs) <= 197.444047
    assert jobs[1]["plan"] == repair.PLAN
    seen = []
    def stage(output):
        seen.append(output)
        return dict(status="WIRING_ONLY", qualified=False)
    monkeypatch.setattr(repair, "repair_stage", stage)
    output = tmp_path/"report.json"
    worker.main(["--stage", jobs[1]["name"], "--device", "cpu", "--output", str(output)])
    assert seen == [output]
