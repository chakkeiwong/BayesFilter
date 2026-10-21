"""Independent endpoint binding, reference admission and closed comparison checks."""
from types import SimpleNamespace

import pytest

from bayesfilter.inference import surrogate_hmc_equivalence as equivalence
from bayesfilter.inference import surrogate_hmc_program as master
from scripts import run_surrogate_hmc_ksc_finite as runner
from scripts.prepare_surrogate_hmc_ksc import save, sha


def test_public_master_dispatches_independent_c(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(runner, "main", lambda args: seen.extend(args) or 9)
    code = master.main(["--stage", "P3-KSC-finite-inference", "--scope", "KSC", "--output", str(tmp_path),
        "--qualification-root", "/q", "--selection", "/s/selection.json", "--posterior-root", "/r",
        "--endpoint-root", "/endpoint", "--finite-reference-root", "/d", "--dry-run"])
    assert code == 9
    assert seen[seen.index("--endpoint-root")+1] == "/endpoint"
    assert seen[seen.index("--finite-reference-root")+1] == "/d"
    assert "--cost-root" not in seen


def test_master_c_requires_its_finite_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "main", lambda args: pytest.fail("missing D launched"))
    with pytest.raises(SystemExit):
        master.main(["--stage", "P3-KSC-finite-inference", "--scope", "KSC", "--output", str(tmp_path),
            "--qualification-root", "/q", "--selection", "/s/selection.json", "--posterior-root", "/r",
            "--endpoint-root", "/endpoint"])


def make_reference(tmp_path, monkeypatch):
    data = tmp_path / "data.json"
    save(data, {"observations": [1., 2.]})
    tensor = tmp_path / "grid.tensor"
    tensor.write_bytes(b"diagnostic tensor fixture")
    item = dict(path=str(tensor), sha256=sha(tensor))
    grid = {name: item for name in ("gamma", "beta", "density", "predictive")}
    exact = dict(means=[.5, .5], scales=[.2, .2], cuts=[[.2, .5, .8]]*2,
                 margins=[.1]*5+[.02]*6+[.03]*2, function_names=[str(i) for i in range(13)],
                 data_sha256=sha(data))
    endpoint = tmp_path / "endpoint"
    endpoint.mkdir()
    save(endpoint / "report.json", {"endpoint_identity": "D"})
    monkeypatch.setattr(runner, "check_endpoint", lambda *args, **kwargs: ({"endpoint_identity": "D"}, {}))
    report = dict(**exact, status="QUALIFIED_FINITE_POSTERIOR_REFERENCE", qualified=True,
        model_id="M03-KSC-T2", full_prior_integrated=True, endpoint_identity="D", selection_sha256="selection",
        endpoint_report_sha256=sha(endpoint / "report.json"), reference_report_sha256="R",
        parameter_intervals=[16, 32, 64, 128], grids=[grid]*4, log_density=item,
        precision=[True]*13, contraction=[True]*13, allowance=[1e-5]*13, expectations=[[0.]*13]*4,
        D_vs_R={"status": "DISCREPANCY"})
    manifest = dict(exit_code=0, source_hashes={}, posterior_report_sha256="R",
        **{key: report[key] for key in ("endpoint_identity", "selection_sha256", "endpoint_report_sha256", "data_sha256")})
    save(tmp_path / "report.json", report)
    save(tmp_path / "run-manifest.json", manifest)
    return dict(exact_reference=exact, endpoint_root=endpoint, selection_sha="selection", posterior_sha="R"), report


def test_reference_discrepancy_does_not_block_independent_c(tmp_path, monkeypatch):
    arguments, _ = make_reference(tmp_path, monkeypatch)
    report = runner.check_finite_reference(tmp_path, **arguments)
    assert report["D_vs_R"]["status"] == "DISCREPANCY"


@pytest.mark.parametrize("mutation", ["endpoint", "precision", "functionals", "corruption"])
def test_reference_rejects_invalid_or_mismatched_evidence(tmp_path, monkeypatch, mutation):
    arguments, report = make_reference(tmp_path, monkeypatch)
    if mutation == "endpoint":
        report["endpoint_identity"] = "R"
    elif mutation == "precision":
        report["precision"][0] = False
    elif mutation == "functionals":
        report["means"][0] = .6
    else:
        (tmp_path / "grid.tensor").write_bytes(b"corrupted")
    save(tmp_path / "report.json", report)
    # Use a separate exact-reference object, like independently loaded artifacts.
    if mutation == "functionals":
        arguments["exact_reference"]["means"] = [.5, .5]
    with pytest.raises(ValueError):
        runner.check_finite_reference(tmp_path, **arguments)


def test_c_binding_uses_finite_endpoint_and_full_field(monkeypatch):
    import bayesfilter.inference as inference
    seen = {}

    def bind(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(binding_hash="bound")

    monkeypatch.setattr(inference, "bind_neural_force_hmc_tuning_runner", bind)
    force, exact, finite = object(), object(), object()
    adapter, _ = runner.bind_finite_arm(SimpleNamespace(force=force, endpoint=exact), finite)
    assert seen["target"] is finite
    assert seen["force"] is force
    assert adapter.target_scope.endswith(":C")


def test_c_d_and_c_r_use_their_own_reference_allowances(monkeypatch):
    seen = []

    def assess(parameters, functions, reference, allowance, margins, **kwargs):
        seen.append((reference.numpy().tolist(), allowance.numpy().tolist(), kwargs))
        return {"status": "INCONCLUSIVE"}

    monkeypatch.setattr(equivalence, "assess_equivalence", assess)
    d = dict(expectations=[[1.]*13], allowance=[.001]*13, margins=[.1]*13)
    r = dict(expectations=[[2.]*13], allowance=[.002]*13, margins=[.1]*13)
    result = runner.assess_comparisons(None, None, {"C/D": d, "C/R": r})
    assert set(result) == {"C/D", "C/R"}
    assert seen[0][:2] == ([1.]*13, [.001]*13)
    assert seen[1][:2] == ([2.]*13, [.002]*13)
    assert all(row[2]["comparison_families"] == 4 for row in seen)


def test_closed_comparison_is_not_reopened_when_other_continues():
    d, r = object(), object()
    status = {"C/D": "EQUIVALENT", "C/R": "INCONCLUSIVE"}
    assert runner.pending_comparisons(status, d, r) == {"C/R": r}
    status["C/R"] = "DISCREPANCY"
    assert runner.pending_comparisons(status, d, r) == {}
