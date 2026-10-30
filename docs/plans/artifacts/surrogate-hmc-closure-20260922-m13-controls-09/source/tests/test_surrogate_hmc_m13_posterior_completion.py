"""Completion retry must reuse exactly the preserved partial refinement."""
from bayesfilter.testing import surrogate_hmc_m13_posterior_completion as completion


def test_missing_cell_set_is_exact():
    assert len(completion.missing_cells()) == 386
    assert {i for i, _ in completion.missing_cells()} == set(completion.MISSING_ROWS)


def test_master_completion_is_within_remaining_extension_pool():
    from bayesfilter.inference import surrogate_hmc_closure_program as master
    stages = master.jobs("m13-posterior-completion")
    assert [stage["name"] for stage in stages] == ["review", "m13-posterior-completion"]
    assert stages[1]["plan"] == completion.PLAN
    assert stages[0]["cpu"] + stages[1]["cpu"] <= 117.22267
    assert stages[1]["timeout"] <= 223.90629
    assert "tests/test_surrogate_hmc_m13_posterior_completion.py" in stages[0]["command"]


def test_worker_reaches_posterior_completion(monkeypatch, tmp_path):
    from bayesfilter.testing import surrogate_hmc_closure_diagnostics as worker
    calls = []
    monkeypatch.setattr(completion, "completion_stage",
                        lambda path: calls.append(path) or
                        dict(status="MECHANICS", qualified=False))
    output = tmp_path/"report.json"
    worker.main(["--stage", "m13-posterior-completion", "--device", "cpu",
                 "--output", str(output)])
    assert calls == [output]
