"""Preserved consumer health remains distinct from generic map health."""
import hashlib
import json
from pathlib import Path

import pytest

from bayesfilter.inference.surrogate_hmc_force_health import retained_original_force_health


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def example(tmp_path, *, inline=False, fallback=False):
    run = tmp_path/"inference"
    path = run/"B/chunk-01/manifest.json"
    path.parent.mkdir(parents=True)
    tensor = path.parent/"flags.tfbin"
    # This reader verifies an already computed diagnostic, without TensorFlow.
    tensor.write_bytes(b"preserved telemetry fixture")
    save(path, dict(health_passed=True,
        chunk=dict(force_fallback=dict(path=tensor.name, sha256=digest(tensor)))))
    counts = [0, 0, int(fallback), 0]
    record = dict(passed=not fallback, force_fallback_count_by_chain=counts,
        transitions_per_chain=250, manifest_sha256=digest(path), telemetry_sha256=digest(tensor))
    chunks = [dict(manifest=str(path), health_passed=True)]
    if inline:
        chunks[0]["original_force_health"] = record
    audit = tmp_path/"surrogate-hmc-all-models-test-sv20-force-health-audit-01"
    source = audit/"source/scripts/audit_surrogate_hmc_sv_force_health.py"
    source.parent.mkdir(parents=True)
    source.write_text("# preserved test issuer\n")
    save(audit/"run-manifest.json", dict(observed_run=str(run), source_sha256=digest(source)))
    save(audit/"accounting.json", dict(exit_code=0))
    save(audit/"report.json", dict(status="COMPLETED", exit_code=0,
        retained_chunks_checked=1, retained_original_force_healthy=True,
        records={"chunk-01": dict(manifest=str(path), retained=True,
            counts_by_chain=counts, transitions_per_chain=250,
            manifest_sha256=digest(path), telemetry_sha256=digest(tensor))}))
    return run, chunks, audit


@pytest.mark.parametrize("inline", [False, True])
def test_preserved_inline_or_external_evidence_is_checked(tmp_path, inline):
    run, chunks, _ = example(tmp_path, inline=inline)
    result = retained_original_force_health(run, chunks, audit_root=tmp_path)
    assert result["passed"] is True
    assert result["source"] == ("consumer_report" if inline else "external_status_audit")
    assert result["chunks"][0]["force_fallback_count_by_chain"] == [0, 0, 0, 0]


def test_inline_finite_fallback_is_a_consumer_veto(tmp_path):
    run, chunks, _ = example(tmp_path, inline=True, fallback=True)
    result = retained_original_force_health(run, chunks, audit_root=tmp_path)
    assert chunks[0]["health_passed"] is True
    assert result["passed"] is False


@pytest.mark.parametrize("corruption", ["telemetry", "source", "manifest", "coverage", "path",
                                        "counts", "missing", "unfinished"])
def test_external_status_audit_cannot_hide_missing_or_changed_evidence(tmp_path, corruption):
    run, chunks, audit = example(tmp_path)
    report = json.loads((audit/"report.json").read_text())
    if corruption == "telemetry":
        (run/"B/chunk-01/flags.tfbin").write_bytes(b"changed")
    elif corruption == "source":
        (audit/"source/scripts/audit_surrogate_hmc_sv_force_health.py").write_text("# changed")
    elif corruption == "manifest":
        (run/"B/chunk-01/manifest.json").write_text("{}")
    elif corruption == "coverage":
        report["records"]["extra"] = report["records"]["chunk-01"]
    elif corruption == "path":
        report["records"]["chunk-01"]["manifest"] = str(tmp_path/"other/manifest.json")
    elif corruption == "counts":
        report["records"]["chunk-01"]["counts_by_chain"] = [0, 1, 0, 0]
    elif corruption == "missing":
        report["records"] = {}
    elif corruption == "unfinished":
        report["status"] = "RUNNING"
    save(audit/"report.json", report)
    with pytest.raises((ValueError, KeyError)):
        retained_original_force_health(run, chunks, audit_root=tmp_path)


@pytest.mark.parametrize("counts", [[0, False, 0, 0], [0, -1, 0, 0], [0, 251, 0, 0], [0, 0]])
def test_invalid_counts_do_not_establish_consumer_health(tmp_path, counts):
    run, chunks, _ = example(tmp_path, inline=True)
    chunks[0]["original_force_health"]["force_fallback_count_by_chain"] = counts
    with pytest.raises(ValueError, match="counts"):
        retained_original_force_health(run, chunks, audit_root=tmp_path)


def test_partial_inline_check_cannot_be_patched_with_an_external_audit(tmp_path):
    run, chunks, _ = example(tmp_path, inline=True)
    chunks.append(dict(manifest=str(run/"B/chunk-02/manifest.json"), health_passed=True))
    with pytest.raises(ValueError, match="partial"):
        retained_original_force_health(run, chunks, audit_root=tmp_path)
