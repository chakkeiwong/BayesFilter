"""Optional original-force guard for consumers whose plans require it.

The generic surrogate-HMC kernel permits a qualified deterministic extension.
This check enforces a stricter declared consumer condition without changing
the proposal map or treating a force failure as endpoint support exclusion.
"""
import hashlib
import json
from pathlib import Path


def original_force_archive_health(manifest_path, *, expected_transitions, expected_chains=4):
    import tensorflow as tf

    if any(type(value) is not int or value < 1 for value in (expected_transitions, expected_chains)):
        raise ValueError("positive transition and chain counts required")
    path = Path(manifest_path)
    raw_manifest = path.read_bytes()
    manifest = json.loads(raw_manifest)
    record = manifest["chunk"]["force_fallback"]
    raw_tensor = (path.parent/record["path"]).read_bytes()
    if hashlib.sha256(raw_tensor).hexdigest() != record["sha256"]:
        raise ValueError("original-force telemetry checksum mismatch")
    flags = tf.io.parse_tensor(raw_tensor, tf.bool)
    if tuple(flags.shape) != (expected_transitions, expected_chains):
        raise ValueError("original-force telemetry shape mismatch")
    counts = tf.reduce_sum(tf.cast(flags, tf.int32), axis=0).numpy().tolist()
    return dict(passed=not any(counts), force_fallback_count_by_chain=counts,
        manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
        telemetry_sha256=record["sha256"], transitions_per_chain=expected_transitions,
        role="predeclared_original_force_condition_not_endpoint_validity")


def _checked_record(manifest_path, record, *, transitions, chains):
    """Check preserved diagnostic provenance without importing TensorFlow."""
    path = Path(manifest_path)
    raw_manifest = path.read_bytes()
    manifest = json.loads(raw_manifest)
    telemetry = manifest["chunk"]["force_fallback"]
    counts = record["force_fallback_count_by_chain"]
    if (not isinstance(counts, list) or len(counts) != chains or
            any(type(count) is not int or not 0 <= count <= transitions for count in counts) or
            record["transitions_per_chain"] != transitions or
            record["passed"] is not (not any(counts))):
        raise ValueError("invalid preserved original-force counts")
    if (record["manifest_sha256"] != hashlib.sha256(raw_manifest).hexdigest() or
            record["telemetry_sha256"] != telemetry["sha256"] or
            hashlib.sha256((path.parent/telemetry["path"]).read_bytes()).hexdigest() != telemetry["sha256"]):
        raise ValueError("preserved original-force evidence changed")
    return record


def retained_original_force_health(run_directory, chunks, *, audit_root,
                                  expected_transitions=250, expected_chains=4):
    """Read inline consumer checks or the separately preserved SV20 status audit.

    The external audit supports the original run whose worker omitted the
    consumer check. It cannot authorize new sampling or replace missing traces.
    Counts were computed by the recorded issuer; this control-plane reader
    verifies their provenance and does not re-run numerical diagnostics.
    """
    if not chunks:
        return dict(passed=True, source="no_retained_chunks", chunks=[])
    present = ["original_force_health" in chunk for chunk in chunks]
    if all(present):
        records = [_checked_record(chunk["manifest"], chunk["original_force_health"],
            transitions=expected_transitions, chains=expected_chains) for chunk in chunks]
        return dict(passed=all(record["passed"] for record in records),
            source="consumer_report", chunks=records)
    if any(present):
        raise ValueError("partial retained original-force evidence")
    matches = []
    for path in Path(audit_root).glob("surrogate-hmc-all-models-*-sv20-force-health-audit-*/run-manifest.json"):
        manifest = json.loads(path.read_text())
        if Path(manifest.get("observed_run", "")).resolve() == Path(run_directory).resolve():
            matches.append((path, manifest))
    if len(matches) != 1:
        raise ValueError("one completed original-force audit is required")
    path, manifest = matches[0]
    report_path = path.parent/"report.json"
    report = json.loads(report_path.read_text())
    accounting = json.loads((path.parent/"accounting.json").read_text())
    source = path.parent/"source/scripts/audit_surrogate_hmc_sv_force_health.py"
    if (report.get("status") != "COMPLETED" or report.get("exit_code") != 0 or
            accounting.get("exit_code") != 0 or report.get("retained_chunks_checked") != len(chunks) or
            report.get("retained_original_force_healthy") is not True or
            hashlib.sha256(source.read_bytes()).hexdigest() != manifest["source_sha256"]):
        raise ValueError("original-force audit is incomplete or invalid")
    retained = {name: row for name, row in report["records"].items() if row.get("retained") is True}
    if set(retained) != {Path(chunk["manifest"]).parent.name for chunk in chunks}:
        raise ValueError("original-force audit coverage mismatch")
    records = []
    for chunk in chunks:
        archive_path = Path(chunk["manifest"])
        row = retained[archive_path.parent.name]
        if Path(row["manifest"]).resolve() != archive_path.resolve():
            raise ValueError("original-force audit observed another archive")
        counts = row["counts_by_chain"]
        records.append(_checked_record(archive_path, dict(passed=not any(counts),
            force_fallback_count_by_chain=counts, transitions_per_chain=row["transitions_per_chain"],
            manifest_sha256=row["manifest_sha256"], telemetry_sha256=row["telemetry_sha256"]),
            transitions=expected_transitions, chains=expected_chains))
    if not all(record["passed"] for record in records):
        raise ValueError("original-force audit conclusion disagrees with its counts")
    return dict(passed=True, source="external_status_audit", chunks=records,
        audit_report=str(report_path.resolve()),
        audit_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest())
