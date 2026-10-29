"""Diagnostic evidence integrity checks, independent of numerical execution."""

import json
import tarfile

import pytest

from scripts import index_filter_repair_endpoint_evidence as index


def evidence(tmp_path):
    root = tmp_path / 'evidence'
    source = tmp_path / 'source'
    directory = root / 'run-00042'
    directory.mkdir(parents=True)
    source.mkdir()
    (source / 'kernel.py').write_text('original source\n')
    manifest = directory / 'run.json'
    manifest.write_text(json.dumps({'state': 'passed', 'device': 'CPU',
        'key': ['test', 'fixture'],
        'source_sha256': {'kernel.py': index.sha(source / 'kernel.py')}}))
    result = directory / 'actual-result.json'
    result.write_text('{"numerical_passed": true}\n')
    report = root / 'comparison.json'
    report.write_text(json.dumps({'comparisons': [{'run': 42,
        'manifest_sha256': index.sha(manifest), 'result_sha256': index.sha(result)}]}))
    return root, source, report, manifest, result


def test_valid_hashes_preserve_review_requirement_and_source_delta(tmp_path):
    root, source, report, _, _ = evidence(tmp_path)
    checked = index.index_report(report, root, source)
    assert checked['run_references'][0]['integrity'] == 'recorded_hashes_verified'
    assert checked['changed_recorded_sources'] == {}
    assert checked['status'] == 'contextual_review_required'
    (source / 'kernel.py').write_text('changed source\n')
    changed = index.index_report(report, root, source)
    assert changed['changed_recorded_sources']['kernel.py']['runs'] == [42]
    assert changed['status'] == 'contextual_review_required'


def test_tampered_result_and_manifest_are_not_accepted(tmp_path):
    root, source, report, manifest, result = evidence(tmp_path)
    result.write_text('{"numerical_passed": false}\n')
    checked = index.index_report(report, root, source)
    assert checked['status'] == 'integrity_failure'
    assert checked['integrity_mismatch_runs'] == [42]
    assert checked['run_references'][0]['matching_result_files'] == []
    manifest.write_text(manifest.read_text() + '\n')
    checked = index.index_report(report, root, source)
    assert checked['run_references'][0]['manifest_hash_matches'] is False


def test_missing_manifest_is_an_integrity_gap(tmp_path):
    root, source, report, manifest, _ = evidence(tmp_path)
    manifest.unlink()
    checked = index.index_report(report, root, source)
    assert checked['status'] == 'integrity_failure'
    assert checked['run_references'][0]['integrity'] == 'missing_manifest'


def test_unrecorded_hashes_and_missing_references_are_explicit(tmp_path):
    root, source, report, _, _ = evidence(tmp_path)
    report.write_text('{"rows": [{"run": 42}]}')
    checked = index.index_report(report, root, source)
    assert checked['run_references'][0]['integrity'] == 'hash_fields_unrecorded'
    report.write_text('{"rows": []}')
    checked = index.index_report(report, root, source)
    assert checked['run_references'] == []
    assert checked['status'] == 'contextual_review_required'


def test_named_artifacts_and_file_tables_check_every_digest(tmp_path):
    root, source, report, manifest, result = evidence(tmp_path)
    files = {p.name: index.sha(p) for p in (manifest, result)}
    report.write_text(json.dumps({'rows': [{'run': 42, 'artifact_sha256': files}]}))
    checked = index.index_report(report, root, source)
    assert checked['run_references'][0]['integrity'] == 'recorded_hashes_verified'
    report.write_text(json.dumps({'files': [{'path': f'run-00042/{name}',
        'sha256': digest} for name, digest in files.items()]}))
    checked = index.index_report(report, root, source)
    assert checked['run_references'][0]['integrity'] == 'recorded_hashes_verified'
    result.write_text('changed payload\n')
    checked = index.index_report(report, root, source)
    assert checked['status'] == 'integrity_failure'
    assert checked['run_references'][0]['artifact_hash_matches']['actual-result.json'] is False


def test_archived_report_requires_archive_and_member_identity(tmp_path):
    root, source, report, _, _ = evidence(tmp_path)
    archive, receipt = root / 'bundle.tar.gz', root / 'receipt.json'
    member = 'analysis/comparison.json'
    with tarfile.open(archive, 'w:gz') as bundle:
        bundle.add(report, arcname=member)
    receipt.write_text(json.dumps({'archive_sha256': index.sha(archive),
        'members': {member: index.sha(report)}}))
    checked = index.index_archived_report(archive, receipt, member, root, source, {})
    assert checked['run_references'][0]['integrity'] == 'recorded_hashes_verified'
    receipt.write_text(json.dumps({'archive_sha256': index.sha(archive),
        'members': {member: 'wrong'}}))
    with pytest.raises(ValueError, match='Archived report hash mismatch'):
        index.index_archived_report(archive, receipt, member, root, source, {})
