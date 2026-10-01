"""Diagnostic index of saved endpoint evidence; never an admission decision.

This standard-library reader preserves historical reports and checks their
recorded file identities. It does not rerun analyzers or numerical kernels.
Source differences require endpoint-specific review before evidence renewal.
"""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

FAMILIES = {
    'block_public': ('block-public-costs-*.json',
        'scripts/analyze_filter_repair_block_public_costs.py'),
    'sequential_public': ('sequential-public-costs-*.json',
        'scripts/analyze_filter_repair_sequential_public_costs.py'),
    'posterior_public': ('posterior-public-costs-*.json',
        'scripts/analyze_filter_repair_posterior_public_costs.py'),
    'posterior_initializer': ('posterior-initializer-cost-*.json',
        'scripts/analyze_filter_repair_posterior_initializer_costs.py'),
    'remaining_svd': ('remaining-svd-cost-*.json',
        'scripts/analyze_filter_repair_remaining_svd_costs.py'),
    'srukf_svd': ('svd-cost-*.json',
        'scripts/analyze_filter_repair_svd_costs.py'),
    'kdm': ('kdm-cost-analysis-*.json',
        'scripts/analyze_filter_repair_kdm_cost.py'),
    'genut': ('genut-bounded-gradient-cost-analysis-*.json',
        'scripts/analyze_filter_repair_genut_gradient_cost.py'),
    'staged_locator': ('staged-public-cost-analysis-*.json', None),
    'single_locator': ('joint-public-cost-analysis-*.json', None),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_references(value):
    """Yield declared integer run references with their immediate hash fields."""
    if isinstance(value, dict):
        if type(value.get('run')) is int:
            yield value
        if type(value.get('reference_run')) is int:
            yield {'run': value['reference_run'], 'role': 'reference'}
        for child in value.values():
            yield from run_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from run_references(child)


def index_report(path, root, source_root, analyzer=None, hash_cache=None, payload=None):
    cache = {} if hash_cache is None else hash_cache

    def current_hash(relative):
        if relative not in cache:
            candidate = source_root / relative
            cache[relative] = sha(candidate) if candidate.is_file() else None
        return cache[relative]

    payload = path.read_bytes() if payload is None else payload
    report = json.loads(payload)
    references = list(run_references(report))
    # Some analyzers record an explicit file table instead of row-local hashes.
    file_tables = {}
    for item in report.get('files', []):
        relative = Path(item['path'])
        if len(relative.parts) == 2 and relative.parts[0].startswith('run-'):
            number = int(relative.parts[0][4:])
            file_tables.setdefault(number, {})[relative.name] = item['sha256']
    references.extend({'run': number, 'artifact_sha256': files}
        for number, files in file_tables.items())
    manifests, checks, changes = {}, [], {}
    for reference in references:
        number = reference['run']
        directory = root / f'run-{number:05d}'
        manifest_path = directory / 'run.json'
        row = {'run': number, 'manifest_exists': manifest_path.is_file()}
        if not manifest_path.is_file():
            row['integrity'] = 'missing_manifest'
            checks.append(row)
            continue
        if number not in manifests:
            manifests[number] = json.loads(manifest_path.read_text())
        run = manifests[number]
        row.update(state=run['state'], device=run['device'], group=run['key'][1])
        artifact_digests = reference.get('artifact_sha256', {})
        row['artifact_hash_matches'] = {name: (directory / name).is_file()
            and sha(directory / name) == digest for name, digest in artifact_digests.items()}
        manifest_digest = reference.get('manifest_sha256', artifact_digests.get('run.json'))
        result_digest = reference.get('result_sha256')
        row['manifest_hash_matches'] = (
            sha(manifest_path) == manifest_digest if manifest_digest else None)
        # A summary may omit the result filename. Search this run's direct
        # JSON files by exact digest, never infer one from the test name.
        row['matching_result_files'] = ([p.name for p in sorted(directory.glob('*.json'))
            if p.name != 'run.json' and sha(p) == result_digest] if result_digest else None)
        named_result = any(name.endswith('.json') and name != 'run.json'
            for name in artifact_digests)
        row['integrity'] = ('mismatch' if row['manifest_hash_matches'] is False
            or row['matching_result_files'] == []
            or not all(row['artifact_hash_matches'].values()) else
            'hash_fields_unrecorded' if not manifest_digest or not (result_digest or named_result) else
            'recorded_hashes_verified')
        for relative, recorded in run.get('source_sha256', {}).items():
            actual = current_hash(relative)
            if actual != recorded:
                change = changes.setdefault(relative, {'current_sha256': actual,
                    'recorded_sha256': set(), 'runs': set()})
                change['recorded_sha256'].add(recorded)
                change['runs'].add(number)
        checks.append(row)
    for change in changes.values():
        change['recorded_sha256'] = sorted(change['recorded_sha256'])
        change['runs'] = sorted(change['runs'])
    recorded_analyzer = report.get('analyzer_sha256')
    actual_analyzer = current_hash(analyzer) if analyzer else None
    mismatches = [row['run'] for row in checks if row['integrity'] in
        ('missing_manifest', 'mismatch')]
    return {'report': path.name, 'report_sha256': hashlib.sha256(payload).hexdigest(),
        'schema': report.get('schema'), 'contract': report.get('contract'),
        'analyzer_path': analyzer, 'recorded_analyzer_sha256': recorded_analyzer,
        'current_analyzer_sha256': actual_analyzer,
        'analyzer_hash_matches': (recorded_analyzer == actual_analyzer
            if recorded_analyzer and actual_analyzer else None),
        'run_references': checks, 'distinct_runs': sorted(manifests),
        'integrity_mismatch_runs': sorted(set(mismatches)),
        'changed_recorded_sources': dict(sorted(changes.items())),
        'status': 'integrity_failure' if mismatches else 'contextual_review_required',
        'limits': ['No numerical/performance gate was recomputed.',
            'Only explicit run references, file tables and recorded digests are verified.',
            'A report without references is unqualified by this index.',
            'Recorded source dictionaries are broad snapshots, not proven dependency closures.',
            'Current source differences do not by themselves identify a runtime regression.',
            'External dynamic callbacks and unrecorded dependencies are not certified.']}


def index_archived_report(archive, receipt_path, member, root, source_root, cache):
    """Read one report without extracting or modifying its preserved archive."""
    receipt = json.loads(receipt_path.read_text())
    if sha(archive) != receipt['archive_sha256']:
        raise ValueError('Evidence archive hash mismatch')
    with tarfile.open(archive, 'r:gz') as bundle:
        payload = bundle.extractfile(member).read()
    if hashlib.sha256(payload).hexdigest() != receipt['members'][member]:
        raise ValueError('Archived report hash mismatch')
    result = index_report(Path(member), root, source_root, hash_cache=cache, payload=payload)
    result['archive'] = archive.name
    result['archive_sha256'] = receipt['archive_sha256']
    result['archive_member'] = member
    return result


def build(root, source_root):
    families, cache = {}, {}
    for name, (pattern, analyzer) in FAMILIES.items():
        families[name] = [index_report(path, root, source_root, analyzer, cache)
            for path in sorted(root.glob(pattern))]
    archive_root = source_root / 'docs/plans/artifacts/filter-gradient-repair-20260917'
    archive = archive_root / 'joint-public-gpu-renewal-evidence-04368.tar.gz'
    receipt = archive_root / 'joint-public-gpu-renewal-verification-04368.json'
    archived_status = 'missing_archive_or_receipt'
    if archive.is_file() and receipt.is_file():
        families['single_locator'].append(index_archived_report(archive, receipt,
            'analysis/joint-public-cost-analysis-gpu-04368.json', root, source_root, cache))
        archived_status = 'archive_and_report_hashes_verified'
    return {'schema': 'filter_endpoint_evidence_index.v1', 'families': families,
        'archived_single_locator_GPU': archived_status,
        'status': 'diagnostic_inventory_only_no_terminal_admission',
        'report_count': sum(len(rows) for rows in families.values()),
        'nonclaims': ['Does not supersede failed or stale historical comparisons.',
            'Does not establish whole-repository coverage or authorize merging.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = build(args.root, args.source_root)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'report_count': report['report_count'], 'families': {
        name: {'reports': len(rows), 'integrity_failures': sum(
            row['status'] == 'integrity_failure' for row in rows)}
        for name, rows in report['families'].items()}}))


if __name__ == '__main__':
    main()
