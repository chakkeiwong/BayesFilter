from pathlib import Path
import hashlib
import json
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
evidence = root / 'docs/plans/artifacts/replication-branch-sync-20261014'
inventory = json.loads((evidence / 'final-scoped-inventory.json').read_text())
scope = {record['path'] for record in inventory['files']}
scope.add('docs/plans/artifacts/replication-branch-sync-20261014/final-scoped-inventory.json')
integration_path = evidence / 'integration-receipt.json'
final_documents = json.loads(integration_path.read_text()).get('final_documentation_sha256', {}) if integration_path.exists() else {}
allowed_documents = {
    'docs/plans/replication-branch-synchronization-20261014.md',
    'docs/plans/replication-sync-recovery-checkpoint-20261014.md',
    'docs/plans/replication-sync-recovery-result-20261014.md',
}
assert not (set(final_documents) - allowed_documents)
hash_failures = [record['path'] for record in inventory['files'] if record['path'] != '.gitignore' and hashlib.sha256((root / record['path']).read_bytes()).hexdigest() != final_documents.get(record['path'], record['sha256'])]
assert not hash_failures, hash_failures
manifest = json.loads((evidence / 'closure-recheck-complete/manifest.json').read_text())
runtime_failures = [path for path, checksum in manifest['current_runtime_sha256'].items() if hashlib.sha256((root / path).read_bytes()).hexdigest() != checksum]
assert not runtime_failures, runtime_failures
def tree(revision):
    output = subprocess.check_output(['git', 'ls-tree', '-r', '-z', revision], cwd=root)
    return {record.split(b'\t', 1)[1].decode(): record.split(b'\t', 1)[0].decode() for record in output.split(b'\0') if record}
incoming_commit = subprocess.check_output(['git', 'rev-parse', 'origin/main'], cwd=root, text=True).strip()
incoming = tree(incoming_commit)
merged = tree('HEAD')
owner_paths = sorted(set(incoming) - scope)
owner_failures = [path for path in owner_paths if incoming[path] != merged.get(path)]
assert not owner_failures, owner_failures
original_ignore = subprocess.check_output(['git', 'show', incoming_commit + ':.gitignore'], cwd=root, text=True).splitlines()
merged_ignore = (root / '.gitignore').read_text().splitlines()
assert not (set(original_ignore) - set(merged_ignore))
implementation_commit = Path('/tmp/bayesfilter-replication-sync-20261014/implementation-commit.txt').read_text().strip()
campaign_ignore = subprocess.check_output(['git', 'show', implementation_commit + ':.gitignore'], cwd=root, text=True).splitlines()
assert not (set(campaign_ignore) - set(merged_ignore))
receipt = dict(
    schema='replication_merge_preservation_check_v1',
    command=[sys.executable, '-B', *sys.argv],
    cwd=str(root),
    checked_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    preserved_incoming_commit=incoming_commit,
    scoped_hashes_checked=len(inventory['files']) - 1,
    ignore_resolution='additive_both_original_rule_sets_preserved',
    numerical_runtime_hashes_checked=len(manifest['current_runtime_sha256']),
    noncampaign_incoming_tree_entries_preserved=len(owner_paths),
    incoming_ignore_lines_preserved=True,
    final_documentation_hashes_checked=len(final_documents),
    failures=[],
)
Path(sys.argv[2]).write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
