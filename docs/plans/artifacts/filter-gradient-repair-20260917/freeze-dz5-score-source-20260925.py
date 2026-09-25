"""Freeze committed BF code plus unchanged diagnostic MF source/inputs."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
BF = Path('/home/ubuntu/workspace/BayesFilter')
RAW = BF / 'docs/plans/artifacts/filter-gradient-repair-20260917'
PREVIOUS = RAW / 'dz5-candidate-source-merged-9d8202b77-r2'
DEST = RAW / 'dz5-candidate-source-score-4c37f9f40-r1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parent = json.loads((PREVIOUS / 'manifest.json').read_text())
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    assert head.startswith('4c37f9f40')
    assert not subprocess.check_output(['git', 'diff', 'HEAD', '--', 'bayesfilter'], cwd=REPO)
    DEST.mkdir(exist_ok=False)
    manifest = {**parent, 'role': 'fresh_score_oracle_snapshot_not_admitted',
        'bayesfilter_commit': head, 'sources': {}, 'inputs': {},
        'previous_snapshot_manifest': str(PREVIOUS / 'manifest.json'),
        'previous_snapshot_manifest_sha256': sha(PREVIOUS / 'manifest.json'),
        'builder_sha256': sha(Path(__file__)),
        'bayesfilter_dirty_status': subprocess.check_output(['git', 'status', '--short'], cwd=REPO, text=True).splitlines(),
        'source_discovery': 'All committed BF package files; unchanged frozen MF closure/inputs; preserved import-only native binary.',
        'actual_loaded_module_audit': 'pending new isolated workers',
        'adapter_installed': False}
    manifest.pop('supersedes_failed_snapshot', None)

    def copy(original, source, field):
        saved = DEST / original.lstrip('/')
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, saved)
        manifest[field][original] = {'saved': str(saved), 'sha256': sha(saved), 'origin': str(source)}

    for field in ('sources', 'inputs'):
        for original, entry in parent[field].items():
            if field == 'sources' and original.startswith(str(BF) + '/') and not original.endswith('.so'):
                continue
            source = PREVIOUS / original.lstrip('/')
            assert sha(source) == entry['sha256']
            copy(original, source, field)
    files = subprocess.check_output(['git', 'ls-files', 'bayesfilter', 'AGENTS.md',
        'docs/reference/neutra-implementation.md'], cwd=REPO, text=True).splitlines()
    for relative in files:
        source = REPO / relative
        committed = subprocess.check_output(['git', 'show', f'{head}:{relative}'], cwd=REPO)
        assert source.read_bytes() == committed
        copy(str(BF / relative), source, 'sources')
    qualifier = '/home/ubuntu/workspace/MacroFinance-dz5-neutra/scripts/qualify_dz5_cdf_runtime.py'
    manifest['frozen_qualifier_sha256'] = manifest['sources'][qualifier]['sha256']
    (DEST / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'snapshot': str(DEST), 'source_count': len(manifest['sources']),
        'input_count': len(manifest['inputs']), 'manifest_sha256': sha(DEST / 'manifest.json')}))


if __name__ == '__main__':
    main()
