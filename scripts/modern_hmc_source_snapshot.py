"""Ordinary source copies and import checks for local HMC diagnostics."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

CONSUMER_FILES = (
    'one_country_affine_macro_phase9_hmc.py',
    'one_country_affine_macro_governance.py',
    'one_country_affine_macro_transition_2x2.py',
    'one_country_affine_macro_prior_measurement_contract.py',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(repo, consumer, destination, *, extra_paths=()):
    """Copy bytes, then verify both ends; never edit the original checkout."""
    repo, consumer, destination = map(lambda p: Path(p).resolve(), (repo, consumer, destination))
    destination.mkdir(parents=True, exist_ok=False)
    source, external = destination/'BayesFilter', destination/'MacroFinance'
    origins = [p for directory in ('bayesfilter', 'scripts')
               for p in (repo/directory).rglob('*')
               if p.is_file() and p.suffix in ('.py', '.so')]
    origins += [repo/p for p in extra_paths]
    origins += [consumer/p for p in CONSUMER_FILES]
    expected = {str(p): digest(p) for p in sorted(set(origins))}
    hashes = {}
    for name, sha in expected.items():
        origin = Path(name)
        target = source/origin.relative_to(repo) if origin.is_relative_to(repo) else external/origin.relative_to(consumer)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target)
        hashes[str(target)] = sha
    if any(digest(p) != sha for p, sha in {**expected, **hashes}.items()):
        raise ValueError('concurrent source edit during snapshot; preserve and capture again')
    return dict(schema='bayesfilter.modern_hmc_source_copy.v1', source_root=str(source),
        consumer_root=str(external), original_root=str(repo), original_consumer_root=str(consumer),
        source_sha256=hashes, original_sha256=expected,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        provenance='Git commit plus exact copied working-tree bytes; dirty dependencies recorded')


def validate_snapshot(manifest, *, expected_root):
    source, external = Path(manifest['source_root']), Path(manifest['consumer_root'])
    if source.resolve() != Path(expected_root).resolve():
        raise ValueError('worker is not executing its declared source copy')
    paths = [Path(p) for p in manifest['source_sha256']]
    for path in paths:
        if not (path.is_relative_to(source) or path.is_relative_to(external)):
            raise ValueError('source outside snapshot')
        if digest(path) != manifest['source_sha256'][str(path)]:
            raise ValueError('snapshot source changed: '+str(path))
    return paths


def audit_imports(manifest, *, modules=None):
    """Check actual local imports, including scripts imported by bare name."""
    source, external = Path(manifest['source_root']), Path(manifest['consumer_root'])
    script_names = {Path(p).stem for p in manifest['source_sha256']
                    if Path(p).parent == source/'scripts'}
    loaded = {}
    for name, module in (sys.modules if modules is None else modules).items():
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        path = Path(filename).resolve()
        local = (name == 'bayesfilter' or name.startswith(('bayesfilter.', 'scripts.',
                 'one_country_affine_macro_')) or name in script_names)
        if local:
            if not (path.is_relative_to(source) or path.is_relative_to(external)):
                raise ValueError('local import escaped source copy: '+name+' -> '+str(path))
            if str(path) not in manifest['source_sha256'] or digest(path) != manifest['source_sha256'][str(path)]:
                raise ValueError('loaded source missing or changed: '+str(path))
            loaded[name] = str(path)
    return loaded
