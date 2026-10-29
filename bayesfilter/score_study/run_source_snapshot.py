"""Capture repository Python sources, including dirty and untracked modules."""

import hashlib
import json
from pathlib import Path
import subprocess


def capture_source_snapshot(repository, destination):
    """Preserve the whole Python package rather than guess its transitive imports."""
    repository = Path(repository).resolve()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    paths = sorted((repository / "bayesfilter").rglob("*.py"))
    if not paths:
        raise ValueError("No BayesFilter Python sources found")
    hashes = {}
    for path in paths:
        relative = path.relative_to(repository)
        contents = path.read_bytes()
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
        hashes[str(relative)] = hashlib.sha256(contents).hexdigest()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    for name, command in (
        ("tracked-source.diff", ["git", "diff", "--binary", "HEAD", "--", "bayesfilter"]),
        ("source-status.txt", ["git", "status", "--porcelain", "--untracked-files=all", "--", "bayesfilter"]),
    ):
        (destination / name).write_bytes(subprocess.check_output(command, cwd=repository))
    result = dict(commit=commit, scope="all bayesfilter Python sources, including untracked files",
                  path=str(destination), source_hashes=hashes)
    (destination / "index.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def verify_source_snapshot(repository, snapshot):
    """Report changes since capture without altering either source or snapshot."""
    repository = Path(repository).resolve()
    changed = []
    missing = []
    for relative, digest in snapshot["source_hashes"].items():
        path = repository / relative
        if not path.is_file():
            missing.append(relative)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            changed.append(relative)
    additional = sorted(str(path.relative_to(repository))
        for path in (repository / "bayesfilter").rglob("*.py")
        if str(path.relative_to(repository)) not in snapshot["source_hashes"])
    return dict(unchanged=not (changed or missing or additional), changed=changed,
                missing=missing, additional=additional)
