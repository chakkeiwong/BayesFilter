#!/usr/bin/env python3
"""Build a bounded fingerprint for the pinned GPU and model runtimes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import stat
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA = "bayesfilter.complete_highdim_leaderboard.runtime_fingerprint.v1"
TF_ENV = Path("/home/chakwong/anaconda3/envs/tf-gpu")
NODE_RUNTIME = Path("/home/chakwong/.nvm/versions/node/v22.23.1")
PACKAGE_NAMES = (
    "tensorflow",
    "tensorflow-probability",
    "tfp-nightly",
    "tf-keras",
    "keras",
    "numpy",
    "scipy",
)
PINNED_FILES = (
    TF_ENV / "bin/python3.11",
    TF_ENV / "lib/python3.11/site-packages/tensorflow/__init__.py",
    TF_ENV / "lib/python3.11/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so",
    TF_ENV / "lib/python3.11/site-packages/tensorflow_probability/__init__.py",
    NODE_RUNTIME / "bin/node",
    NODE_RUNTIME / "lib/node_modules/@openai/codex/package.json",
    NODE_RUNTIME / "lib/node_modules/@openai/codex/bin/codex.js",
    NODE_RUNTIME / "lib/node_modules/@anthropic-ai/claude-code/package.json",
    NODE_RUNTIME / "lib/node_modules/@anthropic-ai/claude-code/bin/claude.exe",
    Path("/usr/lib/wsl/lib/libcuda.so.1.1"),
    Path("/usr/lib/wsl/lib/nvidia-smi"),
)
TREE_ROOTS = (
    ("tensorflow", TF_ENV / "lib/python3.11/site-packages/tensorflow"),
    (
        "tensorflow_probability",
        TF_ENV / "lib/python3.11/site-packages/tensorflow_probability",
    ),
    ("tf_keras", TF_ENV / "lib/python3.11/site-packages/tf_keras"),
    ("keras", TF_ENV / "lib/python3.11/site-packages/keras"),
    ("numpy", TF_ENV / "lib/python3.11/site-packages/numpy"),
    ("scipy", TF_ENV / "lib/python3.11/site-packages/scipy"),
    ("codex_package", NODE_RUNTIME / "lib/node_modules/@openai/codex"),
    (
        "claude_package",
        NODE_RUNTIME / "lib/node_modules/@anthropic-ai/claude-code",
    ),
)
CUDA_LIBRARY_GLOBS = (
    "targets/x86_64-linux/lib/libcudart.so.*",
    "targets/x86_64-linux/lib/libcublas.so.*",
    "targets/x86_64-linux/lib/libcublasLt.so.*",
    "targets/x86_64-linux/lib/libcufft.so.*",
    "targets/x86_64-linux/lib/libcusolver.so.*",
    "targets/x86_64-linux/lib/libcusparse.so.*",
    "lib/libcudnn.so.*",
    "lib/libcudnn_ops.so.*",
    "lib/libcudnn_cnn.so.*",
    "lib/libcudnn_adv.so.*",
    "lib/libcudnn_graph.so.*",
    "lib/libcudnn_heuristic.so.*",
    "lib/libcudnn_engines_precompiled.so.*",
    "lib/libcudnn_engines_runtime_compiled.so.*",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_binding(label: str, root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"runtime tree is missing: {root}")
    records: list[dict[str, Any]] = []
    total_bytes = 0
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name != "__pycache__")
        current_path = Path(current)
        for name in sorted(files):
            if name.endswith((".pyc", ".pyo")):
                continue
            path = current_path / name
            rel = path.relative_to(root).as_posix()
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                records.append(
                    {"path": rel, "kind": "symlink", "target": os.readlink(path)}
                )
            elif stat.S_ISREG(info.st_mode):
                records.append(
                    {
                        "path": rel,
                        "kind": "file",
                        "size": info.st_size,
                        "sha256": _sha256(path),
                    }
                )
                total_bytes += info.st_size
            else:
                raise ValueError(f"runtime tree contains a special file: {path}")
    canonical = json.dumps(records, separators=(",", ":"), sort_keys=True).encode()
    return {
        "label": label,
        "root": str(root),
        "entry_count": len(records),
        "total_file_bytes": total_bytes,
        "tree_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _cuda_library_paths() -> list[Path]:
    values = set()
    for pattern in CUDA_LIBRARY_GLOBS:
        values.update(path.resolve() for path in TF_ENV.glob(pattern) if path.is_file())
    return sorted(values)


def build(
    *,
    tree_roots: Iterable[tuple[str, Path]] | None = None,
    pinned_files: Iterable[Path] | None = None,
    cuda_library_paths: Iterable[Path] | None = None,
) -> dict[str, Any]:
    selected_tree_roots = TREE_ROOTS if tree_roots is None else tree_roots
    selected_pinned_files = PINNED_FILES if pinned_files is None else pinned_files
    packages = []
    for name in PACKAGE_NAMES:
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        metadata_path = Path(distribution._path)  # type: ignore[attr-defined]
        packages.append(
            {
                "requested_name": name,
                "canonical_name": distribution.metadata.get("Name"),
                "version": distribution.version,
                "metadata_path": str(metadata_path),
                "metadata_record_sha256": (
                    _sha256(metadata_path / "RECORD")
                    if (metadata_path / "RECORD").is_file()
                    else None
                ),
            }
        )
    files = []
    for path in selected_pinned_files:
        if not path.is_file():
            raise FileNotFoundError(f"runtime fingerprint file is missing: {path}")
        files.append(
            {"path": str(path), "size": path.stat().st_size, "sha256": _sha256(path)}
        )
    cuda_files = []
    for path in cuda_library_paths or _cuda_library_paths():
        if not path.is_file():
            raise FileNotFoundError(f"CUDA runtime file is missing: {path}")
        cuda_files.append(
            {"path": str(path), "size": path.stat().st_size, "sha256": _sha256(path)}
        )
    trees = [_tree_binding(label, root) for label, root in selected_tree_roots]
    node_versions = {}
    for package_path in (
        NODE_RUNTIME / "lib/node_modules/@openai/codex/package.json",
        NODE_RUNTIME / "lib/node_modules/@anthropic-ai/claude-code/package.json",
    ):
        payload = json.loads(package_path.read_text(encoding="utf-8"))
        node_versions[str(package_path)] = {
            "name": payload.get("name"),
            "version": payload.get("version"),
        }
    return {
        "schema_version": SCHEMA,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "tf_env": str(TF_ENV),
        "node_runtime": str(NODE_RUNTIME),
        "packages": packages,
        "pinned_files": files,
        "runtime_tree_bindings": trees,
        "cuda_library_bindings": cuda_files,
        "node_packages": node_versions,
        "inherited_loader_overrides_required_absent": {
            name: os.environ.get(name)
            for name in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD")
        },
        "scope": (
            "aggregate content-tree digests for TensorFlow, TFP, tf-keras, "
            "Keras, NumPy, SciPy, Codex, and Claude; exact selected CUDA/cuDNN "
            "and WSL driver-facing payload hashes; not a byte hash of unrelated "
            "packages in the full 16 GB conda environment"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    payload = build()
    if args.check:
        observed = json.loads(args.output.read_text(encoding="utf-8"))
        if observed != payload:
            raise ValueError("runtime fingerprint drifted")
        print(f"RUNTIME_FINGERPRINT_CHECK_PASS {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"RUNTIME_FINGERPRINT_WRITTEN {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
