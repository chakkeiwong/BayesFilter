from __future__ import annotations

import json

from scripts import build_complete_highdim_leaderboard_runtime_fingerprint as runtime


def test_runtime_fingerprint_has_required_runtime_payloads(tmp_path, monkeypatch) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "payload.bin").write_bytes(b"tree-payload")
    pinned = tmp_path / "pinned.bin"
    pinned.write_bytes(b"pinned-payload")
    cuda = tmp_path / "libcuda.so"
    cuda.write_bytes(b"cuda-payload")
    monkeypatch.setattr(runtime, "TREE_ROOTS", (("fixture", tree),))
    monkeypatch.setattr(runtime, "PINNED_FILES", (pinned,))
    monkeypatch.setattr(runtime, "_cuda_library_paths", lambda: [cuda])
    payload = runtime.build()

    assert payload["schema_version"] == runtime.SCHEMA
    assert all(entry["sha256"] for entry in payload["pinned_files"])
    assert payload["runtime_tree_bindings"][0]["tree_sha256"]
    assert payload["cuda_library_bindings"][0]["sha256"]
    package_names = {entry["requested_name"] for entry in payload["packages"]}
    assert {"tensorflow", "numpy", "scipy"}.issubset(package_names)
    assert any(name in package_names for name in {"tensorflow-probability", "tfp-nightly"})
    assert all(
        value is None
        for value in payload["inherited_loader_overrides_required_absent"].values()
    )

    output = tmp_path / "runtime.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    assert runtime.main(["--output", str(output), "--check"]) == 0
