"""CPU diagnostic only: exercise a clean C1 source candidate via Phase 9.

The process-local authority path and expected commit are explicit preview
inputs. This does not change the consumer's official runtime configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--consumer", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--row-tests", type=Path, required=True)
    parser.add_argument("--consumer-changes", type=Path)
    parser.add_argument("--bound-tests", type=Path,
                        help="Exercise the explicit public source-binding API without pin injection")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"classification": "test_only_source_admission_preview",
              "official_consumer_admission": False, "status": "RUNNING",
              "runner_sha256": digest(Path(__file__)),
              "command": [sys.executable, *sys.argv], "started_unix": time.time(),
              "python": sys.executable, "plan":
              "docs/plans/bayesfilter-modern-hmc-c2-exact-source-admission-plan-2026-10-28.md",
              "gpu_intentionally_hidden": True,
              "environment": {k: os.environ.get(k) for k in (
                  "CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH",
                  "PYTHONDONTWRITEBYTECODE", "TF_NUM_INTRAOP_THREADS",
                  "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS")},
              "nonclaims": ["posterior convergence", "ranking",
                             "default promotion", "GPU readiness", "release readiness"]}
    inputs = {}
    code = 1
    try:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
        assert os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") == "true"
        assert os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
        assert "tensorflow" not in sys.modules
        candidate = args.candidate.resolve()
        consumer_root = args.consumer.resolve()
        commit = git(candidate, "rev-parse", "HEAD")
        assert not git(candidate, "status", "--porcelain"), "candidate is dirty"
        result["candidate_commit"] = commit
        result["candidate_root"] = str(candidate)
        manifest = json.loads(args.snapshot.read_text())
        dependencies = json.loads(args.dependencies.read_text())["modules"]
        frozen_root = Path(manifest["source_root"])
        frozen_consumer = Path(manifest["consumer_root"])
        frozen = manifest["source_sha256"]
        changes = json.loads(args.consumer_changes.read_text()) if args.consumer_changes else {}
        result["consumer_changes"] = changes
        for relative, change in changes.items():
            live = consumer_root / relative
            assert digest(live) == change["candidate_sha256"], str(live)
            inputs[str(live)] = change["candidate_sha256"]
        assert all(digest(Path(p)) == h for p, h in frozen.items()), "frozen source drift"
        checked = {}
        for name, item in dependencies.items():
            path = candidate / item["path"]
            assert digest(path) == item["frozen_hash"], str(path)
            checked[name] = {"path": str(path), "sha256": digest(path)}
        result["c1_loaded_dependencies"] = checked
        for path, expected in frozen.items():
            path = Path(path)
            if path.is_relative_to(frozen_consumer):
                relative = path.relative_to(frozen_consumer)
                live = consumer_root / relative
                if str(relative) in changes:
                    change = changes[str(relative)]
                    assert change["c1_sha256"] == expected
                    expected = change["candidate_sha256"]
                assert digest(live) == expected, str(live)
                inputs[str(live)] = expected
        # Consumer-first also resolves the consumer's scripts namespace in its
        # existing test, while every BayesFilter origin is checked below.
        sys.path[:0] = [str(consumer_root), str(candidate)]
        import one_country_affine_macro_phase9_hmc as phase9
        import one_country_affine_macro_governance as governance
        result["original_authority"] = {"root": str(phase9.BAYESFILTER_AUTHORITY),
                                         "commit": governance.LIVE_BAYESFILTER_COMMIT}
        if args.bound_tests:
            from one_country_affine_macro_source_binding import Phase9SourceBinding
            binding = Phase9SourceBinding(candidate, commit)
            result["classification"] = "explicit_public_source_binding_integration"
            result["plan"] = "docs/plans/bayesfilter-modern-hmc-c2-scoped-binding-2026-10-28.md"
            result["sampler_transitions"] = 0
            result["global_pin_injected"] = False
            os.environ["C2_BINDING_ROOT"] = str(candidate)
            os.environ["C2_BINDING_COMMIT"] = commit
        else:
            phase9.BAYESFILTER_AUTHORITY = candidate
            governance.LIVE_BAYESFILTER_COMMIT = commit
            result["global_pin_injected"] = True
        result["source_gate_before"] = governance.audit_bayesfilter_authority(
            candidate, expected_commit=commit)
        assert result["source_gate_before"]["status"] == "PASS"
        tf, _kalman, _score, policy = (
            phase9._load_runtime(source_binding=binding) if args.bound_tests
            else phase9._load_runtime())
        result["tensorflow_version"] = tf.__version__
        result["gpu_policy"] = policy
        result["logical_gpus"] = [str(x) for x in tf.config.list_logical_devices("GPU")]
        assert not result["logical_gpus"]
        test_dir = args.output / "test_inputs"
        test_dir.mkdir()
        tests = ([args.bound_tests] if args.bound_tests else
                 [consumer_root / "tests/test_one_country_affine_macro_phase9_hmc.py", args.row_tests])
        copied = []
        for source in tests:
            destination = test_dir / source.name
            shutil.copyfile(source, destination)
            inputs[str(source)] = digest(source)
            assert digest(destination) == inputs[str(source)]
            copied.append(str(destination))
        os.environ["MACROFINANCE_ROOT"] = str(consumer_root)
        write(args.output / "manifest.json", result | {"test_inputs": inputs})
        import pytest

        class Progress:
            def __init__(self):
                self.tests = []
            def pytest_runtest_logreport(self, report):
                if report.when == "call" or report.failed or report.skipped:
                    self.tests.append({"test": report.nodeid, "when": report.when,
                                       "outcome": report.outcome, "duration": report.duration})
                    write(args.output / "progress.json", {
                        "elapsed_seconds": time.monotonic() - started, "tests": self.tests})

        progress = Progress()
        code = int(pytest.main(["-v", "--tb=short", "-p", "no:cacheprovider",
                               "--import-mode=importlib", f"--confcutdir={test_dir}",
                               f"--junitxml={args.output / 'junit.xml'}", *copied],
                              plugins=[progress]))
        result["tests"] = progress.tests
        result["pytest_exit_code"] = code
        imported = {}
        mismatches = []
        for name, module in list(sys.modules.items()):
            filename = getattr(module, "__file__", None)
            if not filename or not Path(filename).is_file():
                continue
            path = Path(filename).resolve()
            if name == "bayesfilter" or name.startswith("bayesfilter."):
                assert path.is_relative_to(candidate), f"import leakage: {name}: {path}"
                expected = frozen.get(str(frozen_root / path.relative_to(candidate)))
                actual = digest(path)
                imported[name] = {"path": str(path), "sha256": actual,
                                  "c1_sha256": expected}
                if actual != expected:
                    mismatches.append(name)
            elif path.is_relative_to(consumer_root):
                imported[name] = {"path": str(path), "sha256": digest(path)}
        result["imported_modules"] = imported
        result["import_mismatches"] = mismatches
        assert not mismatches, f"additional source mismatch: {mismatches}"
        assert all(digest(candidate / x["path"]) == x["frozen_hash"]
                   for x in dependencies.values()), "candidate dependency drift"
        assert all(digest(Path(p)) == h for p, h in inputs.items()), "consumer/test drift"
        result["source_gate_after"] = governance.audit_bayesfilter_authority(
            candidate, expected_commit=commit)
        assert result["source_gate_after"]["status"] == "PASS"
        if args.bound_tests:
            assert str(phase9.BAYESFILTER_AUTHORITY) == result["original_authority"]["root"]
            assert governance.LIVE_BAYESFILTER_COMMIT == result["original_authority"]["commit"]
            result["legacy_configuration_unchanged"] = True
        result["status"] = "PASS" if code == 0 else "TEST_FAILURE"
    except Exception:
        result["status"] = "ERROR"
        result["exception"] = traceback.format_exc()
        traceback.print_exc()
        code = 1
    finally:
        result["inclusive_elapsed_seconds"] = time.monotonic() - started
        result["finished_unix"] = time.time()
        write(args.output / "result.json", result)
    print(json.dumps({k: result.get(k) for k in (
        "status", "candidate_commit", "inclusive_elapsed_seconds", "pytest_exit_code")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
