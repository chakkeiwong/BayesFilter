#!/usr/bin/env python3
"""Pre-load runtime guard for the offline Gate B R3 localizer and its tests."""

from __future__ import annotations

import argparse
import builtins
import io
import os
import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Sequence


ROOT = Path(__file__).resolve().parents[2]
LOCALIZER_PATH = (
    ROOT
    / "docs/benchmarks/localize_kalman_qr_phase6_gateb_r3_autodiff_structure_2026_07_13.py"
).resolve()
TEST_PATH = (
    ROOT / "tests/test_kalman_qr_phase6_gateb_r3_autodiff_structure_localization.py"
).resolve()
SCRATCH_ROOT = Path(
    "/tmp/kalman_qr_phase6_gateb_r3_autodiff_structure_localization"
).resolve()
DURABLE_OUTPUT = (
    ROOT
    / "docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_"
    "autodiff_structure_localization_2026-07-13.json"
).resolve()
LOCALIZER_OUTPUTS = {
    (SCRATCH_ROOT / "run1.json").resolve(),
    (SCRATCH_ROOT / "run2.json").resolve(),
    DURABLE_OUTPUT,
}
GUARD_SLOT = "_kalman_qr_phase6_autodiff_structure_guard_20260713"
FORBIDDEN_IMPORT_PREFIXES = (
    "tensorflow",
    "bayesfilter",
    "scripts.benchmark_kalman_qr_parameter_count_scaling",
    "scripts.kalman_qr_benchmark_contract",
    "docs.benchmarks.diagnose_kalman_qr_phase6_gateb_r3_trace_rejections_2026_07_12",
    "docs.benchmarks.run_kalman_qr_batched_xla_repair_2026_07_11",
)
DEVICE_PATH_PREFIXES = (
    "/dev/dri",
    "/dev/kfd",
    "/proc/driver/nvidia",
    "/sys/bus/pci/drivers/nvidia",
    "/sys/class/drm",
    "/sys/class/kfd",
    "/sys/module/nvidia",
)
WRITE_MODE_CHARACTERS = frozenset("wax+")


class GuardViolation(RuntimeError):
    """Raised when guarded phase code crosses an offline boundary."""


def _is_forbidden_import(name: str) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_IMPORT_PREFIXES)


def check_import(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise GuardViolation("dynamic import name must be a nonempty string")
    if _is_forbidden_import(name):
        raise GuardViolation(f"forbidden import: {name}")


def _coerce_path(value: Any) -> Path | None:
    if isinstance(value, int):
        return None
    try:
        return Path(os.fspath(value)).resolve()
    except (TypeError, ValueError, OSError) as error:
        raise GuardViolation(f"unresolvable filesystem path: {value!r}") from error


def check_device_path(value: Any) -> None:
    path = _coerce_path(value)
    if path is None:
        return
    text = str(path)
    nvidia_device = path.parent == Path("/dev") and path.name.startswith("nvidia")
    if nvidia_device or any(
        text == prefix or text.startswith(prefix + "/")
        for prefix in DEVICE_PATH_PREFIXES
    ):
        raise GuardViolation(f"device access is forbidden: {path}")


def check_write(value: Any, allowed_outputs: frozenset[Path]) -> None:
    path = _coerce_path(value)
    if path is None:
        raise GuardViolation("raw file-descriptor writes are forbidden")
    if path not in allowed_outputs:
        raise GuardViolation(f"write outside exact invocation output: {path}")


def _blocked(operation: str) -> Callable[..., Any]:
    def fail(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise GuardViolation(f"forbidden guarded operation: {operation}")

    fail.__name__ = f"blocked_{operation.replace('.', '_')}"
    setattr(fail, "__kalman_qr_guard_blocked__", True)
    return fail


def _patch_attributes(module: ModuleType, names: Sequence[str], namespace: str) -> None:
    for name in names:
        if hasattr(module, name):
            setattr(module, name, _blocked(f"{namespace}.{name}"))


def _patch_sensitive_module(name: str, module: ModuleType) -> None:
    top = name.split(".", 1)[0]
    if top == "subprocess":
        _patch_attributes(
            module,
            ("Popen", "run", "call", "check_call", "check_output", "getoutput", "getstatusoutput"),
            "subprocess",
        )
    elif top == "socket":
        _patch_attributes(
            module,
            ("socket", "socketpair", "create_connection", "create_server", "fromfd"),
            "socket",
        )
    elif top == "multiprocessing":
        _patch_attributes(
            module,
            ("Process", "Pool", "Manager", "Pipe", "Queue", "SimpleQueue", "get_context"),
            "multiprocessing",
        )
    elif name == "concurrent.futures.process":
        _patch_attributes(module, ("ProcessPoolExecutor",), "concurrent.futures.process")


def _assert_no_tensorflow_modules() -> None:
    imported = sorted(
        name for name in sys.modules if name == "tensorflow" or name.startswith("tensorflow.")
    )
    if imported:
        raise GuardViolation(f"TensorFlow entered sys.modules: {imported[:5]}")


def install_guards(mode: str, output_path: Path | None) -> dict[str, Any]:
    if mode not in {"localizer", "test"}:
        raise GuardViolation(f"unknown guard mode: {mode}")
    if hasattr(builtins, GUARD_SLOT):
        raise GuardViolation("guard slot was already installed")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise GuardViolation("CUDA_VISIBLE_DEVICES must equal -1")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise GuardViolation("PYTHONDONTWRITEBYTECODE must equal 1")
    if mode == "test" and os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") != "1":
        raise GuardViolation("PYTEST_DISABLE_PLUGIN_AUTOLOAD must equal 1")
    _assert_no_tensorflow_modules()

    if mode == "localizer":
        if output_path is None or output_path.resolve() not in LOCALIZER_OUTPUTS:
            raise GuardViolation("localizer mode requires one exact authorized output")
        allowed_outputs = frozenset({output_path.resolve()})
    else:
        if output_path is not None:
            raise GuardViolation("test mode forbids an output path")
        allowed_outputs = frozenset()

    original_import = builtins.__import__
    original_open = builtins.open
    original_io_open = io.open
    original_os_open = os.open
    original_run_path = runpy.run_path
    run_path_count = {"value": 0}

    def guarded_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        package = globals.get("__package__", "") if isinstance(globals, dict) else ""
        if level:
            if not isinstance(package, str) or not package:
                raise GuardViolation("relative import lacks a package boundary")
            check_import(package)
        else:
            check_import(name)
        module = original_import(name, globals, locals, fromlist, level)
        absolute_name = getattr(module, "__name__", name)
        _patch_sensitive_module(absolute_name, module)
        for candidate_name in ("subprocess", "socket", "multiprocessing", "concurrent.futures.process"):
            candidate = sys.modules.get(candidate_name)
            if isinstance(candidate, ModuleType):
                _patch_sensitive_module(candidate_name, candidate)
        return module

    def guarded_open(
        file: Any,
        mode_text: str = "r",
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        check_device_path(file)
        if any(character in mode_text for character in WRITE_MODE_CHARACTERS):
            check_write(file, allowed_outputs)
        return original_open(file, mode_text, *args, **kwargs)

    def guarded_io_open(
        file: Any,
        mode_text: str = "r",
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        check_device_path(file)
        if any(character in mode_text for character in WRITE_MODE_CHARACTERS):
            check_write(file, allowed_outputs)
        return original_io_open(file, mode_text, *args, **kwargs)

    def guarded_os_open(file: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        check_device_path(file)
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        if flags & write_flags:
            check_write(file, allowed_outputs)
        return original_os_open(file, flags, *args, **kwargs)

    def guarded_run_path(
        path_name: str,
        init_globals: dict[str, Any] | None = None,
        run_name: str | None = None,
    ) -> dict[str, Any]:
        path = Path(path_name).resolve()
        expected_run_name = (
            "__main__"
            if mode == "localizer"
            else "kalman_qr_phase6_autodiff_structure_test_subject"
        )
        if path != LOCALIZER_PATH or run_name != expected_run_name or init_globals is not None:
            raise GuardViolation("runpy target or invocation shape is not authorized")
        if run_path_count["value"] != 0:
            raise GuardViolation("localizer may be loaded exactly once")
        run_path_count["value"] += 1
        return original_run_path(path_name, init_globals=init_globals, run_name=run_name)

    builtins.__import__ = guarded_import
    builtins.open = guarded_open
    io.open = guarded_io_open
    os.open = guarded_os_open
    runpy.run_path = guarded_run_path
    _patch_attributes(
        os,
        tuple(
            name
            for name in (
                "system",
                "popen",
                "fork",
                "forkpty",
                "kill",
                "killpg",
                "posix_spawn",
                "posix_spawnp",
                "spawnl",
                "spawnle",
                "spawnlp",
                "spawnlpe",
                "spawnv",
                "spawnve",
                "spawnvp",
                "spawnvpe",
                "execv",
                "execve",
                "execvp",
                "execvpe",
            )
            if hasattr(os, name)
        ),
        "os",
    )
    _patch_attributes(
        os,
        tuple(
            name
            for name in (
                "chmod",
                "chown",
                "fchmod",
                "fchown",
                "link",
                "makedirs",
                "mkdir",
                "mknod",
                "remove",
                "removedirs",
                "rename",
                "renames",
                "replace",
                "rmdir",
                "symlink",
                "truncate",
                "unlink",
            )
            if hasattr(os, name)
        ),
        "os.filesystem_mutation",
    )
    _patch_attributes(
        os,
        tuple(
            name
            for name in ("pwrite", "write", "writev")
            if hasattr(os, name)
        ),
        "os.raw_write",
    )
    for module_name, module in list(sys.modules.items()):
        if isinstance(module, ModuleType):
            _patch_sensitive_module(module_name, module)

    state = {
        "token": "guard_installed_before_subject_load_v1",
        "mode": mode,
        "allowed_outputs": tuple(sorted(str(path) for path in allowed_outputs)),
        "check_import": check_import,
        "check_device_path": check_device_path,
        "check_write": lambda path: check_write(path, allowed_outputs),
        "blocked_os_system": bool(
            getattr(getattr(os, "system", None), "__kalman_qr_guard_blocked__", False)
        ),
        "blocked_os_write": bool(
            getattr(getattr(os, "write", None), "__kalman_qr_guard_blocked__", False)
        ),
        "run_path_count": run_path_count,
        "localizer_path": str(LOCALIZER_PATH),
    }
    setattr(builtins, GUARD_SLOT, state)
    return state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("localizer", "test"))
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    output_path = args.output_json.resolve() if args.output_json is not None else None
    state = install_guards(args.mode, output_path)

    if args.mode == "localizer":
        assert output_path is not None
        sys.argv = [str(LOCALIZER_PATH), "--output-json", str(output_path)]
        try:
            runpy.run_path(str(LOCALIZER_PATH), run_name="__main__")
        except SystemExit as error:
            if error.code not in (None, 0):
                raise
    else:
        import pytest

        logging_module = sys.modules.get("logging")
        pytest_logging = sys.modules.get("_pytest.logging")
        if not isinstance(logging_module, ModuleType) or not isinstance(
            pytest_logging, ModuleType
        ):
            raise GuardViolation("pytest logging modules were not loaded as expected")

        class _MemoryFileHandler(logging_module.StreamHandler):
            def __init__(
                self,
                filename: str,
                mode: str = "a",
                encoding: str | None = None,
                delay: bool = False,
                errors: str | None = None,
            ) -> None:
                del filename, mode, encoding, delay, errors
                super().__init__(io.StringIO())

        pytest_logging._FileHandler = _MemoryFileHandler
        exit_code = pytest.main(
            [str(TEST_PATH), "-p", "no:cacheprovider", "--capture=no"]
        )
        if exit_code != pytest.ExitCode.OK:
            raise GuardViolation(f"focused pytest failed with exit code {exit_code}")

    if state["run_path_count"]["value"] != 1:
        raise GuardViolation("subject localizer load count was not exactly one")
    _assert_no_tensorflow_modules()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
