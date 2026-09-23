"""Optional host profiling; instrumentation cannot change numerical outcomes.

Use the execution flag for comparisons so the scientific design and seeds stay
identical. A host profile includes Python work and synchronization, not device
kernel timing. Profile availability is separate from numerical completion.
"""
from __future__ import annotations

from pathlib import Path
import warnings

from .storage import file_hash, read_json, write_json


def _warn(message):
    # A caller may promote warnings to exceptions. Instrumentation still must
    # not mask the numerical result or its original exception.
    try:
        warnings.warn(message, RuntimeWarning)
    except Exception:
        pass


class HostProfile:
    def __init__(self, prefix, *, requested, scope):
        self.prefix = str(prefix)
        self.requested = requested
        self.scope = scope
        self.profile = None
        self.error = None

    def start(self):
        if self.requested:
            try:
                import cProfile
                self.profile = cProfile.Profile()
                # TensorFlow's C-extension dispatch can leave the default
                # builtins=True profiler state without Python return events.
                # Python frames are the useful attribution boundary here;
                # device/native timing is explicitly outside this profile.
                self.profile.enable(builtins=False)
            except Exception as exc:
                self.error = f"profile setup failed: {type(exc).__name__}: {exc}"

    def finish(self):
        if not self.requested:
            return
        path = self.prefix + "-host.prof"
        record = {"requested": True, "scope": self.scope, "method": "cProfile",
                  "path": path, "status": "unavailable"}
        try:
            if self.profile is not None:
                self.profile.disable()
                if self.error is None:
                    self.profile.dump_stats(path)
                    record.update(status="available", sha256=file_hash(path))
        except Exception as exc:
            self.error = f"profile persistence failed: {type(exc).__name__}: {exc}"
        if self.error is not None:
            record["reason"] = self.error
            _warn("host profile unavailable: " + self.error)
        try:
            write_json(self.prefix + "-profile.json", record)
        except Exception as exc:
            _warn(f"host profile status unavailable: {exc}")


def inspect_profile(prefix):
    """Read existing evidence only, including missing/changed historical files."""
    prefix = str(prefix)
    path = Path(prefix + "-host.prof")
    record = {"path": str(path), "status": "unavailable"}
    try:
        saved = read_json(prefix + "-profile.json")
        if saved["status"] != "available":
            record["reason"] = saved.get("reason", "profile was not saved")
        elif not path.is_file():
            record["reason"] = "saved profile is missing"
        elif saved["sha256"] != file_hash(path):
            record["reason"] = "saved profile checksum changed"
        else:
            record.update(status="available", sha256=saved["sha256"], scope=saved["scope"])
    except Exception as exc:
        record["reason"] = f"profile evidence unavailable: {type(exc).__name__}: {exc}"
    return record


def profile_report(root, *, isolated, requested):
    """Describe attempt profiles without rerunning or rewriting numerical fits."""
    report = {"requested": requested, "status": "not_requested", "profiles": []}
    if not requested:
        return report
    try:
        root = Path(root)
        if isolated:
            receipts = sorted(root.glob("replication-*/process-attempt-*-exit.json"))
            prefixes = [str(p).removesuffix("-exit.json") for p in receipts]
        else:
            prefixes = sorted({str(p).removesuffix(suffix)
                               for suffix in ("-result.json", "-failure.json")
                               for p in root.glob("attempt-*" + suffix)})
        report["profiles"] = [inspect_profile(prefix) for prefix in prefixes]
        available = sum(row["status"] == "available" for row in report["profiles"])
        report["status"] = ("available" if prefixes and available == len(prefixes) else
                            "partially_available" if available else "unavailable")
        if not prefixes:
            report["reason"] = "no completed process attempts with profiling evidence"
    except Exception as exc:
        report.update(status="unavailable", reason=f"profile inspection failed: {exc}")
    return report
