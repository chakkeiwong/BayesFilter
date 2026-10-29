"""Resumable host-side q20 timing records; no posterior or tuning authority."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import time

from bayesfilter.inference.q20_production_config import digest, write_json
from bayesfilter.inference.q20_stage_budget import StageBudgetPause


def resource_observation(probe):
    if probe is None:
        return {"quality": "cpu_reference_gpu_hidden"}
    from bayesfilter.inference.q20_gpu_runtime import GPUResourceUnavailable
    try:
        return {"quality": "no_contention_observed", "receipt": probe()}
    except GPUResourceUnavailable as error:
        return {"quality": "contention_observed", "receipt": error.receipt}
    except Exception as error:
        return {"quality": "resource_history_unknown", "error": type(error).__name__, "message": str(error)}


class PricingLedger:
    """Atomic measurement blocks with identities, original provenance and bounds."""
    def __init__(self, root, identity, *, resume=None, deadline=None, probe=None):
        self.root, self.identity = Path(root), identity
        self.deadline, self.probe = deadline, probe
        if resume is not None:
            shutil.copytree(resume, self.root)
        else:
            self.root.mkdir(parents=True, exist_ok=False)
            write_json(self.root / "identity.json", {"identity": identity, "sha256": digest(identity)})
        saved = json.loads((self.root / "identity.json").read_text())
        if saved != {"identity": identity, "sha256": digest(identity)}:
            raise ValueError("pricing ledger scope changed")

    def record(self, key, inputs):
        path = self.root / (digest(key) + ".json")
        if not path.exists():
            return None
        row = json.loads(path.read_text())
        checksum = row.pop("sha256")
        if checksum != digest(row) or row["key"] != key or row["inputs"] != inputs:
            raise ValueError("pricing block checksum or inputs changed: " + key)
        if row["identity_sha256"] != digest(self.identity):
            raise ValueError("pricing block identity changed: " + key)
        return row

    def run(self, key, inputs, compute, *, reserve_seconds=0., imported=None):
        saved = self.record(key, inputs)
        if saved is not None:
            return saved["value"]
        if not math.isfinite(reserve_seconds) or reserve_seconds < 0:
            raise ValueError("invalid pricing block reserve")
        if imported is not None:
            value, origin = imported
            row = {"timing_quality": origin["timing_quality"], "origin": origin,
                   "measured_here": False, "wall_seconds": 0.}
        else:
            if self.deadline is not None and time.monotonic()+reserve_seconds >= self.deadline:
                write_json(self.root / "pending.json", {"key": key, "inputs": inputs,
                    "required_seconds": reserve_seconds, "reason": "next_measurement_unfunded"}, exclusive=False)
                raise StageBudgetPause("next pricing block exceeds remaining allocation: " + key)
            before = resource_observation(self.probe)
            if before["quality"] == "contention_observed":
                from bayesfilter.inference.q20_gpu_runtime import GPUResourceUnavailable
                raise GPUResourceUnavailable(before["receipt"])
            started = time.monotonic()
            value = compute()
            seconds = time.monotonic()-started
            # Save the computed block before the terminal observation. A killed
            # or unavailable observer must not erase completed numerical work.
            row = {"timing_quality": "resource_history_unknown", "resource_before": before,
                   "measured_here": True, "wall_seconds": seconds}
            self._write(key, inputs, value, row)
            after = resource_observation(self.probe)
            quality = ("contention_observed" if after["quality"] == "contention_observed" else
                       before["quality"] if before["quality"] == after["quality"] else "resource_history_unknown")
            row.update(resource_after=after, timing_quality=quality)
        self._write(key, inputs, value, row)
        pending = self.root / "pending.json"
        if pending.exists() and json.loads(pending.read_text())["key"] == key:
            pending.unlink()
        return value

    def _write(self, key, inputs, value, details):
        row = {"key": key, "inputs": inputs, "identity_sha256": digest(self.identity),
               "value": value, **details}
        write_json(self.root / (digest(key)+".json"), {**row, "sha256": digest(row)}, exclusive=False)


def checked_historical_price(receipt_path, config, current_root):
    """Inspect a reviewed import, preserving the source and resource qualifications."""
    from bayesfilter.inference.q20_training_resume import check_refresh_sources
    receipt = json.loads(Path(receipt_path).read_text())
    raw = Path(receipt["result_path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt["result_sha256"]:
        raise ValueError("historical pricing changed")
    result = json.loads(raw)
    if result["config_hash"] != digest(config):
        raise ValueError("historical pricing protocol changed")
    current, changed = check_refresh_sources(result["sources"], receipt["source_root"], current_root)
    if current != receipt["current_sources"] or changed != receipt["reviewed_changed_paths"]:
        raise ValueError("pricing source import differs from reviewed diff")
    if receipt["timing_quality"] not in {"historical_no_contention_observed", "resource_history_unknown", "contention_observed"}:
        raise ValueError("invalid historical timing qualification")
    return result, {k: receipt[k] for k in ("result_path", "result_sha256", "source_root", "timing_quality")}
