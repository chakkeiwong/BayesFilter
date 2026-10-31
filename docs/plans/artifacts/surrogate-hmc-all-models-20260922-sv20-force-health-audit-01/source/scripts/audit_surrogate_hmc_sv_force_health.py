#!/usr/bin/env python3
"""CPU-only audit of preserved original-force telemetry in the active SV20 run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start, cpu = time.monotonic(), time.process_time()
    parent_manifest = json.loads((run/"run-manifest.json").read_text())
    plan = "docs/plans/surrogate-hmc-sv20-force-health-audit-2026-09-22.md"
    report = dict(status="RUNNING", retained_original_force_healthy=None, records={})
    save(out/"run-manifest.json", dict(command=[sys.executable, *sys.argv],
        environment=sys.prefix, git_commit=parent_manifest["git_commit"],
        cpu_only=True, gpu_intentionally_hidden=True, plan=plan,
        source_sha256=digest(__file__), observed_run=str(run), result=str(out/"report.json")))
    save(out/"report.json", report)
    code = 0
    try:
        import tensorflow as tf
        report["tensorflow"] = tf.__version__
        while True:
            candidates = []
            tuning_path = run/"B/tuning/manifest.json"
            if tuning_path.exists():
                manifest = json.loads(tuning_path.read_text())
                for phase, count in (("warmup", 250), ("verification", 1000)):
                    candidates.append((phase, tuning_path, manifest[phase]["force_fallback"], count, False))
            for path in sorted(run.glob("B/chunk-*/manifest.json")):
                manifest = json.loads(path.read_text())
                candidates.append((path.parent.name, path, manifest["chunk"]["force_fallback"], 250, True))
            for key, path, record, count, retained in candidates:
                fingerprint = digest(path)
                if key in report["records"]:
                    if fingerprint != report["records"][key]["manifest_sha256"]:
                        raise ValueError("completed archive manifest changed: "+key)
                    continue
                tensor_path = path.parent/record["path"]
                if digest(tensor_path) != record["sha256"]:
                    raise ValueError("original-force telemetry checksum mismatch: "+key)
                flags = tf.io.parse_tensor(tf.io.read_file(str(tensor_path)), tf.bool)
                if tuple(flags.shape) != (count, 4):
                    raise ValueError("original-force telemetry shape mismatch: "+key)
                counts = tf.reduce_sum(tf.cast(flags, tf.int32), axis=0).numpy().tolist()
                row = dict(manifest=str(path), manifest_sha256=fingerprint,
                    telemetry_sha256=record["sha256"], counts_by_chain=counts,
                    transitions_per_chain=count, retained=retained)
                report["records"][key] = row
                save(out/"report.json", report)
                print(key, "fallback_counts", counts, flush=True)
                if retained and any(counts):
                    report.update(status="RETAINED_ORIGINAL_FORCE_VETO", retained_original_force_healthy=False)
                    code = 2
                    break
            if code:
                break
            if (run/"accounting.json").exists():
                result = json.loads((run/"report.json").read_text())
                expected = len(result.get("arms", {}).get("B", {}).get("chunks", []))
                checked = sum(row["retained"] for row in report["records"].values())
                if expected != checked:
                    raise ValueError("missing retained original-force telemetry")
                report.update(status="COMPLETED", retained_chunks_checked=checked,
                    retained_original_force_healthy=True if checked else None)
                break
            if time.process_time()-cpu > 29 or time.monotonic()-start > 12600:
                raise ValueError("status-audit budget exhausted")
            time.sleep(5)
    except Exception as exc:
        code = 2
        report.update(status="AUDIT_FAILED", error=f"{type(exc).__name__}: {exc}")
    finally:
        elapsed = time.monotonic()-start
        usage = resource.getrusage(resource.RUSAGE_SELF)
        charge = dict(cpu_worker_seconds=usage.ru_utime+usage.ru_stime, gpu_seconds=0.)
        report.update(wall_seconds=elapsed, exit_code=code)
        save(out/"report.json", report)
        save(out/"accounting.json", dict(charged_this_run=charge, exit_code=code, wall_seconds=elapsed))
        print(json.dumps(dict(status=report["status"], charge=charge, output=str(out))), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
