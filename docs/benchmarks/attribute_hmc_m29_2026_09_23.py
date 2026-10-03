"""Summarize audited M29 first-call/warmed chunks without executing a sampler."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(fit):
    groups = defaultdict(list)
    checkpoint = read(fit / "tuning/tuning_checkpoint.json")
    order = {row["work_item_id"]: i
             for i, row in enumerate(checkpoint["result"]["observations"])}
    for path in sorted((fit / "tuning/numerical_evidence").glob("*.json")):
        evidence = read(path)
        # The full M29 audit checked numerical content and its binding; bind
        # this attribution to the exact files as well.
        digest = hashlib.sha256(json.dumps(evidence, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        assert digest == path.stem
        for ordinal, chunk in enumerate(evidence["chunks"]):
            key = (evidence["candidate"]["leapfrog_steps"], chunk["count"])
            groups[key].append({
                "work_id": evidence["work"]["work_item_id"],
                "work_ordinal": evidence["work"]["ordinal"],
                "observation_order": order[evidence["work"]["work_item_id"]],
                "chunk_index": ordinal,
                "evidence_path": str(path),
                "first": chunk["includes_first_runner_trace_or_compilation"],
                "seconds": chunk["elapsed_seconds"],
            })
    rows = []
    for (length, count), chunks in sorted(groups.items()):
        first = [c for c in chunks if c["first"]]
        warm = [c for c in chunks if not c["first"]]
        assert len(first) == 1, (length, count, first)
        # Work ordinals record creation, not scheduler execution priority.
        assert first[0] == min(chunks, key=lambda c: (c["observation_order"], c["chunk_index"]))
        rows.append({"L": length, "count": count, "calls": len(chunks),
            "first_seconds": first[0]["seconds"],
            "warmed_seconds": sum(c["seconds"] for c in warm),
            "first_call": first[0]})
    return {
        "fit": str(fit), "keys": rows, "unique_cache_keys": len(rows),
        "calls": sum(r["calls"] for r in rows),
        "first_call_seconds": sum(r["first_seconds"] for r in rows),
        "warmed_call_seconds": sum(r["warmed_seconds"] for r in rows),
        "interpretation": "First calls include construction, trace, compilation, execution and capture. Later calls differ in states/seeds and are descriptive only; subtraction does not estimate compilation time.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    audit_path = root / "terminal-audit-r1.json"
    audit = read(audit_path)
    assert audit["passed"]
    assert not args.output.exists()
    result = {"audit_path": str(audit_path), "audit_sha256": sha(audit_path),
              "sampler_executed": False, "targets": []}
    for pair in audit["pairs"]:
        result["targets"].append({"target": pair["target"],
            **summarize(Path(pair["parity"]["left"]))})
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([{k: row[k] for k in ("target", "unique_cache_keys",
        "calls", "first_call_seconds", "warmed_call_seconds")}
        for row in result["targets"]], indent=2))


if __name__ == "__main__":
    main()
