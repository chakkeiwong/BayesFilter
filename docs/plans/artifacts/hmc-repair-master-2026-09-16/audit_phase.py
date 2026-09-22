"""Shared offline integrity checks for the final HMC repair phases.

The caller chooses an inspected frozen reader version and explicitly lists the
suite/run inventory. Scientific outcomes are reported by phase-specific notes.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PhaseAudit:
    def __init__(self):
        self.invalid = []
        self.outstanding = []
        self.costs = {"cpu_reference": 0., "gpu": 0.}
        self.attempts = []
        self.inventories = []
        self.receipts = 0
        self.tensors = 0
        self.statuses = {}
        self.sources = []
        self.seen_roots = set()

    def snapshot(self, source):
        from bayesfilter.inference.hmc_candidate_set_tuning import _sha256
        source = Path(source)
        manifest = read(source / "source_snapshot.json")
        actual = {str(p.relative_to(source)): sha(p) for p in sorted((source / "bayesfilter").rglob("*.py"))}
        if actual != manifest["source_files"] or _sha256(actual) != manifest["source_identity"]:
            self.invalid.append(str(source) + ": source snapshot changed")
        self.sources.append({"path": str(source), "identity": manifest["source_identity"], "files_checked": len(actual)})
        return manifest["source_identity"]

    def runtime(self, runtime, device, name):
        if device == "gpu":
            memory = runtime["memory_policy"]
            if not (runtime["jit_compile"] and memory["all_physical_devices_memory_growth"]
                    and memory["configured_before_logical_device_initialization"]):
                self.invalid.append(name + ": GPU launch provenance")
        elif runtime.get("gpu_intentionally_hidden") is not True:
            self.invalid.append(name + ": CPU provenance")

    def suite(self, directory, identity):
        from bayesfilter.inference.hmc_candidate_set_tuning import _sha256
        directory = Path(directory)
        index = read(directory / "run_index.json")
        if index["source"]["identity"] != identity:
            self.invalid.append(str(directory) + ": suite source")
        planned = {j["design"]["design_id"]: j for j in index["plan"]["jobs"]}
        if set(planned) != set(index["jobs"]):
            self.outstanding.append(str(directory) + ": unstarted or missing planned jobs")
        self.statuses[str(directory)] = dict(Counter(j["status"] for j in index["jobs"].values()))
        for name, job in index["jobs"].items():
            device = planned[name]["design"]["device"]
            if job["status"] not in {"complete", "failed", "timed_out", "unfunded"}:
                self.outstanding.append(name)
            for attempt in job.get("attempts", []):
                self.costs[device] += attempt["elapsed_seconds"]
                self.attempts.append({"design_id": name, "device": device, **attempt})
            if job.get("result"):
                path = Path(job["result"])
                if sha(path) != job["result_sha256"]:
                    self.invalid.append(name + ": result checksum")
                manifest = read(path.with_name(path.name.replace("-result.json", "-manifest.json")))
                if manifest["source"]["identity"] != identity or _sha256(manifest["design"]) != planned[name]["identity"]:
                    self.invalid.append(name + ": manifest source/design")
                self.runtime(manifest["runtime"], device, name)
                self.artifacts(path.parent)
            else:
                self.artifacts(directory / name)

    def diagnostic(self, path, *, artifacts=False):
        path = Path(path)
        record = read(path)
        device = record["device"]
        self.costs[device] += record["elapsed_seconds"]
        self.attempts.append({"path": str(path), **record})
        if record.get("runtime"):
            self.runtime(record["runtime"], device, str(path))
        if artifacts:
            self.artifacts(path.parent)

    def artifacts(self, root):
        from bayesfilter.inference.hmc_candidate_set_tuning import _sha256
        from bayesfilter.inference.hmc_candidate_set_artifacts import load_candidate_set_result_payload
        from bayesfilter.testing.inference_validation.engines.pipeline import check_inventory
        root = Path(root).resolve()
        if root in self.seen_roots:
            return
        self.seen_roots.add(root)
        for path in sorted(root.rglob("candidate_set_result.json")):
            payload = load_candidate_set_result_payload(path)
            inventory = check_inventory(payload)
            self.invalid.extend(str(path) + ": " + item for item in inventory["failures"])
            self.inventories.append({"path": str(path), **inventory})
            for receipt in payload["verification_receipts"]:
                digest = receipt.get("numerical_evidence_hash")
                if digest is None:
                    # Explicit mechanics-only controllers cannot issue numerical receipts.
                    continue
                evidence = read(path.parent / "numerical_evidence" / (digest + ".json"))
                if _sha256(evidence) != digest or evidence["candidate"]["candidate_record_hash"] != receipt["candidate_record_hash"]:
                    self.invalid.append(str(path) + ": verification receipt")
                self.receipts += 1
        for path in root.rglob("*.tensor.json"):
            if sha(path.with_suffix("")) != read(path)["sha256"]:
                self.invalid.append(str(path) + ": tensor checksum")
            self.tensors += 1
        def walk(node, base):
            if isinstance(node, dict):
                if "tensor" in node:
                    record = node["tensor"]
                    self.tensors += 1
                    if sha(base / record["file"]) != record["sha256"]:
                        self.invalid.append(str(base / record["file"]) + ": checkpoint tensor")
                else:
                    for child in node.values():
                        walk(child, base)
            elif isinstance(node, list):
                for child in node:
                    walk(child, base)
        for path in root.rglob("bundle.json"):
            if "posterior_chunks" in path.parts:
                walk(read(path)["tree"], path.parent)

    def finish(self, opening, limits, overhead):
        opening = read(opening)["budget_seconds"]
        total = {key: self.costs[key] + overhead.get(key, 0.) for key in self.costs}
        charged = {key: opening["cumulative_charged"][key] + total[key] for key in total}
        remaining = {key: opening["authorized"][key] - charged[key] for key in total}
        return {"terminal": not self.outstanding, "source_snapshots": self.sources,
            "attempts": self.attempts, "job_statuses": self.statuses,
            "candidate_inventories": self.inventories, "numerical_receipts_checked": self.receipts,
            "tensor_checksums_checked": self.tensors, "invalid_artifacts": self.invalid,
            "outstanding_workers": self.outstanding,
            "budget_seconds": {"authorized": opening["authorized"],
                "opening_charged": opening["cumulative_charged"], "measured": self.costs,
                "overhead": overhead, "phase_total": total, "cumulative_charged": charged,
                "remaining": remaining, "exceeded": min(remaining.values()) < 0 or
                any(total[key] > limits[key] for key in total)}}
