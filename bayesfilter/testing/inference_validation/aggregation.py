"""Diagnostic aggregation of predeclared, independently seeded SBC shards.

Only one suite and frozen source version may contribute to a group. Missing
workers and fits remain missing; retries never become additional replications.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .designs import ValidationDesign, digest
from .storage import file_hash, read_json, write_json


def validate_groups(suite):
    designs={d["design_id"]: ValidationDesign.from_payload(d) for d in suite["designs"]}
    names=set()
    assigned=set()
    for group in suite.get("aggregate_groups", ()):
        if set(group)!={"group_id","design_ids","replications"}:
            raise ValueError("SBC groups require exact membership and replication count")
        ids=group["design_ids"]
        if (not ids or len(ids)!=len(set(ids)) or any(i not in designs for i in ids)
                or assigned.intersection(ids) or group["group_id"] in names):
            raise ValueError("invalid or duplicated SBC group membership")
        rows=[designs[i] for i in ids]
        def comparable(d):
            payload=d.payload()
            for key in ("design_id","budget_seconds","replications"):
                payload.pop(key)
            return digest(payload)
        if any(d.engine!="sbc" or comparable(d)!=comparable(rows[0]) for d in rows):
            raise ValueError("SBC shards must have identical numerical and statistical settings")
        total=sum(d.replications for d in rows)
        if type(group["replications"]) is not int or total!=group["replications"]:
            raise ValueError("SBC group replication count disagrees with its members")
        replace(rows[0],design_id=group["group_id"],replications=total)
        names.add(group["group_id"])
        assigned.update(ids)


def aggregate_groups(root):
    from .engines.sbc import summarize_datasets

    root=Path(root)
    index=read_json(root/"run_index.json")
    plan=index["plan"]
    groups=plan.get("aggregate_groups", [])
    suite={"designs":[j["design"] for j in plan["jobs"]],"aggregate_groups":groups}
    validate_groups(suite)
    jobs={j["design"]["design_id"]:j for j in plan["jobs"]}
    summaries=[]
    for group in groups:
        records=[]
        workers=[]
        known_unstarted = 0
        missing_from_started = 0
        for name in group["design_ids"]:
            job=index["jobs"].get(name,{"status":"not_run"})
            workers.append({"design_id":name,"status":job["status"]})
            if job["status"]!="complete":
                count = jobs[name]["design"]["replications"]
                if job["status"] in {"not_run", "unfunded"} and not job.get("attempts"):
                    known_unstarted += count
                else:
                    missing_from_started += count
                continue
            result_path=Path(job["result"])
            if file_hash(result_path)!=job["result_sha256"]:
                raise ValueError("SBC shard result checksum mismatch")
            result=read_json(result_path)
            manifest=read_json(result_path.with_name(result_path.name.replace("-result.json","-manifest.json")))
            if (result["design_identity"]!=jobs[name]["identity"] or result["execution_status"]!="complete"
                    or manifest["source"]["identity"]!=index["source"]["identity"]
                    or digest(manifest["design"])!=jobs[name]["identity"]):
                raise ValueError("SBC shard source or design identity mismatch")
            rows=result["assessment"]["datasets"]
            if (len(rows)>jobs[name]["design"]["replications"]
                    or [r["dataset_id"] for r in rows]!=list(range(len(rows)))):
                raise ValueError("SBC shard has duplicated, missing or reordered dataset identities")
            records.extend({**r,"shard_id":name} for r in rows)
            known_unstarted += jobs[name]["design"]["replications"] - len(rows)
        base=ValidationDesign.from_payload(jobs[group["design_ids"][0]]["design"])
        design=replace(base,design_id=group["group_id"],replications=group["replications"])
        result=summarize_datasets(design,records,
            unstarted_datasets=known_unstarted if missing_from_started == 0 else None)
        result.update(known_unstarted_datasets=known_unstarted,
                      missing_records_from_incomplete_shards=missing_from_started)
        result.update(group_id=group["group_id"],workers=workers,source_identity=index["source"]["identity"],
                      aggregation="predeclared independent dataset shards; no cross-version or retry pooling")
        write_json(root/(group["group_id"]+"-aggregate.json"),result)
        summaries.append(result)
    return summaries
