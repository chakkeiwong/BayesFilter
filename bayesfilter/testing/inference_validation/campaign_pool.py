"""Framework-free scheduling of individual fits from a bounded shared pool.

Quanta govern fairness, not numerical stopping criteria. The executor owns
checkpoint validation; a completed assessment is terminal even when unfavorable.
"""
from __future__ import annotations

from collections import deque
import math
import time

from .campaign_recovery import RESOURCE_STOPS, durable_progress


def run_pool(jobs, execute, *, deadline, quantum_seconds, max_attempts,
             checkpoint=lambda value: None, clock=time.monotonic,
             discover=lambda rows: ()):
    """Run one quantum per eligible fit before revisiting unfinished peers.

    ``execute(job, allowance, absolute_deadline, attempt)`` returns a native
    receipt. Unexpected exceptions stop the pool. Known numerical failures
    terminate only that fit. Every attempt, including setup, consumes wall time.
    ``max_attempts=None`` allows productive retries until the deadline.
    ``discover`` can add newly eligible jobs after a peer finishes; an existing
    job definition cannot change and terminal jobs are never rescheduled.
    """
    if (type(quantum_seconds) not in (int, float) or not math.isfinite(quantum_seconds)
            or quantum_seconds <= 0 or not math.isfinite(deadline)
            or (max_attempts is not None and
                (type(max_attempts) is not int or max_attempts < 1))):
        raise ValueError("finite positive quantum, deadline and attempt cap required")
    jobs = [dict(j) for j in jobs]
    ids = [j["job_id"] for j in jobs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate fit in common pool")
    queue, definitions, rows = deque(), {}, {}

    def add_jobs(additions):
        for item in additions:
            job = dict(item)
            q = job.get("first_quantum_seconds", quantum_seconds)
            if type(q) not in (int, float) or not math.isfinite(q) or q <= 0:
                raise ValueError("invalid first quantum")
            ident = job["job_id"]
            if ident in definitions:
                if definitions[ident] != job:
                    raise ValueError("discovered job definition changed: " + ident)
                continue
            definitions[ident] = job
            rows[ident] = {"job_id": ident, "case": job["case"],
                           "status": "unstarted", "attempts": []}
            queue.append(job)

    add_jobs(jobs)
    while True:
        add_jobs(discover(list(rows.values())))
        if not queue:
            break
        job = queue.popleft()
        row = rows[job["job_id"]]
        attempt = len(row["attempts"]) + 1
        allowance = job.get("first_quantum_seconds", quantum_seconds) if attempt == 1 else quantum_seconds
        remaining = deadline - clock()
        if allowance > remaining:
            row["status"] = "incomplete_pool_deadline" if row["attempts"] else "unstarted_pool_deadline"
            checkpoint({"current": None, "rows": list(rows.values())})
            continue  # A peer with a smaller quantum may still fit.
        checkpoint({"current": job["job_id"], "rows": list(rows.values())})
        started = clock()
        receipt = execute(job, allowance, min(deadline, started + allowance), attempt)
        if receipt.get("status") not in RESOURCE_STOPS | {
                "complete", "reused_final_assessment", "failed", "no_resumable_numerical_checkpoint",
                "ineligible_or_allocation_exhausted"}:
            raise ValueError("unexpected common-pool result: " + str(receipt.get("status")))
        row["attempts"].append({"allocated_seconds": allowance,
            "invocation_seconds": max(0., clock() - started), **receipt})
        row["status"] = receipt["status"]
        if receipt["status"] in RESOURCE_STOPS:
            if not durable_progress(receipt):
                row["status"] = "incomplete_no_progress"
            elif max_attempts is not None and attempt >= max_attempts:
                row["status"] = "incomplete_attempt_limit"
            elif max_attempts is None and clock() <= started:
                raise ValueError("productive retry did not advance monotonic time")
            else:
                queue.append(job)
        checkpoint({"current": None, "rows": list(rows.values())})
    return {"rows": list(rows.values()), "attempted": sum(bool(r["attempts"]) for r in rows.values()),
            "completed": sum(r["status"] in {"complete", "reused_final_assessment"} for r in rows.values()),
            "queue_closed": True, "all_workloads_complete": all(
                r["status"] in {"complete", "reused_final_assessment"} for r in rows.values())}
