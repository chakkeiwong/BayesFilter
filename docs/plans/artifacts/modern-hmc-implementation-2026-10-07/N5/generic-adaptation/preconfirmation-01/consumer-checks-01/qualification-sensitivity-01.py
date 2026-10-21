"""CPU diagnostic design sensitivity; synthetic scores do not estimate power."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
SOURCE = ROOT / "docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/full-model-source-e68cdf1c4"
sys.path.insert(0, str(SOURCE))
import tensorflow as tf
from bayesfilter.inference.hmc_acceptance_protocol import HMCReplicatedAcceptancePolicy
from bayesfilter.inference.hmc_acceptance_statistics import replicated_acceptance_statistics

started = time.monotonic()
rows = []
for cap in (4, 8, 12):
    for count in (32, 64, 128, 256):
        p = HMCReplicatedAcceptancePolicy(trial_num_results=64, discarded_prefix=0,
            base_repetitions=32, max_repetitions=count, max_candidates=cap,
            search_family_alpha=.05, verification_family_alpha=.05,
            diagnostic_family_alpha=.05, temporal_tolerance=.1,
            qualification_mode="lower_bound", qualification_region=(.5, 1.))
        rungs = tuple(r for r in (1, 2, 4, 8) if 32*r <= count)
        for value in (.65, .70, .75, .85, .95, 1.):
            stages = {}
            for stage in ("measurement", "verification"):
                report = replicated_acceptance_statistics(tf.fill([count, 4], tf.constant(value, tf.float64)),
                    policy=p, stage=stage, evidence_rungs=rungs, jit_compile=False)
                stages[stage] = dict(decision=report["acceptance_decision"],
                    start_lower=report["start_intervals"][0][0], alpha=report["one_sided_alpha"])
            rows.append(dict(candidate_cap=cap, repetitions=count, constant_score=value, stages=stages,
                both_stages_pass=all(r["decision"] == "passed" for r in stages.values())))
record = dict(source=json.loads((SOURCE / ".validation-source.json").read_text()),
    question="sensitivity of scientific-floor qualification to counts and reserved candidate error allowance",
    rows=rows, scientific_acceptance_floor=.5, interval_method=p.method, gpu_intentionally_hidden=True,
    power_estimated=False, model_specific_acceptance_distribution=False, tuning_artifact_authority=False,
    role="deterministic planning diagnostic only", command=[sys.executable, str(Path(__file__).resolve())],
    environment=sys.executable, wall_seconds=time.monotonic()-started,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    plan="docs/plans/bayesfilter-modern-hmc-adaptive-consumer-repair-2026-10-12.md")
with Path(__file__).with_suffix(".json").open("x") as out:
    out.write(json.dumps(record, indent=2) + "\n")
for cap in (4, 8, 12):
    print(cap, {value: next((r["repetitions"] for r in rows if r["candidate_cap"] == cap
            and r["constant_score"] == value and r["both_stages_pass"]), None)
        for value in (.65, .70, .75, .85, .95, 1.)})
print("diagnostic_seconds", record["wall_seconds"])
