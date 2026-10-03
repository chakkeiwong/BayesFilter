"""Tiny CPU counterexample for repair routing; not scientific HMC evidence."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["BAYESFILTER_TEST_DEVICE_SCOPE"] = "cpu"
REPOSITORY = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY))
started = time.perf_counter()
import tensorflow as tf
from tests.test_hmc_candidate_set_execution import execution_config, make_binding
from bayesfilter.inference import HMCAcceptancePolicy, HMCControllerConfig, tune_hmc_kernel

root = Path(__file__).parent / "gaussian-oversized-step-r1"
root.mkdir(exist_ok=False)
binding = make_binding(config=execution_config(measurement_num_results=64,
    verification_num_results=64, num_warmup_steps=0, acceptance_policy=HMCAcceptancePolicy(),
    target_status_trace_policy="per_chain_step"), epsilon_domain=(.01, 10.),
    max_repairs_per_family=2, repair_factor=2.)
config = HMCControllerConfig(primary_l_grid=(3,), epsilon_by_l=((3, (3.,)),),
    total_budget_units=20, repair_reserve_units=6)
run = tune_hmc_kernel(adapter=binding._base_adapter, initial_position=binding.initial_active_state,
    candidate_set_adapter=binding.typed_adapter, config=config, output_dir=root)
with (root / "numerical-evidence.json").open("x") as stream:
    json.dump(binding._evidence, stream, indent=2, allow_nan=False)
observations = []
for evidence in binding._evidence.values():
    analysis = evidence["analysis"]
    observations.append(dict(candidate=evidence["candidate"], seed=evidence["seed"],
        decision=analysis["decision"], policy_decision=analysis["acceptance_evidence"]["acceptance_decision"],
        acceptance=analysis["acceptance"], hard_vetoes=analysis["hard_vetoes"],
        rhat=evidence["rhat_reporting_only"]))
payload = dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip(),
    command=[sys.executable, *sys.argv], environment=sys.executable,
    device="CPU; GPUs intentionally hidden; non-XLA engineering exception",
    target="two-dimensional standard Gaussian; no observed data", root_seed=binding.config.seed,
    exact_config=binding.config.payload(), observations=observations,
    plan="docs/plans/bayesfilter-hmc-whole-procedure-gap-review-2026-09-14.md",
    final_status=run.result.final_status, repairs=len(run.result.repair_actions),
    states=run.result.candidate_states, wall_seconds=time.perf_counter()-started)
(root / "result.md").write_text("# Actual oversized-step Gaussian counterexample\n\n```json\n"
    + json.dumps(payload, indent=2) + "\n```\n")
assert len(observations) == 1 and observations[0]["policy_decision"] == "repair_step_lower"
assert observations[0]["decision"] == "failed" and not run.result.repair_actions
print(json.dumps({"confirmed": True, "acceptance": observations[0]["acceptance"],
    "policy_decision": observations[0]["policy_decision"], "adapter_decision": observations[0]["decision"],
    "repairs": len(run.result.repair_actions), "seconds": payload["wall_seconds"], "artifact": str(root / "result.md")}))
