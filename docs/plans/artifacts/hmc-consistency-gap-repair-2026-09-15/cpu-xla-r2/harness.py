"""Bounded XLA compilation/replay diagnostic; no posterior or ranking claim."""
import os
import sys

if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
    raise RuntimeError("Memory growth must be configured before framework import")
mode = sys.argv[2]
if mode not in {"cpu", "gpu"}:
    raise ValueError("select cpu or gpu explicitly")
if mode == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("CPU debugging requires hidden GPUs before import")

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import traceback
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=False)
(root / "harness.py").write_text(Path(__file__).read_text())
started = time.monotonic()
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory = (configure_tensorflow_gpu_memory_growth(tf, require_gpu=True) if mode == "gpu"
          else {"mode": "CPU debugging; GPU devices intentionally hidden"})
tf.config.set_soft_device_placement(False)
import tensorflow_probability as tfp
from bayesfilter.inference import (
    HMCAcceptancePolicy, HMCCandidateExecutionConfig, HMCControllerConfig, HMCKernelTuningConfig,
    tune_hmc_kernel, resume_hmc_candidate_set_tuning, load_hmc_candidate_retained_runner,
    build_retained_bound_hmc_archive_runner_from_candidate_set_result,
    build_claim_bearing_retained_frozen_kernel_hmc_adapter_from_candidate_set_result,
)
from tests.test_hmc_candidate_set_execution import GaussianTarget
from tests.test_hmc_tuning_dispatch import _Adapter, _binding, _config as position_config

manifest = {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
    capture_output=True, text=True, check=True).stdout.strip(), "command": sys.argv,
    "mode": mode, "gpu_memory_policy": memory, "jit_compile": True,
    "python": platform.python_version(), "tensorflow": tf.__version__, "tfp": tfp.__version__,
    "environment": {key: os.environ.get(key) for key in (
        "CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
    "plan": "docs/plans/bayesfilter-hmc-consistency-gap-repair-plan-2026-09-15.md",
    "data": "none; exact Gaussian mechanics fixture",
    "seeds": {"tuning": [20260915, 31], "retained": [20260915, 91]},
    "source_hashes": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (ROOT / "bayesfilter/inference").glob("*.py")},
    "role": "XLA engineering diagnostic; CPU mode cannot establish GPU compatibility"}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
try:
    execution = HMCCandidateExecutionConfig(measurement_num_results=128, verification_num_results=128,
        num_warmup_steps=8, seed=(20260915, 31), use_xla=True, target_status_trace_policy="none",
        acceptance_policy=HMCAcceptancePolicy(practical_region=(.55, .85), repair_region=(.5, .9)))
    search = HMCControllerConfig(primary_l_grid=(3, 5),
        epsilon_by_l=((3, (1.1, 1.3, 1.5)), (5, (1.1, 1.3, 1.5))), evidence_rungs=(1,),
        total_budget_units=40, repair_reserve_units=3, max_candidates=6)
    target = GaussianTarget()
    starts = tf.constant([[-1., -.5], [-.3, .2], [.4, -.2], [1., .5]], tf.float64)
    first = tune_hmc_kernel(adapter=target, initial_position=starts,
        config=HMCKernelTuningConfig.smoke(mass_policy="fixed_identity", use_xla=True, target_scope="candidate-bridge-test",
                                          max_attempts=1), execution_config=execution, search_config=search,
        target_lineage={"model": "standard Gaussian", "data": "none", "prior": "standard normal"},
        source_paths=[__file__], output_dir=root / "ordinary", max_work_items=1)
    run = resume_hmc_candidate_set_tuning(root / "ordinary/tuning_checkpoint.json", adapter=target)
    assert run.result.completion_status == "complete" and run.result.verified_candidate_ids
    binding = run.adapter._execution_binding
    builder = (build_claim_bearing_retained_frozen_kernel_hmc_adapter_from_candidate_set_result
               if mode == "gpu" else build_retained_bound_hmc_archive_runner_from_candidate_set_result)
    member = builder(candidate_set_result=run.result, candidate_id=run.result.verified_candidate_ids[0],
                     retained_binding=binding)
    loaded = load_hmc_candidate_retained_runner(member.export(root / "member.json"), adapter=target,
                                                claim_eligible=mode == "gpu")
    block = loaded.run(num_results=16, seed=(20260915, 91), output_dir=root / "retained")
    position = tune_hmc_kernel(adapter=_Adapter(), initial_position=starts,
        parameter_scales=tf.ones([2], tf.float64), runner_binding=_binding(),
        config=replace(position_config(), use_xla=True, non_xla_reason=None, max_leapfrog_steps=5),
        execution_config=replace(execution, pilot_num_results=64),
        search_config=HMCControllerConfig(primary_l_grid=(3,), epsilon_by_l=((3, (.2,)),),
            pilot_enabled=True, evidence_rungs=(1,), total_budget_units=12, repair_reserve_units=3, max_candidates=1),
        output_dir=root / "position", max_work_items=1)
    observation = position.result.observations[0]["observation"]
    assert observation["evidence_validity"] == "valid" and not observation["hard_vetoes"]
    assert sum(chunk["count"] for chunk in observation["chunks"]) == 72
    assert not position.numerical_handoff_authority
    result = {"passed": True, "mode": mode, "jit_compile": True,
        "completion": run.result.completion_status, "verified_count": len(run.result.verified_candidate_ids),
        "resumed_more_work": len(run.result.observations) > len(first.result.observations),
        "samples_devices": sorted({row["samples_device"] for row in binding._evidence.values()}),
        "retained_device": block["samples"].device, "position_pilot_transitions": 72,
        "runtime": binding._runtime, "posterior_convergence_authority": False}
    if mode == "gpu":
        result["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
except Exception as exc:
    result = {"passed": False, "exception": type(exc).__name__, "reason": str(exc),
              "traceback": traceback.format_exc()}
result["wall_seconds_including_imports"] = time.monotonic() - started
(root / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
print(json.dumps(result, indent=2, default=str))
raise SystemExit(0 if result["passed"] else 1)
