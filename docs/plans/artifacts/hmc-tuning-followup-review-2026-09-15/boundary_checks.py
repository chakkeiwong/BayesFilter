"""CPU debugging checks for public tuning boundaries; no posterior claims."""
import os
if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("This diagnostic requires intentionally hidden GPUs")

import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).parent
started = time.monotonic()
import tensorflow as tf
from bayesfilter.inference import (
    HMCControllerConfig, tune_hmc_kernel, load_hmc_candidate_retained_runner,
    build_retained_bound_hmc_archive_runner_from_candidate_set_result,
)
from bayesfilter.inference.neutra_hmc import SequentialNeuTraHMCConfig, sequential_chunk_seed
from tests.test_hmc_candidate_set_execution import GaussianTarget, make_binding, execution_config
from docs.examples.hmc_tuning_neural_force_binding import tune_deterministic_field


class ScalarGaussian(GaussianTarget):
    def adapter_signature(self):
        return "review-scalar-gaussian-20260915"

    def log_prob_and_grad(self, theta):
        tf.debugging.assert_rank(theta, 1, message="scalar exact target")
        return super().log_prob_and_grad(theta)


class NoTelemetryGaussian(GaussianTarget):
    target_status_telemetry = None

    def adapter_signature(self):
        return "review-no-telemetry-gaussian-20260915"


class ClassifiedDomainTarget(GaussianTarget):
    def classify_target_exception(self, error):
        return isinstance(error, tf.errors.InvalidArgumentError) and "review domain" in str(error)


def sequential_config(member, **overrides):
    values = dict(step_size=member.step_size, num_leapfrog_steps=member.num_leapfrog_steps,
        warmup_seed=(20260915, 17), retained_seed=(20260915, 19),
        jit_compile=member._binding.config.use_xla,
        warmup_chunk_results=16, warmup_min_results=16, warmup_check_window_results=16,
        warmup_max_results=16, retained_chunk_results=16, retained_min_results=16,
        retained_max_results=16)
    return SequentialNeuTraHMCConfig(**(values | overrides))


report = {"role": "CPU/non-XLA debugging; injected errors are not numerical failures",
          "gpu_devices": "intentionally hidden", "cases": {}}

for name, target in (("scalar_target", ScalarGaussian()), ("no_telemetry", NoTelemetryGaussian())):
    tick = time.monotonic()
    binding = make_binding(target=target,
        config=execution_config(target_status_trace_policy="none"), source_paths=[__file__])
    run = tune_hmc_kernel(adapter=target, initial_position=binding.initial_active_state,
        config=HMCControllerConfig(primary_l_grid=(3,), epsilon_by_l=((3, (1.3,)),),
            evidence_rungs=(1,), total_budget_units=10, repair_reserve_units=3),
        candidate_set_adapter=binding.typed_adapter, output_dir=OUT / name)
    row = {"tuning_completion": run.result.completion_status,
           "verified_count": len(run.result.verified_candidate_ids)}
    if run.result.verified_candidate_ids:
        member = build_retained_bound_hmc_archive_runner_from_candidate_set_result(
            candidate_set_result=run.result, candidate_id=run.result.verified_candidate_ids[0], retained_binding=binding)
        try:
            result = member.run_sequential(config=sequential_config(member), parameter_names=("x", "y"))
            row["sequential"] = {"returned": True, "passed": result.get("passed")}
        except Exception as exc:
            row["sequential"] = {"exception": type(exc).__name__, "reason": str(exc)}
    row["wall_seconds"] = time.monotonic()-tick
    report["cases"][name] = row

target = ClassifiedDomainTarget()
binding = make_binding(target=target, source_paths=[__file__])
attempted = []
def domain_error(candidate, state, count, seed):
    attempted.append(candidate.leapfrog_steps)
    raise tf.errors.InvalidArgumentError(None, None, "review domain failure")

with patch.object(binding, "_run", domain_error):
    try:
        tune_hmc_kernel(adapter=target, initial_position=binding.initial_active_state,
            config=HMCControllerConfig(primary_l_grid=(3, 5),
                epsilon_by_l=((3, (1.3,)), (5, (1.3,))), total_budget_units=10, repair_reserve_units=3),
            candidate_set_adapter=binding.typed_adapter, output_dir=OUT / "classified_domain")
    except Exception as exc:
        checkpoint = json.loads((OUT / "classified_domain/tuning_checkpoint.json").read_text())
        report["cases"]["classified_domain"] = {"exception": type(exc).__name__, "reason": str(exc),
            "attempted_L": attempted, "checkpoint_status": checkpoint["result"]["completion_status"],
            "target_classifier_recognizes": binding._active_adapter.classify_target_exception(exc)}

try:
    tune_deterministic_field(adapter=GaussianTarget(), initial_position=tf.zeros([4, 2], tf.float64),
        parameter_scales=[1., 1.], output_dir=OUT / "guide_position")
except Exception as exc:
    report["cases"]["guide_position_example"] = {"exception": type(exc).__name__, "reason": str(exc)}

tick = time.monotonic()
original = ROOT / "docs/plans/artifacts/hmc-consistency-gap-repair-2026-09-15/cpu-xla-r2/member.json"
member = load_hmc_candidate_retained_runner(original, adapter=GaussianTarget())
report["cases"]["existing_member_replay"] = {"loaded": True, "candidate_id": member.candidate.candidate_id,
    "evidence_count": len(member._binding._evidence), "wall_seconds": time.monotonic()-tick}

used = sorted(member._tuning_seeds())
collision = next(seed for seed in used if seed[1] > 1009 and (seed[0], seed[1]-1009) not in used)
cfg = sequential_config(member, warmup_seed=(collision[0], collision[1]-1009))
with patch("bayesfilter.inference.hmc_candidate_set_retained._run_sequential_member", return_value={"delegated": True}):
    delegated = member.run_sequential(config=cfg)
report["cases"]["derived_seed_reuse"] = {"root_seed": cfg.warmup_seed,
    "first_chunk_seed": sequential_chunk_seed(cfg.warmup_seed, 0), "tuning_seed": collision,
    "bridge_delegated": delegated["delegated"], "chains_executed": False}
cfg = sequential_config(member, warmup_seed=(17, 0), retained_seed=(17, 1009), warmup_max_results=32)
report["cases"]["posterior_phase_seed_overlap"] = {"config_accepted": True,
    "warmup_second_chunk": sequential_chunk_seed(cfg.warmup_seed, 1),
    "retained_first_chunk": sequential_chunk_seed(cfg.retained_seed, 0), "chains_executed": False}

report["wall_seconds"] = time.monotonic()-started
(OUT / "boundary-checks.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
