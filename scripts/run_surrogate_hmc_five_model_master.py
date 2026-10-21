#!/usr/bin/env python3
"""Five-model FP32/N4032 master: independent endpoints, filter tuning and surrogate HMC.

Use run for likelihood/force readiness, campaign for the complete phased runner,
and continue-advance for adaptive HMC after performance-master preparation.
Every phase preserves evidence and refreshes next-phase-plan.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from bayesfilter.testing.surrogate_hmc_scope_source_compatibility import (
    scope_source_hashes, source_migration_provenance,
)
PLAN = ROOT / "docs/plans/surrogate-hmc-five-model-n4032-master-2026-10-11.md"
DEFAULT_OUTPUT = ROOT / "docs/plans/artifacts/surrogate-hmc-five-model-n4032-20261011-01"
PYTHON = Path("/home/chakwong/anaconda3/envs/tftwogpu/bin/python")
MODELS = ("lgssm", "ksc", "predator_prey", "sir_d18", "range_bearing")
HORIZONS = {m: 50 for m in MODELS}
HORIZONS["range_bearing"] = 20
PARTICLES = 4032
LITERATURE_PRIOR = "literature-model-priors-v1"
BENCHMARK_PRIOR = "benchmark-normal-v1"
FOUR = ROOT / "docs/plans/artifacts/ledh-independent-validation-20261008-01"
RANGE = ROOT / "docs/plans/artifacts/ledh-range-bearing-particle-ladder-20261010-02/N4032"


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def clean(value):
    if hasattr(value, "numpy"):
        return clean(value.numpy())
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sources(model):
    common = [Path(__file__), PLAN, ROOT / "bayesfilter/testing/surrogate_hmc_five_model.py",
              ROOT / "bayesfilter/testing/surrogate_hmc_scope_source_compatibility.py",
              ROOT / "bayesfilter/testing/surrogate_hmc_literature_priors.py",
              ROOT / "bayesfilter/highdim/covariance_proposal_tf.py",
              ROOT / "bayesfilter/highdim/covariance_proposal_transport_tf.py",
              ROOT / "docs/benchmarks/run_ledh_covariance_proposal.py",
              ROOT / "scripts/run_range_bearing_covariance_comparison.py"]
    common += [ROOT / "bayesfilter/highdim" / name for name in (
        "analytical_jet_tf.py", "covariance_proposal_score_tf.py", "covariance_proposal_score_batch_tf.py", "covariance_proposal_score_moments_tf.py")]
    common += [ROOT / "bayesfilter/inference" / name for name in (
        "neural_force_hmc.py", "hmc_tensorflow_tuning.py", "hmc_tensorflow_verified.py",
        "hmc_sequential_verification.py", "hmc_posterior_diagnostics.py",
        "hmc_tuning_dispatch.py", "tuning_contract.py")]
    if model == "range_bearing":
        return common + [RANGE / "tuning.json", RANGE / "data/1704-t20.json",
                         ROOT / "experiments/dpf_implementation/tf_tfp/fixtures/range_bearing_tf.py"]
    return common + [FOUR / "frozen" / f"{model}-tuning.json",
                     FOUR / "inputs" / f"{model}-d26100831/dataset.json",
                     ROOT / f"bayesfilter/highdim/sqmc_{'nonlinear' if model in ('predator_prey','sir_d18') else model}_tf.py"]


def static_audit():
    # This is a deliberate CPU-only configuration check; the workers use GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("BAYESFILTER_PRELOAD_CUSTOM_OP", "0")
    from bayesfilter.inference.tuning_contract import HMC_TUNING_INTERFACE_CAPABILITIES
    public = [x for x in HMC_TUNING_INTERFACE_CAPABILITIES
              if x.interface_name == "tune_hmc_kernel" and x.artifact_authority]
    if len(public) != 1 or "mechanics-only" not in public[0].target_contract:
        raise RuntimeError("public position-field capability changed; re-audit required")
    from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
    chunk = select_transport_chunks(PARTICLES)
    if chunk.row_chunk_size != 2016 or chunk.col_chunk_size != 2016:
        raise RuntimeError("N4032 transport chunk policy mismatch")
    missing = [str(path) for model in MODELS for path in sources(model) if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing evidence/source dependencies: " + ", ".join(sorted(set(missing))))
    return {"status": "PASS_READINESS_ONLY", "particle_count": PARTICLES,
            "models": list(MODELS), "horizons": HORIZONS, "dtype": "float32",
            "tf32": False, "chunk": chunk.__dict__,
            "hmc_public_tuner": public[0].qualified_name,
            "hmc_position_field_authority": "conditional_mechanics_only",
            "surrogate_force_endpoint_separation": "required_not_yet_admitted"}


def worker(model, output):
    if model not in MODELS:
        raise ValueError("unknown model")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    begin = time.monotonic()
    env = {"TF_FORCE_GPU_ALLOW_GROWTH": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH"),
           "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
    manifest = {"schema": "bayesfilter.surrogate_hmc_five_model_force_preflight.v1",
                "status": "STARTING", "model": model, "horizon": HORIZONS[model],
                "particles": PARTICLES, "dtype": "float32", "tf32": False,
                "jit_compile": True, "device": "GPU", "cpu_only": False,
                "gpu_trust_basis": "escalated_local_GPU_execution",
                "command": [sys.executable, *sys.argv], "python": sys.executable,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources(model)},
                "plan": str(PLAN), "output": str(output), "started_utc": stamp(),
                "environment": env,
                "evidence_role": "force_readiness_and_cost_only_not_hmc_sampling_or_tuning"}
    write(output / "manifest.json", manifest)
    tf = None
    try:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
        os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        if tf.config.experimental.tensor_float_32_execution_enabled():
            raise RuntimeError("TF32 unexpectedly enabled")
        from bayesfilter.highdim.covariance_proposal_tf import ProposalControls, make_filter
        from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
        from docs.benchmarks.run_ledh_covariance_proposal import spec_for
        from scripts import run_range_bearing_covariance_comparison as range_base
        dtype = tf.float32
        t = HORIZONS[model]
        if model == "range_bearing":
            tuning = read(RANGE / "tuning.json")
            if tuning["scope"]["particles"] != PARTICLES:
                raise RuntimeError("range tuning particle scope mismatch")
            controls = tuning["selected"]["new"]["controls"]
            weights = tuning["selected"]["new"]["beta"]
            with tf.device("/CPU:0"):
                _, spec = range_base.specs(tf, dtype)
                spec64, _ = range_base.specs(tf, tf.float64)
                saved = read(RANGE / "data/1704-t20.json")
                obs = tf.cast(tf.cast(tf.constant(saved["observations"], tf.float64), dtype), dtype)
                fixed = range_base.inputs(tf, spec64, PARTICLES, 31001, dtype)[:3]
                theta = tf.constant(range_base.POINTS[0], dtype)
            tuning_path = RANGE / "tuning.json"
            data_path = RANGE / "data/1704-t20.json"
        else:
            tuning_path = FOUR / "frozen" / f"{model}-tuning.json"
            data_path = FOUR / "inputs" / f"{model}-d26100831/dataset.json"
            tuning = read(tuning_path)
            saved = read(data_path)
            controls, weights = tuning["scope"]["controls"], tuning["beta"]
            spec = spec_for(model, lgssm_family="p44")
            if (tuning["scope"]["target_id"] != spec.target_id
                    or saved["target_id"] != spec.target_id or saved["data_seed"] != 26100831):
                raise RuntimeError("saved model/data identity mismatch")
            with tf.device("/CPU:0"):
                observation64 = tf.constant(saved["observations"], tf.float64)
                if hashlib.sha256(tf.io.serialize_tensor(observation64).numpy()).hexdigest() != saved["observation_sha256"]:
                    raise RuntimeError("observation checksum mismatch")
                obs = tf.cast(observation64, dtype)
                from bayesfilter.highdim.sqmc_campaign_tf import random_inputs
                design = random_inputs("iid_dual_cap", 261008301, PARTICLES, spec.dimension,
                                       t, tf.float64, jit_compile=True)
                extra = tf.random.stateless_uniform([t, PARTICLES, 2], [261008301, 991], dtype=tf.float64)
                fixed = tuple(tf.cast(x, dtype) for x in (*design[:2], extra))
                theta = tf.cast(spec.default_theta(tf.float64), dtype)
        beta = tf.constant(weights, dtype)
        obs_dim = getattr(spec, "observation_dimension", spec.dimension)
        if obs.shape != (t, obs_dim) or theta.shape != (spec.parameter_count,):
            raise RuntimeError("target tensor shape mismatch")
        program = make_filter(spec, PARTICLES, t, ProposalControls(**controls), dtype, jit_compile=True)
        directions = tf.eye(spec.parameter_count, dtype=dtype)
        with tf.device("/GPU:0"):
            calls = [program(theta, directions[j], obs, *fixed, beta)
                     for j in range(spec.parameter_count)]
        values = [float(row["value"]) for row in calls]
        scores = [float(row["score"]) for row in calls]
        warm_begin = time.monotonic()
        with tf.device("/GPU:0"):
            repeated = [program(theta, directions[j], obs, *fixed, beta)
                        for j in range(spec.parameter_count)]
        repeated_scores = [float(row["score"]) for row in repeated]
        warm_full_score_seconds = time.monotonic() - warm_begin
        if repeated_scores != scores:
            raise RuntimeError("fixed-design force replay changed its score")
        status = [bool(row["valid"]) and int(row["steps_completed"]) == t
                  for row in calls]
        complete = all(status) and all(math.isfinite(v) for v in values + scores)
        delta = max(values) - min(values) if complete else None
        valid = complete and delta <= 1e-4
        result = {"model": model, "status": "PASS_FORCE_READINESS" if valid else "FAIL_FORCE_READINESS",
                  "value": values[0], "score": scores, "directional_values": values,
                  "all_directions_valid": status, "primal_direction_difference": delta,
                  "warm_full_score_seconds": warm_full_score_seconds,
                  "parameter_count": spec.parameter_count, "state_dimension": spec.dimension,
                  "observation_dimension": obs_dim,
                  "horizon": t, "particles": PARTICLES,
                  "tuning_role": "saved_FP64_controls_warm_start_only_not_N4032_FP32_tuning",
                  "controls": controls, "beta": weights,
                  "data_sha256": digest(data_path), "tuning_sha256": digest(tuning_path),
                  "memory_policy": memory, "tensorflow": tf.__version__,
                  "tf32_effective": tf.config.experimental.tensor_float_32_execution_enabled(),
                  "execution_device": calls[0]["value"].device,
                  "chunk": select_transport_chunks(PARTICLES).__dict__,
                  "wall_seconds": time.monotonic() - begin,
                  "hmc_status": "NOT_RUN_ENDPOINT_AND_SCOPE_TUNING_PENDING"}
        if "GPU:" not in result["execution_device"]:
            raise RuntimeError("force did not execute on a GPU")
        write(output / "force-preflight.json", result)
        manifest.update(status=result["status"], memory_policy=memory,
                        tensorflow=tf.__version__, tf32_effective=result["tf32_effective"])
        return 0 if valid else 2
    except Exception:
        manifest.update(status="WORKER_ERROR", traceback=traceback.format_exc())
        raise
    finally:
        manifest.update(finished_utc=stamp(), wall_seconds=time.monotonic() - begin)
        if tf is not None:
            try:
                manifest["gpu_allocator"] = tf.config.experimental.get_memory_info("GPU:0")
            except Exception:
                manifest["gpu_allocator"] = "unavailable"
        write(output / "manifest.json", manifest)


def next_phase(output, ledger):
    rows = ledger["models"]
    remaining = [m for m in MODELS if rows[m]["status"] == "PENDING"]
    passed = [m for m in MODELS if rows[m]["status"] == "PASS_FORCE_READINESS"]
    return {"schema": "bayesfilter.surrogate_hmc_five_model_next_phase.v1",
            "updated_utc": stamp(), "current_phase": "N4032_FP32_FORCE_READINESS",
            "remaining_force_preflights": remaining, "passed_force_readiness": passed,
            "failed_or_unresolved": {m: rows[m] for m in MODELS
                                     if rows[m]["status"] not in ("PENDING", "PASS_FORCE_READINESS")},
            "hmc_sampling_started": False,
            "next_action": ("Complete remaining bounded force preflights" if remaining else
                            "Audit and implement independent endpoints, priors and charts; "
                            "retune every exact FP32/N4032 scope, cost the public position-field tuner, "
                            "then run cumulative HMC within a declared inference budget"),
            "nonclaims": ["no HMC sampling or posterior result from this phase",
                          "saved controls are not exact-scope tuning artifacts",
                          "a passing force preflight does not establish score accuracy"]}


def run(output, *, budget, per_model_timeout, gpu):
    if output.exists():
        raise FileExistsError("fresh versioned output root required: " + str(output))
    if budget <= 0 or budget > 3600 or per_model_timeout <= 0 or per_model_timeout > 900:
        raise ValueError("readiness budget exceeds reviewed cap")
    output.mkdir(parents=True)
    begin = time.monotonic()
    ledger = {"schema": "bayesfilter.surrogate_hmc_five_model_master.v1",
              "status": "RUNNING_FORCE_READINESS", "started_utc": stamp(),
              "plan": str(PLAN), "budget_gpu_wall_seconds": budget,
              "per_model_timeout_seconds": per_model_timeout,
              "gpu_visible_ordinal": gpu, "particles": PARTICLES,
              "dtype": "float32", "tf32": False,
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "audit": static_audit(),
              "models": {m: {"status": "PENDING"} for m in MODELS}}
    write(output / "master.json", ledger)
    write(output / "next-phase-plan.json", next_phase(output, ledger))
    for model in MODELS:
        remaining = budget - (time.monotonic() - begin)
        if remaining <= 0:
            ledger["models"][model] = {"status": "BUDGET_EXHAUSTED"}
            continue
        cmd = [str(PYTHON), str(Path(__file__).resolve()), "worker", "--model", model,
               "--output", str(output / model)]
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), TF_FORCE_GPU_ALLOW_GROWTH="true",
                   TF_CPP_MIN_LOG_LEVEL="2", BAYESFILTER_PRELOAD_CUSTOM_OP="0")
        row = {"status": "RUNNING", "command": cmd, "started_utc": stamp()}
        ledger["models"][model] = row
        write(output / "master.json", ledger)
        start = time.monotonic()
        with (output / f"{model}.log").open("w") as log:
            try:
                code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                                      stderr=subprocess.STDOUT,
                                      timeout=min(per_model_timeout, remaining), check=False).returncode
                row["returncode"] = code
                result = output / model / "force-preflight.json"
                row["status"] = read(result)["status"] if result.exists() else "WORKER_ERROR"
            except subprocess.TimeoutExpired:
                row.update(status="TIMEOUT", returncode=None)
        row.update(finished_utc=stamp(), wall_seconds=time.monotonic() - start,
                   log=str(output / f"{model}.log"))
        write(output / "master.json", ledger)
        write(output / "next-phase-plan.json", next_phase(output, ledger))
        print(model, row["status"], round(row["wall_seconds"], 1), flush=True)
    ledger.update(status="FORCE_READINESS_PHASE_FINISHED_HMC_NOT_RUN",
                  finished_utc=stamp(), wall_seconds=time.monotonic() - begin)
    write(output / "master.json", ledger)
    write(output / "next-phase-plan.json", next_phase(output, ledger))
    return 0 if all(ledger["models"][m]["status"] == "PASS_FORCE_READINESS" for m in MODELS) else 2


PHASES = ("endpoint-audit", "filter-tuning", "mechanics", "inference")
RESULT_NAMES = {"endpoint-audit": "endpoint-audit.json", "filter-tuning": "filter-selection.json",
                "mechanics": "pilot.json", "inference": "hmc.json"}
PASS_STATUS = {"endpoint-audit": {"PASS_ENDPOINT_MECHANICS"},
               "filter-tuning": {"FILTER_SCOPE_SELECTED_FOR_EXPLORATORY_HMC"},
               "mechanics": {"PASS_HMC_MECHANICS_ONLY"}, "inference": {"VERIFIED"}}


def restore_selected_scope(runtime, case, selection_path):
    """Restore declared arithmetic before validating the complete frozen scope.

    The cost-repair master can select mixed precision and fresh tuning seeds.
    Loading a default case first is insufficient for its inference consumer.
    The runtime still validates the full scope, source hashes and selection.
    """
    selected = read(selection_path)
    signature = selected.pop("selection_hash")
    if signature != runtime.hashed(selected):
        raise ValueError("selection checksum mismatch")
    scope = selected["scope"]
    if scope["model"] != case.name:
        raise ValueError("selection model mismatch")
    case = runtime.configure_precision(
        case, scope.get("precision_policy", "uniform"),
        scope.get("tuning_seed_offset", 0))
    runtime.load_selection(case, selection_path)
    return case


def validate_stage_prior(phase, prior):
    """Require a declared prior, and confine the benchmark fixture to mechanics."""
    if phase == "inference" and prior != LITERATURE_PRIOR:
        raise ValueError("Scientific posterior sampling requires model-specific priors; "
                         "benchmark-normal-v1 is mechanics-only")
    if prior not in (LITERATURE_PRIOR, BENCHMARK_PRIOR):
        raise ValueError("a supported explicit prior choice is required")


def stage_worker(args):
    validate_stage_prior(args.phase, args.priors)
    if args.output.exists():
        raise FileExistsError("stage output already exists: " + str(args.output))
    args.output.mkdir(parents=True)
    start = time.monotonic()
    manifest = dict(schema="bayesfilter.five_model_hmc_stage.v1", model=args.model,
        phase=args.phase, particles=PARTICLES, filter_dtype="float32", tf32=False,
        jit_compile=True, device="GPU", cpu_only=False, started_utc=stamp(),
        command=[sys.executable, *sys.argv], python=sys.executable, plan=str(PLAN),
        output=str(args.output), prior_choice=args.priors,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources(args.model)},
        gpu_trust_basis="escalated_local_GPU_execution", status="RUNNING")
    write(args.output/"manifest.json", manifest)
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        os.environ.setdefault("BAYESFILTER_PRELOAD_CUSTOM_OP", "0")
        os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
        os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        from bayesfilter.testing import surrogate_hmc_five_model as runtime
        case = runtime.configure_prior(runtime.load_case(args.model), args.priors)
        if args.selection and args.phase in ("mechanics", "inference"):
            case = restore_selected_scope(runtime, case, args.selection)
        manifest.update(memory_policy=memory, tensorflow=tf.__version__, scope=case.scope(),
                        filter_dtype=case.force_dtype.name,
                        source_migration=source_migration_provenance(ROOT),
                        tf32_effective=tf.config.experimental.tensor_float_32_execution_enabled())
        write(args.output/"manifest.json", manifest)
        with tf.device("/GPU:0"):
            if args.phase == "endpoint-audit":
                result = runtime.endpoint_audit(case, args.output)
            elif args.phase == "filter-tuning":
                result = runtime.tune_filter_scope(case, args.output)
            elif args.phase == "mechanics":
                if args.selection:
                    runtime.load_selection(case, args.selection)
                result = runtime.pilot(case, args.output)
            else:
                if not args.selection:
                    raise ValueError("inference requires an exact-scope filter selection")
                result = runtime.inference(case, args.output, args.selection,
                    resume_manifest=args.resume_manifest,
                    target_accept_prob=args.hmc_target_accept,
                    initial_step_size=args.hmc_initial_step_size)
        manifest.update(status=result["status"], gpu_allocator=tf.config.experimental.get_memory_info("GPU:0"))
        return 0 if result["status"] in PASS_STATUS[args.phase] else 2
    except Exception:
        manifest.update(status="STAGE_ERROR", traceback=traceback.format_exc())
        raise
    finally:
        manifest.update(finished_utc=stamp(), wall_seconds=time.monotonic()-start)
        write(args.output/"manifest.json", manifest)



def valid_selected_precision(model, scope):
    """Keep FP32 default and only the measured SIR recurrence exception."""
    policy = scope.get("precision_policy", "uniform")
    pair_dtype = scope.get("pairwise_dtype", scope.get("dtype"))
    return (scope.get("tf32") is False and pair_dtype == "float32" and
            ((policy == "uniform" and scope.get("dtype") == "float32") or
             (model == "sir_d18" and policy == "fp64_recursion_fp32_pairs"
              and scope.get("dtype") == "float64")))


def prepared_selections(prepared_root, prior_choice=None):
    """Validate frozen preparation before continuing in a fresh campaign root."""
    prepared_root = Path(prepared_root)
    previous = read(prepared_root / "master.json")
    if (previous.get("status") != "PREPARATION_COMPLETE_HMC_NOT_RUN"
            or previous.get("through_phase") != "filter-tuning"
            or previous.get("particles") != PARTICLES or previous.get("dtype") != "float32"
            or previous.get("tf32") is not False or previous.get("horizons") != HORIZONS):
        raise ValueError("prepared campaign has no matching completed FP32/N4032 filter phase")
    selections = {}
    for model in MODELS:
        history = previous["models"].get(model, {})
        if any(history.get(phase, {}).get("status") not in PASS_STATUS[phase]
               for phase in ("endpoint-audit", "filter-tuning")):
            raise ValueError(f"prepared campaign is incomplete for {model}")
        endpoint = read(prepared_root / model / "endpoint-audit" / "endpoint-audit.json")
        if (endpoint.get("status") != "PASS_ENDPOINT_MECHANICS"
                or endpoint.get("valid") is not True
                or endpoint.get("fixed_input_replay_equal") is not True
                or not isinstance(endpoint.get("value"), (int, float))
                or not math.isfinite(endpoint["value"])):
            raise ValueError(f"prepared endpoint failed validity/replay for {model}")
        path = prepared_root / model / "filter-tuning" / "filter-selection.json"
        selected = read(path)
        signature = selected.pop("selection_hash")
        check = hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
        if (signature != check
                or selected.get("status") != "FILTER_SCOPE_SELECTED_FOR_EXPLORATORY_HMC"
                or not selected.get("selected", {}).get("valid")):
            raise ValueError(f"prepared selection integrity failed for {model}")
        scope = selected["scope"]
        if prior_choice is not None and scope.get("prior", {}).get("id") != prior_choice:
            raise ValueError(f"prepared prior changed for {model}; fresh scope tuning required")
        if (scope.get("model") != model or scope.get("particles") != PARTICLES
                or not valid_selected_precision(model, scope)
                or scope.get("horizon") != HORIZONS[model]):
            raise ValueError(f"prepared scope changed for {model}")
        data_path = (RANGE / "data/1704-t20.json" if model == "range_bearing"
                     else FOUR / "inputs" / f"{model}-d26100831/dataset.json")
        if digest(data_path) != scope.get("data_sha256"):
            raise ValueError(f"prepared data changed for {model}")
        actual_sources = scope_source_hashes({
            relative: digest(ROOT / relative) for relative in scope["source_sha256"]})
        for relative, expected in scope["source_sha256"].items():
            if actual_sources[relative] != expected:
                raise ValueError(f"prepared source changed for {model}: {relative}")
        selections[model] = path
    return previous, selections


def advance_prepared_selections(roots, models, prior_choice):
    """Read actual cost-master evidence without relabeling its preparation files."""
    completed = {}
    for root in map(Path, roots):
        advance = read(root / "advance.json")
        for item in advance["models"]:
            model = item["model"]
            if model not in models:
                continue
            if model in completed:
                raise ValueError("duplicate prepared scope: " + model)
            if (item.get("filter-tuning", {}).get("status") not in PASS_STATUS["filter-tuning"]
                    or item.get("mechanics", {}).get("status") not in PASS_STATUS["mechanics"]):
                raise ValueError("advance preparation incomplete: " + model)
            selection = root/model/"filter-tuning"/model/"filter-selection.json"
            selected = read(selection)
            signature = selected.pop("selection_hash")
            if (signature != hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
                    or selected.get("status") not in PASS_STATUS["filter-tuning"]
                    or selected.get("selected", {}).get("valid") is not True):
                raise ValueError("advance selection integrity failed: " + model)
            scope = selected["scope"]
            if (scope.get("model") != model or scope.get("particles") != PARTICLES
                    or scope.get("horizon") != HORIZONS[model]
                    or scope.get("prior", {}).get("id") != prior_choice
                    or not valid_selected_precision(model, scope)):
                raise ValueError("advance scope/prior/precision mismatch: " + model)
            data = (RANGE/"data/1704-t20.json" if model == "range_bearing"
                    else FOUR/"inputs"/f"{model}-d26100831/dataset.json")
            if digest(data) != scope.get("data_sha256"):
                raise ValueError("advance data changed: " + model)
            if not scope.get("source_sha256"):
                raise ValueError("advance source provenance missing: " + model)
            actual_sources = scope_source_hashes({
                relative: digest(ROOT / relative) for relative in scope["source_sha256"]})
            for relative, expected in scope["source_sha256"].items():
                if actual_sources[relative] != expected:
                    raise ValueError("advance source changed: " + relative)
            tuning = read(selection.parent/"result.json")
            if (tuning.get("scope") != scope or tuning.get("selected") != selected["selected"]
                    or tuning.get("status") not in PASS_STATUS["filter-tuning"]):
                raise ValueError("advance tuning result mismatch: " + model)
            directory = root/model/"mechanics"/model
            mechanics = read(directory/"result.json")
            pilot_path = directory/"pilot.json"
            pilot = read(pilot_path)
            endpoint_path = directory/"endpoint-audit.json"
            endpoint = read(endpoint_path)
            command = mechanics.get("command", [])
            selected_arg = (command[command.index("--selection")+1]
                            if "--selection" in command else None)
            bound_selection = Path(selected_arg) if selected_arg else None
            if bound_selection is not None and not bound_selection.is_absolute():
                bound_selection = ROOT/bound_selection
            if (mechanics.get("status") not in PASS_STATUS["mechanics"]
                    or mechanics.get("scope") != scope or pilot.get("scope") != scope
                    or bound_selection is None or bound_selection.resolve() != selection.resolve()
                    or pilot.get("status") not in PASS_STATUS["mechanics"]
                    or pilot.get("force_role") != "selected_controls"
                    or pilot.get("finite") is not True or pilot.get("divergences") != 0
                    or pilot.get("force_fallback_count") != 0):
                raise ValueError("advance mechanics failed or selection changed: " + model)
            for health in ("cold_run_health", "warmed_run_health"):
                checks = pilot.get(health, {})
                if (checks.get("finite") is not True or checks.get("veto") is not False
                        or checks.get("force_fallback_count") != 0 or checks.get("divergences") != 0):
                    raise ValueError("advance mechanics health missing/failed: " + model)
            if (endpoint.get("status") not in PASS_STATUS["endpoint-audit"]
                    or endpoint.get("valid") is not True
                    or endpoint.get("fixed_input_replay_equal") is not True
                    or not isinstance(endpoint.get("value"), (int, float))
                    or not math.isfinite(endpoint["value"])):
                raise ValueError("advance endpoint failed validity/replay: " + model)
            completed[model] = dict(selection=str(selection), selection_hash=signature,
                pilot=str(pilot_path), endpoint=str(endpoint_path), scope=scope,
                prepared_root=str(root), posterior_sampling_complete=False)
    if set(completed) != set(models):
        raise ValueError("missing prepared models: " + ", ".join(sorted(set(models)-set(completed))))
    return completed


def continue_advance(args):
    """Consume completed preparation and charge adaptive HMC to its original ledger."""
    from scripts.run_surrogate_hmc_performance_repair import (
        campaign_remaining, campaign_attempts_remaining)
    validate_stage_prior("inference", args.priors)
    if not (0 < args.budget_seconds <= 172800 and 60 < args.per_model_timeout <= 172800):
        raise ValueError("finite bounded inference budgets required")
    if not args.advance_root or not args.campaign_ledger:
        raise ValueError("continuation requires advance roots and the cumulative campaign ledger")
    if args.resume_manifest and not args.model:
        raise ValueError("a resume manifest belongs to exactly one selected model")
    if args.resume_manifest and (args.hmc_target_accept is not None or args.hmc_initial_step_size is not None):
        raise ValueError("resumption must preserve the frozen HMC kernel")
    order = (args.model,) if args.model else ("ksc", "range_bearing", "lgssm", "predator_prey", "sir_d18")
    prepared = advance_prepared_selections(args.advance_root, order, args.priors)
    budget, _ = campaign_remaining(args.campaign_ledger)
    if any(row.get("status") == "RUNNING" for row in budget["attempts"]):
        raise ValueError("another campaign worker is running; finish it before continuation")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    state = dict(schema="bayesfilter.five_model_advance_continuation.v1", status="RUNNING",
        started_utc=stamp(), plan=str(PLAN), command=[sys.executable, *sys.argv],
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        campaign_ledger=str(args.campaign_ledger), prior_choice=args.priors,
        phase_budget_seconds=args.budget_seconds, models={m: dict(prepared=prepared[m]) for m in order},
        redundant_validation_phase=False, scientific_default_promoted=False)
    def refresh(model=None):
        ledger, remaining = campaign_remaining(args.campaign_ledger)
        state.update(wall_seconds=time.monotonic()-start, remaining_campaign_seconds=remaining,
                     remaining_campaign_attempts=campaign_attempts_remaining(ledger))
        write(args.output/"master.json", state)
        next_plan = dict(status=state["status"], current_model=model, part_of_master=True,
            automatic_continuation=state["status"] == "RUNNING", updated_utc=stamp(),
            retained_rule="cumulative modern R-hat <=1.01 or 20000 samples per chain",
            warmup_rule="at least2000; recent1000 R-hat<=1.05; at most10000",
            redundant_validation_phase=False, remaining_campaign_seconds=remaining,
            action="continue shared adaptive controller within the declared budget"
                   if state["status"] == "RUNNING" else "inspect retained diagnostics and remaining model/budget gaps",
            models=state["models"])
        write(args.output/"next-phase-plan.json", next_plan)
        campaign_next = Path(args.campaign_ledger).parent/"next-phase-plan.json"
        if campaign_next.resolve() != (args.output/"next-phase-plan.json").resolve():
            write(campaign_next, dict(next_plan, active_phase_plan=str(args.output/"next-phase-plan.json")))
    refresh()
    for model in order:
        ledger, remaining = campaign_remaining(args.campaign_ledger)
        stage_budget = min(remaining, args.budget_seconds-(time.monotonic()-start), args.per_model_timeout)
        if stage_budget <= 60 or campaign_attempts_remaining(ledger) == 0:
            state["models"][model]["inference"] = dict(status="CAMPAIGN_BUDGET_OR_ATTEMPT_STOP")
            refresh(model)
            continue
        out = args.output/model/"inference"
        command = [str(PYTHON), str(Path(__file__).resolve()), "stage-worker", "--model", model,
            "--phase", "inference", "--output", str(out), "--gpu", str(args.gpu),
            "--priors", args.priors, "--selection", prepared[model]["selection"]]
        for flag, value in (("--resume-manifest", args.resume_manifest),
                            ("--hmc-target-accept", args.hmc_target_accept),
                            ("--hmc-initial-step-size", args.hmc_initial_step_size)):
            if value is not None:
                command.extend((flag, str(value)))
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TF_FORCE_GPU_ALLOW_GROWTH="true",
            BAYESFILTER_PRELOAD_CUSTOM_OP="0", TF_CPP_MIN_LOG_LEVEL="2",
            BAYESFILTER_WORKER_WALL_DEADLINE=str(time.monotonic()+stage_budget-60))
        entry = dict(model=model, phase="inference", status="RUNNING", command=command,
            output=str(out), started_utc=stamp(), reserved_seconds=stage_budget,
            result=str(out/"hmc.json"))
        ledger["attempts"].append(entry)
        ledger["next_action"] = "running adaptive HMC for " + model
        write(args.campaign_ledger, ledger)
        state["models"][model]["inference"] = entry
        refresh(model)
        begin = time.monotonic()
        try:
            with (args.output/(model+"-inference.log")).open("w") as log:
                child = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                    stderr=subprocess.STDOUT, timeout=stage_budget, check=False)
            result = read(out/"hmc.json") if (out/"hmc.json").exists() else {}
            status = result.get("status", "STAGE_ERROR")
            if child.returncode != 0 and status == "VERIFIED":
                status = "WORKER_EXIT_ERROR"
            entry.update(status=status, returncode=child.returncode)
        except subprocess.TimeoutExpired:
            entry.update(status="STAGE_TIME_BUDGET_STOP", returncode=None,
                next_action="inspect durable public-tuner progress before resuming")
        except Exception:
            entry.update(status="STAGE_ERROR", traceback=traceback.format_exc())
            state.update(status="INCOMPLETE_WORKER_ERROR", finished_utc=stamp())
            raise
        finally:
            entry.update(wall_seconds=time.monotonic()-begin, finished_utc=stamp())
            ledger["used_seconds"] = sum(float(r.get("wall_seconds", r.get("reserved_seconds", 0.)))
                                         for r in ledger["attempts"])
            ledger["remaining_seconds"] = max(0., ledger["authorized_gpu_wall_seconds"]-ledger["used_seconds"])
            ledger["next_action"] = "inspect " + entry["result"] + "; continue the same master within budget"
            write(args.campaign_ledger, ledger)
            refresh(model)
        print(model, "inference", entry["status"], round(entry["wall_seconds"], 1), flush=True)
    done = all(state["models"][m]["inference"]["status"] == "VERIFIED" for m in order)
    state.update(status="SELECTED_MODELS_HMC_SAMPLING_COMPLETE" if done else "INCOMPLETE_SEE_MODEL_STATUS",
                 finished_utc=stamp())
    refresh()
    return 0 if done else 2


def campaign(args):
    validate_stage_prior(args.through, args.priors)
    if not (0 < args.budget_seconds <= 172800):
        raise ValueError("campaign wall budget must be explicit and at most 48 hours")
    prepared = prepared_selections(args.prepared_root, args.priors) if args.prepared_root else None
    if prepared is not None and args.through not in ("mechanics", "inference"):
        raise ValueError("prepared-root continuation must run mechanics or inference")
    root = args.output
    if root.exists():
        raise FileExistsError("fresh versioned output root required: " + str(root))
    root.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + args.budget_seconds
    ledger = dict(schema="bayesfilter.five_model_hmc_campaign.v1", status="RUNNING",
        started_utc=stamp(), particles=PARTICLES, dtype="float32", tf32=False, horizons=HORIZONS,
        plan=str(PLAN), prior_choice=args.priors, through_phase=args.through, wall_budget_seconds=args.budget_seconds,
        command=[sys.executable, *sys.argv], audit=static_audit(),
        prepared_root=str(args.prepared_root) if prepared else None,
        models={m: ({phase: dict(status=prepared[0]["models"][m][phase]["status"],
                                 reused_from=str(args.prepared_root),
                                 result=prepared[0]["models"][m][phase]["result"])
                    for phase in ("endpoint-audit", "filter-tuning")} if prepared else {})
                for m in MODELS})
    def refresh(phase, model=None):
        ledger["wall_seconds"] = time.monotonic()-started
        ledger["remaining_wall_seconds"] = max(0., deadline-time.monotonic())
        write(root/"master.json", ledger)
        if phase != "terminal":
            action = "continue eligible models through the requested master phases"
        elif ledger["status"] == "ALL_MODELS_HMC_SAMPLING_COMPLETE":
            action = "all five models completed retained HMC sampling; review diagnostics and posterior agreement"
        elif ledger["status"] == "INCOMPLETE_SEE_MODEL_STATUS":
            action = "requested phase incomplete; inspect each model status and repair or stop under the budget"
        elif args.through == "filter-tuning":
            action = "preparation complete; continue mechanics and adaptive HMC with this declared prior within the campaign budget"
        elif args.through == "mechanics":
            action = "mechanics complete; assess measured cost and target choice before inference"
        else:
            action = "requested non-inference phase complete; review model diagnostics"
        write(root/"next-phase-plan.json", dict(updated_utc=stamp(), current_phase=phase,
            current_model=model, part_of_master=True,
            automatic_continuation=phase != "terminal",
            remaining_wall_seconds=ledger["remaining_wall_seconds"],
            action=action,
            retained_rule="cumulative modern R-hat <=1.01 or 20000 samples per chain",
            warmup_rule="at least 2000; latest1000 modern R-hat <=1.05; at most10000",
            redundant_validation_phase=False, models=ledger["models"]))
    refresh("starting")
    # Audit and tune every model before allowing expensive inference to consume the budget.
    # Cheaper targets are first in inference; every model still receives earlier stages.
    for phase in PHASES[:PHASES.index(args.through)+1]:
        if prepared is not None and phase in ("endpoint-audit", "filter-tuning"):
            continue
        order = (("ksc", "range_bearing", "lgssm", "predator_prey", "sir_d18")
                 if phase in ("mechanics", "inference") else MODELS)
        for model in order:
            row = ledger["models"][model]
            prerequisites = PHASES[:PHASES.index(phase)]
            if any(row.get(p, {}).get("status") not in PASS_STATUS[p] for p in prerequisites):
                row[phase] = dict(status="PREREQUISITE_REPAIR_REQUIRED",
                                  reason="preserve failed evidence; do not promote invalid endpoint/filter/mechanics")
                refresh(phase, model)
                continue
            remaining = deadline-time.monotonic()
            if remaining <= 60:
                row[phase] = dict(status="CAMPAIGN_BUDGET_STOP")
                refresh(phase, model)
                continue
            estimate = None
            if phase == "inference":
                cost = read(root/model/"mechanics"/"pilot.json")["seconds_per_four_chain_transition"]
                estimate = dict(hours_at_pilot_rate=cost*(768+2000+1000)/3600,
                    available_hours=remaining/3600,
                    role="compilation-inclusive L1 timing; not a lower bound or convergence forecast")
            stage_out = root/model/phase
            cmd = [str(PYTHON), str(Path(__file__).resolve()), "stage-worker", "--model", model,
                   "--phase", phase, "--output", str(stage_out), "--gpu", str(args.gpu)]
            if args.priors:
                cmd.extend(["--priors", args.priors])
            if phase in ("mechanics", "inference"):
                cmd.extend(["--selection", str(prepared[1][model] if prepared else root/model/"filter-tuning"/"filter-selection.json")])
            # Reserve a fair share for each still-eligible model; unused time carries forward.
            eligible_left = sum(
                all(ledger["models"][m].get(p, {}).get("status") in PASS_STATUS[p]
                    for p in prerequisites)
                for m in order[order.index(model):])
            stage_budget = (remaining / eligible_left if phase == "inference"
                            else min(remaining, 3600.))
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TF_FORCE_GPU_ALLOW_GROWTH="true",
                       BAYESFILTER_PRELOAD_CUSTOM_OP="0", TF_CPP_MIN_LOG_LEVEL="2",
                       BAYESFILTER_WORKER_WALL_DEADLINE=str(time.monotonic()+stage_budget-60))
            log_path = root/(model+"-"+phase+".log")
            item = dict(status="RUNNING", command=cmd, started_utc=stamp(), log=str(log_path),
                        runtime_estimate=estimate, stage_wall_budget_seconds=stage_budget,
                        inference_budget_policy="equal_remaining_share_with_unused_time_carried_forward"
                        if phase == "inference" else None)
            row[phase] = item
            refresh(phase, model)
            begin = time.monotonic()
            with log_path.open("w") as log:
                try:
                    proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                          timeout=stage_budget, check=False)
                    result_path = stage_out/RESULT_NAMES[phase]
                    result = read(result_path) if result_path.exists() else {}
                    status = result.get("status", "STAGE_ERROR")
                    if proc.returncode != 0 and status in PASS_STATUS[phase]:
                        status = "WORKER_EXIT_ERROR"
                    item.update(status=status, returncode=proc.returncode, result=str(result_path))
                except subprocess.TimeoutExpired:
                    item.update(status="STAGE_TIME_BUDGET_STOP", returncode=None,
                                next_action="inspect durable public-tuner progress before any continuation")
            item.update(wall_seconds=time.monotonic()-begin, finished_utc=stamp())
            refresh(phase, model)
            print(model, phase, item["status"], round(item["wall_seconds"], 1), flush=True)
    passed = [m for m in MODELS if ledger["models"][m].get("inference", {}).get("status") == "VERIFIED"]
    through_passed = [m for m in MODELS if ledger["models"][m].get(args.through, {}).get("status") in PASS_STATUS[args.through]]
    stage_complete = len(through_passed) == len(MODELS)
    completion = {"endpoint-audit": "ENDPOINT_AUDIT_COMPLETE_HMC_NOT_RUN",
                  "filter-tuning": "PREPARATION_COMPLETE_HMC_NOT_RUN",
                  "mechanics": "HMC_MECHANICS_COMPLETE_POSTERIOR_NOT_RUN",
                  "inference": "ALL_MODELS_HMC_SAMPLING_COMPLETE"}
    ledger.update(status=completion[args.through] if stage_complete else "INCOMPLETE_SEE_MODEL_STATUS",
                  completed_models=passed, finished_utc=stamp(), scientific_default_promoted=False)
    refresh("terminal")
    return 0 if stage_complete else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "worker", "run", "status", "campaign", "stage-worker", "continue-advance"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--budget-seconds", type=float, default=3600.)
    parser.add_argument("--per-model-timeout", type=float, default=900.)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--through", choices=PHASES, default="inference")
    parser.add_argument("--priors", choices=(LITERATURE_PRIOR, BENCHMARK_PRIOR), default=LITERATURE_PRIOR)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--resume-manifest", type=Path)
    parser.add_argument("--hmc-target-accept", type=float)
    parser.add_argument("--hmc-initial-step-size", type=float)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--advance-root", type=Path, nargs="+")
    parser.add_argument("--campaign-ledger", type=Path)
    args = parser.parse_args(argv)
    if args.command == "stage-worker":
        if args.model is None or args.phase is None:
            parser.error("stage-worker requires --model and --phase")
        return stage_worker(args)
    if args.command == "campaign":
        return campaign(args)
    if args.command == "continue-advance":
        return continue_advance(args)
    if args.command == "audit":
        print(json.dumps(static_audit(), indent=2))
        return 0
    if args.command == "status":
        print(json.dumps(read(args.output / "master.json"), indent=2))
        return 0
    if args.command == "worker":
        if args.model is None:
            parser.error("worker requires --model")
        return worker(args.model, args.output)
    return run(args.output, budget=args.budget_seconds,
               per_model_timeout=args.per_model_timeout, gpu=args.gpu)


if __name__ == "__main__":
    raise SystemExit(main())
