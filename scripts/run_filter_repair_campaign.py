"""Bounded local driver for the September 17 filter execution repair.

This is host orchestration, not a numerical implementation. The command prefix
is deliberately stable. Only registered test groups and fixtures can execute;
there is no arbitrary command, network, installation, merge, or push action.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from enforce_filter_gradient_policy import verify as verify_source_policy
from filter_repair_endpoint_fixtures import FIXTURES as ENDPOINT_FIXTURES
from filter_repair_additional_fixtures import FIXTURES as ADDITIONAL_FIXTURES

ROOT = Path(__file__).resolve().parents[1]


def campaign_output_root(root):
    """Share artifacts and the cumulative budget across linked worktrees."""
    common = subprocess.check_output([
        "git", "rev-parse", "--path-format=absolute", "--git-common-dir",
    ], cwd=root, text=True).strip()
    return Path(common).parent / "docs/plans/artifacts/filter-gradient-repair-20260917"


OUTPUT = campaign_output_root(ROOT)
PLAN = "docs/plans/filter_gradient_repair_master_20260917.md"
BASELINE = "3582b4ac5fea67fb5da7fa60a1cbfaf19df35adf"
BASELINE_ROOT = Path("/tmp/bayesfilter-filter-repair-baseline-3582b4ac")
BASELINE_PARENT_PACKAGES = (
    "experiments/__init__.py", "experiments/dpf_implementation/__init__.py",
)
BUDGET_SECONDS = {"CPU": 8 * 3600, "GPU": 4 * 3600}
TEST_GROUPS = {
    "teacher_identity": ("tests/test_filter_repair_dispatch_identity.py",
        "tests/highdim/test_ledh_contract_e_schema_v2_factory.py",
        "tests/highdim/test_zhao_cui_moment_teacher_integration.py::test_factory_identity_binds_teacher_particle_controls_and_source",
        "tests/highdim/test_zhao_cui_moment_teacher_nonlinear.py::test_repository_factory_binds_nonlinear_model_and_prepared_program",
        "tests/highdim/test_zhao_cui_moment_teacher_actual_sv.py::test_factory_binds_actual_sv_and_rejects_cross_model_substitution"),
    "teacher_consumers": ("tests/highdim/test_zhao_cui_moment_teacher_integration.py",
        "tests/highdim/test_zhao_cui_moment_teacher_nonlinear.py",
        "tests/highdim/test_zhao_cui_moment_teacher_actual_sv.py"),
    "moment_teacher": ("tests/test_filter_repair_moment_teacher.py",),
    "ukf_initializer": ("tests/test_filter_repair_ukf_initializer.py", "tests/highdim/test_p76_ukf_initializer.py"),
    "tp_recursions": ("tests/test_filter_repair_tp_recursions.py",),
    "information_recursions": ("tests/test_filter_repair_information_recursions.py",),
    "remaining_routes": ("tests/test_filter_repair_remaining_routes.py",),
    "tt_preparation": ("tests/test_filter_repair_tt_preparation.py",),
    "filtering_wrappers": ("tests/test_filter_repair_filtering.py", "tests/highdim/test_filtering_kalman_exact.py", "tests/highdim/test_zhao_cui_hmc_default_route_policy.py"),
    "sv_sgqf": ("tests/test_filter_repair_sv_sgqf.py",),
    "start_bank": ("tests/test_filter_repair_start_bank.py", *tuple(
        "tests/test_hmc_warmup.py::" + name for name in (
            "test_start_bank_selector_is_byte_identical_to_frozen_oracle",
            "test_start_bank_selector_preserves_transform_scaling_and_greedy_order",
            "test_start_bank_selector_keeps_less_than_or_equal_tolerance_boundary",
            "test_start_bank_endpoint_exclusion_precedes_prior_eligible_exclusion",
            "test_start_bank_failures_match_oracle_and_carry_no_public_diagnostic",
            "test_start_bank_combined_interpretations_are_shadow_decision_inert",
            "test_start_bank_shadow_failures_are_fixed_bounded_codes",
            "test_start_bank_diagnostic_schema_is_finite_fixed_and_private_safe",
            "test_start_bank_failure_carrier_requires_concrete_validated_type",
        ))),
    "signatures": ("tests/test_filter_repair_signatures.py", "tests/test_kalman_covariance_derivatives_tf.py", "tests/test_linear_correlated_kalman_tf.py", "tests/highdim/test_zhao_cui_moment_teacher_xla.py"),
    "contract_e_reset": ("tests/test_filter_repair_contract_e_reset.py",),
    "student_t_selection": ("tests/highdim/test_c2_student_t_floor.py::test_student_t_margin_vs_dense_grid_max", "tests/highdim/test_c2_student_t_floor.py::test_student_t_nu_criterion_well_posed", "tests/test_filter_repair_student_t_selection.py"),
    "moment_hints": ("tests/test_filter_repair_moment_hints.py",),
    "qr": ("tests/test_filter_repair_qr.py",),
    "tt_scalar": ("tests/test_filter_repair_scalar_tt.py", "tests/highdim/test_zhao_cui_fixed_adjacent_tt_tf.py"),
    "tt_scalar_retained": ("tests/test_filter_repair_scalar_retained.py", "tests/highdim/test_p30_sv_short_sequential_tt_value_path.py", "tests/highdim/test_fixed_branch_derivatives.py::test_scalar_fixed_design_tt_score_path_matches_same_branch_fd_for_exact_transformed_sv", "tests/highdim/test_fixed_branch_derivatives.py::test_scalar_fixed_design_tt_score_path_rejects_missing_manual_model_score_method"),
    "tt_panel_retained": ("tests/test_filter_repair_panel_retained.py",),
    "fixed_fit": ("tests/highdim/test_fixed_branch_fit.py",),
    "fixed_fit_cache": ("tests/highdim/test_fixed_branch_fit.py::test_public_native_fit_matches_original_als_and_reuses_signature",),
    "model_simulation": ("tests/test_filter_repair_model_simulation.py",),
    "squared_density": ("tests/highdim/test_squared_tt_density.py", "tests/highdim/test_failure_exits.py", "tests/test_filter_repair_squared_density.py"),
    "ttsirt": ("tests/highdim/test_zhao_cui_frozen_ttsirt_apf_compiler.py", "tests/highdim/test_p57_m2_fixed_ttsirt_transport_contract.py", "tests/test_filter_repair_ttsirt.py"),
    "tt_algebra": ("tests/highdim/test_tt_algebra.py", "tests/highdim/test_fixed_branch_derivatives.py"),
    "tt_scalar_smoke": ("tests/test_filter_repair_scalar_tt.py::test_complete_value_score_and_fit_veto_parity[False-True]",),
    "factor_geometry": ("tests/test_factor_correlation_geometry.py",),
    "fixed_geometry": ("tests/test_fixed_center_curvature.py", "tests/test_filter_repair_host_io.py", "tests/test_posterior_curvature_refinement.py"),
    "block_geometry": ("tests/test_block_score_geometry.py",),
    "sequential_geometry": ("tests/test_sequential_map_covariance.py",),
    "quadratic_geometry": ("tests/test_quadratic_geometry.py", "tests/test_filter_repair_geometry_parity.py"),
    "posterior_initializer": ("tests/test_posterior_local_initializer.py",),
    "primitives": ("tests/highdim/test_retained_moments.py", "tests/test_filter_repair_primitives.py"),
    "kalman": ("tests/test_compiled_filter_parity_tf.py", "tests/test_compiled_kalman_ukf_runtime.py"),
    "sgqf": ("tests/test_fixed_sgqf_tf.py", "tests/test_fixed_sgqf_scores_tf.py", "tests/test_fixed_sgqf_integration_tf.py", "tests/test_predator_prey_sgqf_neutra_target.py"),
    "genut": ("tests/highdim/test_cubature_genut_batch.py", "tests/highdim/test_genut_batch_primal_parity.py", "tests/highdim/test_genut_batch_general_route_parity.py", "tests/highdim/test_ledh_contract_e_canonical_lgssm_phase5.py"),
    "genut_graph": ("tests/highdim/test_genut_batch_primal_parity.py::test_public_score_traces_in_graph_and_xla",),
    "genut_targets": ("tests/test_genut_neutra_targets.py",),
    "dense_ledh": ("tests/test_experimental_batched_ledh_pfpf_ot_tf.py",),
    "particle": ("tests/test_filter_repair_particles.py",),
    "sinkhorn": ("tests/test_filter_repair_sinkhorn.py",),
    "annealed": ("tests/test_filter_repair_annealed.py",),
    "alg1_ukf": ("tests/test_ledh_pfpf_alg1_ukf_tf.py", "tests/test_filter_repair_slogdet.py"),
    "native_execution": ("tests/test_filter_repair_native_execution.py",),
    "ot_endpoints": ("tests/test_filter_repair_ot_endpoints.py",),
    "random": ("tests/test_filter_repair_random.py::test_uniform_words_preserve_non_xla_double_stream",),
    "random_gpu": ("tests/test_filter_repair_random.py::test_gpu_categorical_engine_diagnostic",),
    "geometry_random": ("tests/test_filter_repair_geometry_random.py",),
    "contract_e": ("tests/highdim/test_ledh_contract_e_canonical_lgssm_phase5.py",),
    "contract_e_strict": tuple("tests/highdim/test_ledh_contract_e_canonical_lgssm_phase5.py::" + name for name in (
        "test_one_batch_one_step_active_reset_and_all_parameter_sensitivity",
        "test_exact_chunk_mixed_reset_full_graph_matches_baseline_ad_rounding",
        "test_float32_shared_core_manual_jvp_matches_forward_autodiff",
    )),
    "cpu_pool": ("tests/test_ssl_lstm_process_parallel.py",),
    "cpu_target": ("tests/test_filter_repair_cpu_target.py",),
    "identity": ("tests/test_hmc_identity.py", "tests/test_filter_repair_host_io.py"),
    "target_failure": tuple("tests/test_common_inference_runtime_contracts.py::" + name for name in (
        "test_target_failure_policy_valid_gaussian_does_not_use_fallback",
        "test_target_failure_policy_declared_support_error_gets_finite_fallback",
        "test_target_failure_policy_nonfinite_value_gradient_is_ambiguous",
        "test_target_failure_policy_does_not_mask_programmer_or_shape_errors",
        "test_target_failure_policy_labels_are_bounded_and_backend_breakdown_is_separate",
        "test_target_failure_policy_classifies_sampler_energy_error_after_valid_target",
    )) + ("tests/test_linear_kalman_svd_tf.py::test_target_failure_policy_does_not_activate_on_valid_lgssm_value",),
    "joint_center": ("tests/test_exact_incumbent.py", "tests/test_joint_center.py"),
    "apf": ("tests/highdim/test_zhao_cui_frozen_proposal_apf_tf.py", "tests/highdim/test_c2_sv_frozen_proposal_apf_tf.py"),
    "preparation": ("tests/test_backend_readiness.py", "tests/highdim/test_bases.py", "tests/highdim/test_c2_hermite_basis.py", "tests/highdim/test_p86_lagrangep_mass_integral.py", "tests/highdim/test_retained_moments.py", "tests/test_filter_repair_primitives.py", "tests/test_filter_repair_consumers.py"),
    "tt": ("tests/highdim/test_squared_tt_density.py", "tests/highdim/test_zhao_cui_actual_sv_batched_tt_tf.py", "tests/highdim/test_zhao_cui_frozen_proposal_apf_tf.py"),
    "tt_actual": ("tests/highdim/test_zhao_cui_actual_sv_batched_tt_tf.py",),
    "tt_contractions": ("tests/highdim/test_p1a_retained_quadratic_form.py", "tests/highdim/test_squared_tt_density.py", "tests/test_filter_repair_tt.py"),
    "tt_value": ("tests/highdim/test_p3_xla_value_parity.py",),
    "tt_maps": ("tests/test_filter_repair_tt_maps.py",),
    "tt_gaussian_consumers": ("tests/highdim/test_c2_gaussian_frozen_target_diagnostics.py", "tests/highdim/test_c2_student_t_floor.py::test_student_t_floor_lane_parity"),
    "tt_adjoint_nodes": ("tests/highdim/test_p2_adjoint_nodes.py", "tests/highdim/test_p2_adjoint_vs_forward_jvp.py"),
    "tt_adjoint": ("tests/highdim/test_p2_adjoint_engine_fd.py",),
    "consumers": ("tests/test_filter_repair_consumers.py",),
    "policy": ("tests/test_filter_repair_campaign.py", "tests/test_filter_repair_policy.py"),
}
FIXTURES = ("rectangular", "factor", "covariance", "sinkhorn_jvp", "sqmc", "dns", "retained_moments", "sgqf_derivatives", "joint_target", "genut", "contract_e", "tt", "tt_adapted", "tt_gaussian", "tt_actual", "tt_adjoint", "tt_scalar", "apf", "particle", "particle_alg1", "cpu_pool", "squared_density", "ttsirt_preparation", "simulation_sv", "simulation_sir", "simulation_predator_prey", "tt_scalar_retained", "tt_panel_retained", "tt_panel_ksc", *ENDPOINT_FIXTURES)


TEST_DEVICES = {"random_gpu": "GPU"}
FIXTURES += ADDITIONAL_FIXTURES


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measurement_harness(fixture):
    names = ("filter_repair_benchmark_worker.py", "measure_filter_xla_memory.py",
             "filter_repair_endpoint_fixtures.py")
    if fixture in ADDITIONAL_FIXTURES:
        names += ("filter_repair_additional_worker.py", "filter_repair_additional_fixtures.py")
    return {name: sha(ROOT / "scripts" / name) for name in names}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def source_hashes():
    paths = git("ls-files", "--cached", "--others", "--exclude-standard", "bayesfilter", *BASELINE_PARENT_PACKAGES, "experiments/dpf_implementation/tf_tfp", "scripts", "tests", "docs/benchmarks").splitlines()
    return {p: sha(ROOT / p) for p in sorted(set(paths))
            if (p.endswith(".py") or p == "scripts/filter_gradient_runtime_policy.json")
            and (ROOT / p).is_file()}


def records():
    return [json.loads(p.read_text()) for p in sorted(OUTPUT.glob("run-*/run.json"))]


def charged_seconds(rows, device):
    # A crashed/unfinished run is charged its whole reserved timeout on resume.
    recorded = sum(row.get("elapsed_seconds", row["timeout_seconds"]) for row in rows if row["device"] == device)
    supplemental = [json.loads(path.read_text()) for path in OUTPUT.glob("supplemental-compute-*.json")]
    return recorded + sum(row["charged_seconds"] for row in supplemental if row["device"] == device)


def test_evidence(run):
    """A pytest exit alone cannot certify that required checks executed."""
    path = Path(run["result"]).with_name("junit.xml")
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return dict(passed=False, reason="missing_or_invalid_junit")
    cases = list(root.iter("testcase"))
    counts = {name: sum(len(case.findall(name)) for case in cases)
              for name in ("failure", "error", "skipped")}
    return dict(passed=bool(cases) and not any(counts.values()), tests=len(cases), **counts)


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def ensure_baseline():
    marker = BASELINE_ROOT / "source-manifest.json"
    if marker.exists():
        manifest = json.loads(marker.read_text())
        if manifest["commit"] != BASELINE or any(sha(BASELINE_ROOT / p) != digest for p, digest in manifest["files"].items()):
            raise RuntimeError("Baseline snapshot changed; comparison is invalid")
        paths = tuple(path for path in BASELINE_PARENT_PACKAGES if path not in manifest["files"])
        if not paths:
            return
        if any((BASELINE_ROOT / path).exists() for path in paths):
            raise RuntimeError("Unrecorded baseline package marker; inspect before recovery")
    else:
        BASELINE_ROOT.mkdir(exist_ok=False)
        manifest = {"commit": BASELINE, "files": {}}
        paths = ("bayesfilter", *BASELINE_PARENT_PACKAGES, "experiments/dpf_implementation/tf_tfp")
    # Parent markers keep Python from resolving the live regular package in
    # preference to an incomplete baseline namespace package.
    archive = subprocess.check_output(["git", "archive", BASELINE, *paths], cwd=ROOT, timeout=60)
    with tarfile.open(fileobj=io.BytesIO(archive)) as handle:
        files = tuple(member.name for member in handle.getmembers() if member.isfile())
        handle.extractall(BASELINE_ROOT, filter="data")
    manifest["files"].update({path: sha(BASELINE_ROOT / path) for path in files})
    save_json(marker, manifest)


def run_job(args):
    rows = records()
    device = args.device
    timeout = 900 if args.action == "test" else 300
    if charged_seconds(rows, device) + timeout > BUDGET_SECONDS[device]:
        raise RuntimeError(f"{device} campaign budget exhausted")
    key = [args.action, args.group, args.arm, args.fixture, args.jit, args.size, args.repeat, device]
    hashes = source_hashes()
    attempts = [row for row in rows if row["key"] == key and row["source_sha256"] == hashes]
    if len(attempts) >= 3:
        raise RuntimeError("Three attempts consumed for this exact job; inspect/repair scope before retry")
    if device == "GPU" and getattr(args, "gpu_preflight", None) is None:
        args.gpu_preflight = check_gpu_idle()
    directory = OUTPUT / f"run-{len(rows) + 1:05d}"
    directory.mkdir(exist_ok=False)
    result = directory / "result.json"
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "2" if device == "GPU" else "-1", "TF_FORCE_GPU_ALLOW_GROWTH": "true", "TF_NUM_INTRAOP_THREADS": "2", "TF_NUM_INTEROP_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"})
    if args.action == "test":
        if args.arm == "before":
            ensure_baseline()
            env["FILTER_REPAIR_SOURCE_ROOT"] = str(BASELINE_ROOT)
        command = [sys.executable, "scripts/filter_repair_test_worker.py", "-q", *TEST_GROUPS[args.group], f"--junitxml={directory / 'junit.xml'}"]
    elif args.action == "measure":
        ensure_baseline()
        source = BASELINE_ROOT if args.arm == "before" else ROOT
        worker = "filter_repair_additional_worker.py" if args.fixture in ADDITIONAL_FIXTURES else "filter_repair_benchmark_worker.py"
        command = [sys.executable, str(ROOT / "scripts" / worker), "--source-root", str(source), "--fixture", args.fixture, "--jit", args.jit, "--size", str(args.size), "--device", device, "--output", str(result)]
    elif args.action == "audit":
        command = [sys.executable, "scripts/audit_filter_gradient_policy.py", "--output", str(directory / "audit.json.gz"), "--markdown", str(directory / "audit.md")]
    elif args.action == "compare":
        command = [sys.executable, "scripts/compare_filter_repair_campaign.py", "--output", str(result)]
    else:
        raise ValueError(args.action)
    record = {"schema": "filter_repair_run.v1", "key": key, "started_utc": datetime.now(timezone.utc).isoformat(), "state": "running", "device": device, "timeout_seconds": timeout, "command": command, "cwd": str(ROOT), "environment": {k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONHASHSEED")}, "git_head": git("rev-parse", "HEAD"), "git_diff_stat": git("diff", "--stat"), "source_sha256": hashes, "plan": PLAN, "result": str(result), "log": str(directory / "process.log")}
    if getattr(args, "gpu_preflight", None) is not None:
        record["gpu_preflight"] = args.gpu_preflight
    save_json(directory / "run.json", record)
    started = time.monotonic()
    print(json.dumps({"run": str(directory), "command": command}), flush=True)
    with (directory / "process.log").open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 130 if isinstance(exc, KeyboardInterrupt) else 124
    if args.action == "test":
        record["test_evidence"] = test_evidence(record)
        if code == 0 and not record["test_evidence"]["passed"]:
            code = 1
    record.update(state="passed" if code == 0 else "failed", exit_code=code, elapsed_seconds=time.monotonic() - started)
    save_json(directory / "run.json", record)
    print(json.dumps({"state": record["state"], "elapsed_seconds": record["elapsed_seconds"], "log": record["log"]}), flush=True)
    if code:
        print((directory / "process.log").read_text()[-14000:])
    return code


def gate():
    try:
        policy = verify_source_policy(ROOT, ROOT / "scripts/filter_gradient_runtime_policy.json")
    except (OSError, ValueError, KeyError, SyntaxError) as exc:
        policy = {"passed": False, "error": str(exc)}
    ledger = json.loads((ROOT / "docs/plans/filter_gradient_repair_ledger_20260917.json").read_text())
    pending = [item["id"] for item in ledger["findings"] if item["status"] != "closed" or not item.get("evidence") or any(not (ROOT / path).is_file() for path in item.get("evidence", []))]
    if {item["id"] for item in ledger["findings"]} != {f"F{i:02d}" for i in range(1, 21)}:
        pending.append("incomplete_finding_inventory")
    rows = records()
    current = source_hashes()
    missing = []
    for group in TEST_GROUPS:
        candidates = [row for row in rows if row["key"][:3] == ["test", group, "after"]
                      and row["state"] == "passed" and row["source_sha256"] == current
                      and row["device"] == TEST_DEVICES.get(group, "CPU")
                      and test_evidence(row)["passed"]]
        if not candidates:
            missing.append(group)
    comparisons = [row for row in rows if row["key"][0] == "compare" and row["state"] == "passed" and row["source_sha256"] == current]
    complete = not pending and not missing and bool(comparisons) and policy["passed"]
    print(json.dumps({"merge_allowed": complete, "open_findings": pending, "missing_current_tests": missing, "current_comparison": bool(comparisons), "source_policy": policy}, indent=2))
    return 0 if complete else 1


def check_gpu_idle():
    samples, consecutive_idle = [], 0
    # Utilization is sampled over an interval and can outlive the prior worker.
    for attempt in range(6):
        output = subprocess.check_output([
            "nvidia-smi", "-i", "2", "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=5)
        memory, utilization = (int(value.strip()) for value in output.strip().split(","))
        samples.append(dict(memory_mib=memory, utilization_percent=utilization))
        consecutive_idle = consecutive_idle + 1 if memory <= 100 and utilization <= 5 else 0
        if consecutive_idle == 2:
            return samples
        if attempt < 5:
            time.sleep(2)
    raise RuntimeError(f"GPU2 contention veto after bounded recheck: {samples}")


def check_matrix_state(frozen):
    if (OUTPUT / "pause-request.json").exists():
        raise RuntimeError("Campaign paused between workers; restart the matrix to resume")
    if source_hashes() != frozen:
        raise RuntimeError("Source changed during the campaign matrix")


def run_matrix(args):
    """Resume registered jobs sequentially; failures retain their original evidence."""
    from compare_filter_repair_campaign import (
        ONE_SIZE, baseline_compilation_failure, compare_pair, current_provenance,
    )

    frozen = source_hashes()
    if args.stage == "tests":
        for group in TEST_GROUPS:
            check_matrix_state(frozen)
            device = TEST_DEVICES.get(group, "CPU")
            if any(row["key"][:3] == ["test", group, "after"] and row["state"] == "passed"
                   and row["source_sha256"] == frozen and row["device"] == device
                   and test_evidence(row)["passed"] for row in records()):
                continue
            job = argparse.Namespace(**vars(args))
            job.action, job.group, job.arm, job.device = "test", group, "after", device
            if device == "GPU":
                job.gpu_preflight = check_gpu_idle()
            code = run_job(job)
            if code:
                return code
        return 0

    ensure_baseline()
    marker = json.loads((BASELINE_ROOT / "source-manifest.json").read_text())
    available = {}
    for row in records():
        if row["key"][0] != "measure" or not Path(row["result"]).is_file():
            continue
        value = json.loads(Path(row["result"]).read_text())
        try:
            current_provenance(row, value, row["key"][2], measurement_harness(row["key"][3]), marker["files"])
        except (ValueError, KeyError):
            continue
        available[tuple(row["key"][2:])] = (row, value)

    def execute(name, size, repeat, arm, mode):
        check_matrix_state(frozen)
        device = "CPU" if name == "cpu_pool" else "GPU"
        key = (arm, name, mode, size, repeat, device)
        if key in available:
            row, value = available[key]
        else:
            job = argparse.Namespace(**vars(args))
            job.action, job.fixture, job.size, job.repeat = "measure", name, size, repeat
            job.arm, job.jit, job.device = arm, mode, device
            if device == "GPU":
                job.gpu_preflight = check_gpu_idle()
            code = run_job(job)
            row = records()[-1]
            if not Path(row["result"]).is_file():
                raise RuntimeError(f"Missing measurement artifact after exit {code}: {row['log']}")
            value = json.loads(Path(row["result"]).read_text())
            current_provenance(row, value, arm, measurement_harness(name), marker["files"])
            available[key] = row, value
        if value["status"] != "passed" and (arm != "before" or not baseline_compilation_failure(value)):
            raise RuntimeError(f"Measurement failure requires repair: {row['result']}")
        return row, value

    fixtures = ((args.fixture,) if args.selection == "fixture" else
                ADDITIONAL_FIXTURES if args.selection == "additional" else
                ENDPOINT_FIXTURES if args.selection == "new" else FIXTURES)
    for name in fixtures:
        for size in ((1,) if name in ONE_SIZE else (1, 2)):
            for repeat in (range(1) if args.stage == "qualify" else range(3)):
                for mode in ("off", "on"):
                    before_run, before = execute(name, size, repeat, "before", mode)
                    baseline_attempt = before_run["result"]
                    if before["status"] != "passed" and mode == "on":
                        before_run, before = execute(name, size, repeat, "before", "off")
                    if before["status"] != "passed":
                        before_run, before = execute(name, size, repeat, "before", "eager")
                    after_run, after = execute(name, size, repeat, "after", mode)
                    error = compare_pair(before, after)
                    print(json.dumps({"parity": "passed", "fixture": name, "size": size,
                        "repeat": repeat, "mode": mode, "max_absolute_error": error,
                        "before": before_run["result"], "after": after_run["result"],
                        "baseline_attempt": baseline_attempt}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "test", "measure", "matrix", "pause", "audit", "compare", "gate"))
    parser.add_argument("--stage", choices=("qualify", "repeat", "tests"), default="qualify")
    parser.add_argument("--selection", choices=("fixture", "new", "additional", "all"), default="all")
    parser.add_argument("--group", choices=tuple(TEST_GROUPS), default="policy")
    parser.add_argument("--fixture", choices=FIXTURES, default="covariance")
    parser.add_argument("--arm", choices=("before", "after"), default="after")
    parser.add_argument("--jit", choices=("on", "off", "eager"), default="on")
    parser.add_argument("--size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--repeat", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--device", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.action == "pause":
        save_json(OUTPUT / "pause-request.json", {"requested_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"state": "pause_requested", "active_worker": "allowed_to_finish"}))
        return 0
    with (OUTPUT / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "status":
            rows = records()
            print(json.dumps({"branch": git("branch", "--show-current"), "baseline": BASELINE, "runs": len(rows), "budget_seconds": BUDGET_SECONDS, "charged_seconds": {d: charged_seconds(rows, d) for d in BUDGET_SECONDS}, "artifact_root": str(OUTPUT)}, indent=2))
            return 0
        if args.action == "gate":
            return gate()
        if args.action == "matrix":
            (OUTPUT / "pause-request.json").unlink(missing_ok=True)
            return run_matrix(args)
        return run_job(args)


if __name__ == "__main__":
    raise SystemExit(main())
