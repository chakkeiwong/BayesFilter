"""Preparation and measured-cost decisions for the K0--K7 diagnostic campaign.

Planning is framework-free. Numerical references are imported only by prepare,
with GPUs explicitly hidden. This module does not implement a tuner.
"""
from __future__ import annotations

from dataclasses import replace
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .designs import ScenarioSpec, ValidationDesign, digest, seed_for
from .ssm_campaign_profiles import PROFILES
from .storage import read_json, write_json

PLAN = "docs/plans/bayesfilter-hmc-state-space-48h-plan-2026-09-25.md"
COUNTS = dict(warmup_chunk_results=500, warmup_min_results=2000,
              warmup_check_window_results=1000, warmup_max_results=10000,
              retained_chunk_results=500, retained_min_results=1000,
              retained_max_results=10000)
C1_SERVICE = "bayesfilter-hmc-c1-repaired-20260925-r1.service"
C1_RECEIPT = "docs/plans/artifacts/hmc-budget-debug-2026-09-25/confirmation-r1/execution.json"
GRANT = "docs/plans/artifacts/hmc-additional-gpu-2026-09-25/grant-ledger.json"
K6_FILES = (
    "docs/benchmarks/configs/multidim_lgssm_full_estimation_rerun_2026_07_13.json",
    "docs/benchmarks/artifacts/multidim_lgssm_full_estimation_rerun_2026_07_13/fixture_T120_seed20260709_301.json",
    "docs/plans/artifacts/multidim-triangular-lgssm-neutra-hmc-2026-07-08/lower_triangular_lgssm_contract_v1.json",
)
SNAPSHOT_FILES = (*K6_FILES, PLAN, "scripts/prepare_hmc_state_space_campaign.py",
    "scripts/run_hmc_c1_recovery_then_ssm.py",
    "scripts/run_hmc_funded_ssm_queue.py",
    "tests/inference_validation/test_ssm_xla_full_chain.py",
    "docs/plans/bayesfilter-hmc-ssm-xla-preflight-repair-2026-09-29.md",
    "docs/plans/bayesfilter-hmc-shared-gpu-recovery-2026-09-29.md",
    "docs/plans/bayesfilter-hmc-c1-recovery-and-ssm-launch-2026-09-28.md",
    "docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase15-manual-score-xla-compile-gate-result-2026-07-08.md")


def suite(name, designs, **metadata):
    from .execution import plan_suite
    value = {"schema":"bayesfilter.inference_validation_suite.v1", "suite_id":name,
             "profile":"ssm", "profiles":{"ssm":sorted({d.engine for d in designs})},
             "campaign_metadata":{"plan":PLAN, **metadata},
             "designs":[d.payload() for d in designs],
             "required_coverage":[{**d.coverage_key(), "design_id":d.design_id} for d in designs]}
    plan_suite(value)
    return value


def main_design(profile, dataset, slot):
    precision = {name:(.01 if name.endswith("_sd") else .02) for name in profile.names}
    # K0/K7 are the predeclared sibling-mechanism cases. All other fits assess
    # one member. The full verified set is always preserved.
    members = 2 if profile.case in {"K0", "K7"} else 1
    seconds = 2100 if profile.case not in {"K6", "K7"} else 3600
    return ValidationDesign(
        design_id=f"{profile.case}-{dataset['dataset_id']}-s{slot}", engine="accuracy",
        scenario=ScenarioSpec(profile.target,"ordinary"), replications=1, draws=500,
        seed=seed_for(20260926, profile.case, dataset["dataset_id"], slot, "main")[0],
        budget_seconds=seconds, purpose="K0--K7 actual-filter public procedure validation",
        numerical_provenance=PLAN + "; frozen synthetic stress hypotheses; descriptive few-fit evidence",
        device="gpu", phase="confirmation", posterior_cap=10000, measurement_draws=128,
        l_grid=(3,5,9,13,18,25), mcse_tolerance=.02, accuracy_tolerance=.02,
        options={"data":dataset["data"], "dataset_id":dataset["dataset_id"],
            "data_seed":dataset["data_seed"], "data_version":dataset["data_sha256"],
            "plan_file":PLAN, "native_search":True, "isolate_fits":True,
            "fit_process_timeout_seconds":seconds-10,
            "timeout_policy":{"max_extension_seconds":0., "shutdown_grace_seconds":5.,
                              "extension_mode":"observed_intervals", "gpu_admission_wait_seconds":600.},
            "posterior_members":"selected", "member_rule":"shortest_verified_l",
            "posterior_member_count":members, "posterior_settings":dict(COUNTS),
            "mcse_tolerance_by_parameter":precision, "posterior_precision_method":"lugsail",
            "reference_settings":dataset["reference_settings"], "campaign_case":profile.case})


def build_suites(datasets, *, shared_contention=False):
    main, prices, mechanics, bridge = [], [], [], []
    for target, profile in PROFILES.items():
        entries = datasets[target]
        for dataset in entries:
            for slot in range(4 if profile.case == "K6" else 2):
                main.append(main_design(profile,dataset,slot))
        base = main_design(profile, entries[0], 0)
        options = {**base.options, "fit_process_timeout_seconds":1175.}
        prices.append(replace(base, design_id=f"price-{profile.case}", phase="development",
            seed=seed_for(20260926,profile.case,"cold-fit-pricing")[0], budget_seconds=1185.,
            options=options, purpose="complete cold fit price with unchanged main numerical counts"))
        mechanics.append(replace(base, design_id=f"preflight-{profile.case}", engine="mechanics",
            scenario=ScenarioSpec(target,"frozen"), replications=4, phase="development",
            budget_seconds=60., options={"data":entries[0]["data"], "plan_file":PLAN},
            purpose="GPU/XLA independent value and total-score preflight"))
        if profile.case in {"K0", "K7"}:
            for route in ("ordinary", "prepared"):
                small = dict(warmup_chunk_results=4, warmup_min_results=4,
                    warmup_check_window_results=4, warmup_max_results=8,
                    retained_chunk_results=4, retained_min_results=4, retained_max_results=8)
                bridge.append(replace(base, design_id=f"bridge-{profile.case}-{route}",
                    scenario=ScenarioSpec(target,route), phase="development", draws=8,
                    seed=seed_for(20260926,profile.case,route,"gpu-bridge")[0],
                    budget_seconds=180., posterior_cap=8, measurement_draws=64,
                    l_grid=(2,3), step_size=.25,
                    options={"data":entries[0]["data"], "plan_file":PLAN,
                        "isolate_fits":True, "fit_process_timeout_seconds":170.,
                        "timeout_policy":{"extension_mode":"observed_intervals", "gpu_admission_wait_seconds":600.},
                        "posterior_members":"selected", "member_rule":"first_verified",
                        "acceptance_policy":{"target":.70,"practical_region":[.50,.90],
                                             "repair_region":[.41,.99]},
                        "posterior_settings":small,
                        "search":{"pilot_enabled":False,"refinement_rounds":0,
                                  "total_budget_units":32,"repair_reserve_units":8,"evidence_rungs":[1]},
                        "reference_settings":entries[0]["reference_settings"]},
                    purpose="GPU public ordinary/prepared bridge; no posterior promotion or cost inference"))
    assert len(main) == 32 and len(prices) == 8
    anchor = datasets["ssm_campaign_location"][0]
    stationarity = [ValidationDesign(design_id="K0-stationarity-"+control, engine="invariance",
        scenario=ScenarioSpec("ssm_campaign_location","frozen",control,start="reference"),
        replications=512,draws=64,seed=seed_for(20260926,"stationarity",control)[0],
        budget_seconds=120.,purpose="actual-filter fixed-kernel stationarity and defective energy control",
        numerical_provenance=PLAN+"; fixed engineering epsilon .12, L=3; no tuning or stopping calibration",
        device="cpu_reference",step_size=.12,leapfrog_steps=3,rank_draws=7,
        options={"data":anchor["data"],"plan_file":PLAN}) for control in ("baseline","noop","wrong_energy")]
    if shared_contention:
        def shared(design, extension):
            return replace(design, budget_seconds=design.budget_seconds+extension,
                options={**design.options, "timeout_policy":{
                    **design.options["timeout_policy"], "gpu_admission_mode":"shared",
                    "max_contention_retries":1, "max_extension_seconds":extension}})
        main = [shared(d, 0.) for d in main]  # Measured main margins are set by price().
        prices = [shared(d, 600.) for d in prices]
        bridge = [shared(d, 170.) for d in bridge]
    suites = {"main-unpriced":suite("state-space-48h",main, main_fit_count=32,
                launch_status="unpriced inventory; run only after preflight and measured affordability",
                rhat_role="posterior only, never tuning membership"),
            "pricing":suite("state-space-pricing",prices, cost_scope="one complete fit per K case"),
            "preflight-mechanics":suite("state-space-preflight-mechanics",mechanics),
            "preflight-pipeline":suite("state-space-preflight-pipeline",bridge),
            "stationarity":suite("state-space-cpu-stationarity",stationarity)}
    if shared_contention:
        for value in suites.values():
            value["campaign_metadata"]["shared_contention"] = True
    return suites


def prepare(root):
    """CPU-only data/reference preparation; refuses an existing output root."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("prepare requires CUDA_VISIBLE_DEVICES=-1 before framework import")
    from .ssm_campaign_profiles import generate_data
    from .references import ssm
    from .execution import source_state
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    datasets, checks = {}, {}
    for target, p in PROFILES.items():
        entries = []
        for number in range(1 if p.case == "K6" else 2):
            data_seed = [20260926, 100+10*int(p.case[1:])+number]
            data = (ssm.multivariate_definition()[1]["observations"] if p.case == "K6"
                    else generate_data(target,data_seed))
            name = f"{p.case}-data-{'AB'[number]}"
            settings = {"resolution":161, "sensitivity":[.002,.002]}
            attempts = []
            for resolution in (161,321,641):
                settings["resolution"] = resolution
                _, metadata = ssm.posterior_reference(target,4096,
                    seed_for(20260926,name,"reference")[0],data,**settings)
                attempts.append(metadata)
                if metadata["checked"] or p.reference == "unavailable": break
            entry = {"dataset_id":name,"data_seed":data_seed if p.case != "K6" else [20260709,301],
                "data":data, "data_sha256":digest(data), "reference_settings":dict(settings),
                "reference_checked":metadata["checked"], "profile":p.payload()}
            entries.append(entry)
            checks[name] = {"attempts":attempts, "sanity":ssm.sanity_comparators(target,data),
                            "reference_checked":metadata["checked"]}
            write_json(root/"references"/(name+".json"),checks[name])
        datasets[target] = entries
        write_json(root/"datasets.json",datasets)
    from .references.ssm import nonlinear_latent_reference, nonlinear_likelihood
    short = datasets["ssm_campaign_nonlinear"][0]["data"][:2]
    latent = [nonlinear_latent_reference([.7,.8],short,order) for order in (9,15,21)]
    write_json(root/"nonlinear-approximation.json",{
        "target":"true Model B, T=2", "orders":[9,15,21], "log_likelihood":latent,
        "last_difference":abs(latent[-1]-latent[-2]),
        "refinement_tolerance":.005, "refinement_passed":abs(latent[-1]-latent[-2])<=.005,
        "sigma_point_log_likelihood":float(nonlinear_likelihood([.7,.8],short)),
        "role":"separate approximation-error diagnostic, not a sampler reference",
        "long_horizon_accuracy_established":False})
    for name, value in build_suites(datasets).items():
        write_json(root/(name+".json"),value)
    manifest = {"plan_file":PLAN, "source":source_state(), "command":[sys.executable,*sys.argv],
        "environment":sys.executable, "gpu_intentionally_hidden":True,
        "wall_seconds":time.monotonic()-started, "dataset_count":15, "main_fit_count":32,
        "reference_status":{k:v["reference_checked"] for k,v in checks.items()},
        "gpu_preflight":"pending C1 settlement", "result_file":str(root/"manifest.json")}
    write_json(root/"manifest.json",manifest)
    return manifest


def free_gpu_seconds(ledger, *, c1_active):
    """Check actual settled charges, preserving the separate NeuTra reserve."""
    if c1_active or ledger.get("active_reservation"):
        raise ValueError("C1 or another campaign still has an active reservation")
    records = ledger["records"]
    if not any(str(r["receipt"]).endswith(C1_RECEIPT) for r in records):
        raise ValueError("C1 terminal enclosing receipt has not been settled")
    paths = [str(Path(r["receipt"]).resolve()) for r in records]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate enclosing receipt charges")
    values = [r["elapsed_seconds"] for r in records]
    if any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("invalid settled receipt cost")
    grant = ledger["grant_gpu_seconds"]
    if type(grant) not in (int,float) or not math.isfinite(grant) or grant < 0:
        raise ValueError("invalid grant size")
    remaining = grant-sum(values)
    if not math.isclose(remaining,ledger["remaining_gpu_seconds"],abs_tol=1e-6):
        raise ValueError("grant balance and settled receipts disagree")
    reserved = ledger["allocations"].get("canonical_neutra_pricing_reserved_only",0)
    return max(0.,remaining-reserved)


def freeze_priced_main(template, pilot_index, datasets, *, available_seconds):
    """No censored/failed fit can set a runtime price; preserve every main slot."""
    from .execution import plan_suite
    eligible, dispositions, prices = [], [], {}
    expected = {d["options"]["campaign_case"] for d in template["designs"]}
    if pilot_index.get("source") is None:
        raise ValueError("pilot source provenance missing")
    shared = template.get("campaign_metadata",{}).get("shared_contention",False)
    if type(shared) is not bool:
        raise ValueError("shared_contention must be boolean")
    expected_suites = build_suites(datasets, shared_contention=shared)
    expected_pilots = expected_suites["pricing"]
    if digest(template) != digest(expected_suites["main-unpriced"]):
        raise ValueError("main inventory changed after preparation")
    if pilot_index.get("suite_identity") != digest(expected_pilots):
        raise ValueError("pilot does not match the frozen complete-fit workload")
    if pilot_index.get("execution_options") != {"reuse_leapfrog_graphs":True,"share_unused_budget":False}:
        raise ValueError("pilot graph reuse/budget execution differs from main")
    if not math.isfinite(available_seconds) or available_seconds < 0:
        raise ValueError("available allocation must be finite and nonnegative")
    for case in sorted(expected):
        job = pilot_index["jobs"].get("price-"+case,{})
        reason = "no complete uncensored pilot"
        attempts = job.get("attempts",[])
        cost = None
        if job.get("status") == "complete" and len(attempts) == 1:
            receipt = attempts[0]
            if (receipt.get("exit_code") == 0 and receipt.get("status") == "complete"
                    and not receipt.get("timed_out",False)):
                seconds = receipt.get("elapsed_seconds")
                if type(seconds) in (int,float) and math.isfinite(seconds) and seconds > 0:
                    cost = math.ceil(1.5*seconds)
        rows = [d for d in template["designs"] if d["options"]["campaign_case"] == case]
        # Assessing the full intended member workload is required even if the
        # candidate/posterior fails scientifically. An empty search is not a
        # cheap complete price for posterior work.
        if cost is not None:
            result = read_json(job["result"])
            if result.get("execution_status") != "complete":
                cost, reason = None, "pilot worker did not finish cleanly"
            # Read native pipeline output rather than infer delivery from a
            # posterior finding or the child's exit status.
            pipeline = Path(job["result"]).parent/"replication-0000/pipeline.json"
            payload = read_json(pipeline) if pipeline.exists() else {}
            members = [m for m in payload.get("members",[]) if m.get("status")=="assessed"]
            requested = rows[0]["options"]["posterior_member_count"]
            if payload.get("completion") != "complete" or len(members) != requested:
                cost, reason = None, "pilot did not assess the declared member workload"
            elif any(m.get("recorded_retained_count",0) < COUNTS["retained_min_results"] for m in members):
                cost, reason = None, "pilot did not execute the declared retained sampling workload"
        if cost is not None:
            cap = sum(d["budget_seconds"] for d in rows)
            needed = cost*len(rows)
            reference_ok = all(e["reference_checked"] for e in datasets[rows[0]["scenario"]["target"]]) or case=="K6"
            if not reference_ok: reason = "reference unavailable"
            elif needed > cap: reason = "measured price exceeds lane ceiling"
            elif needed > available_seconds: reason = "insufficient remaining grant"
            else:
                for row in rows:
                    d = ValidationDesign.from_payload(row)
                    # First 1.25x is the nominal limit; the last .25x can be
                    # used only under the existing progress/contention policy.
                    nominal = max(1.,cost/1.5*1.25-10.)
                    options = {**d.options,"fit_process_timeout_seconds":nominal,
                        "timeout_policy":{**d.options["timeout_policy"],
                                          "max_extension_seconds":max(0.,cost-nominal-10.),
                                          "progress_grace_seconds":60.,"shutdown_grace_seconds":5.}}
                    eligible.append(replace(d,budget_seconds=float(cost),options=options))
                available_seconds -= needed
                reason = "funded at measured price"
            prices[case] = {"pilot_seconds":seconds,"allowance_per_fit":cost,
                            "planned_fits":len(rows),"needed_seconds":needed,"lane_ceiling":cap}
        dispositions.extend({"design_id":r["design_id"],"case":case,"disposition":reason} for r in rows)
    resolved = suite("state-space-48h-priced",eligible, main_fit_count=32,
                     funded_fit_count=len(eligible), disposition=dispositions)
    resolved["required_coverage"] = template["required_coverage"]
    plan_suite(resolved)
    return resolved, {"prices":prices,"dispositions":dispositions,
        "remaining_unallocated_seconds":available_seconds,"margin":1.5,
        "runtime_tail_guarantee":False,"original_denominator":32}


def freeze_source(root):
    """Freeze the package, launcher, plan and model dependencies."""
    import shutil
    from .execution import REPO, source_state
    from .storage import file_hash
    root = Path(root).resolve()
    destination = root/"source"
    if destination.exists():
        raise ValueError("source already frozen; use a fresh preparation version")
    before = source_state()
    shutil.copytree(REPO/"bayesfilter",destination/"bayesfilter",
                    ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    for relative in SNAPSHOT_FILES:
        target = destination/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(REPO/relative,target)
    after = source_state()
    if before != after:
        raise ValueError("live source changed during snapshot; preserve this attempt and freeze a fresh version")
    write_json(destination/"source_snapshot.json",{"git_commit":before["commit"],"source":before})
    diff = subprocess.run(["git","diff","--binary","--","bayesfilter",*K6_FILES],
                          cwd=REPO,check=True,capture_output=True).stdout
    (root/"source-diff.patch").write_bytes(diff)
    write_json(root/"source.json",before)
    write_json(root/"dependencies.json",{path:file_hash(destination/path) for path in SNAPSHOT_FILES})
    write_json(root/"prepared-inputs.json",{p.name:file_hash(p) for p in root.glob("*.json")
        if p.name in {"datasets.json","main-unpriced.json","pricing.json",
                      "preflight-mechanics.json","preflight-pipeline.json","stationarity.json"}})
    return before


def c1_is_active():
    result = subprocess.run(["systemctl","--user","show",C1_SERVICE,"--property=ActiveState","--value"],
                            text=True,capture_output=True,check=True,timeout=10)
    return result.stdout.strip() not in {"inactive","failed"}


def completed_fit_runtime(directory):
    """Validate the final child, including a successful contention recovery."""
    from .storage import file_hash
    exits = sorted(Path(directory).glob("process-attempt-*-exit.json"))
    if not exits:
        raise ValueError("preflight lacks a completed child receipt")
    path = exits[-1]
    receipt = read_json(path)
    if (receipt.get("status") != "complete" or receipt.get("exit_code") != 0
            or receipt.get("assessment_sha256") != file_hash(Path(directory)/"independent_assessment.json")):
        raise ValueError("preflight final child or assessment is incomplete")
    return read_json(path.with_name(path.name.replace("-exit.json", "-manifest.json")))["runtime"]


def check_preflight(root):
    """GPU evidence and actual candidate lifecycle, separate from posterior success."""
    root = Path(root)
    source = read_json(root/"source.json")
    for stage in ("preflight-mechanics","preflight-pipeline"):
        index = read_json(root/stage/"run_index.json")
        if index["source"]["identity"] != source["identity"]:
            raise ValueError("preflight source mismatch")
        expected = read_json(root/(stage+".json"))
        if index["suite_identity"] != digest(expected):
            raise ValueError("preflight design mismatch")
        for design in expected["designs"]:
            job = index["jobs"].get(design["design_id"],{})
            if job.get("status") != "complete":
                raise ValueError("preflight incomplete: "+design["design_id"])
            result = read_json(job["result"])
            runtime = result["runtime"]
            if stage == "preflight-mechanics":
                if result["assessment"]["finding"] != "mechanics_passed":
                    raise ValueError("preflight value/score discrepancy")
                if any("GPU" not in result["assessment"].get(k,"") for k in ("value_device","score_device")):
                    raise ValueError("preflight filter value/score did not execute on GPU")
            else:
                directory = Path(job["result"]).parent/"replication-0000"
                runtime = completed_fit_runtime(directory)
                observed = read_json(directory/"independent_assessment.json")
                if observed["inventory"]["failures"]:
                    raise ValueError("preflight candidate inventory failed")
                assessed = [m for m in observed["members"] if m.get("status")=="assessed"]
                if not assessed or any(not m["warmup_exclusion_matches"] for m in assessed):
                    raise ValueError("preflight did not exercise retained-member lifecycle")
            memory = runtime["memory_policy"]
            if (runtime.get("jit_compile") is not True or "GPU" not in runtime.get("gpu_tensor_device","")
                    or not memory.get("configured_before_logical_device_initialization")
                    or not memory.get("all_physical_devices_memory_growth")
                    or not memory.get("physical_devices")):
                raise ValueError("preflight lacks trusted GPU/XLA/growth provenance")
    return {"engineering_preflight_passed":True,"posterior_claim":False}


def price(root, ledger, *, allocation_cap_seconds=None):
    root = Path(root)
    check_preflight(root)
    index = read_json(root/"pricing/run_index.json")
    if index["source"]["identity"] != read_json(root/"source.json")["identity"]:
        raise ValueError("pricing source mismatch")
    available = free_gpu_seconds(ledger,c1_active=c1_is_active())
    state = read_json(root/"checkpoint.json")
    wall_available = max(0., state["started_epoch"]+46*3600-time.time()-30.)
    if allocation_cap_seconds is not None and (not math.isfinite(allocation_cap_seconds)
                                             or allocation_cap_seconds < 0):
        raise ValueError("main allocation cap must be finite and nonnegative")
    cap = min(22*3600., available-4*3600., wall_available,
              allocation_cap_seconds if allocation_cap_seconds is not None else 22*3600.)
    # Preserve the plan's four-hour repair/contention reserve; never allocate
    # another grant or count C1's active reservation as money-free time.
    suite_value, costs = freeze_priced_main(read_json(root/"main-unpriced.json"),index,
        read_json(root/"datasets.json"),available_seconds=max(0.,cap))
    costs["allocation_cap_seconds"] = max(0.,cap)
    costs["remaining_wall_seconds"] = wall_available
    write_json(root/"main.json",suite_value)
    write_json(root/"costs.json",costs)
    return costs


def run_stage(root, stage, ledger_path, *, gpu_admission_mode="idle", allocation_cap_seconds=None):
    """Run the existing executor in a bounded systemd cgroup, then charge once.

    Call from trusted execution. systemd kills the entire group, including fit
    children that create their own process sessions. No live C1 is interrupted.
    """
    from datetime import datetime, timezone
    from .execution import plan_suite
    root, ledger_path = Path(root).resolve(), Path(ledger_path).resolve()
    ledger = read_json(ledger_path)
    available = free_gpu_seconds(ledger,c1_active=c1_is_active())
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH","").lower() != "true":
        raise ValueError("GPU stages require TF_FORCE_GPU_ALLOW_GROWTH=true before launch")
    device = os.environ.get("CUDA_VISIBLE_DEVICES","")
    if not device.startswith("GPU-") or "," in device:
        raise ValueError("select one GPU UUID for unambiguous workload accounting")
    if stage in {"pricing","main"}: check_preflight(root)
    if stage == "main": price(root,ledger,allocation_cap_seconds=allocation_cap_seconds)
    value = read_json(root/(stage+".json"))
    plan = plan_suite(value)
    if not plan["jobs"]:
        return {"status":"unfunded", "reason":"no main lanes met reference and affordability criteria"}
    seconds = plan["maximum_worker_seconds"]
    # Thirty seconds is an operational shutdown/accounting allowance, not a
    # runtime-tail assertion. It is reserved and charged, never hidden overhead.
    reservation = seconds+30.
    if reservation > available:
        raise ValueError("stage exceeds actual remaining grant after other reservations")
    state_path = root/"checkpoint.json"
    state = read_json(state_path) if state_path.exists() else {
        "started_epoch":time.time(),"stages":{},"plan_file":PLAN,"gpu_uuid":device}
    if state.get("gpu_uuid") != device:
        raise ValueError("pricing and main execution must use the same selected GPU UUID")
    now = time.time()
    if now >= state["started_epoch"]+42*3600 or now+reservation > state["started_epoch"]+46*3600:
        raise ValueError("new work would exceed the declared 42/46-hour campaign limits")
    if (root/stage).exists() or stage in state["stages"]:
        raise ValueError("stage already attempted; preserve evidence and use a reviewed repair attempt")
    snapshot = read_json(root/"source.json")
    frozen = root/"source"
    from .storage import file_hash
    if any(file_hash(frozen/path) != sha for path,sha in snapshot["files"].items()):
        raise ValueError("frozen numerical source changed")
    for manifest,base in (("dependencies.json",frozen),("prepared-inputs.json",root)):
        if any(file_hash(base/path) != sha for path,sha in read_json(root/manifest).items()):
            raise ValueError("frozen data or prepared suites changed")
    command = [sys.executable,"-m","bayesfilter.testing.inference_validation","run",
               str(root/(stage+".json")),"--output",str(root/stage),"--max-workers","1"]
    if stage != "preflight-mechanics": command.append("--reuse-leapfrog-graphs")
    unit = "bayesfilter-ssm-"+stage+"-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    launch = ["systemd-run","--user","--wait","--collect","--pipe","--service-type=exec",
        "--unit",unit,"--property=KillMode=control-group",f"--property=RuntimeMaxSec={math.ceil(seconds)}",
        "--property=TimeoutStopSec=10",f"--working-directory={frozen}",
        f"--setenv=CUDA_VISIBLE_DEVICES={device}","--setenv=TF_FORCE_GPU_ALLOW_GROWTH=true",
        "--setenv=TF_NUM_INTRAOP_THREADS=2","--setenv=TF_NUM_INTEROP_THREADS=2",
        "--setenv=OMP_NUM_THREADS=2","--setenv=BAYESFILTER_PRELOAD_CUSTOM_OP=0",*command]
    receipt_path = root/(stage+"-execution.json")
    ledger["active_reservation"] = {"service":unit,"maximum_gpu_seconds_including_shutdown":reservation,
                                     "receipt":str(receipt_path)}
    write_json(ledger_path,ledger)
    write_json(state_path,state)
    manifest = {"command":launch,"source":snapshot,"environment":sys.executable,
        "plan_file":PLAN,"result_file":str(receipt_path),"gpu_uuid":device,
        "memory_growth_required":True,"maximum_gpu_seconds":reservation,
        "gpu_admission_mode":gpu_admission_mode,
        "started_utc":datetime.now(timezone.utc).isoformat(),"suite_identity":digest(value)}
    write_json(root/(stage+"-manifest.json"),manifest)
    started = time.monotonic()
    shutdown_confirmed = False
    try:
        from .timeout_policy import TimeoutPolicy, wait_for_gpu_admission
        admission = wait_for_gpu_admission(device="gpu", deadline=started+seconds,
            policy=TimeoutPolicy(gpu_admission_wait_seconds=600.,gpu_admission_mode=gpu_admission_mode))
        write_json(root/(stage+"-admission.json"),admission)
        remaining = seconds-(time.monotonic()-started)
        if not admission["admitted"] or remaining <= 0:
            code, shutdown_confirmed = 75, True
        else:
            # Admission waiting is inside the already reserved stage envelope.
            # The cgroup must not reclaim that time as a new numerical budget.
            launch = [f"--property=RuntimeMaxSec={math.ceil(remaining)}"
                      if arg.startswith("--property=RuntimeMaxSec=") else arg for arg in launch]
            manifest["command"] = launch
            manifest["admission_wait_seconds"] = admission["wait_seconds"]
            write_json(root/(stage+"-manifest.json"),manifest)
            with (root/(stage+".log")).open("w") as log:
                result = subprocess.run(launch,stdout=log,stderr=subprocess.STDOUT,check=False,
                                        timeout=max(.001,reservation-(time.monotonic()-started)))
            code = result.returncode
            shutdown_confirmed = True  # systemd-run --wait includes unit termination.
    except BaseException:
        # An interrupted client cannot leave unaccounted descendants alive.
        try:
            stop = subprocess.run(["systemctl","--user","stop",unit],check=False,timeout=30)
            shutdown_confirmed = stop.returncode == 0
        except (OSError,subprocess.SubprocessError):
            shutdown_confirmed = False
        code = -1
        raise
    finally:
        elapsed = time.monotonic()-started
        receipt = {"receipt":str(receipt_path),"resource":"gpu",
                   "elapsed_seconds":elapsed if shutdown_confirmed else max(elapsed,reservation),
                   "observed_client_seconds":elapsed,"shutdown_confirmed":shutdown_confirmed,
                   "exit_code":locals().get("code",-1),"ended_utc":datetime.now(timezone.utc).isoformat(),
                   "accounting":"enclosing stage charged once; nested cells not additional charges"}
        write_json(receipt_path,receipt)
        current = read_json(ledger_path)
        if current.get("active_reservation",{}).get("service") != unit:
            raise ValueError("grant reservation changed during stage; reconcile receipts before continuing")
        if not any(r["receipt"]==str(receipt_path) for r in current["records"]): current["records"].append(receipt)
        current["charged_gpu_seconds"] = sum(r["elapsed_seconds"] for r in current["records"])
        current["remaining_gpu_seconds"] = current["grant_gpu_seconds"]-current["charged_gpu_seconds"]
        if shutdown_confirmed:
            current.pop("active_reservation",None)
        write_json(ledger_path,current)
        state["stages"][stage] = receipt
        state["remaining_grant_gpu_seconds"] = current["remaining_gpu_seconds"]
        state["next_action"] = "review stage evidence; continue only within unchanged source, criteria and budget"
        write_json(state_path,state)
    return receipt


def summarize(root):
    """One result per original slot, including unfunded and failed work."""
    root = Path(root)
    template = read_json(root/"main-unpriced.json")
    costs = read_json(root/"costs.json") if (root/"costs.json").exists() else {}
    dispositions = {r["design_id"]:r["disposition"] for r in costs.get("dispositions",[])}
    index = read_json(root/"main/run_index.json") if (root/"main/run_index.json").exists() else {"jobs":{}}
    rows = []
    for d in template["designs"]:
        job = index["jobs"].get(d["design_id"],{})
        row = {"design_id":d["design_id"],"case":d["options"]["campaign_case"],
               "allocation":dispositions.get(d["design_id"],"not yet priced"),
               "execution_status":job.get("status","unstarted"),"posterior_diagnostic_passes":0}
        if job.get("result"):
            result = read_json(job["result"])
            row["assessment"] = result["assessment"]
            pipeline = Path(job["result"]).parent/"replication-0000/pipeline.json"
            if pipeline.exists():
                payload = read_json(pipeline)
                row["verified_candidates"] = len(payload.get("verified_candidate_ids",[]))
                row["posterior_diagnostic_passes"] = sum(m.get("posterior",{}).get("passed",False)
                    for m in payload.get("members",[]) if m.get("status")=="assessed")
        rows.append(row)
    record = {"planned_main_fits":32,"slots":rows,"completed_processes":sum(
        r["execution_status"]=="complete" for r in rows),
        "inference_status":{"hard_veto_screen":"see numerical/member diagnostics in each slot",
            "statistically_supported_ranking":False,"descriptive_only":"few-seed diagnostics and reference differences",
            "default_readiness":False,"next_evidence":"resolve missing slots and assess each lane independently"},
        "decision":"engineering and posterior evidence remain separate; incomplete cases stay in denominator",
        "K6_posterior_oracle":"unavailable", "C1_calibration_gap_closed":False}
    write_json(root/"result.json",record)
    (root/"result.md").write_text(
        "# State-space campaign disposition\n\n"
        f"Completed processes: {record['completed_processes']}/32. Process completion does not establish posterior correctness.\n\n"
        "| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| Assess each K case separately | Per-slot filter and lifecycle checks | Preserve numerical and posterior failures | Few fits; checked numerical references have no rigorous error bound | Review result.json and native receipts | Calibration, superiority or a new default |\n\n"
        "| Inference status | Result |\n| --- | --- |\n"
        "| Hard veto screen | Per-slot recorded diagnostics |\n| Statistically supported ranking | None |\n"
        "| Descriptive differences | Per-member counts, errors, ESS/MCSE and times |\n"
        "| Default readiness | Not established |\n| Next evidence | Missing slots and lane-specific diagnoses |\n\n"
        "A failed candidate is not rejection of state-space inference. Invalid value/score/reference evidence requires repair of the affected lane. "
        "The strongest alternative explanation for poor posterior delivery is geometry or weak identification. K6 has no posterior oracle; K7 samples a sigma-point approximation.\n")
    return record


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command",required=True)
    prep = commands.add_parser("prepare",help="CPU frozen data, checked references and unpriced suites")
    prep.add_argument("--output",type=Path,required=True)
    freeze = commands.add_parser("freeze",help="freeze package and K6 dependencies after tests")
    freeze.add_argument("root",type=Path)
    summary = commands.add_parser("report",help="preserve all 32 slot dispositions")
    summary.add_argument("root",type=Path)
    run = commands.add_parser("run",help="trusted GPU stages after C1 terminal settlement")
    run.add_argument("root",type=Path)
    run.add_argument("--stage",choices=("preflight-mechanics","preflight-pipeline","pricing","main","all"),required=True)
    run.add_argument("--ledger",type=Path,default=Path(GRANT))
    price_cmd = commands.add_parser("price",help="resolve affordable main lanes from complete pilot receipts")
    price_cmd.add_argument("root",type=Path)
    price_cmd.add_argument("--ledger",type=Path,default=Path(GRANT))
    args = parser.parse_args(argv)
    if args.command == "prepare":
        record = prepare(args.output)
        print({"output":str(args.output),"reference_status":record["reference_status"],
               "wall_seconds":record["wall_seconds"]})
    elif args.command == "freeze":
        print(freeze_source(args.root)["identity"])
    elif args.command == "price":
        print(price(args.root,read_json(args.ledger)))
    elif args.command == "report":
        print(summarize(args.root)["completed_processes"])
    elif args.command == "run":
        stages = ("preflight-mechanics","preflight-pipeline","pricing","main") if args.stage=="all" else (args.stage,)
        failed = False
        try:
            for stage in stages:
                record = run_stage(args.root,stage,args.ledger)
                print(record,flush=True)
                if record.get("exit_code",0) != 0:
                    failed = True
                    break
        finally:
            summarize(args.root)
        if failed:
            raise SystemExit(1)
