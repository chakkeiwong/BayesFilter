"""Three fresh single-pair searches under the original failed funnel geometry."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/"source-gpu-r4"
SAVED = ROOT.parent/"m7-r1/pilots-gpu-r1/m7-pilot-funnel/replication-0000/tuning"
PLAN = "docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md"
BUDGET = 1600.


def read(path):
    return json.loads(path.read_text())


def worker(output):
    sys.path.insert(0,str(SOURCE))
    from bayesfilter.testing.inference_validation.execution import configure_worker, source_state
    from bayesfilter.testing.inference_validation.storage import write_json,file_hash
    runtime = configure_worker(SimpleNamespace(device="gpu"))
    write_json(output/"runtime.json",runtime)
    source = source_state()
    assert source["identity"] == read(SOURCE/"source_snapshot.json")["source_identity"]
    from dataclasses import replace
    from bayesfilter.inference import HMCControllerConfig, HMCCandidateExecutionConfig, tune_hmc_kernel
    from bayesfilter.inference.hmc_candidate_set_execution import _issue_binding,_tensor_from_payload
    from bayesfilter.inference.hmc_candidate_set_tuning import _sha256
    from bayesfilter.inference.hmc_candidate_set_artifacts import load_candidate_set_result_payload
    from bayesfilter.testing.inference_validation.targets import ValidationTarget
    from bayesfilter.testing.inference_validation.designs import seed_for
    from bayesfilter.testing.inference_validation.engines.pipeline import check_inventory
    wrapper = read(SAVED/"execution_spec.json")
    spec = wrapper["execution"]
    assert _sha256(spec) == wrapper["binding_hash"]
    old = load_candidate_set_result_payload(SAVED/"candidate_set_result.json")
    assert not old["verified_candidate_ids"]
    lineage = spec["target_lineage"]
    assert lineage["model"] == "funnel" and lineage["control"] == "baseline"
    target = ValidationTarget(lineage["model"],lineage["prior"],lineage["data"],jit_compile=True)
    assert target.adapter_signature() == spec["scope"]["target_signature"]
    search = HMCControllerConfig(primary_l_grid=(18,),initial_epsilon=.2256942307263964,
        pilot_enabled=False,refinement_rounds=0,max_candidates=1,total_budget_units=63,
        repair_reserve_units=1,evidence_rungs=(1,2,4,8,16),max_wall_time_seconds=510.)
    manifest = {"source":source,"runtime":runtime,"command":sys.argv,
        "plan_file":PLAN,"result_file":str(output/"diagnostic.json"),
        "original_binding_hash":wrapper["binding_hash"],
        "saved_inputs":{name:file_hash(SAVED/name) for name in ("execution_spec.json","candidate_set_result.json")},
        "search":search.payload(),"budget_seconds":BUDGET,
        "data_version":"same three-dimensional funnel; saved mass and starts",
        "historical_qualification_reused":False,"replications":3}
    write_json(output/"manifest.json",manifest)
    rows = []
    for replication in range(3):
        execution = replace(HMCCandidateExecutionConfig.from_payload(spec["config"]),
            preparation_elapsed_seconds=0.,measurement_num_results=128,verification_num_results=128,
            seed=seed_for(2026091840,"saved-pair-evidence-extension",replication))
        assert execution.use_xla
        binding = _issue_binding(adapter=target,layers=spec["layers"],
            initial_active_state=_tensor_from_payload(spec["initial_active_state"]),
            target_scope=spec["target_scope"],target_lineage=lineage,
            preparation={"source":"saved_funnel_geometry_evidence_extension",
                         "original_binding_hash":wrapper["binding_hash"]},
            config=execution,source_paths=[__file__],scope_id="m8-saved-funnel-evidence-extension",
            search_id=f"fresh-{replication}",epsilon_domain=tuple(spec["scope"]["epsilon_domain"]),
            repair_factor=spec["scope"]["repair_factor"],max_repairs_per_family=0)
        assert binding.scope.mass_signature == spec["scope"]["mass_signature"]
        assert binding.scope.start_bank_signature == spec["scope"]["start_bank_signature"]
        directory = output/f"replication-{replication:04d}"
        result = tune_hmc_kernel(adapter=target,initial_position=binding.initial_active_state,
            config=search,candidate_set_adapter=binding.typed_adapter,output_dir=directory).result
        payload = read(directory/"candidate_set_result.json")
        inventory = check_inventory(payload)
        assert not inventory["failures"]
        row = {"replication":replication,"seed":list(execution.seed),"inventory":inventory,
            "completion":result.completion_status,"verified":list(result.verified_candidate_ids),
            "candidate_states":dict(result.candidate_states),"binding_hash":binding.binding_hash,
            "mass_signature":binding.scope.mass_signature,"start_bank_signature":binding.scope.start_bank_signature,
            "observations":[{"stage":o["stage"],"observation":o["observation"]} for o in payload["observations"]]}
        write_json(output/f"replication-{replication:04d}.json",row)
        rows.append(row)
    write_json(output/"diagnostic.json",{"rows":rows,"planned":3,
        "posterior_assessed":False,"default_promoted":False,"ranking_supported":False,
        "interpretation":"fresh longer evidence for one previously unresolved pair; no search-wide success claim"})


def main():
    output = ROOT/"funnel-evidence-extension-gpu-r1"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    assert os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH","").lower() == "true"
    if "--worker" in sys.argv:
        worker(output)
        return
    output.mkdir(exist_ok=False)
    command = [sys.executable,str(Path(__file__).resolve()),"--worker"]
    started = time.monotonic()
    when = datetime.now(timezone.utc).isoformat()
    with (output/"worker.log").open("x") as log:
        try:
            result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=BUDGET)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
    record = {"command":command,"started_utc":when,"elapsed_seconds":time.monotonic()-started,
        "returncode":code,"device":"gpu","environment":sys.executable,"plan_file":PLAN,
        "result_file":str(output/"diagnostic.json"),"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_commit":read(SOURCE/"source_snapshot.json")["git_commit"],
        "runtime":read(output/"runtime.json") if (output/"runtime.json").exists() else None}
    (output/"gpu-diagnostic-run.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record,indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
