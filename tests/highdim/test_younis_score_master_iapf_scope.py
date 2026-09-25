"""Adaptive-count evidence and actual selection/consumer integration checks."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from bayesfilter.score_study.iapf_adapter import iteration_decision
from bayesfilter.score_study.iapf_scope import cost_record, validate_adaptive_ledger


def growth_ledger():
    counts, values, records = [], [], []
    n = 16
    for index, value in enumerate([0., -1., -1.]):
        counts.append(n)
        values.append(value)
        action = iteration_decision(values, counts, k=1, tau=100., max_particles=64)
        records.append(dict(iteration=index, particles=n, log_value=value, **action))
        n = action["next_particles"]
    return records


def test_realized_count_and_separate_work_are_not_initial_count():
    ledger = growth_ledger()
    counts = validate_adaptive_ledger(ledger, initial_particles=16, max_particles=64, k=1, tau=100.)
    assert counts["count_history"] == [16, 16, 32]
    work = cost_record(ledger=ledger, fit_calls=2, fit_optimizer_steps=19,
        final_particle_count=32, horizon=2, offline_seconds=3., final_seconds=1.)
    assert work["offline_particle_time_points"] == 128
    assert work["offline_fit_input_particle_time_points"] == 64
    assert work["final_particle_time_points"] == 64
    assert work["total_wall_seconds"] == 4.


@pytest.mark.parametrize("mutation", [
    lambda a: a[0].update(particles=32),
    lambda a: a[1].update(next_particles=64),
    lambda a: a[2].update(particles=16),
    lambda a: a[1].update(log_value=math.nan),
    lambda a: a[1].update(action="final"),
    lambda a: a[2].update(cv=1.),
])
def test_invalid_adaptive_history_is_rejected(mutation):
    ledger = growth_ledger()
    mutation(ledger)
    with pytest.raises(ValueError):
        validate_adaptive_ledger(ledger, initial_particles=16, max_particles=64, k=1, tau=100.)


def _lifecycle(root):
    from bayesfilter.score_study.runtime import configure_runtime
    configure_runtime(device="CPU", tf32=False, jit_compile=True)
    from tests.highdim.test_younis_score_master_provider_consumers_tf import fixture
    from bayesfilter.score_study.adapters import evaluate_gaussian
    from bayesfilter.score_study.contracts import digest
    from bayesfilter.score_study.coordinator import execute, report, write_json
    from bayesfilter.score_study.iapf_scope import validate_result_accounting
    from bayesfilter.score_study.registry import default_registry
    from bayesfilter.score_study.tuning import issue_selection, consume_selection, scope
    row, context = fixture("iapf")
    config = dict(k=1,tau=100.,max_iterations=4,max_particles=128,mean_bound=4.,sd_lower=.2,sd_upper=4.,
        max_fit_steps=2000,max_backtracks=30,fit_tolerance=1e-7,floor_ratio=.01,
        fit_theta=context["study"]["settings"]["theta"],fit_dtype="float64")
    study = {**context["study"], "schema":"younis_score_study_v1", "phase":"0E", "version":1,
        "plan":"docs/plans/younis-score-iapf-adaptive-scope-2026-09-16.md",
        "budget":{"wall_seconds":400,"max_attempts":2,"max_attempts_per_row":1},
        "partitions":{"calibration":[800],"validation":[810],"claim":[820]},
        "required_proposals":["iapf"],"tuning_candidate_family":[{"iapf":config}],
        "rows":[{**row,"iapf":config,"id":role,"role":role,"dataset":dataset}
            for role,dataset in (("calibration",800),("validation",810))]}
    registry = default_registry()
    execute(study, registry, root/"tuning")
    summary = report(root/"tuning", registry)
    assert summary["execution_status"] == "complete", summary
    selection_path = root/"selection.json"
    selected = issue_selection(root/"tuning",selection_path)
    assert selected["scope"]["reset_contract_id"] == "not_applicable"
    assert selected["scope"]["particle_count_contract"]["initial_particles"] == 16
    evidence = selected["candidates"][0]["adaptive_evidence"]
    assert len(evidence) == 2
    assert all(e["work_accounting"]["offline_fit_calls"] == 2 for e in evidence)
    claim = {**row,"id":"claim","role":"claim","dataset":820,"tuning_selection":str(selection_path)}
    context = {"study":{**study,"rows":[claim]},"registry":registry}
    result = evaluate_gaussian(claim,context)
    diag = result["diagnostics"]
    assert diag["iapf_configuration"] == config
    assert diag["fit_observation_digest"] == diag["data_version"]
    assert diag["fit_digest"] not in {e["fit_digest"] for e in evidence}
    assert diag["work_accounting"]["fixed_final_count_for_score"]
    other = evaluate_gaussian({**claim,"replicate":1},context)
    assert other["diagnostics"]["fit_digest"] == diag["fit_digest"]
    assert other["diagnostics"]["fit_seed_records"]["iapf_final_process"] != diag["fit_seed_records"]["iapf_final_process"]
    for dataset in (800,810):
        with pytest.raises(ValueError,match="leaked"):
            consume_selection({**claim,"dataset":dataset},context)
    for field,value in (("particles",32),("horizon",3)):
        changed=deepcopy(context);changed["study"]["settings"][field]=value
        with pytest.raises(ValueError,match="scope mismatch"):
            consume_selection(claim,changed)
    changed=deepcopy(context);changed["study"]["tuning_candidate_family"][0]["iapf"]["max_particles"]=256
    with pytest.raises(ValueError,match="scope mismatch"):
        consume_selection(claim,changed)
    with pytest.raises(ValueError,match="contradicts"):
        evaluate_gaussian({**claim,"iapf":{**config,"max_particles":256}},context)
    with pytest.raises(ValueError,match="requires"):
        evaluate_gaussian({**claim,"tuning_selection":None},context)
    changed=deepcopy(selected);changed["candidates"][0]["realized_particle_counts"]=[999]
    write_json(root/"edited-selection.json",changed)
    with pytest.raises(ValueError,match="modified"):
        consume_selection({**claim,"tuning_selection":str(root/"edited-selection.json")},context)
    for field in ("actual_particle_count","work_accounting","fit_observation_digest","fit_seed_records","fit"):
        changed=deepcopy(result)
        if field=="actual_particle_count": changed["diagnostics"][field]+=1
        if field=="work_accounting": changed["diagnostics"][field]["final_particle_time_points"]+=1
        if field=="fit_observation_digest": changed["diagnostics"][field]="incorrect"
        if field=="fit_seed_records":
            changed["diagnostics"][field]["iapf_final_process"]=changed["diagnostics"][field]["iapf_fit0_process"]
        if field=="fit":
            changed["diagnostics"][field]["centers"][0][0]+=1.
            changed["diagnostics"]["fit_digest"]=digest(changed["diagnostics"][field])
        with pytest.raises(ValueError): validate_result_accounting(changed,{**claim,"iapf":config},study["settings"])
    state=json.loads((root/"tuning/state.json").read_text())
    saved=state["rows"]["calibration"]
    path=root/"tuning"/saved["result_path"]
    original=path.read_text(); original_state=(root/"tuning/state.json").read_text()
    changed=json.loads(original);changed["diagnostics"]["actual_particle_count"]+=1
    write_json(path,changed);saved["result_digest"]=digest(changed);write_json(root/"tuning/state.json",state)
    with pytest.raises(ValueError,match="realized particle scope"):
        consume_selection(claim,context)
    path.write_text(original);(root/"tuning/state.json").write_text(original_state)
    write_json(root/"checks.json",{"actual_consumer":True,"selection_consumed":True,
        "adaptive_evidence":evidence,"claim_count":diag["actual_particle_count"],"default_ready":False})


def test_actual_adaptive_selection_and_consumer(tmp_path):
    result = subprocess.run([sys.executable,"-c",
        "import runpy,sys; from pathlib import Path; runpy.run_path(sys.argv[1])['_lifecycle'](Path(sys.argv[2]))",
        str(Path(__file__).resolve()),str(tmp_path)], cwd=Path(__file__).resolve().parents[2],
        env={**os.environ,"CUDA_VISIBLE_DEVICES":"-1","BAYESFILTER_PRELOAD_CUSTOM_OP":"0"},
        capture_output=True,text=True,timeout=480)
    assert result.returncode==0,result.stdout+result.stderr
