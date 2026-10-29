"""CPU-only evidence wiring for the M13 energy repair."""
import json
from dataclasses import replace
from types import SimpleNamespace
import pytest
from bayesfilter.testing import surrogate_hmc_m13_energy_repair as repair
from bayesfilter.testing.surrogate_hmc_m13_public_tuning import tuning_config


def fixture(tmp_path):
    binding=SimpleNamespace(payload=lambda: {"binding_hash":"fixture"})
    cfg=tuning_config("repair-fixture")
    cfg=replace(cfg,nomination=replace(cfg.nomination,max_step_size=repair.STEP_CAP,energy_error_max=1000.))
    path=tmp_path/"report.json"
    probe=dict(status="DIAGNOSTIC_COMPLETE",repair_screen_passed=True,source_unchanged=True,
        device="gpu",jit_compile=True,counts_per_chain=64,multipliers=[.125,1.],
        step_cap=repair.STEP_CAP,original_step=2*repair.STEP_CAP,binding=binding.payload(),
        replications=[dict(health_passed=True,checks={k:True for k in repair.HEALTH_KEYS}) for _ in range(2)])
    path.write_text(json.dumps(probe))
    path.with_name("guard-receipt.json").write_text(json.dumps(dict(status="COMPLETED",accounting_complete=True)))
    return path,probe,binding,cfg


def test_actual_binding_scope_and_completed_health_are_required(tmp_path):
    path,probe,binding,cfg=fixture(tmp_path)
    result=repair.checked_energy_repair_probe(binding,cfg,path=path)
    assert result["passed"] and result["step_cap"]==repair.STEP_CAP
    with pytest.raises(ValueError,match="binding"):
        repair.checked_energy_repair_probe(SimpleNamespace(payload=lambda:{"binding_hash":"changed"}),cfg,path=path)
    with pytest.raises(ValueError,match="config"):
        repair.checked_energy_repair_probe(binding,replace(cfg,energy_error_max=2000.),path=path)


@pytest.mark.parametrize("failure",["energy","missing_movement","source","incomplete","wrong_step"])
def test_incomplete_or_unhealthy_probe_cannot_nominate_repair(tmp_path,failure):
    path,probe,binding,cfg=fixture(tmp_path)
    if failure=="energy": probe["replications"][0]["checks"]["energy_valid"]=False
    if failure=="missing_movement": del probe["replications"][0]["checks"]["all_chains_moved"]
    if failure=="source": probe["source_unchanged"]=False
    if failure=="wrong_step": probe["step_cap"]*=2
    if failure=="incomplete":
        path.with_name("guard-receipt.json").write_text(json.dumps(dict(status="RUNNING",accounting_complete=False)))
    path.write_text(json.dumps(probe))
    with pytest.raises(ValueError,match="health screen"):
        repair.checked_energy_repair_probe(binding,cfg,path=path)
