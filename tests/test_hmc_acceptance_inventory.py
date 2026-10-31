"""The coverage inventory must describe executable cases without claiming passes."""
from pathlib import Path

import pytest

from bayesfilter.testing.acceptance_validation_inventory import COVERAGE, MODEL_IDS, model_configuration
from tests.test_hmc_acceptance_protocol import policy


def test_every_root_cause_group_has_an_oracle_tier_and_existing_test_source():
    root=Path(__file__).resolve().parents[1]
    assert len({row[0] for row in COVERAGE})==len(COVERAGE)
    for name,tier,oracle,path in COVERAGE:
        assert oracle and tier in {"regression","calibration","scheduled_stress"}
        assert (root/path).is_file(), (name,path)


@pytest.mark.parametrize("case_id",MODEL_IDS)
def test_every_model_is_an_explicit_valid_configuration(case_id):
    from bayesfilter.testing.acceptance_decision_models import validate_configuration
    p=policy(base_repetitions=4,max_repetitions=4,max_candidates=2)
    cfg=model_configuration(case_id,policy_payload=p.payload(),evidence_rungs=(1,),
        seed=(20261002,601),wall_seconds=120,classification="development",expected_outcome="bounded_diagnostic")
    actual,search=validate_configuration(cfg)
    assert actual==p
    assert len(cfg["active_starts"])==4
    assert search.replicated_acceptance_policy==p
    assert cfg["expected_outcome"]=="bounded_diagnostic"


def test_stress_profiles_actually_initialize_the_declared_regime():
    import math
    p=policy().payload()
    for case in ("k2","k3"):
        cfg=model_configuration(case,policy_payload=p,evidence_rungs=(1,),seed=(1,2),
            wall_seconds=30,classification="development",expected_outcome="bounded_diagnostic")
        g=cfg["geometry"]
        points=[[a+b*s for a,b,s in zip(g["center"],g["scale"],z)] for z in cfg["active_starts"]]
        if case=="k2":
            assert all(.96 < .999*math.tanh(q[0]) < .999 for q in points)
        else:
            assert all(math.exp(q[1]) < .025 for q in points)


def test_outcome_oracle_rejects_uninformative_or_wrong_terminal_states():
    from bayesfilter.testing.acceptance_decision_models import outcome_check
    passed = {"a":"verified","b":"inconclusive_at_cap"}
    assert outcome_check("positive_delivery",states=passed,completion="complete",
        exported=["a"],receipts=[]) is True
    assert outcome_check("positive_delivery",states=passed,completion="complete",
        exported=[],receipts=[]) is False
    assert outcome_check("positive_delivery",states=passed,completion="paused_infrastructure",
        exported=["a"],receipts=[]) is False
    for name in ("inconclusive_at_cap","preparation_review_required"):
        assert outcome_check(name,states={"a":name},completion="complete",
            exported=[],receipts=[]) is True
        assert outcome_check(name,states=passed,completion="complete",
            exported=["a"],receipts=[]) is False
    assert outcome_check("budget_deferred",states={"a":"validating"},
        completion="partial_budget",exported=[],receipts=[]) is True
    assert outcome_check("bounded_diagnostic",states={"a":"verified"},
        completion="complete",exported=["a"],receipts=[]) is None


def test_bounded_diagnostic_cannot_be_relabelled_as_confirmation():
    from bayesfilter.testing.acceptance_decision_models import validate_configuration
    p=policy(base_repetitions=4,max_repetitions=4,max_candidates=2)
    cfg=model_configuration("lgssm_qr",policy_payload=p.payload(),evidence_rungs=(1,),
        seed=(20261002,601),wall_seconds=120,classification="confirmation",
        expected_outcome="bounded_diagnostic")
    with pytest.raises(ValueError,match="cannot establish"):
        validate_configuration(cfg)
