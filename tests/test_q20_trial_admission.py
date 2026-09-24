"""Consumer and source-boundary regressions for trial-only map admission."""
import copy
import json
from types import SimpleNamespace

import pytest

from bayesfilter.inference import q20_production_hmc
from bayesfilter.inference.q20_master_program import choose_maps
from bayesfilter.inference.neutra_post_training import POST_TRAINING_SCHEMA, PROBE_SCHEMA
from bayesfilter.inference.q20_production_config import (
    digest, frozen_scope_hash, protocol_template, training_cohort, write_json,
)


@pytest.mark.parametrize("trial,legacy,reliable,accepted", [
    (True, True, True, True),
    (False, True, True, False),
    (None, True, True, False),
    (True, True, False, False),
])
def test_serious_consumers_require_current_trial_assessment(tmp_path, monkeypatch, trial, legacy, reliable, accepted):
    config = protocol_template()
    decision = {"development_eligible": legacy}
    if trial is not None:
        decision["hmc_trial_eligible"] = trial
    record = {"frozen_scope_hash": frozen_scope_hash(config), "sources": {"fixture": "same"},
        "role": "development", "assessment": {"beta": 1., "map_reliability": {"passed": reliable},
        "decision": decision, "current_map_hash": "map-state",
        "post_training": {"schema": POST_TRAINING_SCHEMA, "training_state_hash": "checkpoint",
            "map_hash": "map-state", "beta": 1., "numerical_check_passed": True,
            "hmc_trial_eligible": True, "geometry": {"schema": PROBE_SCHEMA,
                "finite": True, "complete": True, "rows": 1000, "valid_rows": 1000,
                "summary_status": "all_declared_rows_summarized",
                "score_residual_norm": {"rows": 1000, "min": 0., "median": .1,
                    "mean": .1, "rms": .2, "p95": .3, "p99": .4, "max": .5,
                    "exceedance_fraction": {"1.0": 0.}, "fraction_larger_than_gaussian_score": 0.},
                "r_log_target_over_gaussian_up_to_constant": {"rows": 1000, "range": .5}}}},
        "training_checkpoint_hash": "checkpoint",
        "frozen_transport": {"fixture": "frozen-map", "training_state_hash": "checkpoint"}}
    monkeypatch.setattr(q20_production_hmc, "source_snapshot", lambda: {"fixture": "same"})
    loaded = object()
    monkeypatch.setattr(q20_production_hmc, "load_frozen_neutra_artifact", lambda *a, **k: loaded)
    adapter = SimpleNamespace(adapter_signature=lambda: "fixture")
    if accepted:
        assert q20_production_hmc.validate_training_export(record, config, adapter, 1.) is loaded
        changed = copy.deepcopy(record)
        changed["assessment"]["beta"] = .5
        with pytest.raises(ValueError, match="temperature"):
            q20_production_hmc.validate_training_export(changed, config, adapter, 1.)
    else:
        with pytest.raises(ValueError, match="HMC trial|numerical reliability"):
            q20_production_hmc.validate_training_export(record, config, adapter, 1.)
    candidate = training_cohort(config, method="neutra")[0]
    export = tmp_path / "map.json"
    export.write_text(json.dumps(record))
    checkpoint = tmp_path / "cohort.json"
    checkpoint.write_text(json.dumps({"cohort": {candidate["id"]: {"exports": {"1.0": str(export)}}}}))
    assert bool(choose_maps(config, checkpoint, schedule="direct", count=1)) is accepted
    if accepted:
        record["assessment"].pop("post_training")
        export.write_text(json.dumps(record))
        assert not choose_maps(config, checkpoint, schedule="direct", count=1)
        with pytest.raises(ValueError, match="post-training"):
            q20_production_hmc.validate_training_export(record, config, adapter, 1.)


@pytest.mark.parametrize("failure", ["deterioration_repair_trigger", "numerically_invalid"])
def test_repeated_import_cannot_erase_a_saved_training_veto(tmp_path, failure):
    from pathlib import Path
    from bayesfilter.inference.q20_campaign_runtime import source_snapshot
    from bayesfilter.inference.q20_training_resume import import_training_checkpoint, read_training_checkpoint
    repo = Path(__file__).resolve().parents[1]
    config = protocol_template()
    candidate = training_cohort(config, method="neutra")[0]
    sources = source_snapshot(repo)
    # This is an import-ledger fixture, with no claim to be a restorable map.
    session = {"scope": {"config_hash": digest(config), "candidate": candidate, "sources": sources},
               "iteration": 512, "level_updates": 512, "rng_index": 512, "map": {"beta": 1.}}
    session["state_hash"] = digest(session)
    item = {"session": session, "baseline": None, "previous": None,
            "assessments": [{"beta": 1., "updates": 512, "decision": {"status": failure}}],
            "status": failure, "exports": {}, "plateaus": 0}
    state = {"schema": "bayesfilter.q20.training_cohort.v1", "config_hash": digest(config),
             "sources": sources, "cohort": {candidate["id"]: item}}
    path = tmp_path / "original.json"
    write_json(path, {**state, "checkpoint_hash": digest(state)})
    for index in range(2):
        receipt = import_training_checkpoint(path, config, tmp_path / f"import-{index}",
            previous_root=repo, current_root=repo)
        path = Path(receipt["checkpoint"])
        current = read_training_checkpoint(path, config)["cohort"][candidate["id"]]
        assert current["reassessment_requires_update"] is True
        assert not current["exports"]
