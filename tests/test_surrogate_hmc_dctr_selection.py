"""Offline imperfect-force selection must never acquire exact-score authority."""
import json
import hashlib
from pathlib import Path
import pytest

from bayesfilter.inference.surrogate_hmc_dctr_selection import first_stable_pair, issue_selection, load_selection


def test_stress_guard_checks_each_row_when_comparator_has_a_failure():
    from scripts.calibrate_surrogate_hmc_lg1_controls import no_new_failures
    assert not no_new_failures([True, False], [False, True])
    assert no_new_failures([True, True], [False, True])
    with pytest.raises(ValueError):
        no_new_failures([True], [True, False])


def pair(passed):
    return dict(non_harm_passed=passed, healthy_nominal=True, stress_safe=True,
                no_new_stress_failure=True, normalized_maximum_change=0. if passed else .1,
                tolerance=1e-3)


def test_requires_both_tuning_partitions_and_first_shared_pass():
    a = [pair(x) for x in (True,True,True)]
    b = [pair(x) for x in (False,True,True)]
    assert first_stable_pair(dict(calibration=a,validation=b)) == 1
    assert first_stable_pair(dict(calibration=b,validation=[pair(False)]*3)) is None
    with pytest.raises(ValueError):
        first_stable_pair(dict(calibration=a,claim=b))
    with pytest.raises(ValueError,match="measured"):
        first_stable_pair(dict(calibration=[{**pair(False),"non_harm_passed":True}]*3,validation=a))


@pytest.mark.parametrize("model,attempt", [("LG1", 4), ("LG2", 1)])
def test_scoped_artifact_rejects_exact_score_promotion_and_scope_drift(tmp_path, model, attempt):
    evidence, data = {}, {}
    scope, hashes = {"model":model,"attempt":attempt,"n":1024}, {"source":"abc"}
    for partition in ("calibration","validation"):
        evidence[partition] = tmp_path / (partition+"-curve.json")
        data[partition] = tmp_path / (partition+"-data.json")
        data[partition].write_text(json.dumps(dict(partition=partition,attempt=attempt,model_id=model)))
        evidence[partition].write_text(json.dumps(dict(scope=scope,source_hashes=hashes,
            data_sha256=hashlib.sha256(data[partition].read_bytes()).hexdigest(),
            pairs=[pair(x) for x in (False,True,True)])))
    path=tmp_path/"selection.json"
    original=issue_selection(output=path,scope=scope,evidence_paths=evidence,
                             partition_data_paths=data,source_hashes=hashes)
    assert load_selection(path,expected_scope=scope,expected_source_hashes=hashes)["selected_index"]==1
    with pytest.raises(ValueError,match="scope"):
        load_selection(path,expected_scope={"model":"LG2"},expected_source_hashes=hashes)
    path.write_text(json.dumps({**original,"exact_score_admission":True}))
    with pytest.raises(ValueError,match="proposal-field"):
        load_selection(path,expected_scope=scope,expected_source_hashes=hashes)
    path.write_text(json.dumps(original))
    evidence['validation'].write_text('{}')
    with pytest.raises(ValueError,match="checksum"):
        load_selection(path,expected_scope=scope,expected_source_hashes=hashes)


@pytest.mark.parametrize("scope", [{"model":"LG2","attempt":4}, {"model":"LG1","attempt":1}, {"model":"LG2"}])
def test_selection_rejects_wrong_model_or_attempt_before_reading_curves(tmp_path, scope):
    data = {}
    for partition in ("calibration", "validation"):
        data[partition] = tmp_path / (partition + ".json")
        data[partition].write_text(json.dumps(dict(partition=partition, attempt=1, model_id="LG2")))
    with pytest.raises(ValueError, match="model|attempt"):
        issue_selection(output=tmp_path / "selection.json", scope=scope,
                        evidence_paths={p:tmp_path / "missing" for p in data},
                        partition_data_paths=data, source_hashes={})
