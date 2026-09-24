"""Frozen-design boundary tests; numerical issuance is covered by CLI evidence."""
import copy
import json

import pytest
from bayesfilter.score_study import fd_selection as selection


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    row = dict(model="gaussian_all_parameters", proposal="ledh", estimator="symmetric_fd",
               comparison_target="model_score", fd_stencil="central3", fd_h=.02,
               fd_coupling="common_innovations", fd_directions=[[1.]],
               fd_error_budget={}, fd_lower=[-1.], fd_upper=[1.], dataset=3, role="claim")
    study = dict(settings={}, fd_candidate_family=[selection.fd_signature(row)],
                 evidence_class="mechanics")
    derived = dict(source_run="frozen-run", scope=selection.fd_scope(study,row),
                   selected_signature=selection.fd_signature(row), evidence_class="mechanics",
                   partitions={"calibration":[1], "validation":[2], "claim":[3]})
    derived = copy.deepcopy(derived)
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(derived))
    row["fd_selection"] = str(path)
    monkeypatch.setattr(selection,"derive_fd_selection",lambda _:copy.deepcopy(derived))
    return row, {"study":study}, path


def test_frozen_design_is_consumed(frozen):
    row,context,_ = frozen
    assert selection.consume_fd_selection(row,context)["selected_signature"]["fd_h"] == .02


@pytest.mark.parametrize("fault",["changed_step","leaked_data","changed_scope","claim_promotion","forged_selection"])
def test_misuse_rejected(frozen,fault):
    row,context,path = frozen
    if fault == "changed_step": row["fd_h"] = .01
    if fault == "leaked_data": row["dataset"] = 1
    if fault == "changed_scope": context["study"]["settings"]["particle_count"] = 100
    if fault == "claim_promotion": context["study"]["evidence_class"] = "scientific"
    if fault == "forged_selection":
        data = json.loads(path.read_text()); data["selected_signature"]["fd_h"] = .01
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError): selection.consume_fd_selection(row,context)
