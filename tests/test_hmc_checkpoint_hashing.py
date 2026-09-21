"""Checkpoint specialization preserves identities and live mutation checks."""
import json

import pytest

from bayesfilter.inference.hmc_candidate_set_checkpoint import (
    _json_native_sha256, write_numerical_tuning_checkpoint)
from bayesfilter.inference.hmc_candidate_set_tuning import _sha256


@pytest.mark.parametrize("payload", [
    {"unicode": "θ", "number": -0.0, "integer": 2**64, "none": None},
    {"nested": ({"2": [True, False, 1.234e-77]},), "bytes": "AAECAw=="},
    {"trace": {"status": [0, 0, 1]}, "scientific": "not admissible"},
])
def test_json_native_hash_matches_general_identity(payload):
    normalized = json.loads(json.dumps(payload))
    assert _json_native_sha256(normalized) == _sha256(normalized)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), object()])
def test_non_json_or_nonfinite_native_record_is_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        _json_native_sha256({"value": value})


def test_actual_checkpoint_rechecks_mutation_after_cached_persistence(tmp_path):
    from bayesfilter.inference import run_typed_hmc_candidate_set, load_numerical_tuning_checkpoint
    from tests.test_hmc_candidate_set_execution import make_binding, GaussianTarget
    from tests.test_hmc_candidate_set_tuning import _config

    binding = make_binding()
    result = run_typed_hmc_candidate_set(binding.typed_adapter,
        _config(grid=(3,), epsilons=((3, (1.3,)),)), output_dir=tmp_path).result
    assert binding._persisted_files and binding._evidence
    restored, controller = load_numerical_tuning_checkpoint(tmp_path / "tuning_checkpoint.json",
        adapter=GaussianTarget())
    assert controller.result().verified_candidate_ids == result.verified_candidate_ids
    key = next(iter(binding._evidence))
    binding._evidence[key]["analysis"]["acceptance"] = .123
    with pytest.raises(ValueError, match="corrupt live numerical evidence"):
        write_numerical_tuning_checkpoint(binding, result, tmp_path)
    restored._spec["config"]["measurement_num_results"] += 1
    with pytest.raises(ValueError, match="corrupt execution specification"):
        write_numerical_tuning_checkpoint(restored, controller.result(), tmp_path)
