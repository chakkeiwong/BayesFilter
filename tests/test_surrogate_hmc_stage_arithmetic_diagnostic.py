"""CPU-only checks of identical-input arithmetic localization; no admission."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("BAYESFILTER_PRELOAD_CUSTOM_OP", "0")
from dataclasses import replace
import json
import tensorflow as tf
from bayesfilter.testing import surrogate_hmc_five_model as runtime
from bayesfilter.testing import surrogate_hmc_stage_arithmetic_diagnostic as diagnostic


def test_precision_difference_uses_batch_axis_and_exact_casts():
    x = tf.constant([[[1., 2.], [3., 4.]], [[2., 4.], [8., 16.]]])
    y = tf.cast(x, tf.float64)
    actual = diagnostic.difference(x, y, 1)
    assert actual["finite"]
    assert actual["maximum_scaled"] == 0
    assert actual["per_row_maximum_scaled"].shape == (2,)
    y = tf.tensor_scatter_nd_add(y, [[1, 0, 1]], [4.])
    assert diagnostic.difference(x, y, 1)["per_row_maximum_scaled"].numpy().tolist() == [.5, 0.]
    boolean = tf.constant([True, False])
    converted = diagnostic.cast_inputs((x, boolean), tf.float64)
    assert converted[0].dtype == tf.float64
    assert converted[1].dtype == tf.bool


def test_localization_records_local_and_propagated_errors_and_fused_check(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "N", 16)
    case = runtime.load_case("lgssm")
    case.controls = replace(case.controls, flow_steps=2, reset_steps=2, correction_steps=0)
    result = diagnostic.run_diagnostic(case, tmp_path, horizon=2)
    assert result["status"] == "STAGE_ARITHMETIC_LOCALIZED"
    assert result["cpu_only"] is True
    assert result["posterior_complete"] is False
    assert result["staging_preserves_fused_outputs"] is True
    assert result["trace_counts"] == {"float32": [1,1,1,1], "float64": [1,1,1,1]}
    assert len(result["stages"]) == 2
    for step in result["stages"]:
        assert step["healthy"]
        assert set(step["local"]) == {"proposal", "density", "reset"}
        assert set(step["independent_trajectory"]) == {"value", "score"}
        saved = step["saved_reset_inputs"]["child"]
        tensor = tf.io.parse_tensor(tf.io.read_file(saved["path"]), tf.float32)
        assert tensor.shape == (2,16,3)
    artifact = json.loads((tmp_path/"stage-arithmetic.json").read_text())
    assert artifact["status"] == result["status"]
    assert set(artifact["fused_comparison"]) == {"float32", "float64"}
