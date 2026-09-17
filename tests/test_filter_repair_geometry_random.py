"""Independent checks for the owner-approved versioned geometry stream."""

import numpy as np
import pytest

from bayesfilter.ops.geometry_random_tf import (
    STREAM_ID,
    GeometryTensorStream,
    _ball_kernel,
    _draw_kernel,
)


def test_seed_call_order_reproducibility_and_compilation():
    first, second = GeometryTensorStream((11, 22)), GeometryTensorStream((11, 22))
    for stream in (first, second):
        assert stream.call_index == 0
    np.testing.assert_array_equal(
        first.normal(size=(31, 4)), second.normal(size=(31, 4))
    )
    np.testing.assert_array_equal(first.permutation(31), second.permutation(31))
    np.testing.assert_array_equal(
        first.ball(31, 4, radius=0.5), second.ball(31, 4, radius=0.5)
    )
    assert first.call_index == second.call_index == 3
    assert STREAM_ID == "geometry_tf_philox_cpu_xla_v1"
    assert not np.array_equal(
        first.normal(size=(31, 4)), GeometryTensorStream((11, 22)).normal(size=(31, 4))
    )
    assert not np.array_equal(
        first.normal(size=(31, 4)), GeometryTensorStream((11, 23)).normal(size=(31, 4))
    )
    for kernel in (
        _draw_kernel("normal", (31, 4)),
        _draw_kernel("permutation", (31,)),
        _ball_kernel(31, 4),
    ):
        concrete = kernel.get_concrete_function()
        assert concrete.function_def.attr["_XlaMustCompile"].b
        assert kernel.experimental_get_tracing_count() == 1
        assert not {"PyFunc", "EagerPyFunc"} & {
            node.op for node in concrete.graph.as_graph_def().node
        }


@pytest.mark.parametrize("minimum", [0.0, 0.25])
def test_ball_support_and_radial_law(minimum):
    points = GeometryTensorStream((819, 527)).ball(
        8192, 3, radius=1.7, minimum_uniform=minimum
    )
    norms = np.linalg.norm(points, axis=1)
    uniforms = (norms / 1.7) ** 3
    assert np.all(np.isfinite(points))
    assert np.all(norms <= 1.7)
    assert np.all(uniforms >= minimum - 1e-14)
    assert abs(uniforms.mean() - 0.5 * (1.0 + minimum)) < 0.02
    assert np.max(np.abs(np.mean(points / norms[:, None], axis=0))) < 0.03


def test_permutation_empty_shapes_and_validation():
    stream = GeometryTensorStream((12, 73))
    np.testing.assert_array_equal(np.sort(stream.permutation(128)), np.arange(128))
    assert stream.normal(size=(0, 2)).shape == (0, 2)
    assert stream.permutation(0).shape == (0,)
    assert stream.ball(0, 2, radius=1.0).shape == (0, 2)
    with pytest.raises(ValueError):
        stream.ball(1, 0, radius=1.0)
    with pytest.raises(ValueError):
        stream.ball(2, 3, radius=float("nan"))
