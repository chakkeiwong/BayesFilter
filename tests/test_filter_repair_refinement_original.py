"""Original-source complete refinement authority; intermediate evidence retained."""

import textwrap
from types import FunctionType

import pytest

from tests.test_filter_repair_lifecycle_original import (
    _compare_original,
    _original_checkpoint,
)
from tests.test_filter_repair_sequential_refinement import (
    test_complete_refinement_records as _checkpoint_test,
)


def _original_step(frozen, source):
    # Original source spells out the search cloud/selection before fitting.
    # Extract that exact body; the intermediate helper call does not exist.
    start = source.index("        cloud_builder = (", source.index("    radius = cfg.initial_radius\n"))
    stop = source.index("        _emit_progress(", start)
    block = textwrap.dedent(source[start:stop])
    wrapper = (
        "def execute(value_and_score_fn, batched_value_and_score_fn, dimension, cfg, "
        "center, center_value, center_score, scale_tf, radius, stalled, index):\n"
        "    assert not cfg.dimension_scaled_search and not cfg.record_refinement_movement_diagnostics\n"
        "    evaluations, history = 0, []\n"
        "    search_sample_count = cfg.search_sample_count\n"
        "    structured_fresh_count = cfg.structured_fresh_sample_multiplier * dimension\n"
        "    for attempt in (index,):\n" + textwrap.indent(block, "        ")
        + '    return {"center": center, "center_value": center_value, "center_score": center_score, '
        '"radius": radius, "stalled": stalled, "evaluations": evaluations, "history": history}\n'
    )
    namespace = dict(vars(frozen))
    exec(compile(wrapper, "3582b4ac:refinement_exact_excerpt", "exec"), namespace)  # noqa: S102 - frozen diagnostic
    return namespace["execute"]


@pytest.mark.parametrize("case", ["symmetric_proposal", "symmetric_recenter", "symmetric_reject",
                                 "factor_one", "factor_two", "factor_two_reuse"])
def test_original_complete_refinement_records(case, request):
    namespace = {**_checkpoint_test.__globals__, "FrozenCheckpoint": _original_checkpoint,
                 "_compare": _compare_original, "_old_step": _original_step}
    test = FunctionType(_checkpoint_test.__code__, namespace)
    test(case, request)
