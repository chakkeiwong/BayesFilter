"""Complete lifecycle records against the pre-repair numerical authority.

The intermediate cfbc32d2 comparisons stay available separately. That checkpoint
contains a now-demonstrated sequential eigensystem error, so it cannot be the
sole precision authority for the corrected runtime.
"""

from types import FunctionType

import pytest

from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_fixed_stability import _compare
from tests.test_filter_repair_lifecycle_actual import (
    CASES,
)
from tests.test_filter_repair_lifecycle_actual import (
    test_actual_lifecycle_complete_records_and_calls as _checkpoint_test,
)


def _original_checkpoint(_intermediate_revision, label):
    return FrozenCheckpoint('3582b4ac', 'original_' + label)


def _without_new_execution_metadata(value):
    if isinstance(value, dict):
        if value.get('schema') == 'bayesfilter.factor_correlation_geometry.v1':
            # The original artifact predates this explicit execution-mode field.
            # Preserve every original field, including all numerical diagnostics.
            diagnostics = dict(value['diagnostics'])
            if 'jit_compile' in diagnostics:
                assert diagnostics.pop('jit_compile') is True
            value = {**value, 'diagnostics': diagnostics}
        return {key: _without_new_execution_metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_without_new_execution_metadata(item) for item in value]
    return value


def _compare_original(actual, expected):
    _compare(_without_new_execution_metadata(actual), _without_new_execution_metadata(expected))


@pytest.mark.parametrize('case', CASES)
def test_original_full_lifecycle_records_and_target_order(case, request):
    # Reuse the exact qualified observation/materialization harness; substitute
    # only its explicitly recorded Git reference and the new metadata handling.
    namespace = {**_checkpoint_test.__globals__, 'FrozenCheckpoint': _original_checkpoint,
        '_compare': _compare_original}
    test = FunctionType(_checkpoint_test.__code__, namespace)
    test(case, request)
