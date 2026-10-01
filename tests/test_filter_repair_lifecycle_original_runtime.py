"""Original-reference lifecycle resource checks after eigenpair correction."""

import json
from pathlib import Path
from types import FunctionType

import pytest

from tests.test_filter_repair_initializer_rounding import _record_differences
from tests.test_filter_repair_lifecycle_original import (
    _compare_original,
    _original_checkpoint,
    _without_new_execution_metadata,
)
from tests.test_filter_repair_lifecycle_runtime import (
    test_actual_lifecycle_runtime_inputs_and_resource_lifetime as _checkpoint_test,
)


@pytest.mark.parametrize('dimension', [3, 5])
def test_original_lifecycle_runtime_inputs_and_resource_lifetime(dimension, request):
    comparisons = []

    def record_comparison(actual, expected):
        actual = _without_new_execution_metadata(actual)
        expected = _without_new_execution_metadata(expected)
        comparisons.append({'actual': actual, 'expected': expected,
            'differences': _record_differences(actual, expected)})

    namespace = {**_checkpoint_test.__globals__, 'FrozenCheckpoint': _original_checkpoint,
        '_compare': record_comparison}
    test = FunctionType(_checkpoint_test.__code__, namespace)
    try:
        test(dimension, request)
    finally:
        directory = Path(request.config.getoption('xmlpath')).parent
        with (directory / f'lifecycle-original-runtime-comparisons-{dimension}.json').open('x') as handle:
            json.dump({'dimension': dimension, 'comparisons': comparisons,
                'scope': 'Complete original numerical comparisons; failure is raised after compiler/resource observations.'},
                handle, indent=2)
            handle.write('\n')
    for row in comparisons:
        _compare_original(row['actual'], row['expected'])
