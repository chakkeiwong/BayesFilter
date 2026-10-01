"""Dense spectral boundary regressions before enclosing uniform XLA use."""

import json
from pathlib import Path

import numpy as np
import pytest

from tests.filter_repair_dense_extreme_comparison import compare_huge_dense_records
from tests.test_filter_repair_quadratic_numerics import (
    equal_dense_fields,
    inputs,
    original,
    program,
    serializable,
)


@pytest.mark.parametrize('dimension', [3, 5])
@pytest.mark.parametrize('case', ['indefinite', 'zero', 'condition', 'repeated', 'tiny', 'huge', 'near_identity'])
def test_dense_spectral_boundaries_preserve_original_fields(dimension, case, request):
    arguments = inputs('dense', dimension, case)
    expected = program('dense', dimension, source='original')(*arguments)
    results = {mode: program('dense', dimension, jit=mode == 'xla')(*arguments) for mode in ('graph', 'xla')}
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / f'dense-boundary-{case}-{dimension}.json').open('x') as handle:
        json.dump({'original': serializable(expected),
            **{mode: serializable(row) for mode, row in results.items()},
            'baseline': '3582b4ac', 'source_sha256': original()[0].hashes()}, handle, indent=2, allow_nan=False)
        handle.write('\n')
    comparisons = {}
    for mode, row in results.items():
        if case == 'huge':
            comparisons[mode] = compare_huge_dense_records(row, expected,
                dimension=dimension, case=case, offsets=arguments[1],
                response=arguments[0][None, :] - arguments[2])
        else:
            equal_dense_fields(row, expected, dimension)
        np.testing.assert_allclose(row['raw_eigenvalues'], np.linalg.eigvalsh(row['raw_precision']),
                                   atol=1e-10, rtol=1e-10)
    if comparisons:
        with (directory / f'dense-boundary-huge-comparison-{dimension}.json').open('x') as handle:
            json.dump(comparisons, handle, indent=2, allow_nan=False)
            handle.write('\n')
