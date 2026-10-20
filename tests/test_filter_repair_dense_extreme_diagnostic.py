"""Independent high-precision and original self-sensitivity diagnostics only."""

import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from tests.filter_repair_dense_extreme_comparison import exact_symmetric_fit
from tests.test_filter_repair_quadratic_numerics import (
    inputs,
    original,
    program,
    serializable,
)


@pytest.mark.parametrize('dimension',[3,5])
def test_original_extreme_precision_self_sensitivity(dimension,request):
    arguments=inputs('dense',dimension,'huge')
    x=arguments[1].numpy()
    response=arguments[0].numpy()[None,:]-arguments[2].numpy()
    exact=exact_symmetric_fit(x,response,100)
    higher=exact_symmetric_fit(x,response,160)
    np.testing.assert_array_equal(exact,higher)
    baseline=program('dense',dimension,source='original')(*arguments)
    results={'original':baseline,
        'graph':program('dense',dimension,jit=False)(*arguments),
        'xla':program('dense',dimension,jit=True)(*arguments)}
    perturbations={}
    for row,column in ((0,0),(1,dimension-1),(7,dimension//2)):
        for direction in (-np.inf,np.inf):
            changed=x.copy()
            changed[row,column]=np.nextafter(changed[row,column],direction)
            values=(arguments[0],tf.constant(changed,tf.float64),*arguments[2:])
            result=program('dense',dimension,source='original')(*values)
            difference=~np.isclose(result['raw_precision'],baseline['raw_precision'],atol=1e-10,rtol=1e-10)
            perturbations[f'{row}:{column}:{direction}']={'result':serializable(result),
                'strict_raw_precision_failed_entries':int(np.count_nonzero(difference)),
                'max_error_over_matrix_scale':float(np.max(np.abs(result['raw_precision']-baseline['raw_precision']))/np.max(np.abs(baseline['raw_precision'])))}
    errors={name:{'max_error_over_reference_scale':float(np.max(np.abs(value['raw_precision']-exact))/np.max(np.abs(exact))),
        'relative_response_residual':float(np.linalg.norm((x @ value['raw_precision'].numpy()-response)/np.max(np.abs(response)))/np.linalg.norm(response/np.max(np.abs(response))))}
        for name,value in results.items()}
    directory=Path(request.config.getoption('xmlpath')).parent
    with (directory/f'dense-extreme-diagnostic-{dimension}.json').open('x') as handle:
        json.dump({'role':'independent_explanation_only_not_parity_waiver','baseline':'3582b4ac',
            'original_source_sha256':original()[0].hashes(),'precision_digits':[100,160],
            'exact_symmetric_precision':exact.tolist(),'design_condition':float(np.linalg.cond(x)),
            'results':{name:serializable(value) for name,value in results.items()},
            'errors':errors,'original_one_ulp_perturbations':perturbations},handle,indent=2,allow_nan=False)
        handle.write('\n')
    # Self-sensitivity remains explanatory. Mandatory complete-record, reference
    # and residual comparisons now run in dense_boundaries under owner approval.
    assert all(row['max_error_over_reference_scale']<1e-12 for row in errors.values())
    assert all(row['relative_response_residual']<1e-12 for row in errors.values())
