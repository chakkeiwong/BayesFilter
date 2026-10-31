"""Explanatory kernel-cost attribution; not full endpoint timing or admission."""

import json
import statistics
import time
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.mass_matrix_tf import eigenpair_program
from bayesfilter.inference.score_curvature_tf import _singular_value_program
from bayesfilter.ops.qr_lstsq_tf import complete_orthogonal_lstsq
from tests.test_filter_repair_quadratic_numerics import inputs, original

D = tf.float64


@pytest.mark.parametrize('dimension',[3,5])
def test_dense_dependency_component_costs(dimension,request):
    arguments=inputs('dense',dimension)
    center,offsets,scores,_,_=arguments
    response=center[None,:]-scores
    precision=original()[1]['dense'].fit_dense_score_precision_tf(center,offsets,scores)['raw_precision']
    gpu=bool(tf.config.list_logical_devices('GPU'))
    cases={
        'native_cod':(complete_orthogonal_lstsq,(offsets,response)),
        'design_svd':(lambda matrix: tf.linalg.svd(matrix,compute_uv=False),(offsets,)),
        'reduced_design_svd':(lambda matrix: tf.linalg.svd(
            tf.linalg.qr(matrix, full_matrices=False)[1], compute_uv=False),(offsets,)),
        'precise_reduced_design_svd':(lambda matrix: _singular_value_program(dimension)(
            tf.linalg.qr(matrix, full_matrices=False)[1]),(offsets,)),
        'precision_eigh':(tf.linalg.eigvalsh,(precision,)),
        'refined_precision_eigh':(lambda matrix: eigenpair_program(dimension)(matrix)[0],(precision,)),
    }
    observations={}
    for name,(function,values) in cases.items():
        modes={}
        for jit in (False,True):
            numerical = tf.linalg.eigvalsh if name == 'refined_precision_eigh' and not jit else function
            if name == 'precise_reduced_design_svd' and not jit:
                numerical = cases['design_svd'][0]
            kernel=tf.function(numerical,input_signature=[tf.TensorSpec(v.shape,v.dtype) for v in values],
                autograph=False,jit_compile=jit)
            times=[]
            first=None
            for _ in range(41):
                started=time.perf_counter()
                result=kernel(*values).numpy()
                times.append(time.perf_counter()-started)
                if first is None:
                    first=result.tolist()
                else:
                    assert first==result.tolist()
            assert kernel.experimental_get_tracing_count()==1
            graph=kernel.get_concrete_function().graph.as_graph_def()
            modes['xla' if jit else 'graph']={'cold_seconds':times[0], 'warm_seconds':times[1:],
                'warm_median_seconds':statistics.median(times[1:]),'result':first,
                'nodes':len(graph.node)+sum(len(f.node_def) for f in graph.library.function)}
        np.testing.assert_allclose(modes['xla']['result'],modes['graph']['result'],atol=1e-10,rtol=1e-10)
        observations[name]=modes
    directory=Path(request.config.getoption('xmlpath')).parent
    with (directory/f'dense-components-{dimension}.json').open('x') as handle:
        json.dump({'role':'explanatory_dependency_cost_attribution','dimension':dimension,'gpu':gpu,
            'components':observations,'baseline_source_sha256':original()[0].hashes(),
            'nonclaims':['Separate launches have different fusion/dispatch overhead; these times are not additive.',
                'This is not a replacement for full original/graph/XLA endpoint comparisons.']},handle,indent=2,allow_nan=False)
        handle.write('\n')
