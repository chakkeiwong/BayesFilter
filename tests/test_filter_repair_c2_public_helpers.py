"""Independent-original diagnostics for C2 public sampling helpers."""

import inspect
import json
import os
from pathlib import Path

import pytest
import tensorflow as tf

from tests.test_filter_repair_c2_preparation import (
    BASELINE,
    MaterializedCheckpoint,
    compare,
    jsonable,
)

D = tf.float64


def proposal(module, changed=False):
    return module.FrozenGaussianStateProposal(
        mean=tf.constant([.2, -.1] if not changed else [.31, .08], D),
        chol=tf.constant([[1., 0.], [.1, .9]] if not changed else [[.8, 0.], [-.2, 1.1]], D),
        time_index=1, family='standalone_reference_fixture')


def sample_inputs(changed=False):
    means=tf.constant([[.2, -.1], [-.1, .3], [.4, .2], [-.3, -.2]], D)
    chol=tf.broadcast_to(tf.constant([[1., 0.], [.1, .9]], D), [4, 2, 2])
    if changed:
        means=means+tf.constant([.07, -.11], D)
    return {'posterior_means': means,
        'posterior_covariances': tf.matmul(chol, chol, transpose_b=True),
        'posterior_cholesky': chol,
        'lookahead_log_likelihood': tf.constant([-.1, -.3, -.2, -.4], D),
        'log_parent_weights': tf.math.log(tf.constant([.1, .2, .3, .4], D)),
        'seed': (-813, 91) if changed else (813, 17)}


def test_original_standalone_contract(request):
    """Freeze accepted edge cases and invalid-size/seed errors before edits."""
    old=MaterializedCheckpoint(BASELINE, 'c2_public_helpers_contract')
    module=old.load('bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf')
    rows=[]
    for count, seed in ((0, (813,17)), (1, (813,17)), (-1, (813,17)), (4, (813,)),
                        (4, (2**34+7, -814)), (4, (813,17,99))):
        for jit in (False, True):
            row={'count':count, 'seed':seed, 'jit_compile':jit}
            try:
                row['result']=jsonable(proposal(module).sample_with_seed(count, seed, jit_compile=jit))
            except (IndexError, tf.errors.InvalidArgumentError) as error:
                row['error']={'class':type(error).__name__, 'message':str(error)}
            rows.append(row)
    output=Path(request.config.getoption('xmlpath')).parent
    (output/'c2-public-helper-original-contract.json').write_text(
        json.dumps({'rows':rows,'baseline_sources':old.hashes()},indent=2)+'\n')


@pytest.mark.parametrize('jit', (False,True))
def test_standalone_gaussian_preserves_original_stream(jit, request):
    from bayesfilter.highdim import c2_sv_frozen_proposal_apf_tf as current
    old=MaterializedCheckpoint(BASELINE, 'c2_public_helpers_gaussian')
    original=old.load('bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf')
    rows=[]
    report={'max_abs':0.}
    for changed, count, seed in ((False,16,(813,17)), (True,16,(-814,91)),
                                 (False,0,(813,17)), (False,1,(2**34+7,-814))):
        expected=proposal(original,changed).sample_with_seed(count,seed,jit_compile=jit)
        actual=proposal(current,changed).sample_with_seed(count,seed,jit_compile=jit)
        compare(jsonable(expected),jsonable(actual),report=report)
        rows.append({'changed':changed,'count':count,'seed':seed,
                     'original':jsonable(expected),'current':jsonable(actual)})
    if jit:
        assert inspect.signature(current.FrozenGaussianStateProposal.sample_with_seed).parameters['jit_compile'].default is True
        actual=proposal(current).sample_with_seed(16,(813,17))
        compare(jsonable(actual),rows[0]['current'])
    owner=current._gaussian_seed_sampler(16,2,jit)
    assert owner.experimental_get_tracing_count()==1
    concrete=owner.get_concrete_function()
    assert concrete.inputs and not concrete.captured_inputs
    assert all(node.op not in ('PyFunc','EagerPyFunc') for node in concrete.graph.as_graph_def().node)
    if jit:
        p=proposal(current)
        hlo=owner.experimental_get_compiler_ir(p.mean,p.chol,tf.constant([813,17],tf.int64))(stage='hlo')
        assert 'HloModule' in hlo
    output=Path(request.config.getoption('xmlpath')).parent
    (output/f'c2-public-helper-gaussian-{jit}.json').write_text(json.dumps({
        'rows':rows,'maximum_absolute_error':report['max_abs'],
        'trace_count':owner.experimental_get_tracing_count(),'cache':current._gaussian_seed_sampler.cache_info()._asdict(),
        'baseline_sources':old.hashes()},indent=2)+'\n')


@pytest.mark.parametrize('jit', (False,True))
def test_k1_convenience_retains_original_compiled_stream(jit, request):
    from bayesfilter.highdim import c2_mixture_ukf_apf_tf as current
    old=MaterializedCheckpoint(BASELINE, 'c2_public_helpers_k1')
    original=old.load('bayesfilter.highdim.c2_mixture_ukf_apf_tf')
    records=[]
    for changed in (False, True, False):
        kwargs=sample_inputs(changed)
        expected=original.sample_k1_apf_step(**kwargs,jit_compile=jit)
        actual=current.sample_k1_apf_step(**kwargs,jit_compile=jit)
        compare(jsonable(expected),jsonable(actual))
        records.append({'changed':changed,'original':jsonable(expected),'current':jsonable(actual)})
    owner=current._retained_k1_apf_sampler(4,2,jit)
    assert owner.experimental_get_tracing_count()==1
    assert not owner.get_concrete_function().captured_inputs
    output=Path(request.config.getoption('xmlpath')).parent
    (output/f'c2-public-helper-k1-{jit}.json').write_text(json.dumps({
        'records':records,'trace_count':owner.experimental_get_tracing_count(),
        'baseline_sources':old.hashes()},indent=2)+'\n')


def test_standalone_configuration_caches_are_bounded():
    from bayesfilter.highdim import c2_mixture_ukf_apf_tf as ukf
    from bayesfilter.highdim import c2_sv_frozen_proposal_apf_tf as gaussian
    # Factory-only pressure checks the ownership bound, not native eviction.
    for count in range(20,29):
        gaussian._gaussian_seed_sampler(count,2,True)
        ukf._retained_k1_apf_sampler(count,2,True)
    assert gaussian._gaussian_seed_sampler.cache_info().currsize==4
    assert ukf._retained_k1_apf_sampler.cache_info().currsize==4


def test_standalone_gaussian_original_edge_contract():
    from bayesfilter.highdim import c2_sv_frozen_proposal_apf_tf as current
    number=5526 if os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' else 5527
    raw=Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
    reference=json.loads((raw/f'run-{number:05d}'/'c2-public-helper-original-contract.json').read_text())
    for row in reference['rows']:
        p=proposal(current)
        if 'result' in row:
            actual=p.sample_with_seed(row['count'],tuple(row['seed']),jit_compile=row['jit_compile'])
            assert actual['physical_points'].shape==(row['count'],2)
            compare(row['result'],jsonable(actual))
        else:
            with pytest.raises(Exception) as caught:
                p.sample_with_seed(row['count'],tuple(row['seed']),jit_compile=row['jit_compile'])
            assert type(caught.value).__name__==row['error']['class']
            if row['count']<0:
                assert f"Dimension {row['count']} must be >= 0" in str(caught.value)
            else:
                assert str(caught.value)==row['error']['message']
    # Seed coercion still precedes TensorFlow shape rejection.
    with pytest.raises(IndexError,match='tuple index out of range'):
        proposal(current).sample_with_seed(-1,(813,))
