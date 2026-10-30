"""Exact parity against frozen source-08 health logic, including hostile traces.

The copied reference contains only the affected functions from the frozen
release source; it has no runtime authority. Other dependencies are unchanged.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
import tensorflow as tf

from bayesfilter.inference import hmc_verification as verification
from bayesfilter.inference.hmc_candidate_set_execution import HMCCandidateExecutionBinding
from bayesfilter.inference.hmc_candidate_health import native_health_program


def reference_namespace():
    namespace = dict(vars(verification))
    code = (Path(__file__).parent/'data/hmc_health_source08_reference.txt').read_text()
    exec(compile(code, 'frozen-source08-health-reference', 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('case', ['integer32', 'unsigned64', 'negative_floor',
    'failed_negative_floor', 'conditioning_float', 'conditioning_integer',
    'conditioning_complex', 'failed_nonfinite', 'bad_optional_dtype',
    'two_bad_optional_fields', 'floor_before_bad_dtype'])
def test_fused_status_matches_original_for_dtypes_and_malformed_fields(reference, case):
    dtype = tf.uint64 if case == 'unsigned64' else tf.int32
    telemetry = dict(status_code=tf.zeros([2,4], dtype),
        valid_pre_regularized_score=tf.ones([2,4], tf.bool),
        floor_count_value=tf.zeros([2,4], dtype))
    optional = verification.TARGET_STATUS_TELEMETRY_OPTIONAL_CONDITIONING_FIELDS
    if case == 'unsigned64':
        telemetry['floor_count_value'] = tf.fill([2,4], tf.constant(2**64-1, tf.uint64))
        telemetry['status_code'] = tf.fill([2,4], tf.constant(2**63, tf.uint64))
        # Include a valid row with a large unsigned floor value.
        telemetry['status_code'] = tf.tensor_scatter_nd_update(telemetry['status_code'], [[0,0]], [tf.constant(0, tf.uint64)])
    if case in {'negative_floor', 'failed_negative_floor', 'floor_before_bad_dtype'}:
        telemetry['floor_count_value'] = tf.fill([2,4], -1)
    if case in {'failed_negative_floor', 'failed_nonfinite'}:
        telemetry['valid_pre_regularized_score'] = tf.zeros([2,4], tf.bool)
    if case.startswith('conditioning') or case in {'failed_nonfinite', 'bad_optional_dtype', 'two_bad_optional_fields', 'floor_before_bad_dtype'}:
        optional_dtype = {'conditioning_integer': tf.int64, 'conditioning_complex': tf.complex128}.get(case, tf.float64)
        telemetry.update({name: tf.ones([2,4], optional_dtype) for name in optional})
        if case in {'conditioning_complex', 'failed_nonfinite', 'two_bad_optional_fields'}:
            value = complex(1., float('inf')) if case == 'conditioning_complex' else float('nan')
            telemetry[optional[0]] = tf.fill([2,4], tf.constant(value, optional_dtype))
        if case in {'bad_optional_dtype', 'two_bad_optional_fields', 'floor_before_bad_dtype'}:
            telemetry[optional[-1]] = tf.fill([2,4], 'invalid')

    def outcome(function):
        try:
            return ('result', function(telemetry, expected_shape=(2,4)))
        except (ValueError, TypeError) as exc:
            return (type(exc).__name__, str(exc))

    assert outcome(verification.target_status_telemetry_has_failure) == outcome(reference['target_status_telemetry_has_failure'])


@pytest.fixture(scope='module')
def reference():
    return reference_namespace()


def native_fixture():
    initial = tf.reshape(tf.cast(tf.range(8), tf.float64), [4, 2])
    samples = initial[None] + tf.cast(tf.range(1, 69), tf.float64)[:, None, None]/100.
    accepted = tf.ones([68, 4], tf.bool)
    trace = dict(proposed_state=samples, is_accepted=accepted,
        initial_momentum=tf.ones_like(samples), final_momentum=tf.ones_like(samples),
        log_accept_ratio=tf.fill([68,4],tf.constant(-.35,tf.float64)),
        log_acceptance_correction=tf.zeros([68,4],tf.float64),
        target_log_prob=-tf.reduce_sum(samples**2,axis=-1),
        proposed_target_log_prob=-tf.reduce_sum(samples**2,axis=-1),
        target_score_finite=accepted)
    status=dict(status_code=tf.zeros([68,4],tf.int32),
        valid_pre_regularized_score=accepted,floor_count_value=tf.zeros([68,4],tf.int32))
    trace.update(target_status_telemetry=dict(status),proposed_target_status_telemetry=dict(status))
    binding=SimpleNamespace(initial_active_state=initial,
        config=SimpleNamespace(use_xla=True,target_status_trace_policy='per_chain_step'))
    return binding,initial,samples,trace


def change(tensor, value, *, first=True):
    index = [0]*tensor.shape.rank if first else [int(d)-1 for d in tensor.shape]
    return tf.tensor_scatter_nd_update(tensor,[index],[tf.constant(value,tensor.dtype)])


@pytest.mark.parametrize('damage',[
    'healthy','rejected_first','bad_rejected_state','initial_nan','sample_nan',
    'proposal_inf','overflow_displacement','initial_momentum_nan','final_momentum_inf',
    'log_accept_nan','target_inf','proposed_target_inf','correction_nan','score_failed',
    'no_correction','divergence','target_status_failed','proposal_status_failed',
    'negative_floor','missing_status_field','malformed_momentum',
])
def test_native_reason_codes_match_frozen_implementation(reference, damage):
    binding,initial,samples,trace=native_fixture()
    expected_reason=None
    if damage=='rejected_first':
        trace['is_accepted']=change(trace['is_accepted'],False)
        samples=tf.tensor_scatter_nd_update(samples,[[0,0]],[initial[0]])
    elif damage=='bad_rejected_state':
        trace['is_accepted']=change(trace['is_accepted'],False)
        expected_reason='metropolis_state_mismatch'
    elif damage=='initial_nan':
        initial=change(initial,float('nan'));expected_reason='nonfinite_initial_state'
    elif damage=='sample_nan':
        samples=change(samples,float('nan'));expected_reason='nonfinite_state'
    elif damage=='overflow_displacement':
        initial=change(initial,-1.7e308)
        trace['proposed_state']=change(trace['proposed_state'],1.7e308)
        expected_reason='nonfinite_proposal_displacement'
    elif damage in {'target_status_failed','proposal_status_failed'}:
        key='target_status_telemetry' if damage=='target_status_failed' else 'proposed_target_status_telemetry'
        trace[key]['status_code']=change(trace[key]['status_code'],1)
        expected_reason=key+'_failed'
    elif damage=='negative_floor':
        trace['target_status_telemetry']['floor_count_value']=change(trace['target_status_telemetry']['floor_count_value'],-1)
    elif damage=='missing_status_field':
        trace['target_status_telemetry'].pop('status_code')
    elif damage=='malformed_momentum':
        trace['initial_momentum']=trace['initial_momentum'][:-1]
    elif damage=='divergence':
        trace['divergence']=change(tf.zeros([68,4],tf.bool),True);expected_reason='native_divergence_positive'
    elif damage=='score_failed':
        trace['target_score_finite']=change(trace['target_score_finite'],False);expected_reason='nonfinite_target_score'
    elif damage=='no_correction':
        trace.pop('log_acceptance_correction')
    elif damage not in {'healthy'}:
        field,value={
            'proposal_inf':('proposed_state',float('inf')),
            'initial_momentum_nan':('initial_momentum',float('nan')),
            'final_momentum_inf':('final_momentum',float('inf')),
            'log_accept_nan':('log_accept_ratio',float('nan')),
            'target_inf':('target_log_prob',float('inf')),
            'proposed_target_inf':('proposed_target_log_prob',float('inf')),
            'correction_nan':('log_acceptance_correction',float('nan')),
        }[damage]
        trace[field]=change(trace[field],value)
        expected_reason='nonfinite_'+('proposal' if field=='proposed_state' else field)
    if damage in {'negative_floor','missing_status_field','malformed_momentum'}:
        for function in (reference['health_failures'],HMCCandidateExecutionBinding.health_failures):
            with pytest.raises(ValueError):function(binding,initial,samples,trace)
    else:
        old=reference['health_failures'](binding,initial,samples,trace)
        actual=HMCCandidateExecutionBinding.health_failures(binding,initial,samples,trace)
        assert actual==old
        if expected_reason:assert expected_reason in actual
        else:assert not actual


@pytest.mark.parametrize('shape',[(65,4,1),(65,4,3),(129,4,2)])
@pytest.mark.parametrize('path',['healthy','frozen','cycle','endpoint_return'])
def test_reported_health_payload_is_exactly_unchanged(reference,shape,path):
    draws,chains,dimension=shape
    samples=tf.random.stateless_normal(shape,[20261003,2701],dtype=tf.float64)
    if path=='frozen':samples=tf.zeros_like(samples)
    elif path=='cycle':samples=tf.cast(tf.range(draws)%2,tf.float64)[:,None,None]*tf.ones(shape,tf.float64)
    elif path=='endpoint_return':samples=tf.concat([samples[:-1],samples[:1]],axis=0)
    log_accept=tf.random.stateless_uniform([draws,chains],[20261003,2702],minval=-1.,maxval=.2,dtype=tf.float64)
    kwargs=dict(samples=samples,log_accept_ratio=log_accept,is_accepted=log_accept>-.7,
                target_log_prob=-tf.reduce_sum(samples**2,axis=-1),
                policy=verification.HMCAcceptancePolicy(min_normalized_return_displacement=0.))
    old=reference['_evaluate_hmc_health_context'](**kwargs)
    old=old.evidence if isinstance(old,verification._AcceptanceHealthContext) else old
    new=verification.evaluate_hmc_trial_health(**kwargs)
    assert old.payload()==new.payload()


def test_native_health_graph_has_one_stable_trace():
    program=native_health_program(True)
    binding,initial,samples,trace=native_fixture()
    HMCCandidateExecutionBinding.health_failures(binding,initial,samples,trace)
    assert program.input_signature is not None
    assert program.experimental_get_tracing_count()==1


def test_vector_materialization_is_once_per_vector():
    class Materialized:
        def tolist(self):return [.1,.2,.3,.4]
    class TensorBoundary:
        calls=0
        def numpy(self):self.calls+=1;return Materialized()
        def __iter__(self):raise AssertionError('per-element tensor conversion')
    vector=TensorBoundary()
    assert verification._host_vector(vector)==(.1,.2,.3,.4)
    assert vector.calls==1
