"""Source-bound diagnostic composition; never external consumer admission."""
from __future__ import annotations
import hashlib
import inspect
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
CONSUMER = Path(os.environ.get('BAYESFILTER_DIAGNOSTIC_CONSUMER_ROOT', '/home/ubuntu/python/MacroFinance')).resolve()
EVIDENCE = BASE/'docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/shared-gpu-policy-20261024-01/gpu-01/evidence'


def verify_gpu_evidence(directory=EVIDENCE, *, base=BASE, consumer=CONSUMER):
    directory, base, consumer = Path(directory), Path(base), Path(consumer)
    result=json.loads((directory/'result.json').read_text())
    manifest=json.loads((directory/'manifest.json').read_text())
    required=('same_value_score','accepted_and_proposed_status','gpu_xla_complete_target',
        'cpu_gpu_target_parity','gpu_xla_shared_chain_body','gpu_xla_deterministic_replay',
        'cpu_gpu_shared_chain_parity')
    if (result.get('status')!='completed' or not result.get('source_unchanged')
            or any(result.get('checks',{}).get(k) is not True for k in required)):
        raise ValueError('complete GPU diagnostic evidence required')
    if (manifest.get('device')!='GPU' or not manifest.get('memory_policy',{}).get(
            'configured_before_logical_device_initialization')
            or not manifest['memory_policy'].get('all_physical_devices_memory_growth')
            or 'device:GPU:' not in result.get('chain_output_device','')):
        raise ValueError('GPU placement and growth evidence required')
    sources=manifest['source_sha256']
    required_paths=[base/'bayesfilter/linear/kalman_qr_derivatives_tf.py',
        base/'bayesfilter/ssm/target_builder.py',consumer/'one_country_affine_macro_phase9_hmc.py']
    if any(str(p) not in sources for p in required_paths):
        raise ValueError('required numerical source absent from evidence')
    checked={}
    for name,digest in sources.items():
        path=Path(name)
        if path.is_relative_to(base/'bayesfilter') or path.is_relative_to(consumer):
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise ValueError('GPU evidence source changed: '+name)
            checked[name]=digest
    return dict(evidence=str(directory), checked_sources=checked,
        manifest_sha256=hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest(),
        result_sha256=hashlib.sha256((directory/'result.json').read_bytes()).hexdigest(),
        role='source-bound full-chain diagnostic capability only')


def build_measured_diagnostic(inputs, memory, *, plan, gpu_evidence=None):
    import tensorflow as tf
    import one_country_affine_macro_phase9_hmc as consumer
    from bayesfilter.linear.kalman_tf import tf_kalman_log_likelihood
    from bayesfilter.linear.kalman_qr_derivatives_tf import tf_qr_sqrt_kalman_score_factors_batched_static as kernel
    from bayesfilter.ssm import build_ssm_posterior_adapter
    evidence=None if gpu_evidence is None else verify_gpu_evidence(gpu_evidence)
    original=consumer._build_generic_target_with_runtime(inputs['observations'],
        dataset_id=inputs['design']['dataset_id'],
        runtime=(tf,tf_kalman_log_likelihood,kernel,memory),
        diagnostic_admission={'role':'source_bound_reference_diagnostic_only','plan':plan})
    observations=tf.constant(inputs['observations'],tf.float64)
    accepted=set(inspect.signature(kernel.python_function).parameters)-{
        'observations','jitter','jitter_updates_filtered_covariance','return_health'}
    def measured(theta):
        payload=consumer._tf_chart_batch_with_derivatives(theta,tf,include_support_checks=True)
        value,score,health=kernel(observations=observations,jitter=tf.constant(0.,tf.float64),
            return_health=True,**{k:v for k,v in payload.items() if k in accepted})
        health=dict(health)
        support=payload['support_valid']
        health['valid_pre_regularized_score'] &= support
        health['status_code']=tf.where((health['status_code']==0)&~support,4,health['status_code'])
        health['chart_support_valid']=support
        health['chart_lyapunov_relative_residual']=payload['lyapunov_relative_residual']
        return value,score,health
    # The analytic Lyapunov/QR implementation requires fully static batches.
    # Bound tracing to the public singleton probe and four-chain execution.
    graphs={n:tf.function(lambda q:measured(q)[2],
        input_signature=[tf.TensorSpec([n,9],tf.float64)],autograph=False,jit_compile=True)
        for n in (1,4)}
    def status(q):
        if q.shape.rank!=2 or q.shape[0] not in graphs:
            raise ValueError('diagnostic health supports static batch sizes 1 and 4')
        return graphs[q.shape[0]](q)
    provenance=dict(schema='macrofinance-measured-filter-health-diagnostic-v2',
        consumer_source_sha256=hashlib.sha256((CONSUMER/'one_country_affine_macro_phase9_hmc.py').read_bytes()).hexdigest(),
        filter_source_sha256=hashlib.sha256((BASE/'bayesfilter/linear/kalman_qr_derivatives_tf.py').read_bytes()).hexdigest(),
        composition_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='diagnostic composition, not installed consumer or posterior evidence',gpu_evidence=evidence)
    diagnostic=build_ssm_posterior_adapter(contract=original.contract,
        prior_log_prob_and_grad=original.prior_log_prob_and_grad,
        filter_log_likelihood_and_grad=original.filter_log_likelihood_and_grad,
        target_status_telemetry=status,target_status_manifest=provenance,
        dtype=tf.float64,target_scope=original.target_scope,evidence_path=plan,
        xla_hmc_ready=evidence is not None,full_chain_xla_diagnostic_ready=evidence is not None,
        nonclaims=('source-checked GPU numerical diagnostic only','external consumer not admitted',
            'no posterior, convergence or default promotion'))
    diagnostic=consumer._AnalyticScoreAdapter(diagnostic,original.analytic_score_provenance())
    if original.value_score_capability().xla_hmc_ready:
        raise ValueError('external consumer capability changed; inspect before continuation')
    return original,diagnostic,measured,provenance
