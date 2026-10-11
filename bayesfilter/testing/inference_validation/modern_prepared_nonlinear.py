"""Full-model local-geometry bindings for explicitly scoped diagnostic campaigns.

This constructs existing repository binding/mass types. It implements no new
transition or tuner and does not admit a CPU local factor as global geometry.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def prepared_binding(target, geometry_directory, *, geometry_kind, seed, use_xla, source_paths=(),
                     geometry_source_policy='strict', transition_profile=None,
                     epsilon_domain=(.001, 1.), max_repairs_per_family=0):
    import tensorflow as tf
    from bayesfilter.inference import HMCCandidateExecutionConfig, PrecomputedMassArtifact
    from bayesfilter.inference.hmc_acceptance_protocol import HMCReplicatedAcceptancePolicy
    from bayesfilter.inference.hmc_candidate_set_execution import bind_hmc_candidate_set_execution
    from bayesfilter.inference.hmc_transition_spec import HMCTransitionProfile
    from bayesfilter.inference.posterior_adapter import value_score_capability
    from .modern_nonlinear import make_nonlinear_comparison_target
    from .modern_bridge_audit import source_dependencies

    if target not in ('NL-STR','NL-PP-UKF','NL-PP-SGQF','NL-SIR') or geometry_kind not in ('local','prior'):
        raise ValueError('unsupported full-model geometry diagnosis')
    if geometry_source_policy not in ('strict', 'warm_start_only'):
        raise ValueError('unsupported geometry source policy')
    directory=Path(geometry_directory)
    geometry=json.loads((directory/'geometry.json').read_text())
    design=json.loads((directory/'design.json').read_text())
    if design['target']!=target or geometry['status']!='local_factor_proposed':
        raise ValueError('wrong target or unavailable local factor')
    fixture=make_nonlinear_comparison_target(target,jit_compile=use_xla)
    if fixture.metadata['data_sha256']!=design['metadata']['data_sha256']:
        raise ValueError('local geometry is for a different dataset')
    tf.debugging.assert_equal(fixture.observations,tf.constant(design['data'],tf.float64))
    center=tf.constant(geometry['center'],tf.float64)
    local_factor=tf.constant(geometry['factor'],tf.float64)
    covariance=tf.constant(geometry['covariance'],tf.float64)
    tf.debugging.assert_near(local_factor@tf.transpose(local_factor),covariance,rtol=1e-9,atol=1e-12)
    tf.debugging.assert_positive(tf.linalg.eigvalsh(covariance))
    dimension=int(center.shape[0])
    offsets=tf.constant([[-.5]*6,[.5,-.5,.5,-.5,.5,-.5],
                         [-.5,.5,-.5,.5,-.5,.5],[.5]*6],tf.float64)[:,:dimension]
    offsets+=.05*tf.random.stateless_normal([4,dimension],[20261009,seed],dtype=tf.float64,alg='philox')
    starts=center+offsets@tf.transpose(local_factor)
    factor=(local_factor if geometry_kind=='local' else
            tf.eye(dimension,dtype=tf.float64)*(.5 if target=='NL-SIR' else 1.))
    mass=PrecomputedMassArtifact(position=center,covariance=factor@tf.transpose(factor),factor=factor,
        adapter_signature=fixture.adapter.adapter_signature(),position_role='reference_center',
        covariance_source='diagnostic_local_curvature_proposal' if geometry_kind=='local' else 'declared_prior_covariance')
    policy=HMCReplicatedAcceptancePolicy(trial_num_results=64,discarded_prefix=0,
        base_repetitions=32,max_repetitions=32,max_candidates=4,qualification_mode='lower_bound',
        qualification_region=(.5,1.),search_family_alpha=.05,verification_family_alpha=.05,
        diagnostic_family_alpha=.05,temporal_tolerance=.1,min_normalized_return_displacement=0.)
    config=HMCCandidateExecutionConfig(measurement_num_results=64,verification_num_results=64,
        num_warmup_steps=0,seed=(20261009,seed+100),acceptance_policy=policy,
        target_status_trace_policy='per_chain_step',use_xla=use_xla,
        non_xla_reason=None if use_xla else 'explicit full-model affine parity reference',
        chain_mode='batched',chunk_max_results=64,
        transition_profile=transition_profile or HMCTransitionProfile('hmc','uniform_duration'))
    root=Path(__file__).resolve().parents[3]
    sources=source_dependencies([__file__,*fixture.source_paths,*source_paths],repository_root=root)
    # The diagnostic driver gained a SIR-only optimizer scaling option after
    # the other geometries were saved. It is not part of target mathematics.
    # All recorded library sources must still match before reuse.
    source_changes=[]
    for path,digest in design.get('source_closure',{}).items():
        original=Path(path)
        # Recorded absolute paths refer to the checkout used to construct the
        # geometry. Compare against this immutable execution snapshot instead.
        parts=original.parts
        if 'bayesfilter' in parts:
            current=root.joinpath(*parts[parts.index('bayesfilter'):])
        elif 'scripts' in parts:
            current=root.joinpath(*parts[parts.index('scripts'):])
        else:
            current=original
        observed=hashlib.sha256(current.read_bytes()).hexdigest() if current.is_file() else None
        if observed != digest:
            source_changes.append(dict(recorded_path=path, recorded_sha256=digest,
                                       current_path=str(current), current_sha256=observed))
            if geometry_source_policy=='strict' and 'bayesfilter' in parts:
                raise ValueError('local geometry numerical source changed: '+path)
    geometry_digest=hashlib.sha256((directory/'geometry.json').read_bytes()).hexdigest()
    binding=bind_hmc_candidate_set_execution(adapter=fixture.adapter,initial_position=starts,
        target_scope=value_score_capability(fixture.adapter).target_scope,
        target_lineage=dict(model=target,data=fixture.metadata['data_sha256'],geometry_kind=geometry_kind,
                            proposed_geometry_sha256=geometry_digest,
                            geometry_source_policy=geometry_source_policy),
        config=config,source_paths=sources,scope_id=f'modern-n5-prepared-{target.lower()}-{geometry_kind}',
        search_id=f'diagnostic-seed-{seed}',epsilon_domain=epsilon_domain,repair_factor=1.25,
        max_repairs_per_family=max_repairs_per_family,mass_artifact=mass)
    metadata=dict(fixture.metadata,geometry_kind=geometry_kind,geometry_center=center,
                  geometry_factor=factor,geometry_sha256=geometry_digest,
                  start_positions=starts,contrasts=offsets,
                  preparation_role='unadmitted_local_proposal',global_coverage_established=False,
                  geometry_source_policy=geometry_source_policy,
                  geometry_source_changes=source_changes,
                  geometry_design_sha256=hashlib.sha256((directory/'design.json').read_bytes()).hexdigest())
    return binding,fixture,metadata
