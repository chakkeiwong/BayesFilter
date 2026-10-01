"""Completed initializer records; formatting never steers numerical execution."""

import tensorflow as tf

from bayesfilter.inference.joint_center_tf import SOURCES
from bayesfilter.inference.joint_center_tf import STATUSES as JOINT_STATUSES
from bayesfilter.inference.mass_matrix import MassMatrixResult, _precision_report
from bayesfilter.inference.quadratic_geometry_full_report import geometry_result
from bayesfilter.inference.quadratic_initializer_tf import EVALUATION_STATUSES, STATUSES
from bayesfilter.inference.quadratic_map_covariance import (
    IterativeQuadraticMapCovarianceResult,
    QuadraticMapCovarianceResult,
    _iterative_diagnostics,
    _iterative_rejected_result,
    _json_ready,
    _rejected_result,
)


def locator_report(raw, config):
    if int(raw['initial_status']):
        return {'status': 'not_run_initial_nonfinite', 'initial_log_prob': float(raw['initial_value']),
                'initial_evaluation_status': EVALUATION_STATUSES[int(raw['initial_status'])]}
    located = raw['locator']
    initial_value, initial_norm = float(raw['initial_value']), float(located['initial_norm'])
    base = {'schema': 'bayesfilter.quadratic_map_covariance.locator.v1',
        'method': 'tfp_lbfgs_minimize_negative_log_prob', 'optimizer_role': 'finite_neighborhood_locator_only',
        'uses_optimizer_inverse_hessian': False, 'jit_compile': True,
        'initial_log_prob': initial_value, 'initial_score_norm': initial_norm, 'config': config.payload()}
    if not config.enabled:
        return {**base, 'status': 'disabled_initial_position', 'accepted_optimizer_position': False,
                'locator_log_prob': initial_value, 'locator_score_norm': initial_norm}
    joint = located['raw']
    accepted, finite = bool(located['accepted']), bool(located['candidate_finite'])
    status = 'finite' if finite else 'nonfinite'
    present, callback_index = bool(joint['best_present']), int(joint['best_callback_index'])
    candidate_value, candidate_norm = float(located['candidate_value']), float(located['candidate_norm'])
    return {**base, 'status': 'tfp_lbfgs_locator_accepted' if accepted else 'tfp_lbfgs_locator_rejected_initial_fallback',
        'accepted_optimizer_position': accepted, 'optimizer_converged': bool(joint['optimizer_converged']),
        'optimizer_failed': bool(joint['optimizer_failed']), 'optimizer_iterations': int(joint['optimizer_iterations']),
        'optimizer_objective_value': -float(joint['endpoint_objective']),
        'optimizer_endpoint_log_prob': float(joint['endpoint_objective']),
        'best_evaluated_source': SOURCES[int(joint['best_source'])] if present else None,
        'best_evaluated_callback_index': callback_index if callback_index >= 0 else None,
        'joint_center_status': JOINT_STATUSES[int(joint['status'])], 'candidate_log_prob': candidate_value,
        'candidate_score_norm': candidate_norm if finite else None, 'candidate_evaluation_status': status,
        'locator_log_prob': candidate_value if accepted else initial_value,
        'locator_score_norm': candidate_norm if accepted else initial_norm,
        'fallback_reason': None if accepted else status}


def transform_report(raw):
    minimum, maximum, all_ones = raw['scale_summary']
    return {'geometry_fit_coordinate_system': 'whitened_z',
        'geometry_coordinate_transform': 'theta = center + scale * z',
        'geometry_precision_coordinate_system': 'z', 'mass_precision_coordinate_system': 'theta',
        'mass_covariance_coordinate_system': 'theta',
        'precision_transform': 'P_theta = diag(1 / scale) @ P_z @ diag(1 / scale)',
        'covariance_transform': 'C_theta = diag(scale) @ C_z @ diag(scale)',
        'scale_min': float(minimum), 'scale_max': float(maximum), 'scale_all_ones': bool(all_ones)}


def mass_result(raw, config, *, iterative):
    precision, covariance, diagnostics, flags, _ = raw['mass']
    report = _precision_report(diagnostics, flags, config.jitter, config.eigenvalue_floor, config.max_condition_number)
    if not config.dense:
        report = {**report, 'diagonal_fallback_used': True, 'diagonal_fallback_source': 'regularized_precision_diagonal'}
    # This public value type recomputes compiled spectral summaries. That work
    # remains a measured reporting cost; it cannot change initializer selection.
    return MassMatrixResult(covariance=covariance,
        source=('iterative_' if iterative else '') + 'low_rank_spd_quadratic_geometry_precision_theta_coordinates',
        matrix_kind='dense' if config.dense else 'diagonal', jitter=config.jitter,
        eigenvalue_floor=float(diagnostics[0]), regularized_precision=precision, regularization_report=report)


def mass_failure_report(raw):
    flags = raw['mass'][3]
    reason = ('precision must be finite' if not bool(flags[0]) else
        'precision eigenvalues must be finite' if not bool(flags[1]) else
        'precision must have a positive eigenvalue; pass eigenvalue_floor' if not bool(flags[2]) else
        'regularized precision diagonal must be positive finite')
    return {'mass_matrix_exception_type': 'ValueError', 'mass_matrix_exception': reason}


def initializer_result(raw, initial, scale, locator_config, geometry_config, mass_config, *,
        iterative_config=None, batched=False, fit_start_callback=None, iteration_callback=None):
    """Materialize completed events in original order after the native call."""
    iterative = iterative_config is not None
    status_code, count = int(raw['status']), int(raw['fit_count'])
    locator_diagnostics = locator_report(raw, locator_config)
    geometry, records = None, []
    for index in range(count):
        row = tf.nest.map_structure(lambda value, i=index: value[i], raw['history'])
        geometry = geometry_result(row['geometry'], row['center'], scale, geometry_config, batched=batched)
        if iterative:
            if fit_start_callback is not None:
                fit_start_callback(index, row['center'])
            record = {'fit_index': index, 'center': row['center'], 'center_log_prob': float(row['value']),
                'center_score_norm': float(row['score_norm']), 'center_score_max_abs': float(row['score_max']),
                'terminal_score_gate': iterative_config.terminal_score_max_abs, 'terminal_before_fit': bool(row['terminal']),
                'geometry_accepted': geometry.accepted, 'geometry_status': geometry.status,
                'center_refinement': dict(geometry.diagnostics.get('center_refinement', {})),
                'geometry_diagnostics': geometry.diagnostics}
            if iteration_callback is not None:
                iteration_callback(_json_ready(record))
            if bool(row['promoted']):
                record.update(exact_incumbent_promoted=True, exact_incumbent_source=geometry.best_evaluated_source,
                              exact_incumbent_value=geometry.best_evaluated_value)
            records.append(record)

    if status_code == 2:
        status = (f'iteration_{int(raw["last_index"])}_' if iterative else '') + f'geometry_{geometry.status}'
    elif status_code in (1, 6):
        status = f'iteration_{int(raw["last_index"])}_{STATUSES[status_code]}'
    else:
        status = STATUSES[status_code]
    nonclaims = {'reports_map_quality': False, 'reports_hmc_convergence': False, 'reports_default_readiness': False}
    common = {'initial_position': initial, 'locator_position': raw['locator_position'],
              'locator_diagnostics': locator_diagnostics}
    if iterative:
        common.update(iterations=tuple(records), terminal_geometry=geometry)
        diagnostics = (_iterative_diagnostics(iterative_config, geometry_config, mass_config, scale)
                       if status_code else nonclaims)
        if status_code == 4:
            diagnostics = {**diagnostics, **mass_failure_report(raw)}
        if status_code != 7:
            return _iterative_rejected_result(status=status, diagnostics=diagnostics, **common)
        mass = mass_result(raw, mass_config, iterative=True)
        return IterativeQuadraticMapCovarianceResult(accepted=True, status=status, dimension=initial.shape[0],
            map_candidate=raw['final_position'], map_candidate_role='iterative_terminal_exact_score_center',
            precision=mass.regularized_precision, covariance=mass.covariance, covariance_source=mass.source,
            mass_matrix=mass, **common, diagnostics={**diagnostics, **transform_report(raw),
                'classification': 'iterative_diagnostic_initializer_accepted',
                'terminal_fit_index': int(raw['last_index']), 'accepted_refinement_step_count': int(raw['last_index']),
                'terminal_score_max_abs': float(raw['history']['score_max'][count - 1]),
                'precision_authority': 'terminal_low_rank_spd_quadratic_geometry_precision_transformed_to_theta',
                'covariance_authority': 'covariance_from_precision'})

    common.update(geometry=geometry)
    diagnostics = {'classification': 'diagnostic_initializer_rejected', **nonclaims}
    candidate, candidate_role = None, 'none_rejected'
    if status_code == 2:
        diagnostics.update(geometry_status=geometry.status, geometry_accepted=geometry.accepted, mass_matrix_attempted=False)
    elif status_code == 8:
        candidate, candidate_role = geometry.best_evaluated_position, f'best_exact_{geometry.best_evaluated_source}'
        diagnostics.update(geometry_status=geometry.status, geometry_accepted=geometry.accepted,
            stale_centered_covariance_prevented=True, covariance_fit_center=geometry.center,
            exact_incumbent_source=geometry.best_evaluated_source, exact_incumbent_value=geometry.best_evaluated_value)
    elif status_code == 4:
        diagnostics.update(geometry_status=geometry.status, mass_matrix_attempted=True,
                           **transform_report(raw), **mass_failure_report(raw))
    if status_code != 7:
        return _rejected_result(status=status, diagnostics=diagnostics, mass_matrix=None,
            map_candidate=candidate, map_candidate_role=candidate_role, **common)
    mass = mass_result(raw, mass_config, iterative=False)
    return QuadraticMapCovarianceResult(accepted=True, status=status, dimension=initial.shape[0],
        map_candidate=geometry.center, map_candidate_role='exact_incumbent_geometry_center',
        precision=mass.regularized_precision, covariance=mass.covariance, covariance_source=mass.source,
        mass_matrix=mass, **common, diagnostics={**nonclaims,
            'classification': 'diagnostic_initializer_accepted',
            'precision_authority': 'low_rank_spd_quadratic_geometry_precision_transformed_to_theta',
            'covariance_authority': 'covariance_from_precision', **transform_report(raw),
            'optimizer_authority': 'locator_only', 'geometry_center_refinement_accepted': geometry.center_refinement_accepted,
            'map_candidate_role': 'exact_incumbent_geometry_center', 'locator_config': locator_config.payload(),
            'quadratic_config': geometry_config.payload(), 'mass_config': mass_config.payload()})
