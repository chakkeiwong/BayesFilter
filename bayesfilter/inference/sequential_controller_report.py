"""Host formatting of completed sequential-controller records only.

All selection, acceptance, movement metrics and mass construction are already
finished. Iteration here serializes bounded histories; it never calls a target
or a numerical decision program.
"""

from dataclasses import replace

import tensorflow as tf

from bayesfilter.inference.mass_matrix import _precision_report
from bayesfilter.inference.sequential_score_fit_tf import partition_schema


def _row(columns, index):
    if isinstance(columns, dict):
        return {key: _row(value, index) for key, value in columns.items()}
    if isinstance(columns, tuple):
        return type(columns)(*(_row(value, index) for value in columns))
    return columns[index]


def terminal_payload(result, cfg):
    computed, seed = result['record'], result['seed']
    status = computed['status']
    if status == -1:
        return {'status': 'insufficient_symmetric_support', 'seed': seed}
    winner = result['has_best']
    payload = {'status': ('rank_deficient_symmetric_fit', 'usable', 'score_holdout_failed')[status],
        'rank': computed['rank'], 'seed': seed,
        'best_exact_value': computed['best_value'] if winner else None,
        'best_exact_position': computed['best_position'] if winner else None,
        'best_exact_score': computed['best_score'] if winner else None,
        'best_exact_source': 'score_fit_cloud' if winner else None}
    if status != 0:
        payload.update(train_score_rmse=computed['train_score_rmse'],
            holdout_score_relative_rmse=computed['holdout_score_relative_rmse'],
            raw_eigenvalues=computed['raw_eigenvalues'], projected_eigenvalues=computed['projected_eigenvalues'],
            projection_relative_frobenius=computed['projection_relative_frobenius'],
            projected_precision_z=computed['projected_precision_z'])
        if cfg.pair_disjoint_score_holdout:
            training, holdout = partition_schema(cfg.terminal_sample_count, cfg.holdout_fraction, pair_disjoint=True)
            payload.update(pair_disjoint_score_holdout=True, training_sample_count=len(training),
                holdout_sample_count=len(holdout))
    return payload


def _factor_payload(computed, decision, prepared, factor_count, cfg, dimension):
    from bayesfilter.inference import sequential_map_covariance as public

    factor_cfg = public.FactorCorrelationGeometryConfig(factor_count=factor_count,
        max_condition_number=cfg.max_condition_number,
        holdout_score_relative_rmse=cfg.structured_holdout_score_relative_rmse)
    fitted = public._factor_result_from_native(computed, factor_cfg, dimension,
        prepared['active_training_rows'], prepared['fresh_holdout_count'], decision=decision)
    winner = prepared['best_index'] >= 0
    metadata = {**prepared, 'best_exact_value': prepared['best_value'] if winner else None,
        'best_exact_position': prepared['best_position'] if winner else None,
        'best_exact_score': prepared['best_score'] if winner else None,
        'best_exact_source': 'structured_fit_cloud' if winner else None}
    return public._factor_fit_payload(fitted, metadata)


def refinement_payload(result, cfg, dimension, radius, index):
    from bayesfilter.inference.sequential_map_covariance import _proposal_gate_payload

    structured = cfg.refinement_geometry_policy == 'factor_correlation'
    first = (_factor_payload(result['first_fit'], result['first_fit']['decision'],
        result['preparation'], 1, cfg, dimension) if structured else
        terminal_payload(result['first_fit'], replace(cfg, terminal_sample_count=cfg.regression_sample_count)))
    action = result['action']
    if action == 0:
        return {'attempt': index, 'action': 'fit_cloud_recentered', 'center_value': result['center_value'],
            'fit': first, 'radius_action': 'retain', 'radius_after': radius,
            'proposal_score_gate': {'policy': cfg.proposal_score_acceptance_policy,
                'active': cfg.require_proposal_score_reduction, 'passed': not cfg.require_proposal_score_reduction,
                'proposal_evaluated': False}}
    base = {'attempt': index, 'radius_before': radius, 'recentered': result['search_recentered'],
        'center_value': result['selected_value'], 'fit': first}
    if action == 1:
        return {**base, 'action': 'fit_rejected_contract'}
    proposals = result['attempts']
    selected = (_factor_payload(proposals['second_fit']['computed'], proposals['second_fit']['decision'],
        result['preparation'], 2, cfg, dimension) if proposals['attempted'] == 2 else first)
    last, rows = proposals['last'], []
    for i in range(proposals['attempted']):
        fitted = first if i == 0 else selected
        if not proposals['evaluated'][i]:
            rows.append({'factor_count': i + 1, 'fit_status': fitted['status'], 'proposal_evaluated': False})
            continue
        row = _row(proposals['histories'], i)
        rows.append({'factor_count': i + 1 if structured else None, 'fit_status': fitted['status'],
            'proposal_evaluated': True, 'actual_improvement': row['actual'],
            'predicted_improvement': row['predicted'], 'rho': row['rho'],
            'score_norm_before': row['old_norm'], 'score_norm_after': row['new_norm'],
            'score_reduction_passed': row['score_passed'],
            'proposal_score_gate': _proposal_gate_payload(row, cfg), 'accepted': row['accepted']})
    gate = (_proposal_gate_payload(last, cfg) if proposals['last_evaluated'] else
        {'policy': cfg.proposal_score_acceptance_policy, 'active': cfg.require_proposal_score_reduction,
            'passed': not cfg.require_proposal_score_reduction})
    return {**base, 'fit': selected, 'action': 'proposal_accepted' if action == 2 else 'proposal_rejected',
        'actual_improvement': last['actual'], 'predicted_improvement': last['predicted'],
        'rho': last['rho'], 'score_norm_before': last['old_norm'], 'score_norm_after': last['new_norm'],
        'proposal_score_gate': gate, 'boundary_active': last['boundary'],
        'radius_action': ('contract', 'expand', 'retain')[proposals['radius_action']],
        'radius_after': result['radius_after'], 'proposal_attempts': rows,
        'exact_incumbent_promoted_without_model_acceptance': proposals['promoted_without_acceptance']}


def _locator_records(located, cfg, start_count, emit):
    if cfg.locator_policy == 'center_first':
        emit('locator_skipped_center_first', locator_policy=cfg.locator_policy, start_count=start_count)
        return [{'finite': True, 'coordinate_system': 'reviewed_exact_center',
            'locator_policy': 'center_first', 'locator_skipped': True, 'skip_reason': 'exact_center_admission'}]
    rows = []
    batched = 'objective_calls' in located
    if 'trace' in located:
        for call, values in enumerate(located['trace'][:located['trace_count']], start=1):
            logp, scaled, transformed, positions = zip(
                *((row[0], row[1], row[2], row[3:]) for row in values), strict=True)
            emit('locator_objective_completed', locator_objective_calls=call, log_posterior=logp,
                max_abs_scaled_score=scaled, max_abs_transformed_gradient=transformed,
                standardized_position=positions, delivery_mode='buffered_after_compiled_locator')
    for index in range(start_count):
        if batched:
            diagnostic = {'converged': located['converged'][index], 'failed': located['failed'][index],
                'iterations': located['iterations'], 'objective_calls': located['objective_calls'],
                'conservative_row_evaluations': located['objective_calls'] * start_count,
                'native_batched_locator': True}
        else:
            converged, failed, iterations, evaluations = located['optimizer_diagnostics'][index]
            diagnostic = {'converged': bool(converged), 'failed': bool(failed),
                'iterations': iterations, 'objective_evaluations': evaluations}
        rows.append({'finite': located['endpoint_finite'][index], **diagnostic,
            'coordinate_system': 'start_centered_prior_standardized_smooth_box',
            'standardized_box_radius': cfg.locator_standardized_box_radius,
            'gradient_tolerance': cfg.locator_gradient_tolerance,
            'stopping_condition': cfg.locator_stopping_condition,
            'endpoint_standardized_norm': located['endpoint_standardized_norm'][index]})
    if batched:
        emit('locator_completed', locator_objective_calls=located['objective_calls'],
            iterations=located['iterations'], converged=located['converged'], failed=located['failed'],
            stopping_condition=cfg.locator_stopping_condition)
    return rows


def sequential_result(computed, cfg, start_count, dimension, progress_callback=None, *, emit_started=True):
    from bayesfilter.inference import sequential_map_covariance as public

    def emit(stage, **payload):
        public._emit_progress(progress_callback, stage, **payload)

    result = tf.nest.map_structure(lambda value: value.numpy().tolist(), computed)
    if emit_started:
        emit('initializer_started', start_count=start_count, dimension=dimension,
            locator_policy=cfg.locator_policy, locator_stopping_condition=cfg.locator_stopping_condition)
    located = result['locator']
    rows = _locator_records(located, cfg, start_count, emit)
    selected = located['selected']
    if result['status'] == 2:
        return public._rejected('no_finite_locator_candidate', result['locator_evaluations'], rows, cfg)
    if result['status'] == 3:
        return public._rejected('maximum_exact_evaluations_after_bounded_locator',
            result['locator_evaluations'], rows, cfg, map_candidate=selected['position'])
    if result['status'] != 0:
        raise RuntimeError('Incomplete sequential controller observations')
    observations = result['observations']
    emit('candidate_selected', finite_candidate_count=selected['finite_count'],
        selected_log_posterior=selected['value'], selected_max_abs_scaled_score=observations['selected_max_score'])
    extra = ({'refinement_movement_initial': {'position_z': observations['initial_position_z'],
        'value': selected['value'], 'score_norm': observations['initial_score_norm']}}
        if cfg.record_refinement_movement_diagnostics else {})
    lifecycle, history = result['lifecycle'], []
    for index in range(lifecycle['attempt_count']):
        row = _row(lifecycle['history'], index)
        if row['terminal_called']:
            fitted = terminal_payload(row['terminal'], cfg)
            emit('terminal_fit_started', attempt=index, exact_evaluations=row['evaluations_before'],
                max_abs_scaled_score=observations['max_score_before'][index], radius=row['radius_before'])
            emit('terminal_fit_completed', attempt=index, exact_evaluations=observations['evaluations_after'][index],
                fit_status=fitted['status'], projection_relative_frobenius=fitted.get('projection_relative_frobenius'))
        event = row['event']
        if event == 1:
            history.append({'attempt': index, 'action': 'terminal_fit_cloud_recentered',
                'center_value': row['terminal']['best_value'], 'fit': fitted})
        elif event == 2:
            history.append({'attempt': index, 'action': 'terminal_fit_rejected', **fitted})
        elif event == 3:
            refined = row['refine']
            record = refinement_payload(refined, cfg, dimension, row['radius_before'], index)
            if cfg.record_refinement_movement_diagnostics and refined['action'] != 0:
                movement = _row(result['movement'], index)
                if not movement.pop('proposal_evaluated'):
                    movement.update(evaluated_proposal_position_z=None,
                        evaluated_proposal_value=None, evaluated_proposal_score_norm=None)
                record['refinement_movement'] = movement
            history.append(record)
            if refined['action'] >= 2:
                emit('refinement_attempt_completed', attempt=index,
                    exact_evaluations=observations['evaluations_after'][index], action=record['action'],
                    max_abs_scaled_score=observations['max_score_after'][index], radius=refined['radius_after'])
    extra['history'] = history
    status = lifecycle['status']
    if status != 0:
        labels = ('usable', 'maximum_exact_evaluations_before_terminal_fit', 'maximum_exact_evaluations',
            'terminal_projection_exceeds_cap', 'sequential_refinement_without_terminal_geometry')
        if status == 3:
            extra['terminal_fit'] = terminal_payload(lifecycle['terminal'], cfg)
        elif status == 4:
            extra['terminal_max_abs_scaled_score'] = lifecycle['max_score']
        return public._rejected(labels[status], lifecycle['evaluations'], rows, cfg,
            map_candidate=lifecycle['center'], extra=extra)
    # This only materializes and checks already-computed status flags. It does
    # not reconstruct a mass matrix or recalculate numerical admission.
    _precision_report(computed['mass_diagnostics'], computed['mass_flags'], 0.,
        cfg.eigenvalue_floor, cfg.max_condition_number)
    terminal = terminal_payload(lifecycle['terminal'], cfg)
    emit('initializer_completed', accepted=True, status='usable', exact_evaluations=lifecycle['evaluations'],
        terminal_max_abs_scaled_score=lifecycle['max_score'])
    return public.SequentialMapCovarianceResult(accepted=True, status='usable',
        map_candidate=lifecycle['center'], precision=computed['precision'], covariance=computed['covariance'],
        diagnostics={'exact_evaluations': lifecycle['evaluations'],
            'terminal_max_abs_scaled_score': lifecycle['max_score'], 'terminal_fit_fresh': True,
            'terminal_fit_attempts': lifecycle['terminal_attempts'], 'search_seed': list(cfg.seed),
            'terminal_seed': terminal['seed'], 'precision_coordinate_system': 'theta',
            'regression_coordinate_system': 'z', 'locator': rows, 'terminal_fit': terminal,
            'proposal_score_acceptance_policy': cfg.proposal_score_acceptance_policy,
            'proposal_score_gate_active': cfg.require_proposal_score_reduction, **extra})
