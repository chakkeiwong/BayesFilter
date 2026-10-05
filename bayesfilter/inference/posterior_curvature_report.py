"""Host formatting of completed fixed-center curvature records only."""

import json

from bayesfilter.inference.posterior_curvature_tf import COUNT_NAMES, STATUS


def posterior_curvature_result(record, center, factor, config):
    """Preserve the public numerical payload, including partial diagnostics."""
    from bayesfilter.inference.posterior_curvature_refinement import (
        PosteriorCurvatureRefinementResult,
    )

    cfg = config
    dimension = int(center.shape[0])
    rows = max(32, 4 * dimension) if cfg.rows_per_partition is None else cfg.rows_per_partition
    padded = ((rows + cfg.batch_size - 1) // cfg.batch_size) * cfg.batch_size
    fit_partitions = 2 * cfg.replicate_count
    roles = (['center'] + [f'training_{index}' for index in range(cfg.replicate_count)]
             + [f'selection_{index}' for index in range(cfg.replicate_count)]
             + ['audit', 'proposal', 'fit', 'replicate_consensus', 'factorization'])
    host = {name: value.numpy().tolist() for name, value in record.items()
            if name not in ('precision', 'covariance', 'refined_factor')}
    status = STATUS[host['status']]
    accepted = host['status'] == 1
    partitions = []
    for index in range(host['partition_count']):
        counts = host['partition_counts'][index]
        partitions.append({'role': roles[index], 'seed': None if index == 0 else host['partition_seeds'][index],
            **{name: counts[slot] for slot, name in enumerate(COUNT_NAMES) if slot < 3 or counts[slot] > 0}})
    details = {}
    if host['fit_count'] > 0 or host['role'] == fit_partitions + 3:
        details['replicates'] = [{
            'precision_z': host['fit_precisions'][index],
            'design_condition': host['fit_metrics'][index][0],
            'precision_condition': host['fit_metrics'][index][1],
            'selection_relative_rmse': host['fit_metrics'][index][2],
            'raw_spd': host['fit_spd'][index], 'design_rank': host['fit_rank'][index],
            'accepted': host['fit_accepted'][index],
        } for index in range(host['fit_count'])]
    if host['spread_done']:
        details['replicate_generalized_eigenvalue_spread'] = host['spread']
    if host['audit_done']:
        details['audit_relative_rmse'] = host['audit_rmse']
    if host['reconstruction_done']:
        details.update(factor_reconstruction_abs=host['factor_abs'], factor_reconstruction_rel=host['factor_rel'])
    if host['norm_done']:
        details['center_score_refined_l2'] = host['center_norm']
    if host['proposal_done']:
        details['proposal_relative_rmse'] = host['proposal_rmse']
    diagnostics = dict(zip(COUNT_NAMES, host['counts'], strict=True))
    diagnostics.update({
        'logical_rows_per_partition': rows, 'physical_rows_per_partition': padded,
        'planned_physical_rows': cfg.batch_size + (fit_partitions + 2) * padded,
        'batch_size': cfg.batch_size, 'max_physical_rows': cfg.max_physical_rows, 'seed': cfg.seed,
        'partitions': partitions, 'failure_partition': None if accepted else roles[host['role']],
        'center_held_fixed': True, 'fit_design': cfg.fit_design,
        'pilot_coordinate_half_width': cfg.coordinate_half_width,
        'proposal_distribution': 'standard_normal_in_refined_factor_coordinates',
        'lineage': json.loads(json.dumps(cfg.lineage, allow_nan=False)), **details,
    })
    return PosteriorCurvatureRefinementResult(accepted, status, center, factor,
        record['refined_factor'] if accepted else None,
        record['covariance'] if accepted else None,
        record['precision'] if accepted else None, diagnostics)
