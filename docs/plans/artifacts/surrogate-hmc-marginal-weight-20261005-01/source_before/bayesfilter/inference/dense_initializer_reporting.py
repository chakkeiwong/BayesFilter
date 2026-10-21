"""Host reporting for the completed native dense-initializer calculation.

Numerical control and values come from DenseInitializerProgram. Python traverses
completed history only to construct records and name archive members; its values
never feed back into a numerical kernel or change a computed acceptance decision.
"""

import math

import tensorflow as tf

from bayesfilter.inference import fixed_center_curvature as fixed
from bayesfilter.inference.batched_local_center import BatchedLocalCenterResult


def _completed_json(value):
    if tf.is_tensor(value):
        return _completed_json(value.numpy().tolist())
    if isinstance(value, dict):
        return {str(name): _completed_json(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_completed_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _raise_partition_error(validation, replicates):
    code = int(validation['error_code'])
    if code == 0:
        return
    messages = ('', 'center must be a nonempty finite vector',
        'center_score_z must be a nonempty finite vector',
        'training offsets/scores must be finite', 'selection offsets/scores must be finite',
        'audit offsets/scores must be finite')
    if code == 6:
        def name(index):
            if index < replicates:
                return f'training[{index}]'
            if index < 2 * replicates:
                return f'selection[{index - replicates}]'
            return 'audit'
        left, right = validation['overlap_pair'].numpy().tolist()
        raise ValueError(f'partition offsets contain copied rows: {name(left)} and {name(right)} overlap')
    raise ValueError(messages[code])


def format_dense_initializer_result(raw, configuration, thresholds, rows, *, trace_count, planned_rows):
    """Decode completed decisions and return the payload plus archive tensors.

    No target, fit, numerical threshold or acceptance calculation runs here.
    Original validation exceptions precede any file write.
    """
    dimension = raw['center'].shape[0]
    attempts, archives = [], {}
    for index in range(int(raw['attempt_count'])):
        location = tf.nest.map_structure(lambda tensor, index=index: tensor[index], raw['location_history'])
        item = tf.nest.map_structure(lambda tensor, index=index: tensor[index], raw['attempt_history'])
        record = {'attempt': index, 'locator': BatchedLocalCenterResult(**location,
            trace_count=trace_count).payload()}
        attempts.append(record)
        if bool(location['accepted']):
            record['scaled_center_score_l2'] = float(raw['scaled_score_norm_history'][index])
        if bool(raw['cloud_ran_history'][index]):
            cloud = item['cloud']
            for partition_index in range(int(cloud['partition_count'])):
                n = rows[partition_index]
                for name, key in (('positions', 'positions'), ('values', 'values'),
                        ('scores', 'scores'), ('valid', 'row_validity')):
                    archives[f'attempt_{index}_partition_{partition_index}_{name}'] = cloud[key][partition_index, :n]
            if bool(cloud['valid']):
                record.update(partition_rows=rows,
                    objective_improvement=float(item['objective_improvement']))
                if int(item['status_code']) == 2:
                    record['curvature_center_moved'] = True
                else:
                    _raise_partition_error(item['fit']['validation'], configuration.replicate_count)
                    result = fixed._fixed_center_result_from_native(item['fit']['fit'], location['center'],
                        item['scaled_center_score'], dimension=dimension, replicates=configuration.replicate_count,
                        training_rows=rows[0], selection_rows=rows[configuration.replicate_count], audit_rows=rows[-1],
                        thresholds=thresholds, factor_max=configuration.factor_max,
                        weights=configuration.shrinkage_weights, structured_target_family=configuration.structured_target_family,
                        lineage={'role': 'fresh_dense_initializer', 'seed': configuration.seed,
                            'attempt': index, 'truth_blind_prior_start': True},
                        jit_compile=bool(raw['jit_compile']))
                    record['curvature'] = result.payload()
    status_code = int(raw['status_code'])
    statuses = {1: f"locator_{attempts[-1]['locator']['status']}",
        2: 'dense_center_score_above_cap', 3: 'dense_curvature_cloud_invalid',
        4: 'dense_curvature_not_centered_within_attempt_budget',
        8: 'dense_initializer_scale_invalid', 9: 'usable_dense_local_initializer'}
    status = (f"dense_{attempts[-1]['curvature']['status']}" if status_code == 7 else statuses[status_code])
    return _completed_json({'passed': bool(raw['usable']), 'method': 'dense_local', 'attempts': attempts,
        'initial_output_shift': raw['initial_output_shift'] if bool(raw['usable']) else None,
        'initial_output_scale_log': raw['initial_output_scale_log'] if bool(raw['usable']) else None,
        'planned_exact_row_ceiling': planned_rows,
        'nonclaims': ['local guide initializer only', 'not posterior covariance', 'not HMC or retained-sampling qualification'],
        'status': status, 'exact_evaluation_count': int(raw['exact_rows']),
        'target_evaluation_accounting': 'logical value/score rows; status callbacks may repeat physical work'}), archives
