"""Host formatting of completed ordered-block records and buffered events."""

import tensorflow as tf

from bayesfilter.inference.block_coordinate_center import (
    BlockCoordinateCenterResult,
    _emit_progress,
    _without_internal_geometry,
)
from bayesfilter.inference.sequential_controller_report import sequential_result

STATUS = ('sweep_incomplete', 'sweep_completed_with_resolvable_progress',
    'sweep_completed_without_resolvable_progress', 'sequential_row_accounting_invalid',
    'invalid_sequential_handoff', 'invalid_sequential_candidate_shape',
    'transaction_objective_decrease', 'material_block_score_reversal',
    'impossible_block_coordinate_cycle')


def block_result(computed, initial, blocks, config, progress_callback=None):
    # One-way result materialization: no value feeds another numerical call.
    host = tf.nest.map_structure(lambda value: value.numpy().tolist(), computed)
    state, history = host['state'], host['history']
    if state['error'] == 1:
        raise ValueError('full target replay must be finite')
    _emit_progress(progress_callback, 'sweep_started', block_count=len(blocks))
    records = []
    for index, block in enumerate(blocks):
        if not history['executed'][index]:
            break
        _emit_progress(progress_callback, 'block_started', block_index=index, block_name=block.name)

        def progress(event, index=index, block=block):
            _emit_progress(progress_callback, 'block_locator_progress', block_index=index,
                block_name=block.name, locator_stage=event.get('stage', 'unknown'))

        located = sequential_result(computed['sequential'][index], block.sequential_config,
            1, block.stop - block.start, progress if progress_callback is not None else None)
        if history['evaluations'][index] < 0:
            raise ValueError('exact_evaluations must be nonnegative')
        if history['has_record'][index]:
            if state['error'] == 2 and not history['committed'][index]:
                raise ValueError('full target replay must be finite')
            reversals = tuple({'block_name': previous.name,
                'post_update_max_abs_scaled_score': history['prior_maxima'][index][j],
                'current_max_abs_scaled_score': history['maxima'][index][j],
                'absolute_resolution_floor': history['floors'][index][j],
                'reversal_ratio_threshold': config.reversal_ratio,
                'material_reversal': history['materials'][index][j]}
                for j, previous in enumerate(blocks[:index])) if history['committed'][index] else ()
            records.append({'name': block.name, 'start': block.start, 'stop': block.stop,
                'handoff_status': located.status, 'exact_evaluations': history['evaluations'][index],
                'committed': history['committed'][index], 'center_before': history['center_before'][index],
                'center_after': history['center_after'][index], 'objective_before': history['value_before'][index],
                'objective_after': history['value_after'][index], 'score_before': history['score_before'][index],
                'score_after': history['score_after'][index], 'displacement_l2': history['displacement'][index],
                'locator_history': _without_internal_geometry(located.diagnostics.get('history', ())),
                'reversal_diagnostics': reversals})
        if history['committed'][index]:
            _emit_progress(progress_callback, 'block_completed', block_index=index, block_name=block.name,
                handoff_status=located.status, exact_evaluations=history['evaluations'][index])
    if state['error']:
        raise RuntimeError('Unreported ordered-block execution failure')
    result = BlockCoordinateCenterResult(completed=host['completed'], status=STATUS[host['status']],
        initial_center=tf.stop_gradient(initial), final_center=computed['state']['center'],
        initial_score=computed['initial_score'], final_score=computed['state']['score'],
        initial_objective=host['initial_value'], final_objective=state['value'],
        initial_score_l2=host['initial_l2'], final_score_l2=host['final_l2'],
        initial_score_max_abs=host['initial_max'], final_score_max_abs=host['final_max'],
        completed_block_count=len(records), accepted_block_count=state['accepted'],
        transaction_rejection_count=state['rejections'], sequential_exact_evaluations=state['sequential_rows'],
        physical_target_rows=state['physical_rows'], maximum_physical_target_rows=config.max_physical_target_rows,
        material_reversal_detected=state['material'], repeat_cycle_detected=state['repeat'],
        two_step_return_cycle_detected=state['two_step'], scheduled_block_maxima_no_worse=host['no_worse'],
        stop_on_material_reversal=config.stop_on_material_reversal,
        require_scheduled_block_maxima_no_worse=config.require_scheduled_block_maxima_no_worse,
        private_block_records=tuple(records), numerical_summary=host['summary'])
    _emit_progress(progress_callback, 'sweep_completed', completed=result.completed,
        status=result.status, physical_target_rows=result.physical_target_rows)
    return result
