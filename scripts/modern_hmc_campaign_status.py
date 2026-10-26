"""Campaign reporting only; numerical checkpoints retain validation authority."""
from __future__ import annotations


def completion_status(value):
    """Never turn numerical invalidity or unknown completion into a budget stop."""
    mapping = {'complete': 'completed', 'partial_budget': 'resource_stopped',
               'shared_invalidity': 'shared_invalidity',
               'paused_infrastructure': 'paused_infrastructure'}
    if value not in mapping:
        raise ValueError('unknown tuning completion status: '+repr(value))
    return mapping[value]


def tuning_outcome(result, *, checkpoint=None, wall_stopped=False):
    if result:
        execution = result.get('execution', {})
        if 'completion_status' not in execution:
            if result.get('status') == 'error':
                return dict(status='error', checkpoint_validation='not_performed')
            raise ValueError('tuning result lacks controller completion')
        status = completion_status(execution['completion_status'])
        # A legacy wrapper can mislabel a failure as resource_stopped. Preserve
        # the controller's more informative failure, but never forgive an error
        # after the controller returned (e.g. source or export validation).
        if result.get('status') == 'error':
            status = 'error'
        elif result.get('status') != status and not (
                result.get('status') == 'resource_stopped'
                and status in ('shared_invalidity', 'paused_infrastructure')):
            raise ValueError('wrapper and controller completion contradict')
        return dict(status=status, completion_status=execution['completion_status'],
                    wrapper_status=result.get('status'), checkpoint_validation='worker_owned')
    if not wall_stopped:
        raise ValueError('missing tuning result without a recorded wall stop')
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('result'), dict):
        raise ValueError('wall stop lacks a readable numerical checkpoint')
    state = checkpoint['result'].get('completion_status')
    status = completion_status(state)
    if status == 'completed':
        status = 'finalization_pending'
    return dict(status=status, completion_status=state, result_missing=True,
                checkpoint_validation='required_before_resume')
