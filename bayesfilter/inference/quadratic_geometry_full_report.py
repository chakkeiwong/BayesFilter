"""Host formatting after all decisions in a complete native geometry attempt."""

from bayesfilter.inference._exact_incumbent import ExactCandidate
from bayesfilter.inference.quadratic_geometry import _base_diagnostics, _rejected_result
from bayesfilter.inference.quadratic_geometry_fit_report import geometry_fit_result
from bayesfilter.inference.quadratic_geometry_fit_tf import SOURCE_ROLES
from bayesfilter.inference.quadratic_geometry_full_tf import STAGES, geometry_extents
from bayesfilter.inference.quadratic_geometry_pilot_report import geometry_pilot_report


def geometry_result(raw, center, scale, config, *, batched=False):
    """No result of formatting feeds back into any target, selection or fit."""
    dimension = center.shape[0]
    rank, required, samples, directions = geometry_extents(dimension, config)
    stage = int(raw['stage'])
    diagnostics = _base_diagnostics(cfg=config, dim=dimension, rank=rank,
        regression_parameter_count=dimension + rank + 2, required_finite_samples=required,
        sample_count=samples, center_value=float(raw['center_value']),
        center_score_norm=float(raw['center_score_norm']) if stage else None)
    if stage:
        pilot = raw['pilot']
        count = 2 * int(pilot['direction_count'])
        compact = {**pilot, 'positions': pilot['positions'][:count], 'values': pilot['values'][:count],
            'scores': pilot['scores'][:count], 'valid': pilot['valid'][:count]}
        _, pilot_report, _ = geometry_pilot_report(compact, rank=rank,
            requested_direction_count=directions, batched=batched, include_candidates=False)
        if rank and not batched:
            pilot_report['evaluation_route'] = 'tensorflow_scalar_row_loop'
        diagnostics['pilot'] = pilot_report
        if stage > 1:
            finite = int(raw['design']['finite_count'])
            diagnostics.update(finite_sample_count=finite, nonfinite_sample_count=samples - finite,
                design_evaluation_route='batched_value_and_score' if batched else 'tensorflow_scalar_row_loop')
    if stage == 3:
        return geometry_fit_result(raw['fit_result'], config, center, scale, diagnostics,
                                   holdout_rows=int(raw['partition']['holdout_count']))
    candidate = raw['incumbent']
    incumbent = (ExactCandidate(candidate['position'], float(candidate['value']), candidate['score'],
        int(candidate['index']), SOURCE_ROLES[int(candidate['source'])]) if bool(candidate['present']) else None)
    return _rejected_result(status=STAGES[stage], center=center, scale=scale, dim=dimension,
        rank=rank, diagnostics=diagnostics, incumbent=incumbent, exact_evaluation_count=int(raw['evaluation_count']))
