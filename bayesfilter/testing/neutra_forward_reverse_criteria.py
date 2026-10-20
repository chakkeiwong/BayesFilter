"""Host-side diagnostic decisions for the approximate warm-start study.

The October 6 owner revision separates intermediate coverage from final fit
accuracy. No numerical kernels or final distribution thresholds change.
"""
import copy
import json
import math
from pathlib import Path

PAIR_CRITERION = 'bayesfilter_neutra_forward_warm_start_then_final_v2'
CRITERION_PLAN = 'docs/plans/bayesfilter-neutra-forward-warm-start-criterion-repair-2026-10-06.md'
MINIMUM_RELATIVE_MODE_MASS = 0.5  # Explicit heuristic: at most twofold underrepresentation.


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def forward_warm_start(endpoint, probe, component_count):
    """Coarse coverage/validity screen; feature z is explanatory at this stage."""
    if type(component_count) is not int or component_count < 0:
        raise ValueError('invalid evaluator component count')
    h = endpoint['heldout']
    reasons = []
    if endpoint.get('checkpoint_reloaded') is not True:
        reasons.append('checkpoint_not_reloaded')
    if h.get('finite') is not True:
        reasons.append('nonfinite_forward_evaluation')
    if not (probe.get('complete') is True and probe.get('finite') is True and
            probe.get('rows') == probe.get('valid_rows') == 1000):
        reasons.append('incomplete_or_nonfinite_probe')
    delta = h.get('cross_entropy_delta')
    if not _finite(delta) or delta > .25:
        reasons.append('identity_density_guard')
    mass_error = h.get('maximum_responsibility_discrepancy')
    if not _finite(mass_error) or not 0 <= mass_error <= .15:
        reasons.append('absolute_mode_mass_error')
    ratios = []
    q, ref = h.get('q_summary', []), h.get('reference_summary', [])
    if len(q) < component_count or len(ref) < component_count:
        reasons.append('missing_mode_mass_evidence')
    else:
        for i in range(component_count):
            if not (_finite(q[i]) and _finite(ref[i]) and 0 <= q[i] <= 1 and 0 < ref[i] <= 1):
                reasons.append('invalid_mode_mass_evidence')
                continue
            ratios.append(q[i] / ref[i])
        if len(ratios) != component_count or any(r < MINIMUM_RELATIVE_MODE_MASS for r in ratios):
            reasons.append('underrepresented_mode')
    return dict(criterion=PAIR_CRITERION, passed=not reasons, reasons=reasons,
        component_mass_ratios=ratios, minimum_relative_mode_mass=MINIMUM_RELATIVE_MODE_MASS,
        full_accuracy_screen_passed=endpoint.get('passed') is True,
        feature_z_role='explanatory_only_for_forward_warm_start',
        feature_z_max=h.get('summary_z_max'), scope='synthetic_evaluator_coverage_not_q20_or_HMC')


def assess_pair_criteria(result, component_count):
    """Return a separate decision view; never rewrite a source worker artifact."""
    if result.get('status') != 'fit_complete':
        return result
    view = copy.deepcopy(result)
    forward = view['forward']
    probe = json.loads(Path(forward['probe_path']).read_text())
    forward['legacy_full_accuracy_passed'] = forward.get('passed') is True
    forward['warm_start'] = forward_warm_start(forward, probe, component_count)
    history = view['training']
    forward_history = [h for h in history if h['phase'] == 'forward']
    for branch in view['branches']:
        reverse_history = [h for h in history if h['phase'] == 'reverse' and
                           h['rate'] == branch['rate'] and h['rung'] <= branch['updates']]
        path = forward_history + reverse_history
        training_valid = bool(forward_history and reverse_history) and all(
            h.get('finite') is True and h['updates'] > 0 and
            0 <= h['clipped_updates'] <= h['updates'] / 2 for h in path)
        branch['legacy_both_endpoint_passed'] = branch.get('legacy_both_endpoint_passed', branch['passed'])
        branch['criterion'] = PAIR_CRITERION
        branch['training_valid'] = training_valid
        branch['passed'] = bool(forward['warm_start']['passed'] and
                                branch['endpoint']['passed'] and training_valid)
    view['criterion'] = PAIR_CRITERION
    return view
