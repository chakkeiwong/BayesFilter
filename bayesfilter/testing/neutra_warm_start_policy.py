"""Deterministic decisions shared by warm-start calibration and supervisors."""
from __future__ import annotations

import math


class MALACalibrationFailure(ValueError):
    """A bounded kernel search found no eligible candidate; not a harness fault."""


def select_mala_candidate(candidates, minimum_acceptance=0.5):
    viable = [row for row in candidates
              if row['invalid_proposals'] == 0
              and math.isfinite(row['acceptance'])
              and row['acceptance'] >= minimum_acceptance
              and math.isfinite(row['squared_displacement'])
              and row['squared_displacement'] > 0]
    if not viable:
        raise MALACalibrationFailure('MALA calibration failed: no finite, moving candidate passed the screen')
    return max(viable, key=lambda row: row['squared_displacement'])


def preserve_deferred(previous, additions, resolved=()):
    resolved = set(resolved)
    rows = {}
    for row in [*previous, *additions]:
        key = (row['target'], row['arm'], row['seed'], row['phase'])
        job = f"repair-{row['target']}-{row['arm']}-s{row['seed']}"
        if job not in resolved:
            rows[key] = row
    return list(rows.values())


def fit_repair_decision(history, *, budget_available=True):
    """Classify measured progress; never infer it from a raw loss decrease."""
    if not history:
        return 'missing_evidence'
    row=history[-1]
    metrics=row.get('assessment',{})
    if row.get('numerical_veto') or not metrics.get('finite',False):
        return 'numerical_failure'
    if metrics.get('passed',False):
        return 'eligible'
    if not budget_available:
        return 'budget_limited_unfinished'
    progress=row.get('paired_progress',{})
    mean=progress.get('mean_log_density_gain',0.)
    se=progress.get('standard_error',math.inf)
    if metrics.get('nonlinear_learning_passed',False) and math.isfinite(mean) and math.isfinite(se) and mean>3*se:
        return 'continue_checkpoint'
    if not metrics.get('nonlinear_learning_passed',False):
        return 'nonlinear_plateau_repair'
    return 'shape_failure_without_supported_progress'


def teacher_repair_settings(level):
    """Isolate mutation work before increasing the number of particles."""
    ladder=((1024,4),(1024,16),(1024,64),(4096,64),(8192,64))
    if not 0<=level<len(ladder):raise ValueError('teacher repair level outside declared ladder')
    particles,steps=ladder[level]
    return {'particles':particles,'mutation_steps':steps,
            'repair_axis':'baseline' if level==0 else 'mutation_steps' if level<3 else 'particles'}
