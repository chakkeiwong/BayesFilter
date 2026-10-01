"""Scope-bound SQMC control selection and untouched diagnostic evaluation.

Selection minimizes calibration mean absolute score error. Validation checks
every replication and does not select another candidate. No Fisher/HMC claim,
statistical superiority, or default promotion is issued by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import hashlib
from pathlib import Path
import statistics

import tensorflow as tf

from bayesfilter.highdim.sqmc_campaign_tf import evaluate_diagnostic, numerical_settings, route_settings
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunk_size


def validate_partitions(calibration, validation, claim):
    parts = [tuple(p) for p in (calibration, validation, claim)]
    if any(not p or len(set(p)) != len(p) for p in parts):
        raise ValueError('each seed partition must be nonempty and contain distinct seeds')
    if any(type(s) is not int or s < 0 for p in parts for s in p):
        raise ValueError('seeds must be nonnegative integers')
    if any(set(parts[i]) & set(parts[j]) for i,j in ((0,1),(0,2),(1,2))):
        raise ValueError('calibration, validation and claim data/randomness must be disjoint')
    return parts


def complete_valid_results(rows, seeds, parameter_count=None):
    if len(rows) != len(seeds) or {r.get('seed') for r in rows} != set(seeds):
        return False
    for row in rows:
        if not row.get('valid', False):
            return False
        for key in ('value','oracle_value','score_l2_error'):
            v = row.get(key)
            if not isinstance(v, (int,float)) or not math.isfinite(v):
                return False
        for key in ('score','oracle_score'):
            values = row.get(key)
            if not isinstance(values, list) or not values or (parameter_count is not None and len(values) != parameter_count):
                return False
            if any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in values):
                return False
    return True


def execution_scope(spec, route, theta, horizon, particle_count, jit_compile):
    # Called only at the host boundary; GPU callers configure growth first.
    gpu = bool(tf.config.list_physical_devices('GPU'))
    return dict(schema='bayesfilter.sqmc_scope.v2', target_id=spec.target_id,
                theta=theta.numpy().tolist(), state_dimension=spec.dimension,
                parameter_count=spec.parameter_count, horizon=horizon, particle_count=particle_count,
                dtype=theta.dtype.name, device_class='gpu' if gpu else 'cpu',
                tf32=bool(tf.config.experimental.tensor_float_32_execution_enabled()), jit_compile=bool(jit_compile),
                route=route, **route_settings(route),
                reset_contract='contract_e_chol_v1', analytical_score='recursive_total_v2',
                chunk_policy='dpf_transport_exact_divisor_cap3000_v1', chunk_size=select_transport_chunk_size(particle_count),
                prepared_data_regime='shared_lgssm_predict_first_stateless_independent_datasets_v2',
                control_family='all_settings_from_sqmc_campaign_tf.numerical_settings',
                source_version='sqmc_repairs_20260925_v2',
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(Path(__file__).parent.glob('*.py'))})


@dataclass(frozen=True)
class CampaignTuningArtifact:
    scope_json: str
    controls_json: str
    calibration_seeds: tuple
    validation_seeds: tuple
    claim_seeds: tuple
    calibration_mean_l2: float
    validation_mean_l2: float

    def public_dict(self):
        return dict(schema='bayesfilter.sqmc_campaign_tuning.v2', scope=json.loads(self.scope_json),
                    controls=json.loads(self.controls_json), calibration_seeds=self.calibration_seeds,
                    validation_seeds=self.validation_seeds, claim_seeds=self.claim_seeds,
                    calibration_mean_l2=self.calibration_mean_l2, validation_mean_l2=self.validation_mean_l2,
                    selection='minimum_calibration_mean_absolute_score_error',
                    validation='all_required_results_present_and_finite',
                    scientific_admission=False, hmc_readiness=False)


_issued = {}


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def tune_campaign(spec, route, candidates, theta, horizon, particle_count,
                  calibration_seeds, validation_seeds, claim_seeds, *, jit_compile=True):
    calibration, validation, claim = validate_partitions(calibration_seeds, validation_seeds, claim_seeds)
    scope = execution_scope(spec, route, theta, horizon, particle_count, jit_compile)
    grid = []
    for controls in candidates:
        settings = numerical_settings(controls)
        rows = [evaluate_diagnostic(spec, route, settings, spec.simulate(theta, horizon, seed, jit_compile=jit_compile), theta,
                                    seed, particle_count, jit_compile=jit_compile) for seed in calibration]
        valid = complete_valid_results(rows, calibration, spec.parameter_count)
        grid.append(dict(controls=settings, seed_results=rows, valid=valid,
                         mean_l2=statistics.fmean(r['score_l2_error'] for r in rows) if valid else None))
    viable = [g for g in grid if g['valid']]
    if not viable:
        return None, dict(scope=scope, grid=grid, decision='no_complete_valid_candidate')
    selected = min(viable, key=lambda g: g['mean_l2'])
    rows = [evaluate_diagnostic(spec, route, selected['controls'], spec.simulate(theta, horizon, seed, jit_compile=jit_compile), theta,
                               seed, particle_count, jit_compile=jit_compile) for seed in validation]
    report = dict(scope=scope, grid=grid, validation=rows, selected_controls=selected['controls'],
                  ranking='calibration_selection_only; no statistically supported superiority',
                  heuristic_dominance='not_assessed; diagnostic tuning does not promote a method')
    if not complete_valid_results(rows, validation, spec.parameter_count):
        return None, dict(report, decision='validation_failed; fresh_partitions_required_for_retuning')
    artifact = CampaignTuningArtifact(_encode(scope), _encode(selected['controls']), calibration, validation, claim,
                                     selected['mean_l2'], statistics.fmean(r['score_l2_error'] for r in rows))
    _issued[id(artifact)] = artifact
    return artifact, dict(report, decision='controls_selected_and_frozen; scientific_accuracy_not_admitted',
                          tuning_artifact=artifact.public_dict())


def evaluate_untouched(spec, route, artifact, theta, horizon, particle_count, seed, *, jit_compile=True):
    if not isinstance(artifact, CampaignTuningArtifact) or _issued.get(id(artifact)) is not artifact:
        raise TypeError('execution requires the repository-issued artifact from completed tuning/validation')
    scope = execution_scope(spec, route, theta, horizon, particle_count, jit_compile)
    if _encode(scope) != artifact.scope_json:
        raise ValueError('tuning scope mismatch')
    if seed not in artifact.claim_seeds or seed in artifact.calibration_seeds + artifact.validation_seeds:
        raise ValueError('seed is not in the reserved untouched partition')
    result = evaluate_diagnostic(spec, route, json.loads(artifact.controls_json), spec.simulate(theta, horizon, seed, jit_compile=jit_compile),
                                 theta, seed, particle_count, jit_compile=jit_compile)
    return dict(result, tuning_status='exact_scope_frozen_controls',
                evidence_role='untouched_diagnostic; no_performance_or_hmc_admission', scope=scope)
