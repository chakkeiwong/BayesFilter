#!/usr/bin/env python3
"""Scope-specific SQMC calibration, validation and untouched diagnostics.

Select on absolute score error against matched Kalman. Old Fisher/HMC
metrics and artifacts are ineligible. No tuning result certifies HMC,
statistical superiority, or production/default readiness.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

GPU_POLICY_RECORD = (dict(configure_tensorflow_gpu_memory_growth(tf, require_gpu=True))
                     if tf.config.list_physical_devices('GPU') else
                     {'mode': 'cpu_reference', 'gpu_intentionally_hidden': os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'})

from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec
from bayesfilter.highdim.sqmc_campaign_tf import evaluate_diagnostic, ROUTES
from bayesfilter.highdim.sqmc_campaign_tuning import complete_valid_results, tune_campaign, evaluate_untouched

DTYPE = tf.float64  # legacy reference callers; CLI chooses its dtype explicitly

@dataclass
class ConfigEvaluation:
    config_idx: int
    objectives: tuple
    controls: dict
    mean_cosine: float | None
    mean_l2: float
    mean_rel_norm: float | None
    valid_fraction: float

    @property
    def valid(self):
        return self.valid_fraction == 1.0 and all(math.isfinite(v) for v in self.objectives)

    def checked(self):
        return self


def _oracle_score(observations, theta):
    return LGSSMSpec('frozen_3d', 3).reference_value_and_score(theta, observations)[1]


def _reset_design(particle_count, dimension):
    from bayesfilter.highdim.sqmc_campaign_tf import reset_design
    return reset_design(particle_count, dimension, DTYPE)


def find_pareto_optimal(grid_results, tuning_seeds):
    """Compatibility name: frontier of the single declared absolute-L2 objective.

    Every requested seed must have a complete, finite value and full score.
    Undefined relative diagnostics never become a zero error or a veto.
    """
    entries=[]
    for i, grid in enumerate(grid_results):
        rows=grid['seed_results']
        if grid.get('valid_fraction') != 1.0 or not complete_valid_results(rows, tuning_seeds):
            continue
        mean=statistics.fmean(r['score_l2_error'] for r in rows)
        def descriptive(key):
            values=[r.get(key) for r in rows]
            return statistics.fmean(values) if all(v is not None and math.isfinite(v) for v in values) else None
        entries.append(ConfigEvaluation(i,(mean,),grid['controls'],descriptive('cosine_similarity'),mean,
                                        descriptive('relative_norm_error'),1.0))
    if not entries:
        return []
    minimum=min(e.mean_l2 for e in entries)
    return [e for e in entries if e.mean_l2 == minimum]


def _evaluate_controls(route, controls, observations, theta, oracle_score, seed, horizon, particle_count,
                       is_ablation=False, state_dim=3, *, jit_compile=True, family=None):
    if is_ablation:
        if route not in {'repaired_permutation','repaired_permutation_ablation'}:
            raise ValueError('cap ablation requires the permutation route')
        route='repaired_permutation_ablation'
    if family is None:
        if state_dim == 3 and theta.shape[0] == 5:
            family='frozen_3d'
        elif theta.shape[0] == 4:
            family='p44'
        else:
            family='diagonal_ar'
    spec=LGSSMSpec(family,state_dim)
    if observations.shape[0] != horizon:
        raise ValueError('observation horizon mismatch')
    _, expected=spec.reference_value_and_score(theta, observations)
    tf.debugging.assert_near(oracle_score,expected,atol=1e-8,rtol=1e-6)
    return evaluate_diagnostic(spec,route,controls,observations,theta,seed,particle_count,jit_compile=jit_compile)


def tune_route(route, horizon, particle_count, tuning_seeds, output_dir, is_ablation=False, *,
               validation_seeds, claim_seeds, spec=None, candidates=None, dtype=tf.float64, jit_compile=True):
    """New partitions are mandatory; historical frozen-data seeds are not holdouts."""
    if is_ablation:
        route='repaired_permutation_ablation'
    spec=spec or LGSSMSpec('frozen_3d',3)
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    theta=spec.default_theta(dtype)
    artifact, report=tune_campaign(spec,route,candidates or _tuning_grid(),theta,horizon,particle_count,
                                   tuning_seeds,validation_seeds,claim_seeds,jit_compile=jit_compile)
    claims=[] if artifact is None else [evaluate_untouched(spec,route,artifact,theta,horizon,particle_count,s,
                                                           jit_compile=jit_compile) for s in claim_seeds]
    report.update(untouched_results=claims, scientific_admission=False, gpu_memory_policy=GPU_POLICY_RECORD,
                  wall_seconds=time.perf_counter()-start, command=sys.argv,
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip(),
                  environment=sys.executable, plan='docs/plans/sqmc-repair-master-program-20260925.md',
                  data_version=spec.target_id, calibration_seeds=list(tuning_seeds),validation_seeds=list(validation_seeds),claim_seeds=list(claim_seeds))
    (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if artifact is not None:
        (out/'tuning_artifact.json').write_text(json.dumps(artifact.public_dict(),indent=2,allow_nan=False)+'\n')
    print(json.dumps({'route':route,'decision':report['decision'],'path':str(out/'result.json')}))
    return report if artifact is not None and complete_valid_results(claims,claim_seeds,spec.parameter_count) else None


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['pilot','full'],default='pilot')
    parser.add_argument('--routes',nargs='+',choices=list(ROUTES),default=list(ROUTES))
    parser.add_argument('--family',choices=['frozen_3d','p44','diagonal_ar'],default='frozen_3d')
    parser.add_argument('--dimension',type=int,default=3)
    parser.add_argument('--horizon',type=int,default=20)
    parser.add_argument('--particles',type=int,default=1008)
    parser.add_argument('--dtype',choices=['float32','float64'],default='float32')
    parser.add_argument('--no-jit-compile',action='store_true',help='Explicit reference/debug exception')
    parser.add_argument('--cpu-reference',action='store_true',help='Require GPUs hidden before import')
    parser.add_argument('--output-root',type=Path,default=Path('artifacts')/('sqmc-repaired-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
    args=parser.parse_args(argv)
    if args.cpu_reference:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
            parser.error('CPU reference requires CUDA_VISIBLE_DEVICES=-1 before import')
    elif not tf.config.list_physical_devices('GPU'):
        parser.error('GPU is the default; use --cpu-reference for an explicit reference run')
    if args.horizon < 1 or args.particles < 2:
        parser.error('horizon and particle count must be positive')
    count=2 if args.mode=='pilot' else 16
    candidates=_tuning_grid()[:2] if args.mode=='pilot' else _tuning_grid()
    good=True
    for route in args.routes:
        result=tune_route(route,args.horizon,args.particles,list(range(71001,71001+count)),args.output_root/route,
                          validation_seeds=list(range(72001,72001+count)),claim_seeds=list(range(73001,73001+count)),
                          spec=LGSSMSpec(args.family,args.dimension),candidates=candidates,dtype=tf.as_dtype(args.dtype),
                          jit_compile=not args.no_jit_compile)
        good=good and result is not None
    return 0 if good else 1


def _frozen_observations(horizon: int) -> tf.Tensor:
    """Return the frozen canonical LGSSM observation sequence.

    This MUST be the same frozen target the UNTUNED baseline diagnostic used
    (`run_sqmc_oracle_characterization.py`), otherwise TUNED and UNTUNED cells
    would be measured on different data and the comparison would be invalid.
    """
    from bayesfilter.highdim.ledh_canonical_neutra_targets_tf import (
        _lgssm_frozen_observations,
    )

    return tf.cast(_lgssm_frozen_observations()[:horizon], DTYPE)


def _tuning_grid() -> List[Dict]:
    """Historical candidate grid, retained only as an explicit warm-start hypothesis.

    These values have no universal numerical-safety or transfer guarantee.
    Each new scope requires its own calibration and independent validation.
    """
    grid = []

    epsilon_values = [4.0, 8.0, 16.0]
    diag_strengths = [0.1, 0.15, 0.2]
    pair_strengths = [0.02, 0.03]  # 0.02 is the baseline value

    for epsilon in epsilon_values:
        for diag_strength in diag_strengths:
            for pair_strength in pair_strengths:
                grid.append({
                    'reset_epsilon': epsilon,
                    'reset_sinkhorn_steps': 8,  # diagnostic candidate setting
                    'reset_balance_steps': 8,
                    'correction_strength': diag_strength,
                    'correction_steps': 4,  # diagnostic candidate setting
                    'pairwise_strength': pair_strength,
                    'pairwise_steps': 4,  # diagnostic candidate setting
                })

    return grid


if __name__ == "__main__":
    raise SystemExit(main())
