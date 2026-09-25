"""Exact preparation-policy snapshots; all numeric choices are fixture baselines."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
from bayesfilter.inference import hmc_kernel_tuning as policy
from bayesfilter.runtime import stable_config_hash
from tests.test_hmc_kernel_tuning_windowed_mass import _geometry


def snapshot():
    records = {}
    for preset in ('smoke', 'standard', 'diagnostic', 'diagnostic_plus', 'serious'):
        for mass in ('windowed_adaptive', 'fixed_identity'):
            cfg = getattr(policy.HMCKernelTuningConfig, preset)(mass_policy=mass, target_scope='configuration-reference')
            loop = policy._public_loop_config(cfg, public_timeout_started_perf_counter_s=10.)
            factory = policy._public_budget_policy_factory(cfg)
            budgets = [factory(dim, attempt).payload() for dim in (1, 2, 9, 256) for attempt in (0, 1, 3, 9)]
            records[preset+'/'+mass] = {
                'config': cfg.payload(), 'config_hash': stable_config_hash(cfg.payload()),
                'geometry': policy._public_geometry_config(cfg).payload(),
                'bootstrap': policy._public_bootstrap_config(cfg).payload(),
                'loop': loop.payload(),
                'windowed': [policy._phase7_windowed_stage_config(loop, attempt_index=i).payload() for i in (0, 1, 9)],
                'budgets': budgets,
                'budget_hashes': [stable_config_hash(b) for b in budgets],
            }
    cfg = policy.HMCKernelTuningConfig.serious(target_scope='configuration-reference')
    geometry = _geometry()
    records['serious_with_geometry'] = {
        'bootstrap': policy._public_bootstrap_config(cfg, geometry=geometry).payload(),
        'budget': policy._public_budget_policy_factory(cfg, geometry)(2, 0).payload(),
    }
    cfg = policy.HMCKernelTuningConfig.standard(
        target_scope='configuration-reference', candidate_search_bound_expansion_steps=1,
        staged_timeout_policy=policy.HMCStagedTimeoutPolicy(),
        staged_timeout_global_started_perf_counter_s=10., staged_timeout_enlargement_rounds={'windowed_mass': 1})
    with patch.object(policy.time, 'perf_counter', return_value=100.):
        loop = policy._public_loop_config(cfg)
        records['staged_timeouts'] = {
            'config': cfg.payload(), 'loop': loop.payload(),
            'windowed': policy._phase7_windowed_stage_config(loop, attempt_index=1).payload(),
        }
    for index, kwargs in enumerate((
        {'preset':'unknown'}, {'max_attempts':0}, {'max_leapfrog_steps':26},
        {'acceptance_band':(.8,.7)}, {'mass_policy':'unknown'},
        {'candidate_search_bound_expansion_steps':-1}, {'step_repair_factor':1.},
        {'public_timeout_budget_s':-1.}, {'metric_update_requirement':'unknown'},
        {'engineering_probe_covariance_multiplier':2.},
    )):
        try:
            policy.HMCKernelTuningConfig(**kwargs)
        except Exception as exc:
            records[f'invalid/{index}']={'exception':type(exc).__name__,'message':str(exc)}
        else:
            raise AssertionError(f'expected invalid config: {kwargs}')
    return json.loads(json.dumps(records, allow_nan=False))


parser=argparse.ArgumentParser()
parser.add_argument('output',type=Path)
parser.add_argument('--compare',type=Path)
args=parser.parse_args()
result=snapshot()
args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if args.compare:
    assert result==json.loads(args.compare.read_text()), 'configuration behavior changed'
print(json.dumps({'exact_parity':bool(args.compare),'records':len(result),
                  'budget_payloads':160,'preset_mass_combinations':10}))
