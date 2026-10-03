"""Bounded complete-fit cost/activation pilot; no SBC power or ranking claim."""
import argparse
import cProfile
import json
import os
from pathlib import Path
import pstats
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--control', choices=('baseline', 'noop', 'location_shift'), required=True)
    parser.add_argument('--seconds', type=float, default=400)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    from bayesfilter.testing.inference_validation.designs import ValidationDesign, ScenarioSpec
    from bayesfilter.testing.inference_validation.procedures import execute_pipeline
    from bayesfilter.testing.inference_validation.engines.pipeline import check_inventory
    from bayesfilter.testing.inference_validation.storage import write_json, read_json

    repo = Path(__file__).resolve().parents[2]
    old = repo/'docs/plans/artifacts/hmc-repair-master-2026-09-16/m22-r1/full-fit-pilot-cpu-r1/m22-full-fit-pilot-baseline'
    settings = read_json(old/'design.json')
    data = read_json(old/'dataset-0000/fit-0000/pipeline.json')['data']
    settings.update(design_id='m25-profile-'+args.control, replications=1,
                    seed=2026092263, budget_seconds=args.seconds,
                    purpose='complete ordinary fit cost and controlled-mutation activation; not power',
                    numerical_provenance='docs/plans/bayesfilter-hmc-gap-closure-continuation-2026-09-22.md')
    settings['scenario']['control'] = args.control
    settings['scenario'] = ScenarioSpec(**settings['scenario'])
    settings['l_grid'] = tuple(settings['l_grid'])
    settings['options']['plan_file'] = settings['numerical_provenance']
    design = ValidationDesign(**settings)
    write_json(args.output/'design.json', design.payload())
    write_json(args.output/'data.json', {'values': data, 'provenance': str(old/'dataset-0000/fit-0000/pipeline.json'),
        'role': 'reused development dataset for cost only; no fresh confirmation'})
    profiler = cProfile.Profile()
    try:
        profiler.enable()
        result = execute_pipeline(design, args.output/'fit', data=data,
                                  deadline=started+args.seconds)
    finally:
        profiler.disable()
        profiler.dump_stats(str(args.output/'profile.pstats'))
    stats = pstats.Stats(profiler)
    top = [{'file': key[0], 'line': key[1], 'function': key[2], 'primitive_calls': value[0],
            'total_calls': value[1], 'own_seconds': value[2], 'cumulative_seconds': value[3]}
           for key, value in sorted(stats.stats.items(), key=lambda item: -item[1][3])[:60]]
    inventory = check_inventory(read_json(result['tuning_path']))
    summary = {'control': args.control, 'command': [sys.executable,*sys.argv],
               'inventory': inventory, 'timing': result['timing'], 'top_cumulative_calls': top,
               'posterior': [{k: m.get(k) for k in ('candidate_id','status','timing','warmup_exclusion_matches')}
                             for m in result['members'] if m.get('status') != 'unassessed_by_design'],
               'unassessed_siblings': sum(m.get('status') == 'unassessed_by_design' for m in result['members']),
               'elapsed_seconds': time.monotonic()-started, 'profiling_overhead_included': True,
               'power_established': False, 'ranking_supported': False,
               'jit_compile': False, 'gpu_intentionally_hidden': True}
    write_json(args.output/'result.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k != 'top_cumulative_calls'}), flush=True)


if __name__ == '__main__':
    main()
