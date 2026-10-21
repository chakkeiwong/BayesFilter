#!/usr/bin/env python3
"""Post-run evidence audit and descriptive tables; no numerical runtime role."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import five_model_precision_diagnostic as campaign


def audit(root):
    execution = campaign.base.load(root/'execution.json')
    assert execution['status'] == 'completed', execution['status']
    assert len(execution['attempts']) == 21
    assert all(a['exit_code'] == 0 for a in execution['attempts'])
    assert execution['source_sha256'] == campaign.provenance(), 'Source changed after measurement'
    total, valid = 0, 0
    for model, n in campaign.SCOPES:
        data_seeds = (1704, 18020, 18021) if model == 'range_bearing' else (26100831, 26100832)
        points = (0, 1) if model == 'range_bearing' else (0,)
        seed_start = 31001 if model == 'range_bearing' else 261008301
        expected = {(d, p, seed_start+100*i+r) for i,d in enumerate(data_seeds)
                    for p in points for r in range(8)}
        paired = {}
        for mode in campaign.MODES:
            directory = root/campaign.label(model,n,mode)
            manifest = campaign.base.load(directory/'manifest.json')
            assert manifest['status'] == 'completed'
            assert manifest['source_sha256'] == execution['source_sha256']
            assert manifest['jit_compile'] and manifest['graph_trace_count'] == 1
            assert manifest['tf32_effective'] == (mode == 'fp32_tf32')
            assert manifest['memory_policy']['all_physical_devices_memory_growth']
            assert manifest['memory_policy']['configured_before_logical_device_initialization']
            rows = campaign.read_rows(directory/'measurements.jsonl')
            assert len(rows) == len(expected)
            keyed = {(r['data_seed'],r['point'],r['seed']): r for r in rows}
            assert set(keyed) == expected
            for row in rows:
                assert 'GPU:' in row['execution_device']
                assert len(row['score']) == manifest['parameter_count']
                assert len(row['steps_completed']) == manifest['parameter_count']
                if row['valid']:
                    assert all(math.isfinite(x) for x in [row['value'],*row['score']])
                    assert all(t == manifest['horizon'] for t in row['steps_completed'])
                    valid_index = row['trace_fields'].index('valid')
                    assert all(len(trace) == manifest['horizon'] for trace in row['directional_traces'])
                    assert all(step[valid_index] == 1 for trace in row['directional_traces'] for step in trace)
                total += 1
                valid += row['valid']
            paired[mode] = keyed
        for key in expected:
            assert len({tuple(paired[mode][key]['input_sha256']) for mode in campaign.MODES}) == 1
    result = dict(status='passed',measurements=total,valid_full_evaluations=valid,
                  invalid_evaluations=total-valid,source_hashes_unchanged=True,
                  exact_inputs_paired=True,scope_and_seed_completeness=True,
                  full_horizon_and_all_step_validity_checked=True,
                  command=[sys.executable,*sys.argv],
                  report_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    campaign.base.save(root/'terminal-audit.json',result)
    return result


def describe(root):
    result = campaign.report(root)
    summaries = []
    for model,n in campaign.SCOPES:
        reference_time = next(t for t in result['timing']
                              if (t['model'],t['particles'],t['mode']) == (model,n,'fp64'))
        for mode in campaign.MODES[1:]:
            cells = [c for c in result['cells'] if (c['model'],c['particles'],c['mode']) == (model,n,mode)]
            timing = next(t for t in result['timing'] if (t['model'],t['particles'],t['mode']) == (model,n,mode))
            complete = all(c['complete'] for c in cells)
            row = dict(model=model,particles=n,mode=mode,complete=complete,
                       full_evaluations=f"{timing['valid']}/{timing['expected']}",
                       screens_passed=sum(c['materiality_screen_pass'] for c in cells),screens_total=len(cells))
            if complete:
                row["relative_one_percent_likelihood_cells"] = sum(c["value_relative_rms"] <= .01 for c in cells)
                row["relative_one_percent_score_cells"] = sum(c["score_relative_rms"] <= .01 for c in cells)
                for key in ('value_difference_rms','score_difference_rms','value_relative_rms',
                            'score_relative_rms','value_sd_ratio','score_spread_ratio'):
                    row['worst_case_'+key] = max(c[key] for c in cells)
                row['observed_median_speed_ratio'] = reference_time['median_full_seconds']/timing['median_full_seconds']
            summaries.append(row)
    campaign.base.save(root/'summary.json',summaries)
    with (root/'final-likelihood-and-scores.csv').open('w',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model','particles','precision','data_seed','parameter_point','valid_designs',
                         'log_likelihood_mean','score_mean','parameters'])
        for model,n in campaign.SCOPES:
            for mode in campaign.MODES:
                rows = campaign.read_rows(root/campaign.label(model,n,mode)/'measurements.jsonl')
                for data,point in sorted({(r['data_seed'],r['point']) for r in rows}):
                    group = [r for r in rows if (r['data_seed'],r['point']) == (data,point)]
                    valid = sum(r['valid'] for r in group)
                    value = statistics.mean(r['value'] for r in group) if valid == 8 else ''
                    score = [statistics.mean(r['score'][j] for r in group) for j in range(len(group[0]['score']))] if valid == 8 else ''
                    writer.writerow([model,n,mode,data,point,valid,value,json.dumps(score),json.dumps(group[0]['theta'])])
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    checked = audit(args.output)
    summary = describe(args.output)
    print(json.dumps(dict(audit=checked,summary=summary),indent=2))
