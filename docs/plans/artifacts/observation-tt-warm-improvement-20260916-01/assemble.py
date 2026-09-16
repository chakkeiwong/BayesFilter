"""Post-run A09 reporting only; standard-library descriptive aggregation."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean


def read(path):
    return json.loads(path.read_text())


def summary(values):
    return dict(mean=mean(values),minimum=min(values),maximum=max(values),count=len(values)) if values else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--calibration',required=True)
    p.add_argument('--confirmation',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();cal=Path(args.calibration);conf=Path(args.confirmation);out=Path(args.output)
    result=read(conf/'result.json');selected=read(cal/'selected-controls.json')
    if result['status']!='COMPLETE' or (conf/'invalidation.json').exists():
        raise ValueError('Completed, non-invalidated confirmation required')
    methods=('baseline','capacity','preservation','tt_pair_block','transition','stationary_prior','sgqf_gaussian','sgqf_joint')
    report=dict(status=result['status'],selected=selected['dimensions'],dimensions={},same_target={},
                inference=result.get('inference'),missing=[],default_ready=False)
    lines=['# A09 fresh confirmation results','',
           'Descriptive sequence averages on matched, reference-valid cases. MSE is normalized by stationary variance. ESS is out of 512.', '',
           '| d | Method | Sequences | MSE | Mean/min ESS | Max weight | Build s | Particle s / repetition |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for d in (1,4):
        entries=[e for e in result['sequences'] if e['dimension']==d]
        matched=[e for e in entries if e['reference_pass'] and all(name in e['metrics'] for name in methods)]
        dimension=dict(total_sequences=len(entries),matched_sequences=len(matched),reference_failures=[],methods={},
                       shared_guide_seconds=summary([e['guide_seconds'] for e in matched]))
        for e in entries:
            if not e['reference_pass']: dimension['reference_failures'].append(e['sequence'])
            if e['failures']: report['missing'].append(dict(dimension=d,sequence=e['sequence'],failures=e['failures']))
        for name in methods:
            records=[e['metrics'][name] for e in matched]
            times=[e['times'][name] for e in matched]
            if not records: continue
            stats=dict(mse=mean(r['regimes']['all']['mse'] for r in records),
                ess_mean=mean(r['particle_diagnostics']['ess']['mean'] for r in records),
                ess_min=min(r['particle_diagnostics']['ess']['minimum'] for r in records),
                maximum_weight=max(r['particle_diagnostics']['maximum_weight']['maximum'] for r in records),
                resampling_rate=mean(r['particle_diagnostics']['resampling_rate'] for r in records),
                rms_sequence_mean_logevidence_bias=math.sqrt(mean(r['log_evidence_bias']**2 for r in records)),
                evidence_screen_failed_sequences=[e['sequence'] for e in matched if not e['metrics'][name]['log_evidence_screen']['passed']],
                cdf_bracket_failures=sum(r['cdf_bracket_failures'] for r in records),
                consumer_invalid_steps=sum(r['consumer_invalid_steps'] for r in records),
                maximum_cdf_residual=max(r['maximum_cdf_residual'] for r in records),
                build_seconds=mean(t['build_seconds'] for t in times),
                particle_seconds_per_rep=mean(t['particle_seconds']/4 for t in times))
            guide_cost=0. if name in ('transition','stationary_prior') else dimension['shared_guide_seconds']['mean']
            stats['single_filter_seconds_including_guide']=guide_cost+stats['build_seconds']+stats['particle_seconds_per_rep']
            stats['amortized_seconds_over_four_repetitions']=(guide_cost+stats['build_seconds'])/4+stats['particle_seconds_per_rep']
            dimension['methods'][name]=stats
            lines.append(f"| {d} | {name} | {len(matched)} | {stats['mse']:.7f} | {stats['ess_mean']:.2f} / {stats['ess_min']:.2f} | {stats['maximum_weight']:.4f} | {stats['build_seconds']:.3f} | {stats['particle_seconds_per_rep']:.3f} |")
        report['dimensions'][str(d)]=dimension
        same={}
        for name in ('baseline','capacity','preservation'):
            rows=[read(f) for f in sorted(cal.glob(f'd{d}-s*/same-target-audit/{name}-t*.json'))]
            same[name]=dict(cases=len(rows),
                fitted_h2=summary([r['audit']['fitted']['defended_h2'] for r in rows]),
                initial_h2=summary([r['audit']['initial']['defended_h2'] for r in rows]),
                amplitude_rms=summary([r['audit']['fitted']['amplitude_rms'] for r in rows]),
                worsened_from_initializer=sum(r['audit']['fitted']['defended_h2']>r['audit']['initial']['defended_h2'] for r in rows),
                kkt=summary([r['fit_diagnostics']['kkt_residual'] for r in rows]),
                condition=summary([r['fit_diagnostics']['maximum_core_gram_condition_capped'] for r in rows]),
                truncation=summary([r['conversion']['truncation_squared_l2'] for r in rows]),
                compression=summary([r['conversion']['compression_squared_l2'] for r in rows]),
                row_ess=summary([r['row_diagnostics']['train']['row_ess'] for r in rows]),
                target_weight_ess=summary([r['row_diagnostics']['train']['target_weight_ess'] for r in rows]))
        report['same_target'][str(d)]=same
    for contrast in report['inference']['primary_contrasts']:
        d=report['dimensions'][str(contrast['dimension'])]
        names=('baseline',contrast['candidate'])
        veto=any(d['methods'].get(name,{}).get('evidence_screen_failed_sequences',[True]) or
                 d['methods'].get(name,{}).get('consumer_invalid_steps',1) or
                 d['methods'].get(name,{}).get('cdf_bracket_failures',1) for name in names)
        contrast['scientific_ranking_admissible']=contrast['mse_improvement_supported'] and not veto and d['matched_sequences']==12
    report['manifests']={name:read(path/'run_manifest.json') for name,path in [('calibration',cal),('confirmation',conf)]}
    report['calibration_limitation']='Nomination/fit diagnostics only: revision-1 scalar seed reuse; no calibration uncertainty or superiority claim.'
    lines += ['', '## Primary filtering contrasts', '',
              'Candidate minus standalone warm baseline; simultaneous 95% exploratory sequence-bootstrap intervals.', '',
              '| d | Candidate | Mean difference | Lower | Upper | Improvement supported |',
              '|---|---|---:|---:|---:|---|']
    for c in report['inference']['primary_contrasts']:
        value=lambda key: 'N/A' if c[key] is None else f"{c[key]:.8f}"
        lines.append(f"| {c['dimension']} | {c['candidate']} | {value('mean_delta')} | {value('lower')} | {value('upper')} | {c['scientific_ranking_admissible']} |")
    lines += ['', '## Fixed-target regression', '',
              'Nine audited target/time cases per dimension and method; descriptive calibration evidence only.', '',
              '| d | Method | Initial H2 | Fitted H2 | Amplitude RMS | Max KKT | Max Gram condition |',
              '|---|---|---:|---:|---:|---:|---:|']
    for d,rows in report['same_target'].items():
        for name,s in rows.items():
            lines.append(f"| {d} | {name} | {s['initial_h2']['mean']:.6f} | {s['fitted_h2']['mean']:.6f} | {s['amplitude_rms']['mean']:.5f} | {s['kkt']['maximum']:.5g} | {s['condition']['maximum']:.4g} |")
    lines += ['', '## Observed conditional heuristic losses', '',
              'Descriptive differences only. These trigger the predeclared default-promotion screen, not rejection of TT repair.', '',
              '| d | TT arm | Heuristic | Observation regime | Sequences | Mean difference |',
              '|---|---|---|---|---:|---:|']
    for c in report['inference']['heuristic_contrasts']:
        if c['observed_loss']:
            lines.append(f"| {c['dimension']} | {c['candidate']} | {c['heuristic']} | {c['regime']} | {c['sequences']} | {c['mean_delta']:.8f} |")
    lines += ['', 'All conditional contrasts, log-evidence screens, resampling rates, guide time and total/amortized timings are retained in report.json.']
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__': main()
