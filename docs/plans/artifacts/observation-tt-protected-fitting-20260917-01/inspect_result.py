"""Post-run diagnostic reporting only; never used for fitting or selection."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


def read(path):
    return json.loads(path.read_text())


def extent(values):
    return dict(mean=mean(values), median=median(values), minimum=min(values),
                maximum=max(values), count=len(values)) if values else None


def finite_numbers(value):
    if isinstance(value, list):
        return all(finite_numbers(v) for v in value)
    return isinstance(value, (int, float)) and math.isfinite(value)


def arm_decisions(summary, inference):
    """Apply A11 screens to each arm; a panel-wide flag is not an arm verdict."""
    decisions = {}
    for name in ('baseline', 'nominee', 'stronger-defense'):
        arm = summary['arms'].get(name)
        if arm is None:
            continue
        complete = (summary['sequences'] == 12 and summary['reference_valid'] == 12
                    and arm['completed'] == 12 and arm['reference_valid_sequences'] == 12)
        valid = (complete and not arm['failures'] and not arm['log_evidence_failures']
                 and arm['cdf_failures'] == 0 and arm['consumer_invalid_steps'] == 0
                 and arm.get('particles', {}).get('finite', False)
                 and arm.get('particles', {}).get('cdf_invalid_steps', 0) == 0)
        losses = [row for row in inference['heuristic_losses'] if row['candidate'] == name]
        contrast = inference['contrasts'].get(name, {})
        interval = contrast.get('simultaneous_interval')
        primary = ('comparator' if name == 'baseline' else
                   'incomplete_or_invalid_comparison' if interval is None else
                   'lower_mse' if interval[1] < 0 else
                   'non_harm' if interval[1] <= .10 else
                   'non_harm_not_established')
        decisions[name] = dict(reference_complete=complete, numerical_screen_passed=bool(valid),
            primary_criterion=primary, contrast=contrast, heuristic_losses=losses,
            promotion_veto=not valid or bool(losses),
            eligible_under_a11=bool(name != 'baseline' and valid and not losses
                                   and primary in ('lower_mse', 'non_harm')),
            default_readiness=False)
    return decisions


def inspect(root):
    run = read(root / 'result.json')
    manifest = read(root / 'run_manifest.json')
    if run['status'] != 'COMPLETE' or manifest['status'] != 'COMPLETE':
        raise ValueError('Only completed attempts can enter the terminal report')
    report = dict(stage=run['stage'], numerical_root=str(root.resolve()),
                  result_sha256=hashlib.sha256((root / 'result.json').read_bytes()).hexdigest(),
                  manifest=manifest, inference=run.get('inference'), dimensions={})
    for d in (1, 4):
        panel = [e for e in run['sequences'] if e['dimension'] == d]
        reference_panel = [e for e in panel if e['reference_pass']]
        names = sorted(set().union(*(set(e['metrics']) | set(e['failures']) for e in panel)))
        methods = Counter()
        for e in panel:
            methods.update(e['guide_methods'])
        summary = dict(sequences=len(panel), reference_valid=len(reference_panel),
                       reference_failures=[e['sequence'] for e in panel if not e['reference_pass']],
                       guide_methods=dict(methods), guide_seconds=extent([e['guide_seconds'] for e in panel]),
                       reference_seconds=extent([e['reference_seconds'] for e in panel]), arms={})
        for name in names:
            present = [e for e in panel if name in e['metrics']]
            ref_present = [e for e in reference_panel if name in e['metrics']]
            metrics = [e['metrics'][name] for e in present]
            arm = dict(completed=len(present), failures={str(e['sequence']):e['failures'][name]
                       for e in panel if name in e['failures']},
                       reference_valid_sequences=len(ref_present),
                       log_evidence_failures=[e['sequence'] for e in present
                                             if not e['metrics'][name]['log_evidence_screen']['passed']],
                       cdf_failures=sum(m['cdf_bracket_failures'] for m in metrics),
                       consumer_invalid_steps=sum(m['consumer_invalid_steps'] for m in metrics),
                       maximum_cdf_residual=max((m['maximum_cdf_residual'] for m in metrics), default=None),
                       regimes={})
            for regime in ('all', 'near_zero', 'ordinary', 'large'):
                pairs = [(e['metrics'][name]['regimes'][regime]['mse'],
                          e['metrics'][name]['regimes'][regime]['count']) for e in ref_present
                         if e['metrics'][name]['regimes'][regime]['count']]
                arm['regimes'][regime] = dict(count=sum(n for _, n in pairs),
                    sequence_count=len(pairs), mse=sum(v*n for v,n in pairs)/sum(n for _,n in pairs)
                    if pairs else None)
            steps, fits, chart_records, standalone, build, particle = [], [], [], [], [], []
            for e in present:
                dest = root / f"d{d}-s{e['sequence']:02d}" / name
                for path in sorted(dest.glob('particles-r*.json')):
                    steps.extend(read(path)['steps'])
                fit_name = 'nominee' if name == 'stronger-defense' else name
                if '-e0.' in fit_name:
                    fit_name = fit_name.rsplit('-e', 1)[0]
                fit_dest = dest.parent / fit_name
                fits.extend(read(p) for p in sorted(fit_dest.glob('fit-*.json')))
                path = fit_dest / 'path.json'
                if path.exists():
                    chart_records.extend(read(path)['charts'])
                times = e['times'][name]
                build_time = e['times'].get(fit_name, times)['build_seconds']
                particle_time = times['particle_seconds']/times['particle_replicates']
                guide_time = 0. if name in ('transition', 'stationary_prior') else e['guide_seconds']
                build.append(build_time)
                particle.append(particle_time)
                standalone.append(guide_time + build_time + particle_time)
            arm['timing_seconds'] = dict(build=extent(build), particle_per_replication=extent(particle),
                                        standalone_sequence=extent(standalone),
                                        caveat='Four-replication amortized particle cost; compilation included; shared fit charged to each standalone TT arm')
            if steps:
                ess = [s['ess'] for s in steps]
                margins = [s['log_density_lower_bound_margin'] for s in steps
                           if 'log_density_lower_bound_margin' in s]
                arm['particles'] = dict(steps=len(steps),
                    finite=all(s.get('finite', True) and finite_numbers([s[k] for k in
                               ('mean', 'covariance', 'log_increment', 'ess', 'maximum_weight')]) for s in steps),
                    native_finite_flag_steps=sum('finite' in s for s in steps),
                    cdf_checked_steps=sum('cdf_bracket_valid' in s for s in steps),
                    cdf_invalid_steps=sum(not s['cdf_bracket_valid'] for s in steps if 'cdf_bracket_valid' in s),
                    ess=extent(ess), maximum_weight=extent([s['maximum_weight'] for s in steps]),
                    low_ess_fraction=sum(x < .05*512 for x in ess)/len(ess),
                    resampling_fraction=mean(float(s['resampled']) for s in steps),
                    physical_bound_steps=len(margins), physical_bound_margin=extent(margins))
            if chart_records:
                arm['chart_floor'] = dict(steps=len(chart_records),
                    minimum_margin=min(r['relative_floor_margin'] for r in chart_records),
                    relative_eigenvalues=extent([v for r in chart_records for v in r['relative_eigenvalues']]))
            if fits:
                arm['fit_diagnostics'] = dict(steps=len(fits),
                    validation_h2=extent([r['validation']['fitted']['defended_h2'] for r in fits]),
                    initial_validation_h2=extent([r['validation']['initial']['defended_h2'] for r in fits]),
                    kkt=extent([r['fit_diagnostics']['kkt_residual'] for r in fits]),
                    row_ess=extent([r['row_diagnostics']['train']['row_ess'] for r in fits]),
                    projection_truncation=extent([r['conversion']['truncation_squared_l2'] for r in fits]),
                    projection_compression=extent([r['conversion']['compression_squared_l2'] for r in fits]))
            summary['arms'][name] = arm
        if run.get('inference'):
            summary['arm_decisions'] = arm_decisions(summary, run['inference'][str(d)])
            summary['panel_promotion_veto_meaning'] = (
                'At least one panel screen failed; use arm_decisions for candidate-specific verdicts')
        report['dimensions'][str(d)] = summary
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = inspect(args.root)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(stage=result['stage'], output=str(args.output),
        references={d:(v['reference_valid'], v['sequences']) for d,v in result['dimensions'].items()})))
