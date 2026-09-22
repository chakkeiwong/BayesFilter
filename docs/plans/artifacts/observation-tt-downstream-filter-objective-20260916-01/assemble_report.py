"""Post-run diagnostic reporting only; does not select or alter any method."""
from pathlib import Path
import collections
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT/'docs/benchmarks/artifacts/observation_tt_downstream_filter_objective_20260916/attempt-full-01'
OUT = Path(__file__).resolve().parent
result = json.loads((RUN/'result.json').read_text())
manifest = json.loads((RUN/'run_manifest.json').read_text())
assert result['status'] == manifest['status'] == 'COMPLETE'
assert len(result['sequences']) == 24
methods = ['transition','stationary_prior','sgqf_gaussian','sgqf_joint',
           'tt_predictive','tt_guided','tt_pair_block','tt_sgqf_safeguard']
regimes = ['all','near_zero','ordinary','large']
tables = []
available_tables = []
matched_sequences = {}
selection = {}
validity = []
refs = []
particle_statistics = []


def summarize_steps(steps):
    """Descriptive summaries of pre-resampling weights; no admission decision."""
    ess = [s['ess'] for s in steps]
    weights = [s['maximum_weight'] for s in steps]
    ancestors = [s['unique_ancestors'] for s in steps]
    for s in steps:
        assert 1-1e-9 <= s['ess'] <= 512+1e-9
        assert 1/512-1e-12 <= s['maximum_weight'] <= 1+1e-12
        assert 1 <= s['unique_ancestors'] <= 512
        assert bool(s['resampled']) == (s['ess'] < 256)
    resampled = [s for s in steps if s['resampled']]
    return dict(step_count=len(steps),mean_ess=statistics.mean(ess),
        mean_ess_fraction=statistics.mean(ess)/512,minimum_ess=min(ess),
        median_ess=statistics.median(ess),
        mean_maximum_weight=statistics.mean(weights),maximum_weight=max(weights),
        resampling_rate=len(resampled)/len(steps),
        mean_unique_ancestors=statistics.mean(ancestors),minimum_unique_ancestors=min(ancestors),
        mean_unique_ancestors_when_resampled=statistics.mean(s['unique_ancestors'] for s in resampled) if resampled else None,
        minimum_log_correction=min(s['minimum_log_correction'] for s in steps),
        maximum_log_correction=max(s['maximum_log_correction'] for s in steps))

def nonfinite_values(value, path=''):
    """Inspect stored numbers and the run serializer's nonfinite strings."""
    if isinstance(value, dict):
        return [p for k,v in value.items() for p in nonfinite_values(v, f'{path}/{k}')]
    if isinstance(value, list):
        return [p for i,v in enumerate(value) for p in nonfinite_values(v, f'{path}/{i}')]
    if isinstance(value, float) and not math.isfinite(value):
        return [path]
    if isinstance(value, str) and value.lower() in ('nan','inf','-inf','infinity','-infinity'):
        return [path]
    return []

def method_summary(dimension, method, records):
    valid = [s for s in records if method in s['metrics']]
    values = {}
    for regime in regimes:
        rows = [s['metrics'][method]['regimes'][regime] for s in valid]
        errors = [r['mse'] for r in rows if r['mse'] is not None]
        values[regime] = {'mean_mse':statistics.mean(errors) if errors else None,
            'sequence_count':len(errors),'coordinate_times':sum(r['count'] for r in rows)}
    times = [s['times'][method] for s in valid]
    return {'dimension':dimension,'method':method,'sequence_ids':[s['sequence'] for s in valid],
        'regimes':values,
        'median_fit_seconds':statistics.median(t['fit_seconds'] for t in times if 'fit_seconds' in t),
        'median_total_seconds':statistics.median(t['total_seconds'] for t in times),
        'max_absolute_log_evidence_bias':max(abs(s['metrics'][method]['log_evidence_bias']) for s in valid),
        'mean_log_evidence_bias':statistics.mean(s['metrics'][method]['log_evidence_bias'] for s in valid)}

for dimension in (1,4):
    records = [s for s in result['sequences'] if s['dimension'] == dimension]
    assert len(records) == 12
    matched = [s for s in records if all(m in s['metrics'] for m in methods)]
    matched_sequences[str(dimension)] = [s['sequence'] for s in matched]
    for method in methods:
        tables.append(method_summary(dimension,method,matched))
        available_tables.append(method_summary(dimension,method,records))
        steps = []
        matched_steps = []
        by_observation = collections.defaultdict(list)
        stored_nonfinite = []
        for s in records:
            dest = RUN/f"d{dimension}-s{s['sequence']:02d}"/method
            sequence_steps = []
            for p in sorted(dest.glob('particles-r*.json')):
                obj = json.loads(p.read_text())
                assert obj['count'] == 512 and len(obj['steps']) == 20
                steps.extend(obj['steps'])
                sequence_steps.extend(obj['steps'])
                stored_nonfinite.extend(nonfinite_values(obj,str(p.relative_to(RUN))))
            if method in s['metrics']:
                assert len(sequence_steps) == 80
                computed = summarize_steps(sequence_steps)
                saved = s['metrics'][method]['particle_diagnostics']
                assert len(saved['per_step']) == 20
                assert math.isclose(computed['mean_ess'],saved['ess']['mean'],rel_tol=1e-12)
                assert math.isclose(computed['minimum_ess'],saved['ess']['minimum'],rel_tol=1e-12)
            if s['sequence'] in matched_sequences[str(dimension)]:
                matched_steps.extend(sequence_steps)
                data=json.loads((dest.parent/'data.json').read_text())
                for step in sequence_steps:
                    magnitude=max(abs(y)/data['beta'] for y in data['observations'][step['time']])
                    label='near_zero' if magnitude<=.5 else 'ordinary' if magnitude<2 else 'large'
                    by_observation[label].append(step)
        particle_statistics.append(dict(dimension=dimension,method=method,
            sequence_ids=matched_sequences[str(dimension)],
            all_steps=summarize_steps(matched_steps),
            transition_steps=summarize_steps([s for s in matched_steps if s['time']>0]),
            whole_vector_observation_regimes_explanatory={k:summarize_steps(v) for k,v in by_observation.items()}))
        validity.append({'dimension':dimension,'method':method,'steps':len(steps),
            'explicit_nonfinite_flags':sum(s.get('finite') is False for s in steps),
            'missing_finite_flag':sum('finite' not in s for s in steps),
            'stored_nonfinite_count':len(stored_nonfinite),'stored_nonfinite_paths':stored_nonfinite,
            'bracket_failures':sum(s.get('cdf_bracket_valid',True) is False for s in steps),
            'bracket_flags_present':sum('cdf_bracket_valid' in s for s in steps),
            'max_cdf_residual':max((s.get('cdf_residual',0) for s in steps),default=0),
            'minimum_ess':min((s['ess'] for s in steps),default=None),
            'maximum_conditional_gaussian_fraction':max((s.get('maximum_conditional_gaussian_fraction',0) for s in steps),default=0),
            'gaussian_component_draws':sum(s.get('gaussian_component_draws',0) for s in steps)})
    names = collections.Counter()
    audit_losses = []
    panels = collections.defaultdict(list)
    l1 = collections.Counter()
    time_regimes = collections.Counter()
    for s in records:
        dest = RUN/f"d{dimension}-s{s['sequence']:02d}"
        refs.append({'dimension':dimension,'sequence':s['sequence'],**s['reference']})
        observations=json.loads((dest/'data.json').read_text())['observations']
        for y in observations:
            magnitude=max(abs(v)/.4 for v in y)
            time_regimes['near_zero' if magnitude<=.5 else 'ordinary' if magnitude<2 else 'large']+=1
        for p in sorted((dest/'tt_sgqf_safeguard').glob('selection-t*.json')):
            obj=json.loads(p.read_text());names[obj['selected']]+=1
            if obj.get('time',0):
                assert obj['audit_used'] is False
                assert obj['validation_h2'][obj['selected']]<=obj['validation_h2']['exact_sgqf_joint']
        for p in sorted((dest/'tt_sgqf_safeguard').glob('fit-t*.json')):
            obj=json.loads(p.read_text());picked=obj['selection']['selected']
            delta=obj['audit_h2'][picked]-obj['audit_h2']['exact_sgqf_joint']
            if delta>0: audit_losses.append({'sequence':s['sequence'],'time':obj['selection']['time'],'delta':delta})
            for split,stats in obj['row_diagnostics'].items(): panels[split].append(stats)
            for family,fit in obj['fits'].items(): l1[family+':'+str(fit['selected_l1'])]+=1
    selection[str(dimension)]={'selected_counts':dict(names),'audit_loss_count':len(audit_losses),
        'audit_loss_max':max((x['delta'] for x in audit_losses),default=0),'audit_losses':audit_losses,
        'whole_vector_max_observation_regimes_explanatory':dict(time_regimes),
        'selected_l1_counts':dict(l1),'row_panels':{k:{'count':len(v),
            'min_target_weight_ess':min(x['target_weight_ess'] for x in v),
            'max_target_weight_share':max(x['maximum_target_weight'] for x in v)} for k,v in panels.items()}}
report={'manifest_status':manifest['status'],'wall_seconds':manifest['wall_seconds'],
    'source_unchanged':manifest.get('source_unchanged'),
    'source_result_sha256':hashlib.sha256((RUN/'result.json').read_bytes()).hexdigest(),
    'reference_pass_count':sum(s['reference_pass'] for s in result['sequences']),
    'failures':[{'dimension':s['dimension'],'sequence':s['sequence'],'failures':s['failures']} for s in result['sequences'] if s['failures']],
    'matched_sequences':matched_sequences,
    'tables':tables,'all_available_tables':available_tables,
    'table_scope':'d1 complete; d4 descriptive only, conditional on all methods completing (sequence 8 excluded)',
    'references':refs,'selection':selection,'validity':validity,'inference':result['inference'],
    'particle_statistics':particle_statistics,
    'particle_statistics_scope':'Pre-resampling particle weights in the exact-importance-corrected PF; single-step ancestor count, not genealogical ESS. Time-zero SGQF and later SGQF selections remain part of the safeguarded policy.',
    'replay_status':'Same data, fit and particle seeds as A06; an instrumented reproducibility rerun, not independent confirmation.',
    'scientific_decision':dict(fit_loss_role='representation diagnostic and repair trigger',
        particle_ess_role='proposal efficiency and collapse diagnostic; no post-hoc ESS pass threshold',
        heuristic_loss_role='limits promotion; does not reject TT or stop repair',
        standalone_tt_recursion_validated=False,default_ready=False)}
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# A08 downstream filtering diagnostic tables','',
       'Primary values are equal-sequence averages of normalized MSE, not pooled time-point averages. Runtime and evidence diagnostics are descriptive.',
       'All methods use the same sequence set within a dimension: all 12 d1 sequences; 11 d4 sequences excluding failed guide sequence 8. The d4 table is conditional on completion and cannot support population ranking. All available outcomes, including the two unguided methods on sequence 8, remain in report.json.','',
       '| d | method | all | near zero | ordinary | large | median fit s | median total s |',
       '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for row in tables:
    cells=['N/A' if row['regimes'][r]['mean_mse'] is None else f"{row['regimes'][r]['mean_mse']:.7f}" for r in regimes]
    lines.append('| '+str(row['dimension'])+' | '+row['method']+' | '+' | '.join(cells)+f" | {row['median_fit_seconds']:.3f} | {row['median_total_seconds']:.3f} |")
lines += ['', 'ESS is computed before resampling from actual particle importance weights; N=512. Means use the same matched sequence sets as the MSE table. These are descriptive statistics, not target-row fitting ESS.', '',
    '| d | method | mean ESS | mean ESS/N | minimum ESS | largest weight | resampling rate | mean unique ancestors |',
    '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for row in particle_statistics:
    p=row['all_steps']
    lines.append(f"| {row['dimension']} | {row['method']} | {p['mean_ess']:.2f} | {p['mean_ess_fraction']:.4f} | {p['minimum_ess']:.2f} | {p['maximum_weight']:.4f} | {p['resampling_rate']:.4f} | {p['mean_unique_ancestors']:.2f} |")
lines += ['', '| d | heuristic | regime | paired delta | simultaneous lower | upper | coverage | precise |',
          '| --- | --- | --- | ---: | ---: | ---: | --- | --- |']
for c in result['inference']['contrasts']:
    lo='N/A' if c['lower'] is None else f"{c['lower']:.7f}"
    hi='N/A' if c['upper'] is None else f"{c['upper']:.7f}"
    lines.append(f"| {c['dimension']} | {c['heuristic']} | {c['regime']} | {c['mean_delta']:.7f} | {lo} | {hi} | {c['sequence_count']} sequences, {c['coordinate_times']} cells | {c['precision_pass']} |")
(OUT/'tables.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'status':manifest['status'],'wall_seconds':manifest['wall_seconds'],
    'references_pass':report['reference_pass_count'],'failures':len(report['failures']),
    'candidate_advances':result['inference']['candidate_advances'],
    'heuristic_verdict':result['inference']['heuristic_dominance_verdict']}))
