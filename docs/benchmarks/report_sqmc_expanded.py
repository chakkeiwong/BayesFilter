"""Post-run SQMC report and completeness audit; standard-library diagnostics."""
from pathlib import Path
import argparse
import csv
import json
import math
import itertools
import statistics

import run_sqmc_expanded_comparison as campaign

LABELS={'iid_dual_cap':'IID','previous_inverse_cdf':'Inverse CDF',
        'repaired_permutation':'Permutation','repaired_permutation_ablation':'Permutation cap .97'}


def number(value):
    return f'{value:.10g}' if isinstance(value,(int,float)) else str(value if value is not None else 'unavailable')


def mean(values):
    return statistics.fmean(values) if values else None



def expected_contracts():
    original=campaign.comparison_contract()
    names=('SCOPES','CAL','VAL','CLAIM','FILTER','ENTRY_POINT','EXTRA_SOURCE_PATHS','EXTRA_CONTRACT','CONTROL_PREPARER')
    saved={name:getattr(campaign,name) for name in names}
    try:
        import run_sqmc_expanded_repair as repair
        repair.configure_profile()
        repaired=campaign.comparison_contract()
    finally:
        for name,value in saved.items():setattr(campaign,name,value)
    return original,repaired


def audit_score_csv(cases, score_rows):
    """Verify every exported coordinate against the preserved numerical result."""
    failures=[]
    expected={}
    for case in cases:
        if len(set(case['parameter_names']))!=len(case['parameter_names']):
            failures.append(f"{case['scope']}/{case['route']}: duplicate coordinate names")
        for row in case['rows']:
            estimates=row.get('score') or row.get('raw_score') or []
            for name,exact,estimate in zip(case['parameter_names'],row['oracle_score'],estimates):
                key=(case['scope'],case['route'],str(row['data_seed']),str(row['filter_seed']),name)
                error=estimate-exact if row['valid'] else None
                expected[key]=dict(program=case['program'],tuning_status=case['tuning_status'],
                    valid=str(row['valid']),exact_score=exact,estimated_score=estimate,
                    signed_error=error,absolute_error=abs(error) if error is not None else None)
    seen=set()
    for exported in score_rows:
        key=tuple(exported.get(k) for k in ('scope','route','data_seed','filter_seed','coordinate'))
        if key in seen:failures.append(f'scores.csv: duplicate coordinate {key}')
        seen.add(key)
        saved=expected.get(key)
        if saved is None:
            failures.append(f'scores.csv: unexpected coordinate {key}');continue
        for field,value in saved.items():
            actual=exported.get(field)
            if field in ('program','tuning_status','valid'):
                agrees=actual==value
            elif value is None:
                agrees=actual==''
            else:
                try:
                    actual_number=float(actual);saved_number=float(value)
                    agrees=actual_number==saved_number or (math.isnan(actual_number) and math.isnan(saved_number))
                except (ValueError,TypeError):
                    agrees=False
            if not agrees:failures.append(f'scores.csv: {field} differs from worker at {key}')
    missing=expected.keys()-seen
    if missing:failures.append(f'scores.csv: {len(missing)} missing coordinates')
    return failures


def report(source, output):
    output.mkdir(parents=True,exist_ok=False)
    data=json.loads((source/'results.json').read_text())
    cases=data['cases']
    lookup={(c['scope'],c['route']):c for c in cases}
    failures=[]
    if len(lookup)!=len(cases):
        failures.append('duplicate scope/route units')
    provenance=[]
    datasets={}
    scope_contracts={}
    numerical_closure=None
    accepted_contracts=expected_contracts()
    for case in cases:
        directory=Path(case['artifact_directory'])
        if not directory.is_absolute():
            directory=source/directory
        meta=json.loads((directory/'manifest.json').read_text())
        memory=meta.get('gpu_memory_policy',{})
        if meta.get('jit_compile') is not True or meta.get('tf32') is not False:
            failures.append(f'{directory}: FP64-reference XLA/TF32 execution declaration mismatch')
        if 'GPU:' not in meta.get('framework_gpu_probe',{}).get('device','') or not meta.get('gpu_uuid'):
            failures.append(f'{directory}: missing verified GPU provenance')
        if (memory.get('mode')!='memory_growth' or
                memory.get('all_physical_devices_memory_growth') is not True or
                memory.get('configured_before_logical_device_initialization') is not True):
            failures.append(f'{directory}: GPU memory-growth policy not verified')
        saved=json.loads((directory/'data.json').read_text())
        tuning=json.loads((directory/'tuning.json').read_text())
        original_case=json.loads((directory/'result.json').read_text())
        for key in ('scope','route','status','parameter_count','parameter_names','selected_controls'):
            if case.get(key)!=original_case.get(key):failures.append(f'{directory}: assembled {key} differs from worker')
        original_rows={(r['data_seed'],r['filter_seed']):r for r in original_case['rows']}
        if len(case['rows'])!=len(original_case['rows']):failures.append(f'{directory}: assembled row count differs from worker')
        for row in case['rows']:
            original=original_rows.get((row['data_seed'],row['filter_seed']))
            if original is None:failures.append(f'{directory}: assembled data/filter pair differs from worker');continue
            for key in ('valid','value','score','raw_value','raw_score','oracle_value','oracle_score','value_error','score_l2_error'):
                if row.get(key)!=original.get(key):failures.append(f'{directory}: assembled {key} differs from worker')
        closure=campaign.numerical_sources(meta['source_sha256'])
        if numerical_closure is not None and closure!=numerical_closure:
            failures.append(f'{directory}: numerical source differs across compared units')
        numerical_closure=closure
        for filename,key in [('result.json','result_sha256'),('tuning.json','tuning_sha256')]:
            if campaign.digest(directory/filename)!=meta[key]:
                failures.append(f'{directory}: {filename} hash mismatch')
        if campaign.digest(directory/'data.json')!=meta['data_version']['sha256']:
            failures.append(f'{directory}: data checksum mismatch')
        if campaign.digest(directory/'random_designs.json')!=meta['random_designs_sha256']:
            failures.append(f'{directory}: random design checksum mismatch')
        if meta.get('source_unchanged') is not True:
            failures.append(f'{directory}: source closure not verified unchanged')
        if meta.get('comparison_contract') not in accepted_contracts:
            failures.append(f'{directory}: comparison specification mismatch')
        contract=meta['comparison_contract']
        if case['scope'] in scope_contracts and scope_contracts[case['scope']]!=contract:
            failures.append(f"{case['scope']}: routes use different tuning/final partitions")
        scope_contracts[case['scope']]=contract
        if 'transport_calibration_sha256' in meta and campaign.digest(directory/'transport_calibration.json')!=meta['transport_calibration_sha256']:
            failures.append(f'{directory}: transport-calibration hash mismatch')
        if case['scope'] in datasets and datasets[case['scope']]!=saved:
            failures.append(f"{case['scope']}: observations/oracles differ across routes")
        datasets[case['scope']]=saved
        if case['status']!='tuning_failed':
            pairs={(r['data_seed'],r['filter_seed']) for r in case['rows']}
            if pairs!=set(itertools.product(contract['final_data'],contract['final_filter'])) or len(case['rows'])!=4:
                failures.append(f'{directory}: final pair design incomplete')
            artifact=tuning.get('tuning_artifact',{})
            if artifact.get('controls')!=case['selected_controls']:
                failures.append(f'{directory}: frozen controls mismatch')
        for row in case['rows']:
            exact=saved['datasets'][str(row['data_seed'])]
            if row.get('oracle_score')!=exact['oracle_score'] or row.get('oracle_value')!=exact['oracle_value']:
                failures.append(f'{directory}: saved oracle does not match evaluated row')
            estimate=row.get('score') or row.get('raw_score') or []
            if len(estimate)!=case['parameter_count'] or len(case['parameter_names'])!=case['parameter_count']:
                failures.append(f'{directory}: score dimension mismatch')
            if row['valid']:
                if not all(math.isfinite(x) for x in estimate):
                    failures.append(f'{directory}: admitted nonfinite score')
                expected=campaign.l2_error(estimate,row['oracle_score'])
                if not math.isclose(expected,row['score_l2_error'],rel_tol=1e-10,abs_tol=1e-10):
                    failures.append(f'{directory}: score error mismatch')
                first=campaign.l2_error(exact['first_observation_score'],row['oracle_score'])
                if not math.isclose(first,row['heuristic_first_observation_only_l2_error'],rel_tol=1e-12,abs_tol=1e-12):
                    failures.append(f'{directory}: first-only baseline target mismatch')
        provenance.append(dict(scope=case['scope'],route=case['route'],status=case['status'],
            directory=str(directory),tuning_path=str(directory/'tuning.json'),manifest_path=str(directory/'manifest.json'),
            wall_seconds=meta.get('wall_seconds'),gpu_uuid=meta.get('gpu_uuid'),jit_compile=meta.get('jit_compile'),tf32=meta.get('tf32'),
            allocator_peak_bytes=meta.get('gpu_allocator_bytes',{}).get('peak'),
            valid_final_cells=sum(r['valid'] for r in case['rows']),recorded_final_cells=len(case['rows'])))
    expected={(s[0],r) for s in campaign.SCOPES for r in campaign.ROUTES}
    missing=sorted(expected-set(lookup))
    unexpected=sorted(set(lookup)-expected)
    if unexpected:failures.append(f'unexpected scope/route units: {unexpected}')
    with (source/'scores.csv').open(newline='') as stream:
        reader=csv.DictReader(stream);score_fields=reader.fieldnames;score_rows=list(reader)
    failures.extend(audit_score_csv(cases,score_rows))
    audit=dict(schema='sqmc_expanded_terminal_audit.v1',engineering_failures=failures,
        missing_units=missing,expected_units=32,recorded_units=len(cases),
        expected_final_cells=128,recorded_final_cells=sum(len(c['rows']) for c in cases),
        valid_final_cells=sum(r['valid'] for c in cases for r in c['rows']),
        expected_score_entries=8480,exported_score_entries=len(score_rows),
        recorded_score_entries=sum(len(r.get('score') or r.get('raw_score') or []) for c in cases for r in c['rows']),
        provenance=provenance,statistical_ranking_supported=False,
        review_type='Codex terminal audit with executable consistency checks; no independent external reviewer')
    campaign.dump(output/'audit.json',audit)
    with (output/'likelihoods.csv').open('w',newline='') as stream:
        fields=['program','tuning_status','scope','route','data_seed','filter_seed','valid',
                'exact_log_likelihood','estimated_log_likelihood','signed_error','absolute_error']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for case in cases:
            for row in case['rows']:
                value=row.get('value') if row['valid'] else row.get('raw_value')
                exact=row.get('oracle_value')
                error=value-exact if row['valid'] and isinstance(value,(int,float)) and exact is not None else None
                writer.writerow(dict(program=case['program'],tuning_status=case['tuning_status'],
                    scope=case['scope'],route=case['route'],data_seed=row['data_seed'],filter_seed=row['filter_seed'],
                    valid=row['valid'],exact_log_likelihood=exact,estimated_log_likelihood=value,
                    signed_error=error,absolute_error=abs(error) if error is not None else None))
    for row in score_rows:
        if row['valid']!='True':row.update(signed_error='',absolute_error='')
    with (output/'scores.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=score_fields);writer.writeheader();writer.writerows(score_rows)
    summaries=[]
    conditional=[]
    for scope,family,d,t,n in campaign.SCOPES:
        lines=[f'# {scope}: actual Kalman and particle scores','',
            'Program: FP64 GPU/XLA reference comparison, TF32 disabled. Each route calibrates flow substeps 2 versus 8 within this exact scope. Full-model repair scopes also calibrate transport smoothing epsilon; the other numerical protections remain inherited hypotheses. Tuning paths are in audit.json. This is not FP32/TF32 production evidence or a statistically supported ranking. Only the P44 partitions reuse disclosed pilot observations; full-model repair partitions are fresh. Invalid rows are retained for diagnosis and excluded from error summaries.','',
            f'N={n}, d={d}, T={t}. '+('Q scores use lower-Cholesky coordinates, with log diagonal; they are not derivatives with respect to covariance entries.' if family=='full_matrix' else 'The four coordinates are persistence, log process-noise scale, log observation-noise scale, and initial-mean scale.'),'']
        contract=scope_contracts.get(scope,accepted_contracts[0])
        for data_seed,filter_seed in itertools.product(contract['final_data'],contract['final_filter']):
            lines += [f'## Data {data_seed}, filter {filter_seed}','',
                'Each method entry is its actual score followed by absolute error in parentheses. CSV files preserve full floating-point precision.','']
            rows={}
            for route in campaign.ROUTES:
                case=lookup.get((scope,route))
                if case:
                    matches=[r for r in case['rows'] if r['data_seed']==data_seed and r['filter_seed']==filter_seed]
                    if matches: rows[route]=matches[0]
            if not rows:
                lines+=['No evaluated final cell for this pair.',''];continue
            likelihood_exact=next(iter(rows.values()))['oracle_value']
            lines += [f'Kalman log likelihood: {number(likelihood_exact)}.','',
                      '| Route | Log likelihood | Absolute error | Valid |',
                      '|---|---:|---:|---|']
            for route in campaign.ROUTES:
                row=rows.get(route)
                if row is None:
                    lines.append(f'| {LABELS[route]} | not evaluated | unavailable | no |')
                    continue
                value=row.get('value') if row['valid'] else row.get('raw_value')
                error=abs(value-likelihood_exact) if row['valid'] and isinstance(value,(int,float)) else None
                lines.append(f"| {LABELS[route]} | {number(value)} | {number(error)} | {row['valid']} |")
            lines += ['']
            exact=next(iter(rows.values()))['oracle_score']
            names=next(lookup[(scope,r)]['parameter_names'] for r in campaign.ROUTES if (scope,r) in lookup)
            lines += ['| Coordinate | Kalman | IID | Inverse CDF | Permutation | Cap .97 |',
                      '|---|---:|---:|---:|---:|---:|']
            for i,name in enumerate(names):
                cells=[]
                for route in campaign.ROUTES:
                    row=rows.get(route)
                    if row is None:
                        cells.append('not evaluated');continue
                    values=row.get('score') or row.get('raw_score')
                    value=values[i]
                    error=abs(value-exact[i]) if row['valid'] and isinstance(value,(int,float)) else None
                    cell=f'{number(value)} ({number(error)})'
                    cells.append(cell if row['valid'] else 'INVALID '+cell)
                lines.append('| '+name+' | '+number(exact[i])+' | '+' | '.join(cells)+' |')
            lines+=['']
        (output/f'{scope}.md').write_text('\n'.join(lines)+'\n')
        for route in campaign.ROUTES:
            case=lookup.get((scope,route))
            if not case:continue
            valid=[r for r in case['rows'] if r['valid']]
            errors=[a-b for r in valid for a,b in zip(r['score'],r['oracle_score'])]
            summary=dict(scope=scope,route=route,status=case['status'],valid_cells=len(valid),
                selected_flow_substeps=(case.get('selected_controls') or {}).get('flow_substeps'),
                selected_balance_steps=(case.get('selected_controls') or {}).get('reset_balance_steps'),
                selected_epsilon=(case.get('selected_controls') or {}).get('reset_epsilon'),
                mean_abs_log_likelihood_error=mean([abs(r['value_error']) for r in valid]),
                mean_score_l2_error=mean([r['score_l2_error'] for r in valid]),
                component_rmse=math.sqrt(mean([e*e for e in errors])) if errors else None,
                max_abs_coordinate_error=max(map(abs,errors)) if errors else None,
                zero_score_observed_losses=sum(r['heuristic_dominance']['zero_score']=='observed_loss' for r in valid),
                first_only_observed_losses=sum(r['heuristic_dominance']['first_observation_only']=='observed_loss' for r in valid),
                iid_observed_losses=sum(r['heuristic_dominance'].get('iid')=='observed_loss' for r in valid),
                block_rmse={k:math.sqrt(mean([r['block_score_rmse'][k]**2 for r in valid]))
                            for k in valid[0].get('block_score_rmse',{})} if valid else {})
            summaries.append(summary)
            for seed in contract['final_data']:
                subset=[r for r in valid if r['data_seed']==seed]
                conditional.append(dict(scope=scope,route=route,data_seed=seed,valid_cells=len(subset),
                    mean_score_l2_error=mean([r['score_l2_error'] for r in subset]),
                    zero_score_l2_error=subset[0]['heuristic_zero_score_l2_error'] if subset else None,
                    first_only_l2_error=subset[0]['heuristic_first_observation_only_l2_error'] if subset else None))
    campaign.dump(output/'summaries.json',dict(summaries=summaries,conditional=conditional))
    with (output/'conditional_heuristics.csv').open('w',newline='') as stream:
        fields=['scope','route','data_seed','valid_cells','mean_score_l2_error','zero_score_l2_error','first_only_l2_error']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows(conditional)
    with (output/'block_errors.csv').open('w',newline='') as stream:
        fields=['scope','route','valid_cells','block','score_component_rmse']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for summary in summaries:
            for block,value in summary['block_rmse'].items():
                writer.writerow(dict(scope=summary['scope'],route=summary['route'],valid_cells=summary['valid_cells'],block=block,score_component_rmse=value))

    lines=['# Expanded SQMC Kalman comparison','',
        'Program: FP64 GPU/XLA reference comparison, TF32 disabled, with exact-scope calibration of flow substeps 2 versus 8. Full-model repair scopes additionally calibrate transport smoothing epsilon on fresh partitions before complete-score tuning. Other numerical protections remain inherited hypotheses. This report does not describe the FP32/TF32 production configuration. Tuning artifacts and manifests for every unit are linked by path in audit.json. A finite candidate is not certified accurate.','',
        f"Recorded {audit['recorded_units']}/32 route/scope units and {audit['recorded_final_cells']}/128 final cells; {audit['valid_final_cells']} final cells are numerically valid. Recorded and verified {audit['recorded_score_entries']}/8480 individual score entries. Invalid final cells: {audit['recorded_final_cells']-audit['valid_final_cells']}. Engineering audit findings: {len(failures)}. Missing units: {len(missing)}.",'',
        'Two data sets, each crossed with two filter designs, permit descriptive comparisons only. Filter repetitions on the same data are not independent data sets. No confidence interval or statistically supported method ranking is inferred. P44 includes two previously inspected pilot pairs; they remain repeated evidence. Full-model repair uses fresh final data and filter partitions. Its inherited generic manifest sentence about pilot reuse applies only to P44; the saved full-model partitions and row labels are correct.','',
        '## Actual scores and likelihoods','',
        '[All score coordinates and absolute errors](scores.csv); [all actual log likelihoods and errors](likelihoods.csv). Separate score tables follow:','']
    for scope,*_ in campaign.SCOPES:
        lines.append(f'- [{scope}]({scope}.md)')
    observed_iid_losses=sum(s['iid_observed_losses'] for s in summaries)
    sqmc_cells=sum(s['valid_cells'] for s in summaries if s['route']!='iid_dual_cap')
    lines+=['',f'IID has a smaller observed score error in {observed_iid_losses} of {sqmc_cells} valid SQMC route/data/design comparisons. These conditional losses veto promotion in those situations. The comparisons share datasets and are not independent trials; their count is not a statistical ranking.','',
        '![Paired score-vector errors](figures/score_errors.png)','',
        '[Score figure, SVG](figures/score_errors.svg); [paired likelihood-error figure, SVG](figures/likelihood_errors.svg). Each line preserves a matched data/filter pair. Axes use separate logarithmic scales; compare routes within a panel.','']
    lines+=['','## Model conditioning','',
        'The transition norm and process-covariance eigenvalues are evaluated at the fixed data-generating parameter. Stable, moderately conditioned test models do not test near-singular or highly persistent regimes.','',
        '| Scope | Maximum absolute row sum of A | Smallest Q eigenvalue | Largest Q eigenvalue | Q condition number |',
        '|---|---:|---:|---:|---:|']
    for scope,*_ in campaign.SCOPES:
        candidate=next((c for c in cases if c['scope']==scope),None)
        if candidate:
            diagnostics=candidate['model_diagnostics']
            lines.append('| '+scope+' | '+' | '.join(number(diagnostics[k]) for k in ('transition_max_abs_row_sum','process_covariance_min_eigenvalue','process_covariance_max_eigenvalue','process_covariance_condition_number'))+' |')
    lines+=['','## Descriptive errors','',
        'Invalid candidates are excluded from these summaries and remain visible in the raw tables. Component RMS accounts for parameter count, but both RMS and vector L2 mix coordinate scales. Compare these summaries within the same model and use the per-coordinate tables for substantive interpretation; normalization alone does not make different models comparable.','',
        '| Scope | Route | Valid / 4 | Flow steps | Epsilon | Mean absolute log-likelihood error | Mean score L2 error | Component RMS error | Largest coordinate error |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append('| '+' | '.join([s['scope'],LABELS[s['route']],str(s['valid_cells']),number(s['selected_flow_substeps']),number(s['selected_epsilon']),
            number(s['mean_abs_log_likelihood_error']),number(s['mean_score_l2_error']),number(s['component_rmse']),
            number(s['max_abs_coordinate_error'])])+' |')
    lines+=['','## Conditional heuristic checks','',
        '[Per-data-set conditional comparisons](conditional_heuristics.csv) and [parameter-block errors](block_errors.csv) preserve the breakdown. Counts are observed losses among valid final cells against zero score, first-observation-only Kalman score, and matched IID particle estimates. An observed loss vetoes promotion in that situation; absence of a loss is not evidence of superiority. Both cheap score baselines are compared with the full-horizon Kalman score. Per-data-seed errors are preserved in summaries.json.','',
        '| Scope | Route | Zero-score losses | First-only losses | IID losses |',
        '|---|---|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['scope']} | {LABELS[s['route']]} | {s['zero_score_observed_losses']} / {s['valid_cells']} | {s['first_only_observed_losses']} / {s['valid_cells']} | {s['iid_observed_losses']} / {s['valid_cells']} |")
    lines+=['','## Interpretation and remaining uncertainty','',
        'The analytical recursion differentiates the finite particle program. Kalman differentiates the exact marginal log likelihood of the matched predict-first Gaussian model. These quantities differ at finite particle count and numerical resolution. Graph/XLA parity and finite differences check implementation behavior; they do not establish equality with the Kalman score.','',
        'The models have full observations, fixed baseline parameters and Gaussian noise. The full model evaluates derivatives in all A entries and lower-Cholesky Q coordinates, including covariance determinant and inverse-covariance terms. These runs do not evaluate nonlinear targets, maximum-likelihood points, partial observations, TF32 accuracy or HMC. The two-setting flow calibration and bounded smoothing ladder do not establish optimality or numerical protection adequacy. Each route calibrates separately; where selected epsilon differs, a route contrast includes that tuning difference and cannot isolate ancestry ordering. Q scores use the listed lower-Cholesky coordinates (log diagonal and unconstrained off-diagonal entries), not derivatives with respect to raw symmetric Q entries.','',
        '## Decision record','',
        '| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |',
        '|---|---|---|---|---|---|',
        f"| Preserve descriptive comparison | {audit['recorded_final_cells']}/128 final cells recorded; no universal accuracy threshold | See invalid cells and heuristic losses | Two independent data sets; narrow calibration | Diagnose specific failures; predeclare further replication if ranking is wanted | Superiority, default readiness, HMC or production admission |",
        '',
        '| Inference status | Finding |','|---|---|',
        '| Hard veto screen | Invalid candidates and any provenance/oracle failures are explicitly separated |',
        '| Statistically supported ranking | None |',
        '| Descriptive-only differences | All likelihood, score-error, runtime and heuristic comparisons |',
        '| Default readiness | Not assessed; reference-precision variant with limited tuning |',
        '| Next evidence needed | Target-specific numerical calibration and a predeclared multi-dataset uncertainty analysis |','']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='provenance'},indent=2))
    if failures:
        raise RuntimeError('terminal engineering audit failed; preserve and investigate')



def report_rejections(source,output):
    """Preserve actual original calibration outputs, with invalid sentinels labeled."""
    output.mkdir(parents=True,exist_ok=False)
    fields=['scope','route','stage','seed','flow_substeps','reset_epsilon','valid','failure_class','coordinate','oracle_score','raw_returned_score','absolute_error_if_valid','score_role']
    likelihood_fields=['scope','route','stage','seed','flow_substeps','reset_epsilon','valid','failure_class','oracle_value','raw_returned_value','absolute_error_if_valid']
    evaluations=[];score_rows=[];likelihood_rows=[]
    for path in sorted(source.glob('full_*/evaluations.jsonl')):
        case=json.loads((path.parent/'result.json').read_text())
        for line in path.read_text().splitlines():
            event=json.loads(line);row=event['row'];controls=event['controls']
            if event['stage']=='final':continue
            common=dict(scope=case['scope'],route=case['route'],stage=event['stage'],seed=row['seed'],flow_substeps=controls['flow_substeps'],reset_epsilon=controls['reset_epsilon'],valid=row['valid'],failure_class=row.get('failure_class',''))
            raw=row.get('score') if row['valid'] else row.get('raw_score')
            exact=row.get('oracle_score')
            if len(raw)!=case['parameter_count'] or len(exact)!=case['parameter_count']:raise RuntimeError('rejection coordinates missing')
            for name,actual,reference in zip(case['parameter_names'],raw,exact):
                score_rows.append(dict(**common,coordinate=name,oracle_score=reference,raw_returned_score=actual,absolute_error_if_valid=abs(actual-reference) if row['valid'] else '',score_role='actual calibration score' if row['valid'] else 'guard-returned invalid sentinel; not an actual score'))
            value=row.get('value') if row['valid'] else row.get('raw_value')
            likelihood_rows.append(dict(**common,oracle_value=row['oracle_value'],raw_returned_value=value,absolute_error_if_valid=abs(value-row['oracle_value']) if row['valid'] else ''))
            evaluations.append(dict(**common,source=str(path),source_sha256=campaign.digest(path)))
    for filename,columns,rows in [('calibration_scores.csv',fields,score_rows),('calibration_likelihoods.csv',likelihood_fields,likelihood_rows)]:
        with (output/filename).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    campaign.dump(output/'audit.json',dict(evaluations=evaluations,score_entries=len(score_rows),valid_evaluations=sum(r['valid'] for r in evaluations),invalid_evaluations=sum(not r['valid'] for r in evaluations),classification='original finite-program candidate rejection; planned stop, not infrastructure crash'))
    (output/'README.md').write_text('# Original full-model calibration failures\n\nThe original flow grid failed validity checks. These are calibration observations, not untouched final comparisons. Every returned coordinate and matched Kalman score is preserved in calibration_scores.csv. Invalid zero returns are guard sentinels, not actual scores, and have no score-accuracy interpretation. calibration_likelihoods.csv preserves raw returned likelihoods. The intentional stop is documented in ../run-01-intentional-stop.json.\n')
    print(json.dumps(dict(calibration_evaluations=len(evaluations),score_entries=len(score_rows),invalid=sum(not r['valid'] for r in evaluations))))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    report(Path(args.source).resolve(),Path(args.output).resolve())
