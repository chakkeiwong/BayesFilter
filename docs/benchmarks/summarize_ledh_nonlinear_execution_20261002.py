#!/usr/bin/env python3
"""Post-run diagnostic reporting only; never selects runtime controls."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'docs/plans/artifacts/ledh-nonlinear-execution-20261002'
OUT = BASE/'comparison'


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def stats(values):
    return dict(mean=statistics.mean(values),
                se=statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else None)


def main():
    OUT.mkdir(exist_ok=True)
    ladder = read(BASE/'bootstrap-reference-04/summary.json')+read(BASE/'bootstrap-reference-05/summary.json')
    refs = [dict(r,kind='approximate',source='bootstrap-reference-05',
                 verification='fixed-state finite differences; independent exact Kalman fixture; four particle rungs, four replicates each',
                 uncertainty='replication jackknife MCSE; finite-particle bias not certified')
            for r in ladder if r['particles']==524288]
    save(OUT/'references.json',refs)
    rows=[]
    for name in ('screen-01','sir-fp64-02','pp-expanded-03'):
        for worker in sorted((BASE/name).glob('worker-*')):
            job=read(worker/'job.json')
            for row in read(worker/'rows.json'):
                ref=next(r for r in refs if all(r[k]==row[k] for k in
                        ('target_id','horizon','observation_sha256','theta','parameter_names')))
                finite=row['log_likelihood'] is not None and all(v is not None for v in row['score'])
                row=dict(row,run=name,dtype=job['dtype'],source_worker=str(worker.relative_to(ROOT)),
                         finite=finite,reference=ref,oracle_available=False)
                row.update(log_likelihood_error=row['log_likelihood']-ref['log_likelihood'] if finite else None,
                           score_error=[a-b for a,b in zip(row['score'],ref['score'])] if finite else None)
                row['score_l2_error']=math.sqrt(sum(v*v for v in row['score_error'])) if finite else None
                rows.append(row)
    save(OUT/'rows-with-reference.json',rows)
    keys=('model','data_seed','dtype','route','arm')
    groups={}
    for row in rows: groups.setdefault(tuple(row[k] for k in keys),[]).append(row)
    cells=[]
    for key,group in groups.items():
        good=[r for r in group if r['finite']]
        cell=dict(zip(keys,key),returned=len(group),finite=len(good),valid=sum(r['valid'] for r in group),
                  reference=group[0]['reference'],log_likelihood=None,score=None)
        if good:
            cell.update(log_likelihood=stats([r['log_likelihood'] for r in good]),
                        score=[stats([r['score'][k] for r in good]) for k in range(len(good[0]['score']))],
                        mean_absolute_likelihood_error=stats([abs(r['log_likelihood_error']) for r in good]),
                        mean_score_l2_error=stats([r['score_l2_error'] for r in good]))
        cells.append(cell)
    lookup={tuple(r[k] for k in keys)+(r['design_seed'],):r for r in rows}
    comparisons=[]
    for cell in cells:
        if cell['arm'] not in ('richer_pairwise','guarded_pairwise'): continue
        for comparator in ('covariance_only','original','richer_marginal'):
            pairs=[]
            for row in rows:
                if tuple(row[k] for k in keys)!=tuple(cell[k] for k in keys): continue
                other=lookup.get(tuple(row[k] for k in keys[:-1])+(comparator,row['design_seed']))
                if other and row['finite'] and other['finite']:
                    pairs.append((row,other))
            if pairs:
                comparisons.append(dict(**{k:cell[k] for k in keys},comparator=comparator,
                    paired_n=len(pairs),both_valid_n=sum(a['valid'] and b['valid'] for a,b in pairs),
                    absolute_likelihood_error_difference=stats([abs(a['log_likelihood_error'])-abs(b['log_likelihood_error']) for a,b in pairs]),
                    unscaled_score_l2_error_difference=stats([a['score_l2_error']-b['score_l2_error'] for a,b in pairs])))
    save(OUT/'summary.json',dict(cells=cells,conditional_heuristic_comparisons=comparisons,
         heuristic_dominance_verdict='not_established_approximate_reference_and_small_replication_count',
         promotion_vetoes=['SIR_FP32_reset_failure','trace_score_parity_failures','scope_controls_untuned','no_heldout_validation'],
         statistical_ranking='not_established',means_include_finite_flagged_rows=True,
         score_l2_role='descriptive_only_unscaled_parameter_units'))
    with (OUT/'actual-values-and-scores.csv').open('w',newline='') as stream:
        fields=[*keys,'design_seed','valid','finite','quantity','actual','reference','error','reference_jackknife_mcse']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            ref=row['reference']
            for name,value,reference,se in zip(['log_likelihood',*row['parameter_names']],
                    [row['log_likelihood'],*row['score']], [ref['log_likelihood'],*ref['score']],
                    [ref['jackknife_mcse_log_likelihood'],*ref['jackknife_mcse_score']]):
                writer.writerow(dict(**{k:row[k] for k in keys},design_seed=row['design_seed'],valid=row['valid'],finite=row['finite'],
                    quantity=name,actual=value,reference=reference,error=value-reference if row['finite'] else None,
                    reference_jackknife_mcse=se))
    lines=['# Nonlinear LEDH execution — 2 October 2026','',
      'Predator–prey returned finite likelihoods and full scores in all 70 T=20/N=1008 runs; 69 passed the separate trace checks. SIR d=18 failed all ten FP32 runs. Four same-data FP64 SIR evaluations were finite, but their likelihoods were far below the independent particle reference. Double precision repairs this observed numerical failure; it does not repair the downstream approximation.','',
      'These are diagnostic results at the declared truth, with untuned inherited controls. They do not establish a ranking between repair variants, scope admission, or HMC readiness. Every realized score coordinate and its reference are in [actual-values-and-scores.csv](../plans/artifacts/ledh-nonlinear-execution-20261002/comparison/actual-values-and-scores.csv).','',
      '## Experiment and independent reference','',
      'The shared canonical analytical LEDH executor used T=20, N=1008, GPU/XLA/FP32/TF32, and memory growth. Dataset 260401 used IID designs 260501–260502 for all five arms and both models. Predator–prey dataset 260402 additionally used designs 260501–260504 under IID, previous inverse-CDF ancestry and repaired permutation ancestry. Precision diagnostics replayed the exact FP32 observations and random inputs in FP64. A fresh FP32 SIR run preserved already-computed stage checks; no validity threshold or numerical protection changed.','',
      'The independent bootstrap particle filter uses the same initial law, transition-before-observation timing, saved observations and parameter values. Its score is the posterior average of accumulated analytical complete-data scores (Fisher identity), rather than a derivative through discrete resampling. Initial-state derivatives vanish. Gaussian mean and covariance derivatives are both retained. This targets the exact observed-data score as particle count increases; the finite-particle estimate is approximate. LEDH instead differentiates its own finite numerical likelihood. Agreement between those different finite computations is an accuracy question, not an identity.','',
      'The old predator–prey bootstrap fixture observed x0 before transitioning, so it was unsuitable for this comparison. The new reference passed fixed-state derivative finite differences in both models and a separate exact linear-Gaussian Kalman fixture. Four independent replications were run at each of N=8192, 32768, 131072, 524288. Estimates below use log of the mean likelihood and likelihood-weighted Fisher scores. Delete-one-replication jackknife MCSE measures replication noise; it does not measure residual particle bias. The larger rungs were added to resolve the observed reference noise, not to select for agreement with LEDH.','',
      '| Model / data seed | Reference particles | Log likelihood | Jackknife MCSE | Score vector | Score MCSE |',
      '|---|---:|---:|---:|---|---|']
    for ref in ladder:
        lines.append(f"| {ref['model']} / {ref['data_seed']} | {ref['particles']} | {ref['log_likelihood']:.6f} | {ref['jackknife_mcse_log_likelihood']:.6f} | "+', '.join(f'{v:.6g}' for v in ref['score'])+' | '+', '.join(f'{v:.4g}' for v in ref['jackknife_mcse_score'])+' |')
    lines += ['', 'Predator–prey score order is (r, K, a, s, u, v). SIR score order is (log infection-rate scale, log removal-rate scale, log observation-noise scale). At N524288 the minimum particle ESS was 11240 and 34022 on the two PP datasets, and 33055 on SIR; final distinct initial ancestors remained at least 12942, 14116 and 7701 respectively. These diagnostics do not certify reference convergence.', '',
      '## Actual LEDH likelihoods and full scores','',
      'The following means include every finite returned evaluation, including the two explicitly flagged trace-parity failures. Counts separate finiteness from validity. The reference above is approximate, not an oracle. Replication standard errors and individual values are retained in the JSON/CSV; small-seed means are descriptive.','',
      '| Model / data / precision | Ancestry / arm | Valid / finite / returned | Mean log likelihood | Mean score vector |',
      '|---|---|---:|---:|---|']
    for c in cells:
        ell='invalid' if c['log_likelihood'] is None else f"{c['log_likelihood']['mean']:.6f}"
        score='invalid' if c['score'] is None else ', '.join(f"{x['mean']:.6g}" for x in c['score'])
        lines.append(f"| {c['model']} / {c['data_seed']} / {c['dtype']} | {c['route']} / {c['arm']} | {c['valid']}/{c['finite']}/{c['returned']} | {ell} | {score} |")
    lines += ['', 'SIR FP64 rows must also be inspected individually because their score variation is large:', '',
      '| Arm / design seed | Log likelihood | Full score | Validity |','|---|---:|---|---|']
    for row in rows:
        if row['model']=='sir_d18' and row['dtype']=='float64':
            lines.append(f"| {row['arm']} / {row['design_seed']} | {row['log_likelihood']:.6f} | "+', '.join(f'{v:.6f}' for v in row['score'])+f" | {'passed' if row['valid'] else 'trace-score parity veto'} |")
    lines += ['', '## Conditional comparisons with simple arms','',
      'Constructed comparators were covariance-only (avoid higher-moment fitting), original capped correction (existing bounded design), and richer marginal-only correction (omit mixed-moment fitting). Comparisons remain conditional on each dataset and ancestry rule. All 25 available matched comparisons are retained in summary.json. The compact table below shows guarded pairwise versus covariance-only. Positive differences mean greater observed error relative to the approximate reference. Score L2 is unscaled across physical parameter units and explanatory only; the coordinate values above are the substantive evidence.','',
      '| Model / data / ancestry | Matched / both-valid | Change in absolute log-likelihood error | Change in unscaled score L2 error |',
      '|---|---:|---:|---:|']
    for c in comparisons:
        if c['arm']=='guarded_pairwise' and c['comparator']=='covariance_only':
            lines.append(f"| {c['model']} / {c['data_seed']} / {c['route']} | {c['paired_n']}/{c['both_valid_n']} | {c['absolute_likelihood_error_difference']['mean']:.6f} | {c['unscaled_score_l2_error_difference']['mean']:.6f} |")
    lines += ['', 'No heuristic dominance or statistical ranking is established. The guarded method is not uniformly descriptively favorable across these situations. Untuned controls, small replication counts, an approximate reference, and missing untouched validation prohibit promotion. In particular, the SIR likelihood gap is hundreds of log units while the reference varies by hundredths at the largest rung. That observed discrepancy cannot be explained by the measured replication MCSE. A systematic reference implementation error is a separate possibility, bounded here by the derivative and linear-Gaussian checks but not logically excluded.','',
      '## Numerical localization and interpretation','',
      'At zero-based time index 3 (observation 4), both localized FP32 SIR arms fail the Contract E reset numerical check. Ancestry, UKF prediction, UKF update and callback checks pass; predicted/posterior covariances, flow states/tangents and posterior logits are finite. Covariance-only reset states are finite but reset tangents are not. The guarded arm has nonfinite reset states and tangents, and its moment-safety input check fails. The first observed failure is therefore in the reset, before a usable protected moment-correction input exists. This does not identify the exact offending factorization or establish a repair.','',
      'Two additional finite runs fail trace-score consistency without nonfinite outputs: original PP/IID/data260402/design260501 differs by 0.000646591 in score coordinate r; guarded SIR/FP64/design260501 differs by 1.70984e-7 in coordinate log-kappa. Their tolerances were not loosened and both vetoes remain. Directional value calls in the finite evaluations agreed within the recorded checks.','',
      'The SIR failures do not invalidate the research direction or the reference harness. They identify a reset precision defect and a separate approximation/calibration gap. Passing in FP64 does not establish that the FP32 default is repaired. Moment residuals and cap activity remain explanatory diagnostics, not substitutes for likelihood/score agreement.','',
      '## Decision and inference status','',
      '| Decision | Primary criterion | Vetoes | Main uncertainty | Next justified action | Not concluded |',
      '|---|---|---|---|---|---|',
      '| Retain PP variants as diagnostic candidates | Finite likelihood/full scores; approximate reference available | One original-arm trace veto; no per-scope tuning | Few seeds, reference bias, unseen parameters/data | Scope-specific calibration/validation and untouched comparison | Repair superiority or default readiness |',
      '| Reject current SIR FP32 evaluations | Nonfinite reset output/derivative | All ten screening runs invalid | Exact reset suboperation still to localize | Replay observation 4 and inspect reset factorization/derivative margins | Rejection of LEDH or higher-moment repair as a direction |',
      '| Keep SIR FP64 as localization evidence only | Finite outputs but large reference discrepancy | One trace veto; no calibration; large errors | Approximation layers and warm-start controls | Diagnose reset, then calibrate at this exact scope with fresh partitions | FP64 as a new default or accuracy repair |',
      '| Keep bootstrap/Fisher result as approximate reference | Exact target/data timing and derivative checks passed | No hard failure; finite-particle bias unresolved | Four replications per rung | Additional independent nonlinear cross-check before certification | Exact nonlinear oracle |','',
      '| Inference status | Finding |','|---|---|',
      '| Hard veto screen | SIR FP32 reset failure; two finite trace-score parity failures |',
      '| Statistically supported ranking | None |',
      '| Descriptive-only differences | All means, score errors, paired arm differences and runtime comparisons |',
      '| Default readiness | Not established; canonical implementation/default direction unchanged |',
      '| Next evidence needed | Reset repair with unchanged healthy trajectories; per-scope calibration; replicated held-out accuracy and reference-bias assessment |','',
      'The next SIR repair should first reproduce observation 4 with the same source cloud and derivative, measure reset covariance/factorization conditioning and compare FP32/FP64 intermediate values. Evaluate any numerics-altering protection against non-harm and finite, flagged behavior before accuracy tuning. Then calibrate correction/OT controls separately on new data at the exact target scope; do not tune on these diagnostic datasets and call them untouched validation.','',
      'Engineering checks: 25 focused CPU reference tests passed, including exact data replay/hash rejection, both nonlinear complete-data score finite differences, Kalman comparison, shared canonical endpoint, and trace/no-trace parity. CPU devices were intentionally hidden. GPU runs used the recorded memory-growth policy and fresh output directories. The new reference uses analytical derivatives and no pfor, resampling autodiff, or NumPy numerical path.','',
      'Post-run red team: the strongest alternative explanation is shared model-adapter error, since the reference and LEDH share the model definition. The exact Kalman fixture and fixed-state finite differences check the estimator and derivatives, not every intended nonlinear modeling assumption. An independent nonlinear reference disagreeing with this ladder would overturn the accuracy interpretation. The weakest evidence is repair ranking across only two/four design seeds at one parameter point; no ranking is claimed.','',
      'Plan: [ledh-nonlinear-execution-20261002.md](../plans/ledh-nonlinear-execution-20261002.md). Full artifacts: [comparison summary](../plans/artifacts/ledh-nonlinear-execution-20261002/comparison/summary.json), [all reference-attached rows](../plans/artifacts/ledh-nonlinear-execution-20261002/comparison/rows-with-reference.json).']
    (ROOT/'docs/benchmarks/ledh-nonlinear-execution-results-20261002.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'rows':len(rows),'cells':len(cells),'conditional_comparisons':len(comparisons)}))
    for c in cells:
        if c['log_likelihood'] and c['arm'] in ('covariance_only','guarded_pairwise'):
            print({k:c[k] for k in (*keys,'log_likelihood','score')})


if __name__=='__main__':
    main()
