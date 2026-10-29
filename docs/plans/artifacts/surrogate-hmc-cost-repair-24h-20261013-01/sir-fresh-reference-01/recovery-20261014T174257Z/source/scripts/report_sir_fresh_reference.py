#!/usr/bin/env python3
"""Post-run diagnostic report, not inference or selection machinery."""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sir_fresh_reference as shared


def quantities(row):
    return [row['log_likelihood'], *row['score']]


def mcse(row):
    return [row['jackknife_mcse_log_likelihood'], *row['jackknife_mcse_score']]


def assess(reference, previous, arm):
    values = quantities(reference)
    shift = [abs(x-y) for x,y in zip(values,quantities(previous))]
    uncertainty = [3*s+d for s,d in zip(mcse(reference),shift)]
    tolerance = [.01*max(1,abs(x)) for x in values]
    delta = [a-b for a,b in zip([arm['value'],*arm['score']],values)]
    return dict(signed_error=delta,reference_mcse=mcse(reference),same_method_shift=shift,
        uncertainty_allowance=uncertainty,materiality_tolerance=tolerance,
        materiality_screen=[abs(d)>u+t for d,u,t in zip(delta,uncertainty,tolerance)],
        resolved_vs_mcse_and_shift=[abs(d)>u for d,u in zip(delta,uncertainty)],
        score_relative_error_percent=[100*abs(d/v) for d,v in zip(delta[1:],values[1:])],
        likelihood_ratio=math.exp(delta[0]))


def report():
    state=shared.read(shared.OUTPUT/'state.json')
    if any(r['status']=='RUNNING' for r in state['runs']):
        raise ValueError('a reference worker is still running')
    inputs=shared.read(shared.OUTPUT/'comparison_inputs.json')
    if shared.digest(Path(inputs['source'])) != inputs['source_sha256']:
        raise ValueError('saved comparison changed')
    groups={'bootstrap':[],'trajectory':[]}
    for run in state['runs']:
        if run['status'] != 'COMPLETE':
            continue
        method=run.get('method','bootstrap')
        path=Path(run['output'])
        summary=shared.read(path/'summary.json')
        if len(summary)!=1:
            raise ValueError('one exact scope expected')
        row=dict(summary[0],output=str(path),wall_seconds=run['wall_seconds'])
        groups[method].append(row)
        if method=='trajectory':
            manifest=shared.read(path/'manifest.json')
            if manifest['observation_sha256']!=inputs['observation_sha256'] or manifest['theta']!=inputs['theta']:
                raise ValueError('reference input mismatch')
            for name,digest in manifest['source_sha256'].items():
                if shared.digest(ROOT/name)!=digest:
                    raise ValueError('reference source changed: '+name)
    for rows in groups.values():
        rows.sort(key=lambda x:x['particles'])
    if len(groups['trajectory'])<2 or not groups['bootstrap']:
        raise ValueError('two trajectory rungs and bootstrap check required')
    ref,prev=groups['trajectory'][-1],groups['trajectory'][-2]
    boot=groups['bootstrap'][-1]
    delta=[a-b for a,b in zip(quantities(ref),quantities(boot))]
    se=[math.hypot(a,b) for a,b in zip(mcse(ref),mcse(boot))]
    arms={arm:assess(ref,prev,inputs[arm]) for arm in ['baseline','candidate']}
    partial=shared.read(shared.OUTPUT/'rung-1048576-attempt-03/replications.json')
    budget=shared.read(shared.CAMPAIGN/'budget.json')
    result=dict(schema='sir_exact_point_reference_comparison_v1',
        status='COMPLETE_WITH_RESOLVED_SCORE_DISCREPANCIES',comparison=inputs,reference_methods=groups,
        reference_role='finite_MC_model_reference_not_exact_oracle',final_reference=ref,arms=arms,
        cross_method=dict(difference=delta,combined_mcse=se,mcse_units=[abs(d)/s for d,s in zip(delta,se)]),
        interrupted_bootstrap=dict(status='INTERRUPTED_FOR_VARIANCE_REPAIR',
            completed_replications=len(partial),rows=[{k:r[k] for k in ['seed','log_likelihood','score']} for r in partial]),
        diagnostic_wall_seconds=state['used_seconds'],remaining_campaign_seconds=budget['remaining_seconds'],
        conclusions=dict(old_score_error='resolved for all three components',
            new_score_error='resolved for kappa and observation noise; nu unresolved',
            default_promotion=False,posterior_impact='not measured',implementation_bug='not established'))
    prior=shared.OUTPUT/'result.json'
    archive=shared.OUTPUT/'bootstrap-only-result.json'
    if prior.exists() and not archive.exists():
        archive.write_bytes(prior.read_bytes())
    shared.save(prior,result)
    labels=['Log likelihood','Score: log kappa scale','Score: log nu scale','Score: log observation-noise scale']
    lines=['# SIR reference at the 1,482.48% discrepancy point','',
        'The old OT score is materially inaccurate relative to the model score at this point. The five-step configuration is closer, but its kappa and observation-noise components still fail the declared materiality screen. This diagnoses accuracy error; it does not establish a coding bug or posterior error.','',
        'Exact scope: fresh dataset 26101401, T=50, 18 states, nine observations, theta=(-0.0175,-0.0175,-0.0175) in the three log-scale coordinates. Saved observations and model-source hashes match. Prior gradients are excluded. Both LEDH arms use N=4032, 16 flow steps, FP64 recursion with FP32 pair computations and TF32 off. Both epsilon and reset iteration count change; their individual effects are not isolated.','',
        '| Quantity | Old epsilon=0.5, 40 steps | New epsilon=1, 5 steps | Reference | MCSE |',
        '|---|---:|---:|---:|---:|']
    for label,a,b,v,s in zip(labels,[inputs['baseline']['value'],*inputs['baseline']['score']],
        [inputs['candidate']['value'],*inputs['candidate']['score']],quantities(ref),mcse(ref)):
        lines.append('| %s | %.6f | %.6f | %.6f | %.6f |'%(label,a,b,v,s))
    lines += ['', '## Independent references', '',
        'The final reference uses 16 independent replications of 131,072 Gaussian-smoothing trajectory proposals (2,097,152 paths). Exact transition-density importance weights correct the proposal, and the analytical complete-data Fisher score targets the model score. The Gaussian approximation changes the proposal only. Finite Monte Carlo error, ratio-estimator bias and unseen importance-weight tails remain; this is not an exact oracle.','',
        '| Method | N | Replications | Log likelihood (MCSE) | Scores | Score MCSE |',
        '|---|---:|---:|---|---|---|']
    for method,rows in groups.items():
        for row in rows:
            scores=', '.join('%.3f'%x for x in row['score'])
            ses=', '.join('%.3f'%x for x in row['jackknife_mcse_score'])
            lines.append('| %s | %d | %d | %.6f (%.6f) | %s | %s |'%(method,row['particles'],row['replications'],row['log_likelihood'],row['jackknife_mcse_log_likelihood'],scores,ses))
    lines += ['', 'Final trajectory minus complete bootstrap differences, in combined MCSE units (value, kappa, nu, noise): '+', '.join('%.2f'%x for x in result['cross_method']['mcse_units'])+'. These references are compatible at this resolution. Both share model callbacks. Seven focused tests passed: data/coordinate checks, SIR/predator-prey fixed-state score finite differences, an exact Kalman fixture, and linear/nonlinear trajectory quadrature fixtures. The nonlinear fixture checks both likelihood and score. This does not independently verify the epidemiological specification.','',
        'Final-rung minimum trajectory ESS: %.1f/%d; maximum normalized weight: %.6f; replication likelihood ESS: %.4f/16. All returned values and covariance factors were finite. Observed ESS does not bound unseen tails.'%(ref['minimum_importance_ess'],ref['particles'],ref['maximum_weight'],ref['between_replication_likelihood_ess']),'',
        'The million-particle bootstrap attempt was deliberately interrupted for variance/cost repair after two saved replications. This is not a numerical failure. Both rows and the original files are preserved; the incomplete rung is excluded from complete-rung certification.','',
        '## Materiality and uncertainty','',
        'Predeclared resolved-error screen: |arm-reference| > 0.01 max(1,|reference|) + 3 MCSE + |last same-method ladder shift|. This is a diagnostic screen, not a simultaneous confidence guarantee or posterior-impact bound.','',
        '| Score | Old absolute error | New absolute error | Threshold | New verdict |',
        '|---|---:|---:|---:|---|']
    old,new=arms['baseline'],arms['candidate']
    for i,label in enumerate(labels[1:],1):
        threshold=new['uncertainty_allowance'][i]+new['materiality_tolerance'][i]
        verdict='Resolved discrepancy' if new['materiality_screen'][i] else 'Unresolved; not certified within 1%'
        lines.append('| %s | %.4f | %.4f | %.4f | %s |'%(label,abs(old['signed_error'][i]),abs(new['signed_error'][i]),threshold,verdict))
    lines += ['', 'The 1,482.48% change divides by the old nu score near zero. That old value was itself far from the model score. This percentage therefore does not demonstrate catastrophic degradation from using five iterations. New nominal relative score errors are '+', '.join('%.1f%%'%x for x in new['score_relative_error_percent'])+'; these remain uncertain, particularly nu.','',
        'Old/new log-likelihood errors: %.6f and %.6f; likelihood ratios to the reference: %.6f and %.6f. Both gaps exceed Monte Carlo uncertainty, but pass the literal 1%%-of-absolute-log-likelihood convention (about 16.76). This convention hides differences in likelihood ratios. No criterion has been silently changed. A theta-constant additive log-likelihood error cancels from the posterior; posterior impact requires multiple parameter points.'%(old['signed_error'][0],new['signed_error'][0],old['likelihood_ratio'],new['likelihood_ratio']),'',
        '## Decision and inference status','',
        '| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |',
        '|---|---|---|---|---|---|',
        '| Old score inaccurate here | Three resolved errors | Accuracy blocks promotion | Finite reference error | Preserve failed comparison | Algorithm-wide failure |',
        '| Five-step setting closer but inaccurate | Kappa/noise resolved; nu unresolved | Accuracy blocks promotion | MCSE and unseen tails | Existing master: same-point value/score consistency, then scoped tuning | HMC or posterior validity |',
        '| Reference subtask complete | Independent methods compatible | No input/numerical veto | No exact nonlinear oracle | Report; no additional reference launches | Global superiority |','',
        '| Inference status | Finding |','|---|---|',
        '| Hard veto screen | No numerical/reference validity failure; six-launch diagnostic cap reached. |',
        '| Statistically supported ranking | New score components are conditionally closer at this exact point; no cross-dataset ranking. |',
        '| Descriptive-only differences | Nominal percentage errors, runtime, ESS and likelihood ratios; posterior impact unmeasured. |',
        '| Default readiness | Neither configuration receives accuracy or HMC promotion. |',
        '| Next evidence needed | Same-program value/score parity, disjoint scoped tuning, independent checks over the posterior region. |','',
        'The next repair belongs to the existing computation/HMC master. This reference subtask is complete; the larger master is not complete. Production defaults are unchanged.','',
        'Post-run red team: shared model and derivative callbacks are the strongest common-error explanation. Finite differences and independent fixtures reduce that risk but do not prove source-model faithfulness. An independent reference outside the reported uncertainty, or a same-program derivative mismatch, would change the interpretation. Kappa/nu precision and unseen weight tails are the weakest evidence.','',
        '## Execution provenance','',
        'Diagnostic worker wall time: %.3f seconds (%.3f hours), including interruption. Trajectory rungs: %.3f seconds. Campaign remaining: %.3f GPU hours.'%(state['used_seconds'],state['used_seconds']/3600,sum(r['wall_seconds'] for r in groups['trajectory']),budget['remaining_seconds']/3600),'',
        'RTX 4080 SUPER; TensorFlow 2.20.0-dev0+selfbuilt; FP64/XLA, TF32 off, verified memory growth. Final-rung allocator peak: 1,314,576,128 bytes. Commands, seeds, hashes and timings are in per-run manifests and state.json. FP64 is a reference exception; FP32/N4032 remains the owner testing direction.','',
        'Plan: docs/plans/sir-fresh-point-reference-2026-10-14.md. Executors: scripts/run_sir_fresh_reference.py and scripts/run_sir_trajectory_reference.py. Machine report: result.json. Earlier interpretation: bootstrap-only-result.json.','']
    (shared.OUTPUT/'result.md').write_text(chr(10).join(lines))
    print(json.dumps(dict(status=result['status'],reference=quantities(ref),mcse=mcse(ref),
        old_screen=old['materiality_screen'],new_screen=new['materiality_screen'],result=str(shared.OUTPUT/'result.md'))))
    return result


if __name__=='__main__':
    report()
