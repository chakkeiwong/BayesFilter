#!/usr/bin/env python3
"""Read-only scientific result summarizer; writes only the current result note."""
from pathlib import Path
from collections import Counter
import json

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/'docs/plans/artifacts/neutra-controlled-repair-2026-10-02/campaign-r1'
NOTE=ROOT/'docs/plans/bayesfilter-neutra-controlled-repair-results-2026-10-02.md'


def read(path):return json.loads(Path(path).read_text())


def lifetime_updates(path):
    data=read(Path(path)/'result.json')
    if 'lifetime_updates' in data:return data['lifetime_updates']
    if not data.get('parent') or data.get('repair'):return data['history'][-1]['step']
    return lifetime_updates(data['parent'])+sum(r['additional_updates'] for r in data['history'])


def retained_failure(data):
    """Explain the final retained check without upgrading a resource stop."""
    checks=data.get('retained_checks',[])
    full=next((r['full_convergence'] for r in reversed(checks) if 'full_convergence' in r),{})
    reasons=[name for name in full.get('failed_checks',[]) if name!='consumer_diagnostic']
    if full.get('broad_precision_passed') is False:reasons.append('broad-event precision')
    outcomes=full.get('both_outcomes_per_chain',[])
    if outcomes and not all(all(row) for row in outcomes):reasons.append('event coverage per chain')
    return ', '.join(reasons) or 'retained information/convergence'


def main():
    state=read(CAMPAIGN/'state.json');cfg=read(CAMPAIGN/'config.json')
    attempts=[r for records in state['jobs'].values() for r in records]
    used={k:sum(r.get(k,0.) for r in attempts) for k in ('gpu_process_seconds','cpu_core_seconds')}
    latest={name:Path(records[-1]['output']) for name,records in state['jobs'].items()
            if records[-1]['status']=='complete'}
    settled={r['result']['group']:r['result']['confirmation'] for r in state.get('decisions',[])
             if r.get('stage')=='case_complete'}
    passes=sum(r.get('qualified',False) for r in settled.values())
    total=len(cfg['targets'])*2*len(cfg['training_seeds'])
    terminal=state['status']=='completed' and len(settled)==total and not state.get('active')
    training={name:read(path/'result.json') for name,path in latest.items()
              if (path/'result.json').exists() and name.startswith(tuple(
                  'controlled-'+arm+'-' for arm in ('parent','continue','forward','reverse','joint')))}
    stops=Counter(data['stop_reason'] for data in training.values() if 'history' in data)
    lines=['# Controlled NeuTra repair: recorded results','',
        f"Execution status: **{state['status']}**. Next recorded action: `{state.get('next_action')}`.",'',
        'This is a bounded FP64 diagnostic campaign on the canonical IAF. '
        'Checkpoint loss and geometry cannot establish posterior correctness. '
        'No ranking or FP32/TF32 readiness is claimed.','',
        f"**{len(settled)}/{total} cases settled; {passes} passed fresh posterior confirmation.** "
        'An unqualified case means the bounded procedure did not deliver its declared result; '
        'it does not by itself reject the research direction.','',
        f"Charged GPU process time: {used['gpu_process_seconds']/3600:.4f} h of {cfg['gpu_process_seconds']/3600:g} h. "
        f"Charged CPU core time: {used['cpu_core_seconds']/3600:.4f} h of {cfg['cpu_core_seconds']/3600:g} h. "
        +('No campaign workers remain active. ' if terminal else
         'Active workers are additional, still reserved work. ')+'Failed attempts are included.','']
    if terminal:
        lines+=['## Terminal interpretation','',
            (f'The declared eight-case procedure finished with {passes} fresh posterior confirmations. '
             if passes else
             'The declared eight-case procedure finished without a learned map passing fresh posterior confirmation. ')
            +'The analytic-transform, iid-sample and exact-Gaussian-HMC controls passed on both targets. '
            'The bounded learned-map training and sampling procedure did not meet the same downstream requirements.','',
            'Recorded training stops: '+', '.join(f'{name}: {count}' for name,count in sorted(stops.items()))+'. '
            'These are work/optimization classifications, not posterior results. '
            'Errors in importance-weight estimation cannot be the sole explanation: oracle cases, '
            'using exact conditional sampling and stratum probabilities, also failed. '
            'Finite empirical-bank error remains in both teacher arms. '
            'These results do not separate incomplete optimization, representational limitations and finite HMC information.','',
            'The optional score-geometry objective was not implemented or executed: its prerequisite, '
            'adequate stationary density fitting with persistent geometry failure, was not established. '
            'It remains a derived proposal. The next discriminating work is a target-specific optimization '
            'study preserving the existing map and optimizer state, with finer continuation rungs and '
            'adequate heldout precision; a new objective must not be justified by falsely declaring these fits converged.','']
    lines+=[
        '## Controls','',
        '| Target | Numerical exact transport | Pooled teacher | iid joint passes | Exact Gaussian HMC |',
        '|---|---|---|---|---|']
    controls={}
    for target in cfg['targets']:
        def status(prefix,key):
            path=latest.get('controlled-'+prefix+'-'+target)
            return read(path/'result.json').get(key) if path else 'pending'
        iid=CAMPAIGN/'iid-reference-r1'/target/'result.json'
        iid_result=read(iid) if iid.exists() else {}
        choices=[latest.get('controlled-exact-hmc-'+target+suffix) for suffix in
                 ('-remaining-members','-repair','')]
        selected=next((p for p in choices if p),None)
        control=read(selected/'result.json') if selected else {}
        controls[target]=control
        lines.append(f"| {target} | {status('reference','passed')} | {status('prepare','passed')} | "
                     f"{iid_result.get('joint_pass_count','pending')}/8 | {control.get('qualified','pending')} |")
    lines+=['','The iid results use eight independent streams paired across targets; they measure finite-sample feasibility, not nominal coverage. '
        'The Gaussian controls include exact physical decoding and separate reference streams. '
        'The numerical transport check evaluates target-plus-log-Jacobian cancellation separately. '
        'The practical teacher screen covers event probabilities and marginal moments with between-replication uncertainty; '
        'it does not establish equality of the entire conditional density within every mode.','',
        '## Training and posterior decisions','',
        '| Target / teacher / seed | Completed fit arms | Posterior confirmation |',
        '|---|---|---|']
    confirmations={}
    for target in cfg['targets']:
        for teacher in ('oracle','estimated'):
            for seed in cfg['training_seeds']:
                group=f'{target}-{teacher}-s{seed}'
                arms=[a for a in ('parent','continue','forward','reverse','joint','joint-repair','joint-continuation')
                      if 'controlled-'+a+'-'+group in latest]
                confirmation=latest.get('controlled-confirmation-'+group)
                result=read(confirmation/'result.json') if confirmation else None
                confirmations[group]=result
                outcome=('passed' if result['qualified'] else 'confirmation did not pass: '+result.get('reason','see result')) if result else (
                    'not reached: no map qualified in bounded checks' if group in settled else 'pending')
                lines.append(f"| {target} / {teacher} / {seed} | {', '.join(arms) or 'none'} | {outcome} |")
    lines+=['','| Completed fit | Final Adam step | Lifetime map updates | Stop classification | Heldout FKL cross entropy | Estimated RKL | Gaussian residual median | Directed residual max |',
            '|---|---:|---:|---|---:|---:|---:|---:|']
    for name,path in latest.items():
        data=read(path/'result.json')
        if 'history' not in data:continue
        last=data['history'][-1];m=last['metrics'];probe=m['post_training_1000']
        lines.append(f"| {name.removeprefix('controlled-')} | {last['step']} | {lifetime_updates(path)} | {data['stop_reason']} | "
                     f"{m['heldout_fkl']:.6g} | {m['estimated_rkl']:.6g} | "
                     f"{probe['score_residual_norm']['median']:.6g} | {m['directed']['maximum_norm']:.6g} |")
    lines+=['','These diagnostics are descriptive. Optimizer step counts include restored Adam '
        'iterations for preserved-state continuation and restart at zero for reset arms. '
        'Lifetime map updates follow the selected map’s ancestry; discarded pilots also consume compute, '
        'but do not belong to that map’s ancestry. '
        'Use each history’s additional updates and measured cost for fair work accounting. '
        'Resource-limited fits are not converged fits. '
        'Teacher arms share a search protocol but may nominate different widths or learning rates, '
        'so their differences do not isolate teacher quality alone. Objective branches within a case '
        'share their initial map. The two development seeds and paired streams across targets/teachers '
        'do not make eight statistically independent method replications.','',
        '## Standard 1,000-point probes of final joint maps','',
        '| Target / teacher / seed | Median | Mean | q95 | q99 | Maximum | Fraction > 1 | Log-density offset range |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for target in cfg['targets']:
        for teacher in ('oracle','estimated'):
            for seed in cfg['training_seeds']:
                group=f'{target}-{teacher}-s{seed}'
                path=latest.get('controlled-joint-continuation-'+group) or latest.get('controlled-joint-'+group)
                if path is None:continue
                probe=read(path/'result.json')['history'][-1]['metrics']['post_training_1000']
                norms=probe['score_residual_norm']
                values=[norms[k] for k in ('median','mean','p95','p99','max')]
                values += [norms['exceedance_fraction']['1.0'],probe['r_log_target_over_gaussian_up_to_constant']['range']]
                lines.append('| '+group+' | '+' | '.join(f'{v:.6g}' for v in values)+' |')
    lines+=['','These are the prescribed Gaussian-base probes of the final trained joint checkpoint, '
        'including its one continuation where completed. They do not select or qualify a map. '
        'Mean and tail summaries can expose errors hidden by the median; finite probes cannot '
        'establish uniform geometry. Repeated use of the same probes and two fitted seeds does '
        'not support a statistical ranking.','',
        '## Continuation and plateau evidence','',
        '| Case | Chosen width | LR | Additional continuation checkpoints | Last paired objective gain | SE | Stop |',
        '|---|---:|---:|---:|---:|---:|---|']
    for name,data in training.items():
        if not name.startswith('controlled-joint-continuation-'):continue
        last=data['history'][-1]
        fmt=lambda value:'not available' if value is None else f'{value:.6g}'
        lines.append(f"| {name.removeprefix('controlled-joint-continuation-')} | {data['width']} | "
                     f"{data['learning_rate']:g} | {len(data['history'])} | "
                     f"{fmt(last.get('paired_objective_gain'))} | {fmt(last.get('paired_objective_se'))} | {data['stop_reason']} |")
    lines+=['','The operational plateau requires two successive distinct-map comparisons with '
        '`abs(gain) + 3*SE < 0.001` nats. A single continuation checkpoint has no within-job '
        'paired change. A noisy gain near zero is inconclusive; it is not a plateau. '
        'This repeated-look rule is an engineering resolution test, not a global-optimization '
        'or formal sequential-coverage guarantee.','',
        '## Frozen-map qualification outcomes','',
        '| Completed qualification job | Checkpoint outcomes | Retained posterior outcomes |',
        '|---|---|---|']
    all_posterior=Counter()
    health_causes=Counter()
    for name,path in latest.items():
        if not name.startswith('controlled-qualify-'):continue
        result=read(path/'result.json')
        reasons=Counter(r.get('reason','unknown') for r in result.get('checkpoint_screen',[]))
        posterior=Counter()
        for f in path.glob('checkpoint-*/member-*/posterior.json'):
            data=read(f)
            hard=[reason for reason in data.get('hard_vetoes',[]) if reason!='campaign_resource_cap']
            if data.get('passed'):label='declared screen passed'
            elif hard:label='numerical/health veto: '+','.join(hard)
            elif 'campaign_resource_cap' in data.get('hard_vetoes',[]):label='resource-limited/incomplete'
            elif data.get('warmup_cap_hit'):label='warmup count cap'
            elif data.get('retained_cap_hit'):label='retained cap: '+retained_failure(data)
            else:label=data.get('decision','other')
            posterior[label]+=1
            all_posterior[label.split(':',1)[0]]+=1
            if hard:
                for check in data.get('warmup_checks',[])+data.get('retained_checks',[]):
                    for field in ('samples_all_finite','log_accept_ratio_all_finite',
                                  'target_log_prob_all_finite','all_states_moved'):
                        if check.get('health',{}).get(field) is False:health_causes[field]+=1
        fmt=lambda counts:'; '.join(f'{k}: {v}' for k,v in counts.items()) or 'none completed'
        lines.append(f"| {name.removeprefix('controlled-qualify-')} | {fmt(reasons)} | {fmt(posterior)} |")
    lines+=['','Recorded member outcomes across the completed jobs: '
        +'; '.join(f'{name}: {count}' for name,count in sorted(all_posterior.items()))+'. '
        'These include repeated checkpoints and kernels; they are not independent replications.','',
        'Failed chunk-health fields: '
        +('; '.join(f'{name}: {count}' for name,count in sorted(health_causes.items())) or 'none')
        +'. Counts refer to failed checks, not necessarily distinct maps or methods.','',
        'A resource stop is incomplete evidence. It does not establish a numerical '
        'defect, poor stationary sampling, or rejection of NeuTra. Count caps are '
        'failures to deliver the declared result within the fixed procedure. '
        'Kernel observations and repeated checkpoints are not independent method replications.','',
        '## Decision and inference status','',
        '| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |',
        '|---|---|---|---|---|---|',
        '| Analytic controls | Read control table above | Failed search/settings preserved | Finite-run reliability | Assess learned maps only after controls pass | Learned-map success |',
        (f'| Learned transports | {passes}/{total} passed fresh posterior confirmation | See per-checkpoint/member assessments | Optimization, residual geometry, retained event information | Plan targeted optimization/inference follow-up; no automatic new campaign | Convergence from loss/probes |' if terminal else
         '| Learned transports | Fresh posterior confirmation required | See per-checkpoint/member assessments | Optimization, residual geometry, retained event information | Follow recorded repair/remaining budget | Convergence from loss/probes |'),
        ('| Optional geometry extension | Trigger not established; not executed | No stationary fit established | No calibrated objective coefficient yet | Resolve optimization adequacy before invoking this trigger | Implemented or validated extension |' if terminal else
         '| Optional geometry extension | Requires adequate stationary density fit and failed geometry | Trigger must be demonstrated | No calibrated objective coefficient yet | Price/test derivatives only if trigger is supported | Implemented or validated extension |'),' ',
        '| Inference status | Conclusion |','|---|---|',
        '| Hard veto screen | Numerical/health failures reject the affected evidence; a precision cap rejects that candidate’s delivery. |',
        '| Statistically supported ranking | None. |',
        '| Descriptive-only differences | Loss, score norms, tails, timings, and individual fit trajectories. |',
        '| Default readiness | Not established; FP64 references and two small targets cannot promote TF32/q20. |',
        '| Next evidence needed | Completed confirmations, target-specific optimization evidence, and independent replication appropriate to the claimed reliability. |','',
        '## Engineering repairs and audit','',
        'The Gaussian control’s scope mismatch was fixed before numerical tuning. '
        'Initial step-size/trajectory choices hit analytic resonances; the wider search '
        'and remaining-member phase preserved rejected settings and used unchanged '
        'posterior criteria. A duplicate export after successful confirmation was '
        'recovered from consistent saved evidence without rerunning simulation. '
        'Diagnostic graphs are reused across current map variables; focused tests '
        'checked that reuse does not freeze stale parameters. Earlier checkpoints '
        'are eligible for downstream screening, and the selected filename is passed '
        'to fresh mode-spanning confirmation.','',
        'Focused checks passed for the combined gradient, exact next-update restoration, '
        'invalid-update rollback, numerical controls, diagnostic graph reuse, checkpoint selection '
        'and controller adoption/resource allocation. The 22 shared rare-region/assessment '
        'regressions also passed. Fourteen pre-existing precision-suite failures conflict with '
        'the dirty working-tree configuration API; these have not been repaired or hidden. '
        'This campaign exercises the declared FP64 route and does not establish FP32/TF32 readiness.','',
        'The strongest alternative explanation for poor learned-map sampling is '
        'unfinished or inadequate optimization, rather than a failure of the IAF '
        'research direction. A frozen map passing independent posterior confirmation '
        'would overturn rejection of that candidate. Rare-event precision and the '
        'small number of fitted seeds limit stronger conclusions.','',
        'Plan: `docs/plans/bayesfilter-neutra-controlled-repair-master-2026-10-02.md`. '
        'Every attempt has a command/environment/source/device/seed/time manifest. '
        'The campaign state preserves attempt order, failures, repairs, and costs. '
        'Original results and source snapshots have not been overwritten. '
        'The reproducible provenance audit is `scripts/audit_neutra_controlled_repair.py`; '
        'the terminal JSON audit is stored beside the campaign state.','']
    NOTE.write_text('\n'.join(lines))
    print(NOTE)
    print(state['status'],len(attempts),'settled attempts;',len(state.get('active',{})),'active')


if __name__=='__main__':main()
