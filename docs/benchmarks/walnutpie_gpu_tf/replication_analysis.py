"""Post-run diagnostic analysis only; NumPy memmaps never drive the sampler."""
import json
import math
from pathlib import Path
from statistics import NormalDist
import numpy as np
from .replication import ARMS, write, sha


def read_draws(directory, arm, n):
    if arm.startswith('author-'):
        paths=[directory/('chain-%d-draws.f64'%i) for i in range(4)]
        if any(p.stat().st_size!=n*11*8 for p in paths):raise ValueError('incorrect reference draw bytes')
        return [np.memmap(p,dtype='<f8',mode='r',shape=(n,11)) for p in paths]
    path=directory/'draws.f64'
    if path.stat().st_size!=n*4*11*8:raise ValueError('incorrect TF draw bytes')
    x=np.memmap(path,dtype='<f8',mode='r',shape=(n,4,11))
    return [x[:,i,:] for i in range(4)]


def batch_summary(x, blocks=100):
    # Diagnostic-only independent batch means, chain-major.
    size=x.shape[1]//blocks
    if size<2:blocks=16;size=x.shape[1]//blocks
    means=x[:,:blocks*size].reshape(4,blocks,size).mean(axis=2)
    return dict(mean=float(x.mean()),batch_se=float(means.std(ddof=1)/math.sqrt(means.size)),
        blocks_per_chain=blocks,batch_size=size,chain_means=x.mean(axis=1).tolist())


def summarize(tf, chains, discard=0):
    from .worker import diagnostics
    n=len(chains[0])-discard
    result=dict(draws=n,discard=discard,coordinates=[],transformed=[],finite=True,movement=True)
    v=np.stack([c[discard:,0] for c in chains])
    for j in range(11):
        raw=np.stack([c[discard:,j] for c in chains])
        z=raw/3. if j==0 else raw*np.exp(-.5*v)
        result['finite'] &= bool(np.all(np.isfinite(raw)) and np.all(np.isfinite(z)))
        result['movement'] &= bool(np.all(np.any(np.diff(raw,axis=1)!=0,axis=1)))
        for name,x in (('coordinates',raw),('transformed',z)):
            d=diagnostics(tf,tf.convert_to_tensor(x.T[:,:,None],tf.float64))
            d.update(index=j,mean_summary=batch_summary(x),variance=float(x.var()),
                batch50=batch_summary(x,50),quantiles=np.quantile(x,[.001,.01,.05,.5,.95,.99,.999]).tolist())
            result[name].append(d)
        print(json.dumps(dict(discard=discard,coordinate=j,draws=n)),flush=True)
    both=result['coordinates']+result['transformed']
    result.update(max_rhat=max(d['max_rhat'] for d in both),
        min_bulk_ess=min(d['min_bulk_ess'] for d in both),min_tail_ess=min(d['min_tail_ess'] for d in both),
        v_range=[float(v.min()),float(v.max())])
    result['tail']={str(t):dict(**batch_summary((v<t).astype(float)),
        analytic=NormalDist(0,3).cdf(t)) for t in (-3,-6)}
    result['posterior_screen']=bool(result['finite'] and result['movement'] and
        result['max_rhat']<=1.01 and min(result['min_bulk_ess'],result['min_tail_ess'])>=400)
    result['uncertainty_note']='Batch intervals assume stationarity and adequate block length; 50/100-block sensitivity is descriptive.'
    return result


def histogram_svg(out,cases):
    colors=['#9c412f','#2b6a99','#417e3b','#7548a4']
    lines=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="580" viewBox="0 0 1000 580">',
        '<rect width="1000" height="580" fill="white"/>',
        '<text x="60" y="30" font-family="sans-serif" font-size="20">Centered funnel: v density, %s iterations per chain</text>'%cases[0]['diagnostics']['draws'],
        '<text x="60" y="555" font-family="sans-serif" font-size="14">Exploration diagnostic; see REPORT.md for convergence and uncertainty.</text>']
    def poly(x,y,color):
        pts=' '.join('%.2f,%.2f'%(60+(a+12)*35,470-b*2500) for a,b in zip(x,y))
        return '<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>'%(color,pts)
    grid=np.linspace(-12,12,241);lines.append(poly(grid,np.exp(-grid*grid/18)/(3*math.sqrt(2*math.pi)),'#222222'))
    for i,case in enumerate(cases):
        h=case['v_histogram'];lines.append(poly(h['centers'],h['density'],colors[i]))
        lines.append('<text x="660" y="%d" fill="%s" font-family="sans-serif" font-size="14">%s</text>'%(65+i*22,colors[i],case['arm']))
    for tick in (-12,-6,0,6,12):
        lines.append('<text x="%d" y="495" font-family="sans-serif">%d</text>'%(60+(tick+12)*35,tick))
    lines.append('</svg>');(out/'v-density.svg').write_text('\n'.join(lines))


def analyze(out):
    from .worker import setup
    tf,env=setup('cpu');cases=[];missing=[]
    manifest=json.loads((out/'manifest.json').read_text())
    evaluation=manifest.get('reference_evaluator','unmodified author targetDistr.funnel10 (scipy)')
    for arm in ARMS:
        directory=out/arm;path=directory/'result.json'
        if not path.exists():missing.append(dict(arm=arm,reason='no completed result'));continue
        meta=json.loads(path.read_text())
        if meta['status']!='completed':missing.append(dict(arm=arm,reason=meta['status']));continue
        if meta['draws']<1024:
            # Concurrent 64-draw launch smoke is engineering only.
            cases.append(dict(arm=arm,status='mechanics_smoke_only',draws=meta['draws']));continue
        chains=read_draws(directory,arm,meta['draws'])
        files=meta.get('files',{})
        if arm.startswith('author-'):
            files={name:entry for chain in meta['chains_report'] for name,entry in chain['files'].items()}
        for name,entry in files.items():
            if sha(directory/name)!=entry['sha256']:raise ValueError('source draw hash mismatch: '+name)
        d=summarize(tf,chains);sensitivity=summarize(tf,chains,512)
        v=np.concatenate([c[:,0] for c in chains]);bins=np.linspace(-12,12,97)
        counts,edges=np.histogram(v,bins=bins);density=counts/(v.size*np.diff(edges))
        case=dict(arm=arm,diagnostics=d,exclude512_sensitivity=sensitivity,
            sampling_seconds=meta['sampling_seconds'],wall_seconds=meta['wall_seconds'],
            v_histogram=dict(centers=(.5*(edges[1:]+edges[:-1])).tolist(),density=density.tolist()),
            files_verified=True)
        cases.append(case);write(out/(arm+'-diagnostics.json'),case)
    comparisons=[];tests=4*13+3*13;critical=NormalDist().inv_cdf(1-.05/(2*tests))
    baseline=next((c for c in cases if c['arm']=='author-corrected' and 'diagnostics' in c),None)
    for case in cases:
        if 'diagnostics' not in case:continue
        d=case['diagnostics'];values={('z%d'%j):x['mean_summary'] for j,x in enumerate(d['transformed'])}
        values.update({('tail'+key):x for key,x in d['tail'].items()})
        for label,s in values.items():
            truth=0. if label.startswith('z') else s['analytic']
            diff=s['mean']-truth;half=critical*s['batch_se']
            comparisons.append(dict(arm=case['arm'],reference='analytic',quantity=label,
                difference=diff,simultaneous95=[diff-half,diff+half],
                detected_difference=abs(diff)>half,eligible=d['posterior_screen']))
            if baseline is not None and case!=baseline:
                bd=baseline['diagnostics']
                ref=bd['transformed'][int(label[1:])]['mean_summary'] if label.startswith('z') else bd['tail'][label[4:]]
                diff=s['mean']-ref['mean'];half=critical*math.hypot(s['batch_se'],ref['batch_se'])
                comparisons.append(dict(arm=case['arm'],reference='author-corrected',quantity=label,
                    difference=diff,simultaneous95=[diff-half,diff+half],detected_difference=abs(diff)>half,
                    eligible=d['posterior_screen'] and bd['posterior_screen'] and case['arm']!='author-original',
                    dependence_note='Different random engines; author original/corrected share seeds, independent-SE formula is descriptive for that pair.'))
    result=dict(cases=cases,missing=missing,comparisons=comparisons,analysis_environment=env,
        reference_evaluator=evaluation,
        critical=critical,bonferroni_family_size=tests,statistically_supported_speed_ranking=False,
        compatibility='Report convergence screens and intervals; absence of a difference is not equivalence.',
        original_author_correction='Kept separate; original code has documented backward-weight omission.')
    write(out/'comparison.json',result)
    lines=['# Million-iteration funnel replication results','',
        'Pinned author algorithm/protocol; four independent method processes. Concurrent wall times include contention.',
        'Python reference evaluator: '+evaluation+'. TF uses the equivalent analytic formula without its constant normalization.',
        'The original and corrected Python arms remain separate. Newly selected h/delta need not equal the paper’s rounded values.',
        '', '| Method | R-hat max | Bulk ESS min | Tail ESS min | Screen | Sampling seconds |',
        '|---|---:|---:|---:|---|---:|']
    for c in cases:
        if 'diagnostics' not in c:lines.append('| %s | N/A | N/A | N/A | mechanics smoke only | N/A |'%c['arm']);continue
        d=c['diagnostics'];lines.append('| %s | %.5f | %.1f | %.1f | %s | %.1f |'%(c['arm'],d['max_rhat'],d['min_bulk_ess'],d['min_tail_ess'],d['posterior_screen'],c['sampling_seconds']))
    lines += ['', '## Decision table','', '| Decision | Primary criterion | Veto | Uncertainty | Next action | Not concluded |',
        '|---|---|---|---|---|---|',
        '| Assess compatibility | Complete draws and posterior screens above | Missing/invalid artifacts or failed screens | Batch MCSE and initialization sensitivity | Inspect comparison.json intervals; investigate differences | Formal equivalence or universal sampler validity |',
        '', '## Inference status','', '| Item | Finding |','|---|---|',
        '| Hard veto screen | Per-arm status above; missing arms: '+str(missing)+' |',
        '| Statistically supported ranking | None; concurrent single-run timing is not a speed ranking |',
        '| Descriptive differences | Runtime, ESS, tails, histograms and failed-screen intervals |',
        '| Default readiness | No change |',
        '| Next evidence | Independent tuning/seed replications and isolated timing if a ranking is needed |',
        '', '## Result review','',
        'Strongest alternative explanations are finite-chain mixing, fixed tuning shared from one author run, and original-source weight error. A passed finite-sample screen is not proof of stationarity. Inspect 50/100-block MCSE sensitivity and simultaneous intervals before interpreting reference agreement. The extreme v=-30 test and exact figure-level replication are separate.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    plotted=[c for c in cases if 'diagnostics' in c]
    if plotted:histogram_svg(out,plotted)
