"""Post-run reporting only; reads frozen metrics and never selects a method."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;started=time.monotonic()
    data=json.loads((out/'uncertainty.json').read_text())
    summary=json.loads((out/'summary.json').read_text())
    if summary['status']!='PASS' or summary['completed_rows']!=10000:
        raise RuntimeError('Reporting requires the complete validated matrix')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    names={'csmc':'Controlled SMC, 128','bpf128':'Bootstrap, 128',
           'bpf5529':'Bootstrap, 5529','ekf':'EKF proposal, 128',
           'quadrature3':'Quadrature proposal, 128'}
    colors={'csmc':'#b52d38','bpf128':'#777777','bpf5529':'#254c75',
            'ekf':'#259178','quadrature3':'#9d6daf'}
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),sharey=False,layout='constrained')
    regimes=[('extreme',2,'Extreme counts: 0–5 or 45–50'),
             ('central',3,'Central counts: 15–35'),('other',4,'Other counts')]
    for ax,(regime,index,title) in zip(axes,regimes):
        if data['regimes'][regime]==0:
            ax.set_title(title+' (n=0)',fontsize=10)
            ax.text(.5,.5,'No observations in this regime\nComparison not evaluated',
                    ha='center',va='center',transform=ax.transAxes,fontsize=10)
            ax.set_axis_off()
            continue
        for method,label in names.items():
            rows=sorted([r for r in data['summaries'] if r['method']==method],key=lambda r:r['process_variance'])
            x=[r['process_variance'] for r in rows]
            ax.plot(x,[r['point'][index] for r in rows],label=label,color=colors[method],linewidth=1.6)
            ax.fill_between(x,[r['ci_lower'][index] for r in rows],[r['ci_upper'][index] for r in rows],color=colors[method],alpha=.1)
        ax.set_title(title+' (n='+str(data['regimes'][regime])+')',fontsize=10)
        ax.set_xlabel('Transition variance');ax.set_yscale('log');ax.grid(alpha=.2)
    axes[0].set_ylabel('Squared error of filtering mean\nversus converged reference')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,fontsize=9)
    fig.suptitle('Same data and original filtering target · 100 runs per cell\nPointwise 95% bootstrap intervals; costs are not matched',fontsize=11)
    for extension in ('png','pdf'):fig.savefig(out/('conditional-filtering-comparison.'+extension),dpi=170)
    plt.close(fig)
    rows=[]
    for method in ('bpf128','bpf5529','ekf','quadrature3'):
        contrasts=[r for r in data['contrasts'] if r['comparator']==method]
        for regime,index,_ in regimes:
            if data['regimes'][regime]==0:
                rows.append(dict(comparator=method,regime=regime,cells=0,
                    status='NOT_EVALUATED_NO_OBSERVATIONS'))
                continue
            rows.append(dict(comparator=method,regime=regime,cells=len(contrasts),
                observed_csmc_larger_error=sum(r['difference'][index]>0 for r in contrasts),
                pointwise_supported_csmc_larger_error=sum(r['ci_lower'][index]>0 for r in contrasts),
                pointwise_supported_csmc_smaller_error=sum(r['ci_upper'][index]<0 for r in contrasts)))
    paper=[]
    for method in ('bpf128','bpf5529','ekf','quadrature3'):
        contrasts=[r for r in data['contrasts'] if r['comparator']==method]
        paper.append(dict(comparator=method,metric='log10_relative_variance_logZ',
            pointwise_supported_csmc_larger=sum(r['ci_lower'][-1]>0 for r in contrasts),
            pointwise_supported_csmc_smaller=sum(r['ci_upper'][-1]<0 for r in contrasts),cells=len(contrasts)))
    result=dict(role='POST_RUN_REPORTING_ONLY',conditional_comparisons=rows,
        paper_statistic_comparisons=paper,pointwise_only=True,
        heuristic_verdict=summary['heuristic_dominance']['verdict'],
        cpu_only=True,gpu_intentionally_hidden=True,wall_seconds=time.monotonic()-started,
        command=['python','-B',__file__,'--output',str(out)])
    (out/'reporting-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
