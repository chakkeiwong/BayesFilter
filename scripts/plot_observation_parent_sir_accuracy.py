"""Post-run scientific plotting only; does not influence runtime or selection."""
import argparse,json,math,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/observation-parent-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    s=json.loads(a.summary.read_text())
    if not s['completeness']['complete']:raise ValueError('final figure requires complete paired grid')
    a.out.mkdir(parents=True,exist_ok=True)
    cases=['nominal','infection0999','infection1001','recovery0999','recovery1001','noise08','noise125']
    labels=['Nominal','Infection ×0.999','Infection ×1.001','Recovery ×0.999','Recovery ×1.001','Noise ×0.8','Noise ×1.25']
    refs={ (r['case'],r['horizon']):r for r in s['reference_cells']}
    stats={(r['case'],r['horizon'],r['arm']):r for r in s['statistics']}
    titles=['Log likelihood','Infection score','Recovery score','Noise score']
    fig,axs=plt.subplots(2,4,figsize=(12,5.3),sharex=True,sharey=True,constrained_layout=True)
    for row,base in enumerate(['original','conditional_uniform']):
        for k in range(4):
            data=[]
            for case in cases:
                vals=[]
                for h in (10,20,50):
                    key=(case,h)
                    if not refs[key]['usable']:vals.append(float('nan'));continue
                    w=stats[(*key,'conditional_weighted')]['rmse_by_reference'];b=stats[(*key,base)]['rmse_by_reference']
                    vals.append(sum(math.log10(w[f][k]/b[f][k]) for f in w)/len(w))
                data.append(vals)
            ax=axs[row,k];im=ax.imshow(data,cmap='RdBu_r',vmin=-3,vmax=3,aspect='auto')
            if row==0:ax.set_title(titles[k],fontsize=11)
            ax.set_xticks([0,1,2],['10','20','50']);ax.set_yticks(range(len(cases)),labels,fontsize=8)
            if row==1:ax.set_xlabel('Horizon T')
            if k==0:ax.set_ylabel('Weighted / '+('original' if row==0 else 'conditional uniform'),fontsize=10)
            for c,case in enumerate(cases):
                for j,h in enumerate((10,20,50)):
                    if not refs[(case,h)]['usable']:ax.text(j,c,'×',ha='center',va='center',color='black');continue
                    comparison=next(x for x in s['paired_comparisons'] if x['case']==case and x['horizon']==h and {x['left'],x['right']}=={'conditional_weighted',base})
                    if comparison['exploratory_supported_lower_mse'][k]:ax.text(j,c,'•',ha='center',va='center',color='black',fontsize=12)
    cb=fig.colorbar(im,ax=axs,shrink=.85,pad=.02);cb.set_label('log10 RMSE ratio (blue favors weighted)')
    fig.savefig(a.out/'accuracy-ratios.pdf');fig.savefig(a.out/'accuracy-ratios.png',dpi=160);plt.close(fig)
    # Print-size panels: avoid shrinking eight panels into unreadable text.
    for base,tag in [('original','original'),('conditional_uniform','uniform')]:
        fig,axs=plt.subplots(2,2,figsize=(7.2,6.0),sharex=True,sharey=True,constrained_layout=True)
        for k,ax in enumerate(axs.flat):
            data=[]
            for case in cases:
                vals=[]
                for h in (10,20,50):
                    key=(case,h)
                    if not refs[key]['usable']:vals.append(float('nan'));continue
                    w=stats[(*key,'conditional_weighted')]['rmse_by_reference'];b=stats[(*key,base)]['rmse_by_reference']
                    vals.append(sum(math.log10(w[f][k]/b[f][k]) for f in w)/len(w))
                data.append(vals)
            im=ax.imshow(data,cmap='RdBu_r',vmin=-3,vmax=3,aspect='auto')
            ax.set_title(titles[k],fontsize=11)
            ax.set_xticks([0,1,2],['10','20','50']);ax.set_yticks(range(len(cases)),labels,fontsize=9)
            if k>=2:ax.set_xlabel('Horizon T')
            for c,case in enumerate(cases):
                for j,h in enumerate((10,20,50)):
                    if not refs[(case,h)]['usable']:ax.text(j,c,'×',ha='center',va='center');continue
                    comparison=next(x for x in s['paired_comparisons'] if x['case']==case and x['horizon']==h and {x['left'],x['right']}=={'conditional_weighted',base})
                    if comparison['exploratory_supported_lower_mse'][k]:ax.text(j,c,'•',ha='center',va='center',fontsize=12)
        cb=fig.colorbar(im,ax=axs,shrink=.85,pad=.02);cb.set_label('log10 RMSE ratio (blue favors weighted)',fontsize=10)
        fig.savefig(a.out/f'accuracy-{tag}.pdf');fig.savefig(a.out/f'accuracy-{tag}.png',dpi=160);plt.close(fig)
    # Weight concentration is shown separately from likelihood/score error.
    fig,ax=plt.subplots(figsize=(7.4,3.2),constrained_layout=True)
    names={'original':'Original','conditional_uniform':'Conditional, uniform','conditional_weighted':'Conditional, weighted','weighted_only':'Conditional-only, weighted'}
    for arm,label in names.items():
        ax.plot([10,20,50],[stats[('nominal',h,arm)]['ess_mean'] for h in (10,20,50)],marker='o',label=label)
    ax.set(xlabel='Horizon T',ylabel='Mean ESS across steps and seeds',ylim=(0,1040),xticks=[10,20,50]);ax.legend(fontsize=8,loc='lower right');ax.grid(alpha=.2)
    fig.savefig(a.out/'nominal-ess.pdf');fig.savefig(a.out/'nominal-ess.png',dpi=160);plt.close(fig)
    print(json.dumps({'figures':[str(p) for p in a.out.glob('*.pdf')],'note':'descriptive ratios; dots mark predeclared exploratory paired criterion; gray cross denotes unusable reference'}))
if __name__=='__main__':main()
