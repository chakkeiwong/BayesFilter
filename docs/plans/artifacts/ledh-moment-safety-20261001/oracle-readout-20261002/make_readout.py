"""Post-run diagnostic reporting only; no filtering, tuning or new experiments."""
from pathlib import Path
import os, json, hashlib, csv, shutil
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['MPLCONFIGDIR']='/tmp/ledh_oracle_readout_mplconfig'
root=Path('/home/chakwong/BayesFilter-SQMC')
source=root/'docs/plans/artifacts/sqmc-ksc-reset-repair-20260929/attempt-01/analysis-01'
out=root/'docs/plans/artifacts/ledh-moment-safety-20261001/oracle-readout-20261002'
out.mkdir(parents=True,exist_ok=False)
summary=json.loads((source/'summary.json').read_text())
routes={'iid_dual_cap':'IID','previous_inverse_cdf':'Inverse CDF','repaired_permutation':'Permutation','repaired_permutation_ablation':'Permutation ablation'}
rows=[]
for c in summary['cells']:
    b,n=c['baseline'],c['candidate']
    assert b['n']==n['n']==8
    rows.append({'method':routes[c['route']],'T':c['horizon'],'data':c['data_seed'],'selected_arm':c['selected_arm'],'n':b['n'],'old_log_error':b['mean_absolute_value_error'],'new_log_error':n['mean_absolute_value_error'],'old_score_error':b['mean_replicate_score_l2_error'],'new_score_error':n['mean_replicate_score_l2_error'],'score_change':c['paired_mean_l2_difference'],'score_ci_low':c['paired_l2_difference_95ci'][0],'score_ci_high':c['paired_l2_difference_95ci'][1],'log_error_percent_change':100*(n['mean_absolute_value_error']/b['mean_absolute_value_error']-1),'score_error_percent_change':100*(n['mean_replicate_score_l2_error']/b['mean_replicate_score_l2_error']-1)})
rows.sort(key=lambda x:(x['T'],x['data'],list(routes.values()).index(x['method'])))
with (out/'numerical-comparison.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
header='''# KSC oracle comparison: numerical results

This is a numerical reading of the saved September 29 residual-design and
coordinate-cap comparison, not a new filtering run. The October safety guard
has not yet been evaluated against these oracles. Every number below comes
from the saved source summary or actual-value table.

The scope is scalar KSC with the full seven-component observation mixture,
N=1008, gamma=1.5, log_beta=0, FP64 GPU/XLA with TF32 off, and horizons 10 and
120. Dataset 243001 is A and 243002 is B. Each untouched cell has eight paired
particle designs. The full-mixture references were checked by quadrature
refinement and an independent density-grid recurrence to 1e-7.

Old denotes the original reset and cap. New denotes the selected richer-design,
identity-core-cap repair with eight moment-correction steps. The terminal OT
regularization is 25.6 for the epsilon25 arm and 102.4 for the steps8 arm; thus
this is a comparison of the selected combined configurations, not an isolated
causal estimate of the residual-design change.

Log-likelihood error is the mean of absolute per-design log-likelihood errors.
Score error is the mean Euclidean norm of the per-design error in the two score
coordinates (gamma, log_beta). These are not the absolute error or score-error
norm of the averaged estimates. A negative score change means lower error.
The saved paired 95% intervals describe particle-design variation conditional
on a fixed dataset; they are exploratory and unadjusted for multiple comparisons.
No likelihood-error interval was supplied by this summary, so those mean changes
are descriptive. No population-wide method ranking is inferred.

## Untouched numerical comparisons

| Method | T | Dataset | Selected arm | Mean absolute log-likelihood error: old → new | Mean score L2 error: old → new | Paired score-error change, 95% interval |
|---|---:|---:|---|---:|---:|---:|
'''
lines=[header]
for r in rows:
    lines.append(f"| {r['method']} | {r['T']} | {r['data']} | {r['selected_arm']} | {r['old_log_error']:.6f} → {r['new_log_error']:.6f} | {r['old_score_error']:.6f} → {r['new_score_error']:.6f} | {r['score_change']:.6f} [{r['score_ci_low']:.6f}, {r['score_ci_high']:.6f}] |")
lines.append('''
## Validation dataset, reported numerically

These frozen-configuration comparisons used two designs per cell on a separate
validation dataset. Their small replication count limits uncertainty assessment.
They show whether the untouched results recur on the earlier validation data.

| Method | T | Mean absolute log-likelihood error: old → new | Mean score L2 error: old → new |
|---|---:|---:|---:|''')
for c in summary['validation']:
    b,n=c['baseline'],c['candidate']
    lines.append(f"| {routes[c['route']]} | {c['horizon']} | {b['mean_absolute_value_error']:.6f} → {n['mean_absolute_value_error']:.6f} | {b['mean_replicate_score_l2_error']:.6f} → {n['mean_replicate_score_l2_error']:.6f} |")
lines.append('''
The inverse-CDF T=120 validation comparison has score error 0.073887 → 0.121654
and log-likelihood error 0.548057 → 0.084772. Thus the score/likelihood trade-off
changes across datasets; the favorable untouched score changes alone do not
establish uniform improvement.

## Fourth-moment mechanism at the saved first reset

The incoming weighted cloud has standardized fourth moment 2.916221. This is a
particle-cloud target, not an oracle posterior fourth moment.

| Quantity | Original reset/cap | Richer design and identity-core cap |
|---|---:|---:|
| Raw reset fourth moment | 1.000473 | 2.972274 |
| After moment correction | 1.577655 | 2.966277 |
| Final fourth moment | 1.328293 | 2.966277 |
| Absolute discrepancy from incoming-cloud target | 1.587928 | 0.050056 |

This local discrepancy is about 96.85% smaller. It explains the mechanism without
establishing the true filtering fourth moment or long-horizon accuracy.

''')
actual=(source/'comparisons.md').read_text().split('## Actual values and scores:',1)[1]
actual='## Actual values and scores:'+actual
actual=actual.split('\nIntervals are conditional',1)[0]
lines.append(actual)
lines.append('''
## Provenance and interpretation

The figure and CSV display all sixteen untouched cells, without binary verdicts.
The original predeclared criteria and decisions remain preserved in the source
report; this readout changes presentation, not the experimental criteria.
The source paths and checksums are recorded in manifest.json. Only deterministic
post-run arithmetic and plotting were executed; no new likelihood/score evidence
was generated. CPU-only reporting; GPUs were intentionally hidden.
''')
(out/'numerical-results.md').write_text('\n'.join(lines)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(17,10),sharey=True,gridspec_kw={'width_ratios':[1,1,1.1]})
old_color='#7b8794'; new_color='#126b92'
labels=[]
for i,r in enumerate(rows):
    y=15-i
    labels.append(f"T={r['T']}, {'A' if r['data']==243001 else 'B'}   {r['method']}")
    for ax,key in zip(axes[:2],['log','score']):
        old=r[f'old_{key}_error'];new=r[f'new_{key}_error']
        ax.plot([old,new],[y,y],color='#bcc5cc',lw=1.5,zorder=1)
        ax.scatter(old,y,s=45,color=old_color,marker='o',zorder=2)
        ax.scatter(new,y,s=48,color=new_color,marker='D',zorder=3)
    axes[2].hlines(y,r['score_ci_low'],r['score_ci_high'],color=new_color,lw=2)
    axes[2].scatter(r['score_change'],y,color=new_color,s=35,zorder=3)
for ax in axes:
    ax.set_ylim(-0.7,15.7);ax.grid(axis='x',color='#e8ecef');ax.set_axisbelow(True)
    for y in (3.5,7.5,11.5): ax.axhline(y,color='#d7dce0',lw=0.8)
axes[0].set_yticks(list(range(15,-1,-1)),labels)
axes[0].set_title('Mean absolute log-likelihood error',loc='left',pad=14)
axes[1].set_title('Mean score-vector L2 error',loc='left',pad=14)
axes[2].set_title('Paired score-error change: new − old',loc='left',pad=14)
axes[0].set_xlim(-0.025,0.70);axes[1].set_xlim(-0.025,0.75)
axes[2].axvline(0,color='#505a63',ls='--',lw=1);axes[2].set_xlim(-0.73,0.075)
axes[0].set_xlabel('Smaller values indicate closer agreement')
axes[1].set_xlabel('Smaller values indicate closer agreement')
axes[2].set_xlabel('Point estimate and paired 95% interval')
fig.suptitle('KSC oracle comparison: size and uncertainty of the observed changes',x=0.02,ha='left',fontsize=17)
fig.text(0.02,0.935,'September residual-design/cap repair · N=1008 · eight paired designs per cell · A=243001, B=243002',fontsize=11)
fig.legend(handles=[Line2D([],[],marker='o',ls='',color=old_color,label='Original'),Line2D([],[],marker='D',ls='',color=new_color,label='Selected repair')],loc='lower center',bbox_to_anchor=(0.51,0.025),ncol=2,frameon=False)
fig.text(0.02,0.012,'Fixed-data exploratory intervals; no likelihood-error intervals shown. The October safety guard is not included.',fontsize=10)
fig.subplots_adjust(left=0.23,right=0.985,top=0.88,bottom=0.12,wspace=0.18)
for ext in ('png','svg','pdf'): fig.savefig(out/f'oracle-error-comparison.{ext}',dpi=160)
shutil.copyfile(__file__,out/'make_readout.py')
manifest={'role':'post-run diagnostic report; no new filter execution','source_files':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source/'summary.json',source/'comparisons.md']},'cpu_only':True,'gpu_intentionally_hidden':True,'comparison':'September design/cap repair; October guard absent','rows':len(rows),'particle_designs_per_cell':8,'script':'make_readout.py','original_criteria_unchanged':True}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'report':str(out/'numerical-results.md'),'figure':str(out/'oracle-error-comparison.png'),'cells':len(rows),'inverse_cdf_effects':[{k:r[k] for k in ['T','data','log_error_percent_change','score_error_percent_change','score_ci_low','score_ci_high']} for r in rows if r['method']=='Inverse CDF']},indent=2))
