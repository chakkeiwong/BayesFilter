"""Standalone post-run diagnostic figure; no inference or runtime selection."""
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/sqmc-ksc-discrepancy-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/plans/artifacts/sqmc-ksc-discrepancy-20260929/attempt-01/analysis-01'
rows=json.loads((OUT/'summary.json').read_text())['replications']
routes=('iid_dual_cap','previous_inverse_cdf','repaired_permutation','repaired_permutation_ablation')
labels=('IID','Inverse CDF','Permutation .98','Permutation .97')
fig,axes=plt.subplots(2,2,figsize=(11,6.4),sharex=True)
colors=('#0072B2','#D55E00','#009E73','#CC79A7')
for col,data in enumerate((213001,213006)):
 for coord in range(2):
  ax=axes[coord,col]
  for i,(route,label,color) in enumerate(zip(routes,labels,colors)):
   a=sorted((r for r in rows if r['data_seed']==data and r['route']==route),key=lambda r:r['n'])
   xs=[j+(i-1.5)*.055 for j in range(3)]
   ys=[r['score_error'][coord]['mean'] for r in a]
   ci=[2.364624251*r['score_error'][coord]['se'] for r in a]
   ax.errorbar(xs,ys,yerr=ci,label=label,color=color,marker='o',markersize=4,capsize=3,linewidth=1.25)
  ax.axhline(0,color='.35',linestyle='--',linewidth=1)
  ax.set_xticks([0,1,2],['1,008','2,016','4,032'])
  ax.set_ylabel(('gamma_raw','log_beta')[coord]+' score error')
  ax.grid(axis='y',alpha=.2)
  if coord==0:ax.set_title('Dataset '+str(data))
  else:ax.set_xlabel('Particle count')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=4,bbox_to_anchor=(.5,.95),frameon=False)
fig.suptitle('KSC score error persists as particle count increases',fontsize=14,y=.995)
fig.text(.5,.017,'T=120; 8 random designs per point; exploratory 95% t intervals; two retrospectively selected datasets.\nFP64 GPU/XLA, TF32 off; frozen untuned diagnostic settings. No method ranking.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.08,1,.89))
for suffix in ('png','pdf'):
 target=OUT/('conditional-score-errors.'+suffix)
 if target.exists():raise RuntimeError('Preserve previous figure')
 fig.savefig(target,dpi=180)
print(str(OUT/'conditional-score-errors.png'))

plt.close(fig)
evaluations=json.loads((OUT/'all-evaluations.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
for ax,data in zip(axes,(213001,213006)):
 row=next(r for r in evaluations if r['phase']=='mechanism' and r['route']=='previous_inverse_cdf' and r['data_seed']==data)
 tr=row['mechanism']['direction_traces'][0]
 ts=list(range(1,len(tr['before_moments'])+1))
 ax.plot(ts,[m[3] for m in tr['before_moments']],label='Weighted particles before reset',color='#0072B2',linewidth=1.2)
 ax.plot(ts,[m[3] for m in tr['after_moments']],label='Particles after reset and correction',color='#D55E00',linewidth=1.2)
 ax.axhline(3,color='.5',linestyle=':',label='Gaussian kurtosis')
 ax.set_title('Dataset '+str(data));ax.set_xlabel('Observation index')
 ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Kurtosis (fourth standardized moment)')
fig.suptitle('Distribution shape before and after the shared reset',fontsize=14)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.045),fontsize=9)
fig.text(.5,.01,'Inverse CDF; N=1,008; original random designs; frozen diagnostic settings. Local shape diagnostic.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.12,1,.93))
for suffix in ('png','pdf'):
 target=OUT/('reset-distribution-shape.'+suffix)
 if target.exists():raise RuntimeError('Preserve previous figure')
 fig.savefig(target,dpi=180)
print(str(OUT/'reset-distribution-shape.png'))
