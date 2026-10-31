#!/usr/bin/env python3
"""Post-run scientific figures from preserved JSON; CPU diagnostic only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import json
import math
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'docs/plans/artifacts/sqmc-ksc-reset-repair-20260929/attempt-01'

def stage():
    src=BASE/'stage__previous_inverse_cdf-attempt-01'
    out=BASE/'stage-figures-01';out.mkdir(exist_ok=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    stages=('source','raw_reset','pre_cap','post_cap','final')
    labels=('Weighted\nsource','Raw\nreset','Moment\ncorrection','Coordinate\ncap','Covariance\nrestoration')
    records=[]
    for arm,label in (('baseline','Original'),('design','Richer design'),
                      ('protection','New protection'),('combined','Both changes')):
        traces=[json.loads((src/f'{arm}-coordinate-{k}.json').read_text()) for k in range(2)]
        kurt=[traces[0][s+'_moments'][0][3] for s in stages]
        delta=[]
        for s in stages:
            delta.append(math.sqrt(sum((t[s+'_next_predictive_score'][0]-t['source_next_predictive_score'][0])**2 for t in traces)))
        records.append(dict(arm=arm,first_step_kurtosis=kurt,first_step_next_score_change_l2=delta))
        axes[0].plot(range(5),kurt,'o-',label=label,linewidth=1.6,markersize=4)
        axes[1].plot(range(5),delta,'o-',label=label,linewidth=1.6,markersize=4)
    for ax in axes:
        ax.set_xticks(range(5),labels,fontsize=8)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Standardized fourth moment (kurtosis)')
    axes[1].set_ylabel('Change in exact next-prediction score (L2)')
    axes[0].legend(fontsize=8)
    fig.suptitle('First reset: same incoming particles and weights in all four interventions',fontsize=11)
    fig.text(.5,.015,'Inverse CDF; N=1008; data 241001, design 251001. Local mechanism diagnostic, not whole-filter accuracy.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.94))
    for suffix in ('png','pdf'):fig.savefig(out/f'stage-comparison.{suffix}',dpi=180)
    (out/'values.json').write_text(json.dumps(records,indent=2)+'\n')
    print(out)

def accuracy():
    out=BASE/'analysis-01';report=json.loads((out/'summary.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,(seed,horizon) in zip(axes.flat,((243001,10),(243002,10),(243001,120),(243002,120))):
        cells=[c for c in report['cells'] if c['data_seed']==seed and c['horizon']==horizon]
        x=list(range(len(cells)))
        ax.bar([i-.18 for i in x],[c['baseline']['mean_replicate_score_l2_error'] for c in cells],.36,label='Original')
        ax.bar([i+.18 for i in x],[c['candidate']['mean_replicate_score_l2_error'] for c in cells],.36,label='Frozen candidate')
        ax.axhline(cells[0]['heuristic_score_l2_errors']['gaussian'],color='crimson',linestyle='--',label='Gaussian heuristic')
        ax.set_xticks(x,['IID','Inverse CDF','Permutation','Ablation'],fontsize=8)
        ax.set_title(f'Data {seed}, T={horizon}; n={cells[0]["candidate"]["n"]} designs',fontsize=10)
        ax.set_ylabel('Mean score-vector L2 error')
        ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Untouched data: error against the full seven-mixture reference',fontsize=12)
    fig.text(.5,.015,'Descriptive means. Paired uncertainty intervals and every score coordinate are in comparisons.md.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95))
    for suffix in ('png','pdf'):fig.savefig(out/f'untouched-score-errors.{suffix}',dpi=180)
    print(out)

if __name__=='__main__':
    {'stage':stage,'accuracy':accuracy}[sys.argv[1]]()
