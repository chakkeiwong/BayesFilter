"""CPU-only post-run diagnostic figures; never imported by numerical routes."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

os.environ['CUDA_VISIBLE_DEVICES']='-1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROUTES=('iid_dual_cap','previous_inverse_cdf','repaired_permutation','repaired_permutation_ablation')
SCOPES=('p44_d3_T10','p44_d3_T120','full_d3_T2','full_d3_T10','full_d3_T120',
        'full_d10_T2','full_d10_T10','full_d10_T120')
POSITIONS=((0,0),(0,1),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2))
COLORS=('#2962a3','#c05a27')
MARKERS=('o','s')


def plot(source, output):
    source=Path(source);output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    cases=json.loads(source.read_text())['cases']
    lookup={(case['scope'],case['route']):case for case in cases}
    valid_cells=sum(row['valid'] for case in cases for row in case['rows'])
    files=[]
    for metric,title,ylabel in (
        ('score_l2_error','Observed score error','Score error, L2'),
        ('value_error','Observed log-likelihood error','Absolute log-likelihood error')):
        fig,axes=plt.subplots(3,3,figsize=(13,10))
        for scope,(i,j) in zip(SCOPES,POSITIONS):
            ax=axes[i,j]
            rows={route:{(row['data_seed'],row['filter_seed']):row
                for row in lookup.get((scope,route),{}).get('rows',[])} for route in ROUTES}
            pairs=sorted({pair for lane in rows.values() for pair in lane})
            data_seeds=sorted({pair[0] for pair in pairs});filter_seeds=sorted({pair[1] for pair in pairs})
            all_errors=[]
            for pair in pairs:
                errors=[abs(rows[route][pair][metric]) if pair in rows[route] and rows[route][pair]['valid'] else None
                    for route in ROUTES]
                all_errors.extend(value for value in errors if value is not None)
                ax.plot(range(4),errors,color=COLORS[data_seeds.index(pair[0])],
                    marker=MARKERS[filter_seeds.index(pair[1])],linewidth=1,markersize=4,alpha=.8)
            if all_errors and min(all_errors)>0:ax.set_yscale('log')
            if not all_errors:ax.text(.5,.5,'No completed final cells',ha='center',va='center',transform=ax.transAxes)
            ax.set_title(scope.replace('_',' '),fontsize=11)
            ax.set_xticks(range(4),('IID','Inv. CDF','Perm.','Cap .97'))
            ax.set_ylabel(ylabel,fontsize=9)
            ax.grid(axis='y',which='both',alpha=.18)
            ax.tick_params(labelsize=9)
        legend_ax=axes[0,2];legend_ax.axis('off')
        handles=[Line2D([0],[0],color=color,marker=marker,linewidth=1,
            label=f'Dataset {i+1}, design {j+1}')
            for i,color in enumerate(COLORS) for j,marker in enumerate(MARKERS)]
        legend_ax.legend(handles=handles,loc='upper left',frameon=False)
        legend_ax.text(0,.49,'Each line pairs the same dataset and\nfilter design across routes.\n\nTwo independent datasets per scope;\nno confidence intervals or ranking.\n\nVertical axes are logarithmic when\nall errors are positive. Scales differ\nbetween panels.',va='top',fontsize=10)
        prefix='Partial comparison: ' if valid_cells<128 else ''
        fig.suptitle(f'{prefix}{title} against matched Kalman ({valid_cells}/128 valid cells)',fontsize=14)
        fig.tight_layout(rect=(0,0,1,.965))
        name='score_errors' if metric=='score_l2_error' else 'likelihood_errors'
        for extension in ('svg','png'):
            filename=output/f'{name}.{extension}'
            fig.savefig(filename,dpi=160,bbox_inches='tight');files.append(filename.name)
        plt.close(fig)
    manifest=dict(schema='sqmc_expanded_diagnostic_figures.v1',created_utc=datetime.now(timezone.utc).isoformat(),
        source=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        command=[sys.executable,*sys.argv],matplotlib=matplotlib.__version__,
        cpu_gpu_status='CPU-only diagnostic rendering; GPU devices intentionally hidden with CUDA_VISIBLE_DEVICES=-1',
        valid_final_cells=valid_cells,files=files,
        interpretation='Paired descriptive errors only; different tuned settings may differ across routes; no superiority inference.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(figures=str(output),valid_final_cells=valid_cells)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();plot(args.source,args.output)
