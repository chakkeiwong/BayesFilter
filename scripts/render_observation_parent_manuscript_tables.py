"""Diagnostic reporting: render saved comparison measurements as LaTeX.

No numerical kernel or scientific experiment is run. The CSV is the checked
post-run report; each printed value is rounded only for presentation.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/plans/artifacts/observation-parent-all-model-comparison-20261028/comparison.csv'
TARGET = ROOT / 'docs/chapters/observation_parent_comparison_tables.tex'
ARMS = ('original_three_branch', 'conditional_uniform', 'conditional_observation_weighted')
MODELS = (('lgssm', 'LGSSM'), ('ksc', 'KSC SV'),
          ('predator_prey', 'Predator--prey'), ('range_bearing', 'Range--bearing'),
          ('sir', 'SIR nominal'))
LABELS = {'transition_parameter': 'Transition', 'log_process_variance': 'Log process variance',
          'log_observation_variance': 'Log obs. variance', 'initial_mean_scale': 'Initial mean scale',
          'persistence_parameter': 'Persistence', 'log_observation_scale': 'Log obs. scale',
          'log_process_scale': 'Log process scale', 'carrying_capacity': 'Carrying capacity',
          'half_saturation': 'Half saturation', 'r': '$r$', 's': '$s$', 'u': '$u$', 'v': '$v$',
          'log infection multiplier': 'Log infection mult.',
          'log recovery multiplier': 'Log recovery mult.',
          'log observation-noise multiplier': 'Log noise mult.'}

def render():
    rows = list(csv.DictReader(SOURCE.open()))
    selected = [r for r in rows if r['model'] != 'sir' or
                (r['case'] == 'nominal' and r['horizon'] == '50')]
    index = {(r['model'], r['arm'], r['quantity']): r for r in selected}
    assert len(index) == len(selected) == 66, len(selected)
    def value(model, arm, quantity):
        return index[model, arm, quantity]
    def num(x):
        return format(float(x), '.6g')
    def cell(row):
        return num(row['mean']) + ' (' + num(row['rmse']) + ')'
    lines = ['% Generated from checked comparison.csv; do not edit numerical cells.']
    def start(columns, header):
        lines.extend([r'\begin{table}[htbp]', r'\centering\small',
                      r'\setlength{\tabcolsep}{4pt}',
                      r'\resizebox{\linewidth}{!}{%', r'\begin{tabular}{' + columns + '}',
                      r'\toprule', header + r'\\', r'\midrule'])
    def end(caption, label):
        lines.extend([r'\bottomrule', r'\end{tabular}}', r'\caption{' + caption + '}',
                      r'\label{' + label + '}', r'\end{table}', ''])
    start('lrrrr', r'Model ($T$) & Reference & Original & Uniform & Weighted')
    for key, label in MODELS:
        ref = value(key, ARMS[0], 'log_likelihood')
        lines.append(label + ' (' + ref['horizon'] + ') & ' + num(ref['reference']) +
                     ' & ' + ' & '.join(cell(value(key,a,'log_likelihood')) for a in ARMS) + r'\\')
    end('Log-likelihood mean (RMSE) across eight paired streams. References have the model-specific limitations described in the text.',
        'tab:parent-cross-model-likelihood')
    for key,label in MODELS:
        start('lrrrr', r'Parameter & Reference & Original & Uniform & Weighted')
        quantities=[r['quantity'] for r in selected if r['model']==key and
                    r['arm']==ARMS[0] and r['quantity']!='log_likelihood']
        for q in quantities:
            ref=value(key,ARMS[0],q)
            lines.append(LABELS[q]+' & '+num(ref['reference'])+' & '+
                         ' & '.join(cell(value(key,a,q)) for a in ARMS)+r'\\')
        end(label + ': score mean (component RMSE) across eight paired streams. Lower observed RMSE is descriptive; no cross-model score scale is imposed.',
            'tab:parent-cross-model-score-'+key.replace('_','-'))
    start('lrrr', r'Model & Original & Uniform & Weighted')
    for key,label in MODELS:
        lines.append(label+' & '+' & '.join(num(value(key,a,'log_likelihood')['ess_mean']) for a in ARMS)+r'\\')
    end('ESS averaged over time and eight streams, with $N=1008$. ESS measures weight concentration rather than likelihood or score accuracy.',
        'tab:parent-cross-model-ess')
    TARGET.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'source':str(SOURCE.relative_to(ROOT)), 'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                      'output':str(TARGET.relative_to(ROOT)), 'measurements':len(selected), 'tables':7}))

if __name__ == '__main__':
    render()
