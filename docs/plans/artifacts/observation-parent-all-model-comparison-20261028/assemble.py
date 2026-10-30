"""Post-run reporting only: recheck and consolidate saved diagnostic results."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "docs/plans/artifacts"
OUT = Path(__file__).resolve().parent
sources = {}


def read(path):
    path = ROOT / path
    raw = path.read_bytes()
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def verify(rows, reference, expected):
    assert len(rows) == 8 and len({r['seed'] for r in rows}) == 8
    assert all(r['valid'] for r in rows)
    quantities = [[r['value']] + r['score'] for r in rows]
    assert all(math.isfinite(v) for row in quantities for v in row)
    means = [statistics.mean(col) for col in zip(*quantities)]
    rmse = [math.sqrt(statistics.mean((row[j] - v) ** 2 for row in quantities))
            for j, v in enumerate([reference['value']] + reference['score'])]
    for a, b in zip(means, [expected['value_mean']] + expected['score_mean']):
        close(a, b)
    for a, b in zip(rmse, [expected['value_rmse']] + expected['score_rmse']):
        close(a, b)
    close(statistics.mean(r['ess_mean'] for r in rows), expected['ess_mean'])
    return sorted(r['seed'] for r in rows)


four = read('docs/plans/artifacts/observation-parent-four-model-check-20261028/summary.json')
new = read('docs/plans/artifacts/observation-parent-nonlinear-extension-20261028/summary.json')
sir = read('docs/plans/artifacts/observation-parent-sir-accuracy-20261028/analysis01/summary.json')
arms = ['original_three_branch', 'conditional_uniform', 'conditional_observation_weighted']
names = ['Original', 'Uniform parents', 'Weighted parents']
models = []
checked_rows = 0
for key, label in [('lgssm', 'LGSSM'), ('ksc', 'KSC SV'),
                   ('predator_prey', 'Predator–prey'), ('range_bearing', 'Range–bearing')]:
    m = new[key] if key in new else next(x for x in four['models'] if x['model'] == key)
    campaign = 'observation-parent-nonlinear-extension-20261028' if key in new else 'observation-parent-four-model-check-20261028'
    folder = Path('docs/plans/artifacts') / campaign / m['folder']
    rows = read(folder / 'rows.json')
    manifest = read(folder / 'manifest.json')
    assert manifest['particles'] == 1008 and manifest['horizon'] == m['horizon']
    assert manifest['theta'] == m['theta']
    pairings = []
    for arm in arms:
        selected = [r for r in rows if r['arm'] == arm]
        assert all(r['finite'] and r['steps_completed'] == m['horizon'] for r in selected)
        assert all(all(all(step) for step in r['validity_trace']) for r in selected)
        pairings.append(verify(selected, m['reference'], m['summary'][arm]))
        checked_rows += len(selected)
    assert pairings[0] == pairings[1] == pairings[2] == manifest['seeds']
    models.append(dict(model=key, label=label, horizon=m['horizon'], theta=m['theta'],
                       parameter_names=m['parameter_names'], reference=m['reference'],
                       summary={a:m['summary'][a] for a in arms}, seeds=pairings[0]))

sir_rows = {}
for path in sir['sources']:
    if 'campaign01/' not in path and 'pilot02/' not in path:
        continue
    for row in read(path):
        key = (row['case'], row['horizon'], row['arm'], row['seed'])
        if key in sir_rows:
            assert row['value'] == sir_rows[key]['value'] and row['score'] == sir_rows[key]['score']
        sir_rows[key] = row
ref_fit = '036-zhao-sir_d18-rank40-fit2-runtime-repair'
sir_arm_map = dict(zip(arms, ['original', 'conditional_uniform', 'conditional_weighted']))
sir_nominal = []
sir_cells = []
for cell in sir['reference_cells']:
    if not cell['usable']:
        continue
    reference = next(r for r in cell['references'] if r['fit'] == ref_fit)
    m = dict(model='sir', label='SIR nominal' if cell['case'] == 'nominal' else 'SIR '+cell['case'],
             case=cell['case'], horizon=cell['horizon'], theta=reference['theta'],
             parameter_names=['log infection multiplier', 'log recovery multiplier', 'log observation-noise multiplier'],
             reference=reference, references=cell['references'], summary={})
    pairings = []
    for arm, stored_arm in sir_arm_map.items():
        s = next(x for x in sir['statistics'] if x['case'] == cell['case'] and
                 x['horizon'] == cell['horizon'] and x['arm'] == stored_arm)
        m['summary'][arm] = dict(value_mean=s['mean'][0], score_mean=s['mean'][1:],
            value_rmse=s['rmse_by_reference'][ref_fit][0], score_rmse=s['rmse_by_reference'][ref_fit][1:],
            ess_mean=s['ess_mean'], rmse_by_reference=s['rmse_by_reference'])
        rows = [r for k,r in sir_rows.items() if k[:3] == (cell['case'],cell['horizon'],stored_arm)]
        pairings.append(verify(rows, reference, m['summary'][arm]))
        checked_rows += len(rows)
    assert pairings[0] == pairings[1] == pairings[2]
    m['seeds'] = pairings[0]
    sir_cells.append(m)
    if cell['case'] == 'nominal':
        sir_nominal.append(m)
models.append(next(m for m in sir_nominal if m['horizon'] == 50))

sir_counts = {q:{'uniform_lower_both_refs':0,'weighted_lower_both_refs':0,'reference_dependent':0}
              for q in ['log_likelihood','infection','recovery','noise']}
for m in sir_cells:
    u = m['summary'][arms[1]]['rmse_by_reference']
    w = m['summary'][arms[2]]['rmse_by_reference']
    for j,q in enumerate(sir_counts):
        which = ('uniform_lower_both_refs' if all(u[k][j] < w[k][j] for k in u) else
                 'weighted_lower_both_refs' if all(w[k][j] < u[k][j] for k in u) else 'reference_dependent')
        sir_counts[q][which] += 1

def number(x):
    return f'{x:.6g}'

def table(header, rows):
    return '\n'.join(['| '+' | '.join(header)+' |', '| '+' | '.join(['---']*len(header))+' |']+
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])

score_rows=[]
likelihood_rows=[]
ess_rows=[]
flat=[]
for m in models:
    ref=m['reference']; sums=m['summary']
    likelihood_rows.append([m['label'],m['horizon'],number(ref['value'])]+
        [f"{number(sums[a]['value_mean'])} ({number(sums[a]['value_rmse'])})" for a in arms])
    ess_rows.append([m['label']]+[number(sums[a]['ess_mean']) for a in arms])
    for j,p in enumerate(m['parameter_names']):
        score_rows.append([m['label'],p,number(ref['score'][j])]+
            [f"{number(sums[a]['score_mean'][j])} ({number(sums[a]['score_rmse'][j])})" for a in arms])
for m in models[:-1]+sir_cells:
    for a in arms:
        s=m['summary'][a]
        for j,q in enumerate(['log_likelihood']+m['parameter_names']):
            flat.append(dict(model=m['model'],case=m.get('case','single_configuration'),horizon=m['horizon'],
                arm=a,quantity=q,reference=([m['reference']['value']]+m['reference']['score'])[j],
                mean=([s['value_mean']]+s['score_mean'])[j],rmse=([s['value_rmse']]+s['score_rmse'])[j],ess_mean=s['ess_mean']))
with (OUT/'comparison.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)

text='''# Original, uniform-parent and weighted-parent comparison

All five models are included. Uniform means observation-conditioned child proposals with equal parent-selection probabilities; weighted uses observation-dependent parent probabilities. Original is the earlier three-branch proposal. Defensive branches remain present in the conditional-mixture arms.

These are saved FP64 GPU/XLA diagnostic runs, with N=1008 and eight paired seeds per model. No new numerical experiment, tuning, default change or algorithm edit was performed for this comparison. The main SIR row is the nominal T=50 case, chosen to align horizons; the CSV contains all 20 usable SIR parameter/horizon cells, including nominal T=10 and T=20. Comparisons are within each model and parameter coordinate; error magnitudes across models or differently scaled parameters are not pooled.

Skeptical reporting audit: preserve each saved comparator and target, use the completed nonlinear extension for KSC/range–bearing, distinguish mean error from RMSE, retain both SIR reference fits, and do not substitute ESS for accuracy. Source hashes and per-seed arithmetic are verified by the accompanying script. Existing run plans remain the governing evidence contracts.

## Log-likelihood estimates

Each method cell is mean estimate (RMSE against the displayed reference). Lower RMSE is favorable descriptively. These are log likelihoods, not raw likelihoods.

'''+table(['Model','T','Reference']+names,likelihood_rows)+'''

## Score estimates

Each method cell is mean score (component RMSE against the displayed reference). The score is the parameter gradient of log likelihood. Means can appear close while individual runs have large errors; RMSE includes both mean error and seed dispersion.

'''+table(['Model','Parameter','Reference']+names,score_rows)+'''

## Particle ESS

These are averages over time and design seeds. Higher ESS alone does not establish a more accurate likelihood or score.

'''+table(['Model']+names,ess_rows)+'''

## Interpretation and reference limitations

Uniform parents have lower observed score RMSE than weighted parents in both range–bearing components, five of six predator–prey components and all three nominal SIR T=50 components. Weighted parents have lower observed score RMSE in both KSC components and three of four LGSSM components. Weighted log-likelihood RMSE is lower for KSC, range–bearing and nominal SIR T=50; uniform is lower for LGSSM and predator–prey. These mixed results do not support a universally more accurate parent rule.

The KSC persistence component is less accurate under either conditional scheme than under the original three-branch proposal. Predator–prey weighted scores have higher RMSE than the original in all six components. These adverse comparisons remain important despite the ESS increase.

LGSSM uses exact Kalman likelihood and score. KSC uses a refined grid for the seven-mixture observation target, not the unapproximated original stochastic-volatility likelihood. Predator–prey uses the mean of two adapted Zhao–Cui local-score fits; fit/support and radius errors are unresolved. Range–bearing uses 16 particle/Fisher reference replicates at 65,536 particles, with score MCSE [0.06502,0.02268] and possible finite-particle bias. SIR uses the saved rank40/fit2 adapted Zhao–Cui reference for the displayed values and retains rank20/fit17 as a sensitivity check. The nominal T=50 second-fit score is [108.57912,-50.42205,-30.38459], compared with [102.85487,-48.60288,-30.30761] in the displayed fit; uniform has lower score RMSE in all three components against either fit.

Across the 20 usable SIR cells, the following counts require the same direction against both reference fits. They describe observed RMSE, not a statistical ranking:

'''+table(['Quantity','Uniform lower','Weighted lower','Reference dependent'],
          [[q,d['uniform_lower_both_refs'],d['weighted_lower_both_refs'],d['reference_dependent']] for q,d in sir_counts.items()])+'''

The implemented analytical score is the derivative of the finite calculation with categorical selections locally fixed. Local finite-difference agreement checks that quantity; it does not prove equality to the model score or to a derivative after averaging over parameter-dependent selections.

## Decision and inference status

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Complete consolidated descriptive comparison | All requested score and likelihood estimates and RMSE shown | Saved validity checks pass; inadequate SIR references excluded | Eight seeds, fixed data, numerical-reference error and untuned conditional controls | Diagnose score propagation and categorical selection effects before accuracy promotion | Universal superiority or scientific default readiness |

| Inference item | Status |
| --- | --- |
| Hard veto screen | No numerical execution failure in the reported arms; 20 SIR reference cells meet saved screens |
| Statistically supported ranking | None newly established; this aggregation performs no new ranking test. The SIR source report retains two exploratory, uncorrected componentwise findings |
| Descriptive differences | All displayed means, RMSE comparisons and counts; conditional methods remain numerically viable |
| Default readiness | Not established; heuristic losses and unresolved score errors remain promotion vetoes under the source plans |
| Next evidence needed | Mechanism checks, scope-specific calibration on disjoint data, stronger references where needed, and predeclared paired uncertainty on independent datasets |

Strongest alternative explanations are finite-reference error, controls calibrated for the original proposal, and the distinction between branchwise finite-program derivatives and model scores. More accurate references and independently calibrated repeated datasets could alter the descriptive ordering. Failure of a candidate on a score component is not a rejection of the research direction.

## Evidence and reproducibility

- Four-model report: observation-parent-four-model-results-20261028.md.
- Nonlinear extension: observation-parent-nonlinear-results-20261028.md.
- SIR campaign: observation-parent-sir-accuracy-result-20261028.md.
- This report's script, CSV, JSON and source-hash manifest: artifacts/observation-parent-all-model-comparison-20261028/.

Status: COMPLETE. This is CPU standard-library post-run reporting only, with no scientific framework import or GPU use. No experiments or background processes remain for this reporting task.
'''
(ROOT/'docs/plans/observation-parent-all-model-comparison-20261028.md').write_text(text)
(OUT/'summary.json').write_text(json.dumps(dict(models=models,sir_nominal=sir_nominal,sir_all_usable_cells=sir_cells,
    sir_comparison_counts=sir_counts,checked_rows=checked_rows,ranking='descriptive only; no new statistical test'),indent=2)+'\n')
(OUT/'manifest.json').write_text(json.dumps(dict(task='Consolidate existing original/uniform/weighted score and log-likelihood comparisons',
    command='python3 docs/plans/artifacts/observation-parent-all-model-comparison-20261028/assemble.py',
    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    cpu_only=True,framework_imports=False,scientific_runs=0,source_sha256=sources,checked_rows=checked_rows,
    scope='read-only source analysis, new report files only'),indent=2)+'\n')
print(json.dumps(dict(status='complete',checked_rows=checked_rows,models=len(models),sir_cells=len(sir_cells),
                     csv_rows=len(flat),sir_counts=sir_counts,report='docs/plans/observation-parent-all-model-comparison-20261028.md')))
