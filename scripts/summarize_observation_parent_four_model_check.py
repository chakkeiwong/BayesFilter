"""Post-run diagnostic reporting; no selection, tuning, or runtime decisions."""
from pathlib import Path
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/plans/artifacts/observation-parent-four-model-check-20261028'
REPORT = ROOT / 'docs/plans/observation-parent-four-model-results-20261028.md'
SEEDS = list(range(281301, 281309))
LABELS = {'original_three_branch': 'Original three-branch',
          'transition_only': 'Transition-only control',
          'conditional_uniform': 'Conditional, uniform parents',
          'conditional_observation_weighted': 'Conditional, weighted parents'}

def read(path):
    return json.loads(path.read_text())

def rmse(errors):
    return math.sqrt(statistics.mean(x*x for x in errors))

def fmt(x):
    return f'{x:.6g}'

def vector(xs):
    return '[' + ', '.join(fmt(x) for x in xs) + ']'

def errors(rows, reference):
    return {'value_rmse': rmse([r['value']-reference['value'] for r in rows]),
            'score_rmse': [rmse([r['score'][j]-v for r in rows])
                           for j, v in enumerate(reference['score'])]}

models = []
count = 0
for folder in sorted(OUT.glob('0*-*')):
    if not folder.is_dir():
        continue
    result = read(folder/'result.json')
    manifest = read(folder/'manifest.json')
    rows = read(folder/'rows.json')
    t = manifest['horizon']
    assert manifest['seeds'] == SEEDS
    assert manifest['particles'] == 1008
    assert result['reference'] == manifest['reference']
    for row in rows:
        assert row['finite'] and row['valid'] and row['steps_completed'] == t
        assert all(math.isfinite(x) for x in [row['value'], *row['score']])
        assert len(row['trace']) == len(row['validity_trace']) == t
        assert all(len(v) == len(row['validity_fields']) and all(x is True for x in v)
                   for v in row['validity_trace'])
        assert len(row['score']) == len(manifest['parameter_names'])
    grouped = {arm: [r for r in rows if r['arm'] == arm] for arm in result['summary']}
    for arm, group in grouped.items():
        assert [r['seed'] for r in group] == SEEDS
        computed = errors(group, result['reference'])
        assert math.isclose(computed['value_rmse'], result['summary'][arm]['value_rmse'], abs_tol=1e-12)
        for actual, saved in zip(computed['score_rmse'], result['summary'][arm]['score_rmse']):
            assert math.isclose(actual, saved, abs_tol=1e-12)
    paired = {}
    weighted = grouped.get('conditional_observation_weighted')
    if weighted:
        for base in ['conditional_uniform', 'original_three_branch']:
            differences = [w['ess_mean']-b['ess_mean'] for w, b in zip(weighted, grouped[base])]
            paired[base] = {'weighted_ess_wins': sum(d > 0 for d in differences),
                            'pairs': len(differences), 'mean_ess_difference': statistics.mean(differences)}
    per_fit = {}
    for fit in result['reference'].get('fits', []):
        per_fit[str(fit['fit_seed'])] = {arm: errors(group, fit) for arm, group in grouped.items()}
    models.append({'model': result['model'], 'folder': folder.name,
                   'target_id': manifest['target_id'], 'horizon': t,
                   'theta': manifest['theta'], 'parameter_names': manifest['parameter_names'],
                   'controls': manifest['controls'], 'beta': manifest['beta'],
                   'conditional_supported': manifest['conditional_supported'],
                   'unsupported_reason': manifest['unsupported_reason'],
                   'reference': result['reference'], 'summary': result['summary'],
                   'paired_ess': paired, 'individual_reference_fit_errors': per_fit,
                   'wall_seconds': result['wall_seconds'], 'rows': len(rows)})
    count += len(rows)
assert len(models) == 4 and count == 96
wall = sum(m['wall_seconds'] for m in models)
summary = {'models': models, 'completed_valid_rows': count, 'all_validity_traces_pass': True,
           'execution_wall_seconds': wall, 'launches': 4, 'budget_seconds': 3600,
           'statistical_ranking': 'not established; predeclared descriptive comparison',
           'heuristic_dominance': 'not established; controls incomplete and range-bearing comparison mixed',
           'default_readiness': 'not assessed by this diagnostic'}
(OUT/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
lines = ['# Four-model likelihood and score check', '',
'All 96 evaluations completed the full horizon with finite values, finite analytical scores, and all recorded validity checks passing. Observation-weighted parents increased mean particle ESS against uniform parents in all eight paired seeds for both supported models. Accuracy did not improve consistently: the predator–prey weighted arm had larger log-likelihood RMSE and larger RMSE in all six score components than the original three-branch arm. These are descriptive comparisons, not statistically established rankings.', '',
'## Scope and reproducibility', '',
'This is one fixed parameter configuration and one saved dataset per model, with N=1008 and eight paired design seeds, 281301–281308. LGSSM, KSC SV and predator–prey use T=50 and dataset 26100831; wrapped range–bearing uses T=20 and dataset 1704. Candidate evaluation used TensorFlow/TFP on GPU, FP64, XLA enabled, TF32 disabled, with verified memory growth. FP64 is an explicit diagnostic exception to the production precision policy. The CPU reference calculations are independent checks.', '',
'The claim being checked is agreement with the model log likelihood and its parameter gradient. The filter returns the finite-particle log-likelihood estimate and its implemented analytical recursive score. The references below approximate or compute the model quantities independently. The measurements do not establish equality of the finite filter score with the exact model score or prove the implementation of every derivative term.', '',
'The conditional route currently requires affine Gaussian observations. KSC SV uses a seven-mixture observation law and range–bearing uses a nonlinear wrapped observation law, so their conditional routes are unsupported. Their runs explicitly select the original proposal; they do not validate observation-weighted parents in those models. The transition-only control uses branch weights [1,0,0] with the same recurrence and OT reset. It is not an independent standard bootstrap particle filter.', '',
'Original-proposal controls and branch weights were frozen from saved settings. They were not retuned for the conditional arms, and the saved datasets were reused for this diagnostic. These are not untouched promotion runs. Each manifest records the actual parameters, data, source hashes, command, environment, device, precision, settings and runtime.', '',
f'The four successful launches consumed {wall:.3f} seconds ({wall/60:.2f} minutes) of the 3,600-second execution budget; no experimental launch required a retry. The initial environment probe exposed a custom-op ABI mismatch in tf-gpu; evaluation used tftwogpu, as specified in the pre-run plan.', '',
'Execution began at commit 14e6fbd59 on branch ledh-refactor-with-policy-fix. The worktree also contained pre-existing modified numerical dependencies. The source-and-input snapshot preserves every loaded local source and reference input by its original manifest hash, including the earlier LGSSM runner version. The later runner adds predator–prey identity assertions and stronger range–bearing reference validation without changing the numerical algorithm. Unrelated live source edits are not included in the new source-code commit.', '',
'Artifacts: [summary](artifacts/observation-parent-four-model-check-20261028/summary.json), [snapshot index](artifacts/observation-parent-four-model-check-20261028/snapshot-index.json), and [pre-run plan](observation-parent-four-model-check-20261028.md). Per-model folders preserve every seed, every score component, ESS and validity traces, manifests, logs and smoke checks.', '',
'## Accuracy by model', '',
'Error is estimate minus reference. RMSE is sqrt(mean(error squared)) across the eight seeds. Score vectors are reported in the parameter order shown for each model; their components have different units, so they are not collapsed into a cross-parameter norm. All likelihood columns refer to log likelihood.', '']
for m in models:
    ref = m['reference']
    lines += ['### '+{'lgssm':'LGSSM','ksc':'KSC SV','predator_prey':'Predator–prey','range_bearing':'Range–bearing'}[m['model']], '',
              f"Target: {m['target_id']}; T={m['horizon']}; theta={vector(m['theta'])}.", '',
              'Score order: '+', '.join(m['parameter_names'])+'.', '',
              f"Reference ({ref['kind']}): log likelihood {fmt(ref['value'])}; score {vector(ref['score'])}.", '',
              '| Arm | Log-likelihood bias | Log-likelihood RMSE | Score RMSE by component | Mean ESS |',
              '|---|---:|---:|---|---:|']
    for arm, s in m['summary'].items():
        lines.append(f"| {LABELS[arm]} | {fmt(s['value_bias'])} | {fmt(s['value_rmse'])} | {vector(s['score_rmse'])} | {fmt(s['ess_mean'])} |")
    lines += ['', '| Arm | Mean score by component | Score bias by component |', '|---|---|---|']
    for arm, s in m['summary'].items():
        lines.append(f"| {LABELS[arm]} | {vector(s['score_mean'])} | {vector(s['score_bias'])} |")
    lines.append('')
    for base, p in m['paired_ess'].items():
        lines += [f"Weighted versus {LABELS[base].lower()}: ESS higher in {p['weighted_ess_wins']}/{p['pairs']} paired seeds; mean ESS difference {fmt(p['mean_ess_difference'])}.", '']
    if m['model'] == 'lgssm':
        lines += ['The exact Kalman solution gives an unusually strong reference here. The three non-transition arms have similar observed errors. Weighted parents produce slightly lower RMSE in two score components and slightly higher RMSE in two relative to the original proposal; likelihood RMSE is also slightly higher. The ESS increase over uniform parents is only about 0.042 out of 1008 particles.', '']
    elif m['model'] == 'ksc':
        lines += [f"The independently refined grid uses 401, 801 and 1201 nodes with bound 40, plus 1201 nodes with bound 48. Maximum reported refinement differences are {ref['maximum_value_difference']:.3g} in log likelihood and {ref['maximum_score_difference']:.3g} in score. This checks the implemented seven-mixture target, rather than the original non-mixture stochastic-volatility likelihood. The observation-scale score has a visible positive mean error despite well-resolved grid refinement. The original proposal has lower observed RMSE than the transition-only control in all reported components.", '']
    elif m['model'] == 'predator_prey':
        lines += ['The central reference is the predeclared mean of two existing adapted Zhao–Cui fits, evaluated at the smallest local score radius, 0.0003125. It is not a certified oracle. The score was estimated by local quadratic fits in the physical parameter coordinates. The saved reference used 128 training and 64 holdout points, 100,000 paths and 20 jackknife groups. Its conditional standard errors exclude fit/support error and radius bias.', '',
                  '| Reference fit seed | Log likelihood | Score | Conditional score SE |', '|---|---:|---|---|']
        for fit in ref['fits']:
            lines.append(f"| {fit['fit_seed']} | {fit['value']:.10f} | {vector(fit['score'])} | {vector(fit['score_conditional_se'])} |")
        lines += ['', 'All individual-fit and radius results are retained. Recomputing the arm errors against each fit separately is included in summary.json; the larger weighted likelihood RMSE and all six larger weighted score RMSEs versus original occur against each individual fit as well. Averaging the fits does not establish that their systematic errors cancel. Relative to uniform parents, weighting raises ESS by about 36 particles, yet log-likelihood RMSE and five of six score RMSEs increase. Relative to the original proposal, log-likelihood RMSE and all six score RMSEs increase. The selected conditional settings or score propagation may warrant investigation; this check does not identify the cause or reject the broader research direction.', '']
    else:
        low, high = ref['refinements']
        lines += [f"The reference averages 16 particle-filter/Fisher replicates at 65,536 particles and retains the 16,384-particle comparison. MCSE for [log likelihood, score components] is {vector(ref['mcse'])}. Increasing the reference particle count changes log likelihood by {fmt(high['value']-low['value'])} and score by {vector([a-b for a,b in zip(high['score'],low['score'])])}. Monte Carlo standard errors do not bound finite-particle bias. The original arm has lower observation-scale score RMSE than the transition-only control, but higher likelihood and process-scale score RMSE. This mixed comparison prevents a dominance claim.", '']
lines += ['## Decision and inference status', '',
'| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |',
'|---|---|---|---|---|---|',
'| Complete the requested diagnostic | Every likelihood and score component compared | All 96 runs finite, valid, full horizon | One configuration/data set per model; numerical references for three models | Inspect the predator–prey component errors before a separate tuned validation | General accuracy or a supported ranking |',
'| Preserve the existing engineering default decision | This run has no promotion threshold | No numerical execution veto; incomplete heuristic comparison | Untuned conditional settings; no KSC/RB conditional implementation | Treat weighted parents as the selected engineering direction, with unresolved accuracy questions | Cross-model scientific default readiness |', '',
'| Inference item | Status |', '|---|---|',
'| Hard veto screen | PASS: every saved validity entry, finite value/score and full horizon checked; target and reference assertions passed |',
'| Statistically supported ranking | None; eight paired seeds were predeclared for descriptive comparison, without a ranking test |',
'| Descriptive-only differences | Component RMSE, bias, seed dispersion, ESS and runtime; the mean ESS rises against uniform in all 16 supported-model pairs |',
'| Default readiness | Not established; KSC/RB conditional support absent, conditional settings not retuned, FP64 diagnostic rather than production-precision admission |',
'| Next evidence needed | Scope-specific calibration on disjoint data, untouched multi-dataset replication with declared paired uncertainty, score-term/parity checks, and stronger numerical references where required |', '',
'The heuristic-dominance check is incomplete. These runs construct a transition-only control and retain original/uniform comparisons where available, but do not supply the full set of independent heuristic filters needed for promotion. Range–bearing is already mixed against the transition-only control. No candidate is rejected by a numerical validity veto; numerical viability is distinct from an accuracy ranking.', '',
'## Post-run skeptical review', '',
'The strongest alternative explanation for the predator–prey result is that controls calibrated for the original proposal fit the changed conditional proposal poorly. Reference approximation and finite-filter derivative behavior are additional explanations. Repeating the comparison with independently calibrated conditional settings and a strengthened score reference could overturn the observed pattern. The present result shows that higher ESS alone cannot certify lower score error.', '',
'The engineering evidence consists of completed runs and checked data/source identity. Numerical evidence consists of finite full-horizon outputs and the saved validity traces. Scientific evidence consists of the descriptive reference errors with the reference limitations above. Passing the first two does not establish the third as a superiority claim. No new mathematical method, Zhao–Cui source-faithfulness claim, canonical LEDH admission, HMC claim or broad default-policy promotion is made.', '']
REPORT.write_text('\n'.join(lines))
print(json.dumps({'report':str(REPORT.relative_to(ROOT)), 'valid_rows':count,
                  'wall_seconds':wall,'summary':str((OUT/'summary.json').relative_to(ROOT))}))
