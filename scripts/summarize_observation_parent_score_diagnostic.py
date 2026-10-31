"""Post-run diagnostic reporting only; no TensorFlow import or admission logic."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'docs/plans/artifacts/ledh-parent-selection-score-20261028'


def read(path):
    return json.loads((ART/path).read_text())


def save(path, data):
    (ART/path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def summary(rows, labels, reference):
    result = []
    for mode, name in enumerate(labels):
        valid = [r for r in rows if r['valid'][mode] and not r['nonfinite_score'][mode]
                 and r['steps'][mode] == 10 and all(r['failed_checks'][mode])]
        values = [[r['score'][mode][k] for r in valid] for k in range(3)]
        result.append(dict(method=name, n=len(valid), requested_or_available_rows=len(rows),
            mean_score=[statistics.mean(x) for x in values],
            mcse=[statistics.stdev(x)/math.sqrt(len(x)) for x in values],
            rmse=[math.sqrt(statistics.mean((x-reference[k])**2 for x in values[k])) for k in range(3)]))
    return result


def paired(a, ai, b, bi, reference):
    bb = {r['seed']: r for r in b}
    aa = [r for r in a if r['seed'] in bb]
    result = []
    for k in range(3):
        differences = [(r['score'][ai][k]-reference[k])**2
                       -(bb[r['seed']]['score'][bi][k]-reference[k])**2 for r in aa]
        n = len(differences)
        mean = statistics.mean(differences)
        se = statistics.stdev(differences)/math.sqrt(n)
        critical = {6: 2.571, 8: 2.365}[n]
        result.append(dict(parameter=k, n=n, mean_squared_distance_change=mean,
            approximate_t95=[mean-critical*se, mean+critical*se],
            candidate_closer_count=sum(x < 0 for x in differences)))
    return result


def table(rows):
    return '\n'.join('| '+r['method']+' | '+str(r['n'])+' | '+
                     ' | '.join(f'{x:.2f}' for x in r['rmse'])+' |' for r in rows)


def main():
    main_rows = read('comparison-01/rows.json')
    controls = read('controls-02/rows.json')
    original_result = read('comparison-01/result.json')
    reference = original_result['reference']
    main_labels = ['Original three-branch', 'Conditional / uniform parents', 'Conditional / weighted parents']
    control_labels = ['Identity only', 'Conditional uniform only', 'Conditional adapted only']
    main_summary = summary(main_rows, main_labels, reference)
    control_summary = summary(controls, control_labels, reference)
    seeds = {r['seed'] for r in controls}
    matched_main = [r for r in main_rows if r['seed'] in seeds]
    matched_summary = summary(matched_main, main_labels, reference)
    assert len(main_rows) == 8 and len(controls) == 6
    assert all(s['n'] == 8 for s in main_summary)
    assert all(s['n'] == 6 for s in control_summary)
    allocation = paired(main_rows, 2, main_rows, 1, reference)
    heuristic = []
    for mi in range(3):
        for ci in range(3):
            heuristic.append(dict(candidate=main_labels[mi], control=control_labels[ci],
                matched_seeds=sorted(seeds),
                observed_candidate_worse_components=[k for k in range(3)
                    if matched_summary[mi]['rmse'][k] > control_summary[ci]['rmse'][k]],
                paired=paired(matched_main, mi, controls, ci, reference)))
    stress_main = read('comparison-01/stress.json')
    stress_control = read('controls-02/stress.json')
    stress = []
    for label, result, mode in [(main_labels[k], stress_main, k) for k in range(3)]+[
            (control_labels[0], stress_control['identity'], 0),
            (control_labels[1], stress_control['conditional'], 0),
            (control_labels[2], stress_control['conditional'], 1)]:
        stress.append(dict(method=label, valid=result['valid'][mode],
            score=result['score'][mode], ess=result['ess'][mode], score_accuracy='not checked: no matched reference'))
    max_cov = max(v for r in main_rows+controls for row in r['covariance_error'] for v in row)
    pure_ess_deviation = max(abs(v-1008) for r in controls for v in r['ess'][2])
    decision = dict(question='Does observation-weighted parent selection improve recursive score accuracy?',
        evidence_class='diagnostic distances to an uncertified approximation',
        reference=reference, reference_conditional_mcse=original_result['reference_conditional_mcse'],
        reference_oracle_status='not_certified', main_eight_streams=main_summary,
        matched_six_streams=matched_summary+control_summary,
        weighted_minus_uniform_paired=allocation, heuristic_comparisons=heuristic,
        stress=stress, hard_numerical_veto='none in completed valid runs',
        invalid_attempt='controls-01: changed target numerator; discarded',
        maximum_covariance_error=max_cov, pure_adapted_maximum_ess_deviation=pure_ess_deviation,
        heuristic_dominance_verdict='PROMOTION_VETO: observed componentwise underperformance; stress accuracy unavailable',
        statistically_supported_accuracy_ranking=False, default_readiness=False,
        allocation_conclusion='Mixed descriptive effect: weighted parents farther on log kappa and observation noise, closer on log nu',
        control_completion='Six of eight requested seeds; hard timeout during seventh, no selective exclusions',
        uncertainty='Intervals exploratory, heavy-tailed small sample, reference approximation error not quantified',
        next_action='Fresh scoped plan: independent model-score reference and particle/seed convergence; no further runs in this budget')
    save('decision.json', decision)
    save('controls-02/recovery-summary.json', dict(status='budget_stopped_external_timeout',
        exit_code=124, completed_seeds=sorted(seeds), requested_seeds=list(range(281101,281109)),
        valid_completed_streams=18, completed_row_elapsed_seconds=342.9768709980417,
        process_timeout_seconds=380, summary=control_summary,
        original_manifest_preserved=True, reconstructed_from='rows.json; no numerical rerun',
        failure='Pre-row elapsed check did not reserve enough time for an entire next row',
        repair='Checkpoint summaries atomically after every complete row and reserve measured next-row time',
        limitation='A first row may still reach the outer timeout; completed evidence remains recoverable'))
    attempts = [
        dict(attempt='smoke-01', seconds=15, timing='conservative charge', outcome='wrong environment ABI; no numerical evidence'),
        dict(attempt='smoke-02', seconds=read('smoke-02/smoke.json')['wall_seconds'], timing='measured internal', outcome='passed'),
        dict(attempt='comparison-01', seconds=original_result['wall_seconds'], timing='measured internal', outcome='24 valid streams'),
        dict(attempt='density-regression', seconds=6.15, timing='pytest reported', outcome='2 passed'),
        dict(attempt='controls-01', seconds=190, timing='conservative charge', outcome='invalid target; interrupted'),
        dict(attempt='final-recurrence-regression', seconds=23.65, timing='pytest reported', outcome='1 passed'),
        dict(attempt='controls-02', seconds=380, timing='outer timeout', outcome='18 valid streams; 2 requested seeds incomplete')]
    subtotal = sum(x['seconds'] for x in attempts)
    save('stage-manifest.json', dict(git_head=read('comparison-01/manifest.json')['git_head'],
        plan='docs/plans/ledh-parent-selection-score-comparison-20261028.md',
        result='docs/plans/ledh-parent-selection-score-result-20261028.md',
        status='diagnostic stage closed; control ladder budget-limited', attempts=attempts,
        budget_seconds=900, charged_subtotal_seconds=subtotal,
        timing_limitation='Aggregate external startup/test overhead was not measured; strict total-budget compliance is not established',
        remaining_compute_budget_seconds=0, repaired_launches=2, cpu_only=True,
        gpu_status='intentionally hidden', dtype='float64', jit_compile=True,
        run_manifests=['smoke-02/manifest.json','comparison-01/manifest.json','controls-02/manifest.json'],
        reporting_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        postrun_regression='stdlib budget-reserve/atomic-checkpoint tests pass; all 10 source snapshot hashes verified'))
    note = f'''# Observation-weighted parent selection: recursive score comparison

28 October 2026. Diagnostic CPU FP64/XLA run, SIR d18, N=1008, T=10, theta=(0,0,0).

The conditional proposals produced much smaller observed score distances than the original three-branch proposal. Observation weighting alone did not improve every component: relative to the same conditional proposal with uniform parents, it was farther from the saved Zhao-Cui reference for log kappa and log observation noise, and closer for log nu. This is evidence to investigate conditional proposal geometry; it is not evidence that the score problem is solved.

## Matched comparison
The eight main streams use seeds 281101 through 281108, identical observations, initial normals, innovations and uniforms. Only the global branch changes; identity/local branches, equal branch probabilities and the complete shared analytical/reset recurrence remain. The uniform conditional arm separates the geometry change from observation-weighted allocation.

The saved Zhao-Cui score is {reference}, with conditional path MCSE {original_result['reference_conditional_mcse']}. Its own artifact says `oracle_status=not_certified`: one rank20/fit17 approximation with 100,000 conditional paths. Those MCSEs exclude rank, fit, support and approximation bias. The exact T10 observations and parameter convention match; serialized-tensor identities are preserved in the manifests. No Zhao-Cui source-route implementation was changed or newly certified.

The table reports root mean squared distance across streams to that fixed approximate score. Lower means closer to this reference; the columns are derivatives with respect to log parameter scales.

| Proposal | Streams | log kappa | log nu | log observation noise |
|---|---:|---:|---:|---:|
{table(main_summary)}

The approximate paired squared-distance intervals for both modified arms versus the original include zero in every component. The original arm has very large derivative tails; eight streams cannot establish a reliable accuracy ranking. Exploratory weighted-minus-uniform intervals appear in `decision.json`: the log-kappa interval barely excludes zero in the unfavorable direction, but it is an unadjusted, small-sample, post-run comparison and does not establish a confirmatory model-score ranking.

## Simple controls and the saved failure
Six of eight requested control seeds completed before the 380-second outer timeout interrupted the seventh. All six complete paired rows are retained; none was selected on its score. For a fair comparison, this table restricts all arms to those same six seeds. The pure controls still use the shared reset recurrence and are not independent exact filters.

| Proposal | Streams | log kappa | log nu | log observation noise |
|---|---:|---:|---:|---:|
{table(matched_summary+control_summary)}

The simple uniform-conditional control has smaller observed log-kappa error than either modified mixture. The pure adapted control has smaller observed observation-noise error, but much larger log-kappa error. Thus neither enhanced mixture clears the componentwise heuristic screen. These are descriptive losses and promotion vetoes, not statistically proven rankings.

On the separate saved seed1104/T2 failure, all six arms are numerically valid. Terminal ESS is approximately 2.40 (original), 6.50 (uniform conditional mixture), 392.51 (weighted conditional mixture), 3.06 (identity only), 6.51 (pure uniform conditional), and 1008 (pure adapted). There is no matched score reference for this failure sequence, so its score accuracy is not checked. Across completed T10 controls the pure adapted ESS differs from N by at most {pure_ess_deviation:.3g}, yet its parameter-score distances remain uneven. ESS is explanatory only.

## What was differentiated and checked
The computed score is the analytical derivative of the same finite recursive likelihood program, including upstream reset sensitivities, conditional means/covariances, predictive probabilities, and proposal density. Categorical indices are held locally fixed. Finite differences on unchanged branches therefore check the finite program; they do not establish an unbiased model-likelihood score, smoothness through parent switches, or HMC suitability.

For this SIR observation model, Q=I, H selects nine odd coordinates, and R=rI with r=100 exp(2 theta_noise). For parent transition mean m_i, ell_i=N(y;Hm_i,(1+r)I), alpha_i=ell_i/sum ell_j, c_i=m_i+H'(y-Hm_i)/(1+r), and C=I-H'H/(1+r). The weighted proposal identity q(x)=g(y|x) p_N(x)/Z_N, with Z_N=(1/N)sum ell_i, preserves the same transition-mixture target numerator. The conditional uniform arm changes c_i and C while leaving parent allocation uniform. This derivation is specific to the stated linear Gaussian observation model.

All 24 main and 18 completed control streams pass the recorded numerical/finite/reset checks; maximum covariance reconstruction error is {max_cov:.3g}. The original hook path is bitwise identical in value, score, particles and particle sensitivities. Nine main finite-difference checks and six pure-control checks pass with unchanged parent indices. Independent density/total-derivative and model tests pass (2 tests); the final shared-recurrence versus independent directional implementation test passes (1 test). The frozen `failed_checks` JSON field is misnamed: it stores all-step pass flags, so true means passed.

## Repairs, provenance and budget
The first smoke used an incompatible TensorFlow environment and failed at compiled-op import. Switching to the previously working tftwogpu environment repaired it without installation. The first pure-control wrapper changed the target numerator along with the denominator; its results are wrong for this comparison and discarded. The repaired optional proposal-density override preserves the numerator. The constant-weight pure-adapted invariant catches this target error, which finite differences alone did not. The main three-arm comparison was unaffected.

The control timeout exposed a reporting problem: the runner only checked elapsed time before starting a row and wrote summaries at normal exit. Completed rows survived. Reporting now reserves time based on previous paired-row duration and atomically checkpoints rows, manifest and summary after each completed row. A standard-library regression checks both behaviors. The recovered six-row summary is a separate file; original manifests and source snapshots are preserved. No numerical rerun followed the timeout.

Recorded/conservatively charged compute totals {subtotal:.2f} seconds against the 900-second allocation. Aggregate external startup/test overhead was not measured, so strict aggregate budget compliance is not established. No compute remains allocated in this stage. Complete per-run commands, seeds, TensorFlow version, Git HEAD, source hashes and logs are in the linked manifests; all ten frozen source files match their recorded hashes. CPU devices were used deliberately; GPU was hidden before framework import.

## Decision
| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain conditional geometry as a diagnostic repair candidate | Much smaller observed distances; no supported ranking | Completed runs valid; heuristic screen not cleared | Heavy tails, eight streams, approximate reference | Independent model-score reference and particle/seed convergence under a fresh scope | General accuracy improvement |
| Do not promote observation weighting as the score solution | Mixed effect across parameters | Simple controls are closer on some components | Parent-switch bias and changed ancestry | Isolate switching/finite-N bias while keeping geometry fixed | Score problem solved |
| Close this bounded stage | Main comparison complete; controls 6/8 | Timeout/budget stop, not scientific-direction rejection | Incomplete control replication | Preserve evidence; no further runs in this allocation | Full ladder completion |

| Inference status | Finding |
|---|---|
| Hard veto screen | Valid completed main/control rows; controls-01 invalid and excluded |
| Statistically supported ranking | None for model-score accuracy |
| Descriptive differences | Large main-arm distance reduction; mixed allocation effect; componentwise heuristic losses |
| Default readiness | No: untuned diagnostic, CPU FP64, discontinuous categorical program, uncertified reference |
| Next evidence needed | Independent reference; larger paired replication; particle convergence; conditional stress accuracy |

Post-run red team: the strongest alternative explanation is that an uncertified reference and finite-program/categorical bias produce apparent accuracy differences. Better agreement with a certified independent model score, replicated across particle counts and stress situations, could overturn the current componentwise interpretation. The weakest evidence is the small number of heavy-tailed streams and missing stress oracle. The timeout invalidates neither the completed rows nor the research direction; it limits the control comparison.

## Evidence paths
- Plan: `ledh-parent-selection-score-comparison-20261028.md`.
- Root: `artifacts/ledh-parent-selection-score-20261028/`.
- Main: `comparison-01/result.json`, `rows.json`, `postanalysis.json`, `manifest.json`, `source-snapshot/`.
- Controls: `controls-02/rows.json`, `recovery-summary.json`, `smoke.json`, `stress.json`, `manifest.json`, `source-snapshot/`.
- Invalid attempt: `controls-01/invalid-attempt.json`; prior evidence preserved.
- Machine-readable assembly: `decision.json`, `stage-manifest.json`; all intervals are exploratory.
- Tests: `density-regression.log`, `final-recurrence-regression.log`.
- Reference: `/home/chakwong/BayesFilter-SQMC/docs/plans/artifacts/ledh-zhao-horizons-20261006-01/038-quadratic-sir_d18-rank20-fit17-T10/result.json`; observations from sibling `037-zhao-sir_d18-rank20-fit17-runtime-repair/sir_austria-input-dataset.json`.
'''
    (ROOT/'docs/plans/ledh-parent-selection-score-result-20261028.md').write_text(note)
    print(json.dumps(dict(status='reported', main_rows=8, control_rows=6,
        charged_seconds=subtotal, main_rmse=[r['rmse'] for r in main_summary],
        control_rmse=[r['rmse'] for r in control_summary],
        report='docs/plans/ledh-parent-selection-score-result-20261028.md')))


if __name__ == '__main__':
    main()
