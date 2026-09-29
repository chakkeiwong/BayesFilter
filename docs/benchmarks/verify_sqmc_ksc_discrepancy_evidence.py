"""Independent post-run arithmetic, pairing, source and budget audit; CPU only."""
from __future__ import annotations
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'docs/plans/artifacts/sqmc-ksc-discrepancy-20260929/attempt-01'
ROUTES=('iid_dual_cap','previous_inverse_cdf','repaired_permutation','repaired_permutation_ablation')


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks=[]
    def check(name, passed, detail=None):
        checks.append(dict(check=name,passed=bool(passed),detail=detail))
    ledger=read(BASE.parent/'budget.json');summary=read(BASE/'analysis-01/summary.json')
    rows=read(BASE/'analysis-01/all-evaluations.json')
    expected={'replay':8,'derivative':24,'particles':192,'numerics':144,'mechanism':8}
    check('evaluation_counts',all(sum(r['phase']==p for r in rows)==n for p,n in expected.items()),expected)
    check('all_evaluations_valid',all(r['valid'] for r in rows))
    check('all_evaluations_marked_untuned',all(not r['claim_eligible'] and 'UNTUNED' in r['tuning'] for r in rows))
    check('all_coordinate_error_arithmetic',all(
        abs(r['value']-r['reference_value']-r['value_error'])<1e-10 and
        all(abs(a-b-e)<1e-11 for a,b,e in zip(r['score'],r['reference_score'],r['score_error'])) and
        abs(math.dist(r['score'],r['reference_score'])-r['score_l2_error'])<1e-11 for r in rows))
    fd=summary['finite_differences']
    check('48_fd_checks_pass',len(fd)==48 and all(x['status']=='pass' for x in fd))
    independent_fd=True
    for x in fd:
        selected=[r for r in x['ladder'] if r['step'] in x['steps']]
        independent_fd &= len(selected)==2 and all(r['branch_changes']==0 for r in selected)
        independent_fd &= abs(selected[0]['centered']-selected[1]['centered'])<=1e-4*(1+abs(x['analytical_score']))
        independent_fd &= abs(selected[1]['centered']-x['analytical_score'])<=2e-4*(1+abs(x['analytical_score']))
    check('fd_classification_recomputed',independent_fd)
    check('chunk_policy',all(r['chunk_size']=={1008:1008,2016:2016,4032:2016}[r['n']] for r in rows))
    numeric=[r for r in rows if r['phase']=='numerics']
    baseline={(r['route'],r['data_seed'],r['design_seed']):r for r in numeric if r['arm']=='baseline'}
    check('numerical_arms_use_identical_inputs',all(r['design_sha256']==baseline[(r['route'],r['data_seed'],r['design_seed'])]['design_sha256'] for r in numeric))
    replication=[r for r in rows if r['phase']=='particles']
    inv={(r['data_seed'],r['n'],r['design_seed']):r for r in replication if r['route']=='previous_inverse_cdf'}
    check('three_sqmc_routes_share_designs',all(r['design_sha256']==inv[(r['data_seed'],r['n'],r['design_seed'])]['design_sha256'] for r in replication if r['route']!='iid_dual_cap'))
    variance_ok=True;mean_ok=True
    for item in summary['replications']:
        group=[r for r in replication if all(r[k]==item[k] for k in ('data_seed','route','n'))]
        decomp=item['empirical_squared_error_decomposition']
        variance_ok &= abs(decomp['mean_squared_norm_error']-decomp['squared_norm_mean_error']-decomp['centered_design_variation'])<1e-10
        for j in range(2):
            es=[r['score_error'][j] for r in group]
            mean_ok &= abs(sum(es)/8-item['score_error'][j]['mean'])<1e-12
            variance_ok &= abs(statistics.variance(es)/8-item['mean_score_error_estimated_covariance'][j][j])<1e-12
    check('replication_mean_arithmetic',mean_ok)
    check('variance_decomposition_and_mean_covariance',variance_ok)
    with (BASE/'analysis-01/values-and-scores.csv').open() as f:
        csv_rows=list(csv.DictReader(f))
    check('csv_complete',len(csv_rows)==len(rows))
    total=sum(a['wall_seconds'] for a in ledger['attempts'])
    check('budget_reconciles',abs(total-ledger['diagnostic_seconds'])<1e-8 and abs(ledger['prior_charged_seconds']+total-ledger['charged_seconds'])<1e-8 and abs(43200-ledger['charged_seconds']-ledger['remaining_gpu_seconds'])<1e-8)
    check('budget_and_deadline_respected',total<=ledger['allocation_seconds'] and ledger['charged_seconds']<=43200 and all(datetime.fromisoformat(a['finished_utc'])<=datetime.fromisoformat(ledger['deadline_utc']) for a in ledger['attempts']))
    runtime_hashes=None;runtime_unchanged=True;all_meta=True;source_ok=True
    for a in ledger['attempts']:
        out=Path(a['output']);m=read(out/'manifest.json')
        all_meta &= a['returncode']==0 and m['status']=='complete' and m['source_unchanged'] and m['jit_compile'] and not m['tf32'] and m['gpu_memory_policy']['all_physical_devices_memory_growth']
        all_meta &= 'GPU:0' in m['framework_gpu_probe']['device'] and out.with_suffix('.log').is_file()
        snapshot=Path(a['source_snapshot'])
        for p,h in m['source_sha256'].items():source_ok &= sha(snapshot/p)==h
        current={p:h for p,h in m['source_sha256'].items() if p.startswith(('bayesfilter/','experiments/'))}
        if runtime_hashes is None:runtime_hashes=current
        runtime_unchanged &= current==runtime_hashes
    check('successful_gpu_xla_provenance_and_logs',all_meta)
    check('saved_source_hashes',source_ok)
    check('same_filter_runtime_all_phases',runtime_unchanged)
    check('mechanism_endpoint_parity',len(summary['mechanism'])==8 and all(r['parity_max_error']<=1e-8 for r in summary['mechanism']))
    increment_ok=True
    for r in rows:
        if r['phase']!='mechanism':continue
        for j,tr in enumerate(r['mechanism']['direction_traces']):
            increment_ok &= abs(sum(tr['actual_log_increment'])-r['value'])<=1e-8
            increment_ok &= abs(sum(tr['actual_score_increment'])-r['score'][j])<=1e-8
    check('trace_increments_sum_to_endpoint',increment_ok)
    check('absolute_errors_match_signed_errors',all(
        all(abs(abs(e)-a)<=1e-12 for e,a in zip(r['score_error'],r['absolute_score_error']))
        for r in rows))
    record=dict(status='pass' if all(c['passed'] for c in checks) else 'fail',checks=checks,
                command=[sys.executable,*sys.argv],cpu_only=True,gpu_seconds=0,
                meaning='Evidence integrity and arithmetic checks; not proof of scientific correctness or superiority.',
                independent_reviewer=False)
    target=BASE/'analysis-01/terminal-validation.json'
    if target.exists():raise RuntimeError('Preserve previous terminal audit; choose a new version')
    target.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(status=record['status'],checks=len(checks),failed=[c for c in checks if not c['passed']])))
    return 0 if record['status']=='pass' else 1


if __name__=='__main__':raise SystemExit(main())
