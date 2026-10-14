"""Post-run source-only descriptive feasibility analysis; no acceptance gate."""
import argparse, hashlib, json, math
from pathlib import Path

ROOT=Path('/home/chakwong/BayesFilter')
SOURCE=ROOT/'docs/plans/artifacts/controlled-smc-kdm-replication-20260928/csmc-qualification-20261009-final-013640/summary.json'

def main(output):
    summary=json.loads(SOURCE.read_text())
    rows=[]
    for row in summary['strata']:
        inf=row['inference']
        variance=inf['source_port_sample_variance'][0]
        count=inf['pairs']
        rows.append(dict(cell=row['cell'],forcing=row['forcing'],attempted=row['attempted'],
            source_valid=row['source_valid'],source_failures=len(row['source_failure_ids']),
            source_variance_logZ=variance,source_residual_effective_count=inf['residual_effective_count'][0],
            source_max_likelihood_share=inf['maximum_likelihood_share'][0],
            descriptive_logZ_mean_se=math.sqrt(variance/count),
            source_sample_scope='SOURCE_VALUES_CONDITIONAL_ON_VALID_PAIR',sample_count=count))
    # The saved Q2 source fields are conditional on both methods being valid.
    # For this dataset all source-valid runs are valid pairs; verify, don't assume.
    assert all(r['source_valid']==r['sample_count'] for r in rows)
    result=dict(schema='csmc_author_feasibility_diagnostic_v1',cpu_only=True,
        gpu_intentionally_hidden=True,source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        role='EXPLORATORY_REUSED_EVIDENCE_NOT_FRESH_CALIBRATION',rows=rows,
        cells=len(rows),source_failures=sum(r['source_failures'] for r in rows),
        source_variance_range=[min(r['source_variance_logZ'] for r in rows),max(r['source_variance_logZ'] for r in rows)],
        residual_effective_count_range=[min(r['source_residual_effective_count'] for r in rows),max(r['source_residual_effective_count'] for r in rows)],
        mean_logZ_se_range=[min(r['descriptive_logZ_mean_se'] for r in rows),max(r['descriptive_logZ_mean_se'] for r in rows)],
        decision='NO_INDEPENDENT_DEGRADATION_BOUND_IDENTIFIED',
        interpretation='Sampling precision varies by cell. Author variability alone cannot set an equivalence margin. The mean-logZ SE is descriptive and is not a likelihood-equivalence test.',
        unmeasured=['fresh source repeatability','reliable tail/moment convergence','fresh runtime/cost','a substantively justified stochastic equivalence bound'],
        replication='STOCHASTIC_EQUIVALENCE_UNRESOLVED_NO_NEW_FAILURE_CLAIM')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('cells','source_failures','source_variance_range','residual_effective_count_range','decision')}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
