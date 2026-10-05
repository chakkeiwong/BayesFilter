"""CPU reference mechanics check; no method ranking or tuning promotion."""
import sys,json
from pathlib import Path
root=Path('/home/chakwong/BayesFilter-SQMC')
sys.path.insert(0,str(root/'docs/benchmarks'))
import tensorflow as tf
import run_sqmc_tuning as runner
from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec
from bayesfilter.highdim.sqmc_campaign_tf import ROUTES
controls=dict(reset_epsilon=.4,reset_sinkhorn_steps=24,reset_balance_steps=12,
              correction_steps=1,correction_strength=.1,pairwise_steps=1,pairwise_strength=.03,
              flow_substeps=2)
spec=LGSSMSpec('p44',3)
out=root/'docs/plans/artifacts/sqmc-repair-20260925/06-tuning-workflow'
summary=[]
for route in ROUTES:
    result=runner.tune_route(route,2,12,[91001,91002],out/route,
        validation_seeds=[92001,92002],claim_seeds=[93001,93002],spec=spec,
        candidates=[controls,dict(controls,correction_strength=.12)],dtype=tf.float64,jit_compile=False)
    assert result is not None, f'{route}: failed complete valid tuning/validation/untouched checks'
    assert len(result['grid'])==2 and all(len(g['seed_results'])==2 for g in result['grid'])
    assert len(result['validation'])==2 and len(result['untouched_results'])==2
    assert all(len(row['score'])==4 and not row['claim_eligible'] for row in result['untouched_results'])
    summary.append(dict(route=route,calibration_cells=4,validation_cells=2,untouched_cells=2,
                        decision=result['decision'],valid=True,scientific_admission=False))
(out/'workflow-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
