"""Independent CPU reference calculation; no retained-chain reanalysis."""
import json
import math
from pathlib import Path

import tensorflow as tf
import tensorflow_probability as tfp

out = Path(__file__).parent
distribution = tfp.distributions.StudentT(tf.constant(36., tf.float64),
    tf.constant(0., tf.float64), tf.constant(1., tf.float64))
rows = []
for functions, families in ((6, 1), (13, 4)):
    critical = float(distribution.quantile(tf.constant(1-.05/(2*functions*2*families), tf.float64)))
    assert math.isfinite(critical) and critical > 0
    row = dict(function_count=functions, comparison_families=families, critical=critical,
        ideal_iid_radius={str(n): critical*math.sqrt(.25/(4*n)) for n in (1000, 2000, 4000, 8000)},
        ideal_draws_per_chain_radius_02=math.ceil(critical**2*.25/(4*.02**2)),
        ideal_draws_per_chain_radius_01=math.ceil(critical**2*.25/(4*.01**2)))
    rows.append(row)
closed = json.loads(Path('docs/plans/artifacts/surrogate-hmc-all-models-20260922-m10-inference-02/P3-M10-inference/report.json').read_text())
look = closed['arms']['B']['looks'][-1]
assert look['family_functions'] == 13 and look['comparison_families'] == 4
critical_from_archived_interval = (look['upper'][0]-look['difference'][0]-look['reference_allowance'][0])/max(row[0] for row in look['batch_mcse'])
assert abs(critical_from_archived_interval-rows[1]['critical']) < 1e-12
report = dict(status='COMPLETED_DESIGN_CALCULATION', rows=rows,
    runtime_critical_parity=True, cpu_only=True, gpu_intentionally_hidden=True,
    posterior_look=False, power_established=False, reference_allowance_assumed=0.,
    interpretation='Expected IID radius is a design diagnostic, not a bound on realized uncertainty or a power calculation')
(out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
print(json.dumps(report))
