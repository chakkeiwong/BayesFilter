"""Post-run fixed-inventory M21 report; never extends a sample count."""
from pathlib import Path
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[4]))
from bayesfilter.testing.inference_validation.engines.controller_stopping_report import summarize_controller_records
inventory=json.loads((ROOT/'confirmation-inventory.json').read_text())
assert (ROOT/'confirmation-cpu-r1/execution.json').is_file(),'wait for all declared cases'
results={};source_files={}
for case in inventory['cases']:
 path=ROOT/'confirmation-cpu-r1'/case/'summary.json'
 rows=json.loads(path.read_text())['rows'] if path.is_file() else [json.loads(p.read_text()) for p in sorted((path.parent).glob('rep-*/result.json'))]
 results[case]=summarize_controller_records(rows,inventory['replications_per_case'],oracle_alpha=.05/6,coverage_floor=.90)
 if path.is_file():source_files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 seeds=[tuple(seed) for row in rows for seed in row.get('streams',{}).values()]
 assert len(seeds)==len(set(seeds)),case
record={'cases':results,'source_files':source_files,'inventory':inventory,
 'reporter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'summary_module_sha256':hashlib.sha256((ROOT.parents[4]/'bayesfilter/testing/inference_validation/engines/controller_stopping_report.py').read_bytes()).hexdigest(),
 'all_oracle_screens_passed':all(r['oracle']['screen_passed'] for r in results.values()),
 'all_planned_replications_complete':all(r['complete'] for r in results.values()),
 'decision':'separate controller delivery, reported interval calibration and exact-oracle checks; no sampler ranking or default change'}
(ROOT/'confirmation-summary.json').write_text(json.dumps(record,indent=2)+'\n')
for case,r in results.items():
 print(case,'posterior',r['posterior_checks_passed']['count'],'warmup_caps',r['warmup_cap_count'],
       'coverage',[r['intervals'][arm]['unconditional']['count'] for arm in ('stopped','fixed')],
       'oracle',r['oracle']['covered_transient_expectation'],r['oracle']['screen_passed'])
