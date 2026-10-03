from pathlib import Path
import contextlib,hashlib,json,shutil,subprocess,sys,tarfile
from datetime import datetime,timezone
ROOT=Path('/home/chakwong/BayesFilter-SQMC')
sys.path.insert(0,str(ROOT/'docs/benchmarks'))
import run_sqmc_expanded_comparison as campaign
import run_sqmc_expanded_repair as repair
import report_sqmc_expanded as report
import plot_sqmc_expanded_diagnostics as plots
base=ROOT/'docs/plans/artifacts/sqmc-expanded-20260928/renewal-01'
budget=json.loads((base/'budget.json').read_text())
assert all(item['status']!='running' for item in budget['launches']), 'Campaign still active'
assert budget['launches'][-1]['action']=='repair-run'
assert budget['launches'][-1].get('return_code')==0
old=json.loads((base/'run-01/results.json').read_text())['cases']
new=json.loads((base/'repair-run-01/results.json').read_text())['cases']
cases=[c for c in old if c['scope'].startswith('p44')]+new
out=base/'final-evidence-01';out.mkdir()
aggregate=out/'aggregate';aggregate.mkdir()
campaign.assemble(aggregate,cases)
with (out/'report.log').open('w') as log,contextlib.redirect_stdout(log):
    report.report(aggregate,out/'report')
audit=json.loads((out/'report/audit.json').read_text())
assert not audit['engineering_failures'] and not audit['missing_units']
assert audit['recorded_units']==32 and audit['recorded_final_cells']==128
assert audit['exported_score_entries']==8480
plots.plot(aggregate/'results.json',out/'report/figures')
provenance=out/'provenance';provenance.mkdir()
repair.configure_profile();hashes=campaign.source_hashes()
current=campaign.numerical_sources(hashes)
for case in cases:
    meta=json.loads((Path(case['artifact_directory'])/'manifest.json').read_text())
    assert campaign.numerical_sources(meta['source_sha256'])==current
for name in ('report_sqmc_expanded.py','plot_sqmc_expanded_diagnostics.py'):
    path=ROOT/'docs/benchmarks'/name;hashes[str(path.relative_to(ROOT))]=campaign.digest(path)
with tarfile.open(provenance/'source-final.tar.gz','x:gz') as archive:
    for name in sorted(hashes):archive.add(ROOT/name,arcname=name)
(provenance/'source-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
shutil.copy2(__file__,provenance/'assemble_final.py')
shutil.copy2(base/'budget.json',out/'budget-at-completion.json')
for label,command in (('git-status',['git','status','--short','--branch']),('git-head',['git','rev-parse','HEAD'])):
    (provenance/f'{label}.txt').write_text(subprocess.check_output(command,cwd=ROOT,text=True))
manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),command=[sys.executable,*sys.argv],
    cpu_gpu_status='CPU-only report assembly; GPU devices intentionally hidden by diagnostic plotter',
    valid_final_cells=audit['valid_final_cells'],recorded_final_cells=audit['recorded_final_cells'],
    recorded_scores=audit['recorded_score_entries'],engineering_findings=audit['engineering_failures'],
    numerical_sources_match_current=True,
    source_archive_sha256=campaign.digest(provenance/'source-final.tar.gz'),
    review_status='Executable terminal audit passed; narrative and final rendered-figure review pending')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(output=str(out),valid_cells=audit['valid_final_cells'],scores=audit['exported_score_entries'])))
