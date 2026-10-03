"""Compare the dated seed map with current named operations; diagnostic only."""
import ast, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
from tests import test_hmc_kernel_tuning_windowed_mass as fixtures
old=json.loads(fixtures._G1A_SOURCE_COVERAGE_MANIFEST.read_text())
current=fixtures._g1a_manifest_sites_from_final_sources()
changes=[]
for a,b in zip(old["sites"],current):
    if a!=b: changes.append({"site_id":a["site_id"],"changes":{k:[a[k],b[k]] for k in a if a[k]!=b[k]}})
print(json.dumps(changes,indent=2))
Path(__file__).with_name("seed-site-differences.json").write_text(json.dumps(changes,indent=2)+"\n")
