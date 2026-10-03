import ast,json
from pathlib import Path
root=Path.cwd()
art=root/'docs/plans/artifacts/hmc-repair-master-2026-09-16/m7-windowed-r1'
p=root/'bayesfilter/inference/hmc_kernel_tuning.py';s=p.read_text(); lines=s.splitlines(keepends=True);tree=ast.parse(s)
names=set(json.loads((art/'definitions.json').read_text()))
nodes=[n for n in tree.body if getattr(n,'name',None) in names]
assert len(nodes)==76
refs={n.id for d in nodes for n in ast.walk(d) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
constants=[]
while True:
 previous=set(refs)
 for n in tree.body:
  if isinstance(n,(ast.Assign,ast.AnnAssign)):
   targets=n.targets if isinstance(n,ast.Assign) else [n.target]
   if any(isinstance(t,ast.Name) and t.id in refs for t in targets):
    if n not in constants: constants.append(n)
    refs.update(x.id for x in ast.walk(n) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Load))
 if previous==refs: break
imports=[]
for n in tree.body:
 if isinstance(n,(ast.Import,ast.ImportFrom)):
  selected=[a for a in n.names if (a.asname or a.name.split('.')[0]) in refs]
  if selected:
   new=ast.Import(names=selected) if isinstance(n,ast.Import) else ast.ImportFrom(module=n.module,names=selected,level=n.level)
   if len(selected)>2 and isinstance(new,ast.ImportFrom):
    imports.append(f'from {new.module} import (\n'+''.join(f'    {ast.unparse(a)},\n' for a in selected)+')')
   else: imports.append(ast.unparse(new))
header='''"""Windowed HMC preparation and frozen geometry/start-bank handoff.

This module owns the operational and historical diagnostic windowed stages.
Preparation does not issue candidate-set tuning authority. Public and historical
imports share these definitions; numerical policies and seed-site IDs are stable.
"""
from __future__ import annotations

'''
header+='\n'.join(imports)+'\nfrom typing import TYPE_CHECKING\n'
header+='''from bayesfilter.inference.hmc_tuning import (
    WindowedMassAdaptationResult,
    build_windowed_warmup_schedule,
    validate_windowed_shrinkage_target,
    welford_covariance,
)

if TYPE_CHECKING:
    from bayesfilter.inference.hmc_kernel_tuning import (
        _HMCAttemptBudgetPolicy, _HMCPhaseAttemptState,
    )
'''
def bounds(n): return min([n.lineno]+[d.lineno for d in getattr(n,'decorator_list',[])])-1,n.end_lineno
def source(n):
 a,b=bounds(n);return ''.join(lines[a:b])
constant_names=[(n.targets[0] if isinstance(n,ast.Assign) else n.target).id for n in constants]
body='\n\n'+ '\n\n'.join(source(n).rstrip() for n in sorted(constants,key=lambda n:n.lineno))+'\n\n\n'
body+='\n\n\n'.join(source(n).rstrip() for n in nodes)+'\n'
old='owner_file="hmc_kernel_tuning.py"'
assert body.count(old)==1
body=body.replace(old,'owner_file="hmc_mass_adaptation.py"')
exports=['HMCStagedTimeoutPolicy','HMCWindowedMassStageConfig','HMCWindowedMassStageResult','WINDOWED_MASS_STAGE_NONCLAIMS','WindowedMassAdaptationConfig','WindowedMassAdaptationResult','build_operational_fixed_mass_hmc_adapter','build_windowed_warmup_schedule','run_hmc_windowed_mass_stage','run_windowed_mass_adaptation_diagnostic','validate_windowed_shrinkage_target','welford_covariance']
body+='\n\n__all__ = [\n'+''.join(f'    "{n}",\n' for n in exports)+']\n'
(root/'bayesfilter/inference/hmc_mass_adaptation.py').write_text(header+body)
remove=set()
for n in nodes+constants:
 a,b=bounds(n);remove.update(range(a,b))
new=''.join(l for i,l in enumerate(lines) if i not in remove)
aliases='from bayesfilter.inference.hmc_mass_adaptation import (\n'+''.join(f'    {n},\n' for n in constant_names+[n.name for n in nodes])+')\n\n'
anchor='from bayesfilter.runtime import stable_config_hash\n'
new=new.replace(anchor,anchor+'\n'+aliases,1)
new=new.replace('controller. Initial geometry and bootstrap screening live in hmc_geometry and\nhmc_bootstrap; imports here preserve historical callers and serialized names.\nWindowed preparation and configuration translation still depend on this module.','controller. Initial geometry, bootstrap screening and windowed preparation live\nin hmc_geometry, hmc_bootstrap and hmc_mass_adaptation. Imports here preserve\nhistorical callers and serialized names. Configuration translation remains here.')
p.write_text(new)
(art/'moved-constants.json').write_text(json.dumps(constant_names,indent=2)+'\n')
(art/'extract_windowed.py').write_text(Path(__file__).read_text())
print({'definitions':len(nodes),'constants':len(constants),'new_lines':len((header+body).splitlines()),'historical_lines':len(new.splitlines())})
