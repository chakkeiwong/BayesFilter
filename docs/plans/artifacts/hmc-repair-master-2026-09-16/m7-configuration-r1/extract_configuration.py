"""Mechanical definition/constant extraction against the saved windowed baseline."""
import ast
import json
from pathlib import Path
import re

root=Path(__file__).resolve().parent
repo=root.parents[4]
p=repo/'bayesfilter/inference/hmc_kernel_tuning.py';s=p.read_text();lines=s.splitlines(keepends=True);tree=ast.parse(s)
names=set(json.loads((root/'definitions.json').read_text()))
nodes=[n for n in tree.body if getattr(n,'name',None) in names]
refs={n.id for d in nodes for n in ast.walk(d) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
constants=[]
while True:
 previous=set(refs)
 for n in tree.body:
  if isinstance(n,(ast.Assign,ast.AnnAssign)):
   targets=n.targets if isinstance(n,ast.Assign) else [n.target]
   if any(isinstance(t,ast.Name) and t.id in refs for t in targets):
    if n not in constants:constants.append(n)
    refs.update(x.id for x in ast.walk(n) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Load))
 if previous==refs:break
imports=[]
for n in tree.body:
 if isinstance(n,(ast.Import,ast.ImportFrom)):
  selected=[a for a in n.names if (a.asname or a.name.split('.')[0]) in refs]
  if selected:
   new=ast.Import(names=selected) if isinstance(n,ast.Import) else ast.ImportFrom(module=n.module,names=selected,level=n.level)
   if isinstance(new,ast.ImportFrom) and len(selected)>2:
    imports.append(f'from {new.module} import (\n'+''.join(f'    {ast.unparse(a)},\n' for a in selected)+')')
   else:imports.append(ast.unparse(new))
header='''"""HMC preparation presets, translation and geometry-scaled budget policy.

These definitions preserve the serialized public and historical policies. They
configure preparation; candidate-set authority belongs to the shared controller.
Historical loop configuration remains readable without importing its executor.
"""
from __future__ import annotations

'''+ '\n'.join(imports)+'\n\n'
def bounds(n):return min([n.lineno]+[d.lineno for d in getattr(n,'decorator_list',[])])-1,n.end_lineno
def source(n):
 a,b=bounds(n);return ''.join(lines[a:b]).rstrip()
ordered=sorted(nodes+constants,key=lambda n:n.lineno)
body='\n\n\n'.join(source(n) for n in ordered)+'\n'
exports=['HMCKernelTuningConfig','HMCTuneVerifyRepairLoopConfig','HMCGeometryScaledBudgetTimingPolicy','resolve_ordinary_hmc_selection_policy','ORDINARY_SHARED_EPSILON_SCREEN_POLICY_ID','ORDINARY_LEGACY_JOINT_L_EPSILON_POLICY_ID','ORDINARY_ENGINEERING_JOINT_L_EPSILON_POLICY_ID']
body+='\n\n__all__ = [\n'+''.join(f'    "{n}",\n' for n in exports)+']\n'
(repo/'bayesfilter/inference/hmc_configuration.py').write_text(header+body)
remove=set()
for n in ordered:
 a,b=bounds(n);remove.update(range(a,b))
new=''.join(l for i,l in enumerate(lines) if i not in remove)
constants_names=[(n.targets[0] if isinstance(n,ast.Assign) else n.target).id for n in constants]
aliases='from bayesfilter.inference.hmc_configuration import (\n'+''.join(f'    {n},\n' for n in constants_names+[n.name for n in nodes])+')\n\n'
anchor='from bayesfilter.runtime import stable_config_hash\n';new=new.replace(anchor,anchor+'\n'+aliases,1)
new=new.replace('historical callers and serialized names. Configuration translation remains here.','historical callers and serialized names. Presets and configuration translation\nare implemented in hmc_configuration.')
# Collapse only blank gaps created by this move.
def context(text,m):return (text[:m.start()].splitlines()[-1],text[m.end():].splitlines()[0])
prior={context(s,m):len(m.group()) for m in re.finditer(r'\n{8,}',s)}
for m in list(re.finditer(r'\n{8,}',new))[::-1]:
 if prior.get(context(new,m))!=len(m.group()):new=new[:m.start()]+'\n\n\n'+new[m.end():]
p.write_text(new)
(root/'moved-constants.json').write_text(json.dumps(constants_names,indent=2)+'\n')
print({'definitions':len(nodes),'constants':len(constants),'owner_lines':len((header+body).splitlines())})
