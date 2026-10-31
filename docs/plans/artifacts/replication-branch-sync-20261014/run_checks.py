from pathlib import Path
import os,sys,json,time,subprocess,ast
root=Path(sys.argv[1]);label=sys.argv[2]
out=Path('/tmp/bayesfilter-replication-sync-20261014')
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='-1';env['TF_FORCE_GPU_ALLOW_GROWTH']='true';env['PYTHONPATH']=str(root);env['PYTHONDONTWRITEBYTECODE']='1'
if label.startswith('branch'):
 paths=json.loads((out/'migration-code-paths.json').read_text())
 for rel in paths:
  if rel.endswith('.py'):ast.parse((root/rel).read_text(),filename=rel)
cmd=[sys.executable,'-B','-m','pytest',*sys.argv[3:],'-q','--tb=short']
start=time.monotonic()
with (out/(label+'.log')).open('w') as log:
 p=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
res=dict(command=cmd,cwd=str(root),cpu_only=True,CUDA_VISIBLE_DEVICES='-1',PYTHONPATH=str(root),exit_code=p.returncode,wall_seconds=time.monotonic()-start,log=str(out/(label+'.log')))
(out/(label+'.json')).write_text(json.dumps(res,indent=2))
print(json.dumps(res));print('\n'.join((out/(label+'.log')).read_text().splitlines()[-20:]))
sys.exit(p.returncode)
