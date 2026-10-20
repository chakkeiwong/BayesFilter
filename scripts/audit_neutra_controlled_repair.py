"""Read-only evidence audit; optional JSON report is the sole output mutation."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import sys

repo=Path('/home/ubuntu/python/BayesFilter')
root=repo/'docs/plans/artifacts/neutra-controlled-repair-2026-10-02/campaign-r1'
shared=repo/'docs/plans/artifacts/neutra-warm-start-master-2026-09-29/campaign-r1'
read=lambda p:json.loads(Path(p).read_text())
state=read(root/'state.json');cfg=read(root/'config.json');shared_state=read(shared/'state.json')
rows=[x for records in state['jobs'].values() for x in records]
charges=[x for x in shared_state['attempts'] if str(x.get('output','')).startswith(str(root))]
errors=[];notes=[];hash_cache={};sources={};manifests=[];input_count=0;fit_counts=Counter()

def digest(path):
    path=Path(path)
    if path not in hash_cache:
        if not path.is_file():
            errors.append({'missing_file':str(path)});return None
        hash_cache[path]=hashlib.sha256(path.read_bytes()).hexdigest()
    return hash_cache[path]

for row in rows:
    out=Path(row['output']);m=read(out/'manifest.json');manifests.append(m)
    matching=[x for x in charges if x['output']==str(out)]
    if len(matching)!=1:errors.append({'charge_count':len(matching),'output':str(out)})
    for key in ('wall_seconds','gpu_process_seconds','cpu_core_seconds'):
        if key in m and not math.isclose(row[key],m[key],abs_tol=1e-7):errors.append({'manifest_cost_mismatch':str(out),'key':key})
        if matching and not math.isclose(row[key],matching[0][key],abs_tol=1e-7):errors.append({'shared_cost_mismatch':str(out),'key':key})
    for field in ('command','environment','git_commit','plan','wall_seconds','cpu_core_seconds'):
        if field not in m:errors.append({'missing_manifest_field':field,'output':str(out)})
    if row.get('source'):
        source=Path(row['source']);sources[str(source)]=read(source/'source.json')
        if m.get('git_commit')!=sources[str(source)]['git_commit']:errors.append({'source_commit_mismatch':str(out)})
        if Path(m['command'][1]).parent.parent!=source:errors.append({'command_source_mismatch':str(out)})
    else:notes.append({'imported_completed_evidence':str(out),'source_manifest':m.get('source_manifest')})
    if row['device']=='gpu':
        policy=m.get('memory_policy',{})
        for field in ('all_physical_devices_memory_growth','configured_before_logical_device_initialization'):
            if policy.get(field) is not True:errors.append({'memory_policy':field,'output':str(out)})
        if m.get('trust_basis')!='trusted_escalated_fixed_campaign_wrapper':errors.append({'trust_basis':str(out)})
        if not m.get('jit_compile'):errors.append({'jit':str(out)})
        if row['status']=='complete' and 'allocator' not in m:errors.append({'missing_allocator':str(out)})
    elif m.get('gpu_devices_intentionally_hidden') is not True:errors.append({'cpu_gpu_visibility':str(out)})
    for p,h in m.get('input_sha256',{}).items():
        input_count+=1
        if digest(p)!=h:errors.append({'input_hash_mismatch':p,'consumer':str(out)})
    if row['status']=='complete':
        result_path=out/'result.json'
        if not result_path.is_file():errors.append({'missing_result':str(out)});continue
        result=read(result_path)
        if 'history' in result:
            fit_counts[result['stop_reason']]+=1
            if not result.get('batch_native') or result.get('sample_wise_target_loop') or not result.get('jit_compile'):
                errors.append({'training_backend':str(out)})
            if result.get('training_converged'):notes.append({'convergence_claim':str(out)})
            last_step=-1
            for entry in result['history']:
                if entry['step']<=last_step:errors.append({'nonincreasing_checkpoint':str(out)})
                last_step=entry['step']
                if entry['metrics']['post_training_1000']['rows']!=1000:errors.append({'probe_rows':str(out)})
                for suffix in ('checkpoint','frozen','metrics','optimizer-diagnostic'):
                    if not (out/(entry.get('stage', 'step-'+str(entry['step']))+'-'+suffix+'.json')).is_file():errors.append({'missing_checkpoint_part':str(out),'stage':entry.get('stage', 'step-'+str(entry['step'])),'part':suffix})
    elif m.get('status')!='failed':errors.append({'unexplained_failure':str(out)})

for source,manifest in sources.items():
    for path,h in manifest['sha256'].items():
        if digest(Path(source)/path)!=h:errors.append({'source_hash_mismatch':source,'file':path})
    h=hashlib.sha256(json.dumps(manifest['sha256'],sort_keys=True).encode()).hexdigest()
    if h!=manifest['signature']:errors.append({'source_signature_mismatch':source})

used={k:sum(x.get(k,0.) for x in rows) for k in ('gpu_process_seconds','cpu_core_seconds')}
for key,value in used.items():
    if value>cfg[key]:errors.append({'budget_exceeded':key})
    if not math.isclose(cfg[key]-value,state['remaining'][key],abs_tol=1e-6):errors.append({'remaining_mismatch':key})
if len(charges)!=len(rows):errors.append({'shared_count_mismatch':[len(charges),len(rows)]})
settled={d['result']['group']:d['result']['confirmation'] for d in state.get('decisions',[]) if d.get('stage')=='case_complete'}
report={'generated_utc':datetime.now(timezone.utc).isoformat(),'status':state['status'],
    'read_only_audit':True,'settled_attempts':len(rows),'statuses':dict(Counter(x['status'] for x in rows)),
    'settled_cases':settled,'active_jobs':list(state.get('active',{})),
    'charged':used,'remaining':state['remaining'],'shared_charge_count':len(charges),
    'source_snapshots_checked':len(sources),'unique_files_hashed':len(hash_cache),'input_bindings_checked':input_count,
    'training_stop_counts':dict(fit_counts),'errors':errors,'notes':notes,
    'passed':not errors,'role':'engineering provenance and budget audit; not posterior validation'}
if len(sys.argv)>1:
    Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('settled_cases','notes')},indent=2))
raise SystemExit(bool(errors))
