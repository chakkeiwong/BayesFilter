"""Read-only evidence audit apart from writing its own new audit report."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    state = json.loads((root/'state.json').read_text())
    if state['active'] is not None:
        raise ValueError('campaign worker remains active')
    shared_path = root.parents[0]/'neutra-warm-start-master-2026-09-29/campaign-r1/state.json'
    shared = json.loads(shared_path.read_text())
    records = []
    seen = set()
    for attempt in state['attempts']:
        path = Path(attempt['output'])
        if str(path) in seen:
            raise ValueError('duplicate local accounting row')
        seen.add(str(path))
        matched = [a for a in shared['attempts'] if a.get('output')==str(path)]
        if len(matched)!=1:
            raise ValueError('missing/duplicate shared charge')
        for k in ('gpu_process_seconds','cpu_core_seconds'):
            if matched[0][k]!=attempt[k]:
                raise ValueError('shared charge differs from local charge')
        row = dict(attempt=path.name,status=attempt['status'],phase=attempt['phase'],
                   wall_seconds=attempt['wall_seconds'],artifact_files=0,source_files=0)
        if attempt['phase']=='engineering-accounting':
            records.append(row)
            continue
        source = json.loads((path/'source.json').read_text())
        for name,h in source['sha256'].items():
            if digest(path/'source'/name)!=h:
                raise ValueError('changed source: '+str(path/'source'/name))
            row['source_files']+=1
        if attempt['status']!='complete':
            row['scientific_result_available']=False
            records.append(row)
            continue
        manifest = json.loads((path/'manifest.json').read_text())
        if not manifest['memory_policy']['all_physical_devices_memory_growth']:
            raise ValueError('GPU memory policy missing')
        if not manifest['jit_compile'] or manifest['status']!='complete':
            raise ValueError('incomplete worker or non-XLA manifest')
        for name,h in manifest['artifact_sha256'].items():
            if digest(path/name)!=h:
                raise ValueError('changed artifact: '+str(path/name))
            row['artifact_files']+=1
        result = json.loads((path/'result.json').read_text())
        row['result_status']=result['status']
        row['result_sha256']=digest(path/'result.json')
        if 'checkpoints' in result:
            for block in result['training_blocks']:
                if (not block['finite'] or block['samplewise_loop'] or
                        block['batch_size']!=64 or not block['jit_compile'] or
                        'GPU:' not in block['device']):
                    raise ValueError('training mechanics invalid')
            full_probes = [c for c in result['checkpoints'] if c['probe']['rows']==1000]
            for checkpoint in full_probes:
                p=checkpoint['probe']
                if not p['complete'] or not p['finite'] or p['valid_rows']!=1000:
                    raise ValueError('incomplete saved 1000-point probe')
            row['complete_1000_point_probes']=[c['label'] for c in full_probes]
            endpoint = result['checkpoints'][-1]
            probe = endpoint['probe']
            if probe['rows']!=1000 or probe['valid_rows']!=1000 or not probe['complete'] or not probe['finite']:
                raise ValueError('incomplete terminal 1000-point probe')
            if not endpoint['checkpoint_reloaded']:
                raise ValueError('checkpoint reload missing')
            row['coverage_passed']=endpoint['coverage_passed']
            row['endpoint']=endpoint['label']
            row['score_residual_norm']=probe['score_residual_norm']
            if result.get('parent_checkpoint'):
                if digest(Path(result['parent_checkpoint']))!=result['parent_sha256']:
                    raise ValueError('reverse continuation parent changed')
        records.append(row)
    computed = {k:state['initial_remaining'][k]-sum(a[k] for a in state['attempts'])
                for k in state['initial_remaining']}
    if computed!=state['remaining']:
        raise ValueError('remaining budget differs from charges')
    return dict(audit_passed=True,active_worker=None,records=records,
        remaining_budget=computed,measured_gpu_process_seconds=sum(a['gpu_process_seconds'] for a in state['attempts']),
        cpu_accounting_note='See engineering-accounting artifact for measured checks and explicit contingency reserve.',
        scope='artifact integrity, finite training mechanics, terminal probe completeness and accounting; no posterior or whitening admission')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('use a fresh audit output')
    result=audit(args.root)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['audit_passed'],attempts=len(result['records']),remaining=result['remaining_budget'])))
