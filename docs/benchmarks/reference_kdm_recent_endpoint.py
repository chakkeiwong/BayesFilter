"""Independent CPU reference export of nine fixed recent KDM checkpoints."""
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path('/home/chakwong/BayesFilter')
sys.path.insert(0,str(ROOT/'docs/benchmarks'))
from reference_kdm_native_evaluation import BASE,PIN,sha,read,write,load_reference

def prepare(out,unit,intervention_training=None):
    ref=load_reference();torch=ref.torch
    cal=BASE/'paired-calibration-20261011/calibration-01'
    manifest=read(cal/'manifest.json');assert manifest['status']=='COMPLETED'
    entry=next(v for v in manifest['completed_units'] if v['unit']==unit)
    report_path=Path(entry['evaluation']);assert sha(report_path)==entry['evaluation_sha256']
    report=read(report_path)
    assert report['decision']['status']=='PASS' and report['source_pin']==PIN
    provenance={str(cal/'manifest.json'):sha(cal/'manifest.json'),str(report_path):sha(report_path)}
    for role in ('source','tf'):
        path=Path(report[role+'_terminal_audit'])
        assert sha(path)==report[role+'_terminal_audit_sha256']
        audit=read(path);assert audit['verdict']=='PASS' and not audit['failures']
        provenance[str(path)]=sha(path)
    data_path=report_path.parent/'evaluation-dataset.pt'
    assert sha(data_path)==report['dataset_sha256']
    raw=torch.load(data_path,map_location='cpu',weights_only=False)
    data={'states':raw['car_states'],'observations':raw['sensor_obs']}
    assert tuple(data['states'].shape)==(256,150,3) and tuple(data['observations'].shape)==(256,150,1)
    assert all(x.dtype==torch.float32 and bool(torch.isfinite(x).all()) for x in data.values())
    provenance[str(data_path)]=sha(data_path)
    constructor=read(Path(entry['reference']));assert constructor['source_pin']==PIN
    provenance[entry['reference']]=sha(entry['reference']);arms=[]
    for saved in report['checkpoint_provenance']:
        variant=saved['variant'];fixture=constructor['fixtures'][variant]
        assert sha(fixture['path'])==fixture['sha256']
        provenance[fixture['path']]=fixture['sha256']
        for role in ('source','tf'):
            path=saved[role+'_checkpoint'];assert sha(path)==saved[role+'_sha256']
            provenance[path]=sha(path)
        state=read(saved['tf_checkpoint'])['model']
        author=torch.load(saved['source_checkpoint'],map_location='cpu',weights_only=True)
        assert set(state['parameters'])<=set(author)
        source_state=copy.deepcopy(state)
        for name,value in state['parameters'].items():
            tensor=author[name]
            assert tensor.dtype==torch.float32 and tuple(tensor.shape)==tuple(torch.tensor(value).shape)
            assert bool(torch.isfinite(tensor).all())
            source_state['parameters'][name]=tensor.tolist()
        for label,model_state in [('source_trained',source_state),('tf_trained',state)]:
            arms.append(dict(variant=variant,checkpoint=label,fixture=read(fixture['path']),state=model_state,
                source_checkpoint_sha256=saved['source_sha256'],tf_checkpoint_sha256=saved['tf_sha256']))
    if intervention_training is not None:
        # Prospective diagnostic: three MDPF checkpoints on the existing data.
        assert unit==2
        result_path=intervention_training/'result.json'
        training=read(result_path)
        assert training['status']=='PASS' and training['selected_stages_complete']
        assert training['unit']==unit and training['seed_base']==1200000000
        assert training['selected_stages']==['mdpf'] and len(training['completed_stages'])==1
        stage=training['completed_stages'][0]
        assert stage['stage']=='mdpf' and stage['full_protocol']
        assert stage['termination'] in ('SOURCE_EARLY_STOP','SOURCE_CAP')
        path=Path(stage['best_checkpoint']);assert sha(path)==stage['best_checkpoint_sha256']
        provenance[str(result_path)]=sha(result_path);provenance[str(path)]=sha(path)
        arms=[arm for arm in arms if arm['variant']=='mdpf']
        arms.append(dict(variant='mdpf',checkpoint='author_start_tf_trained',
            fixture=read(constructor['fixtures']['mdpf']['path']),state=read(path)['model'],
            tf_checkpoint_sha256=sha(path)))
    elif unit==1:
        recovery=BASE/'stage-isolation-20261013/evaluation-repair01'
        rm=read(recovery/'manifest.json');decision=read(recovery/'decision.json')
        assert rm['status']=='COMPLETED' and decision['status']=='PASS'
        assert sha(recovery/'diagnostic/result.json')==rm['result_sha256']
        stage=decision['training_stages'][0]
        assert stage['stage']=='mdpf' and stage['full_protocol'] and stage['termination']=='SOURCE_EARLY_STOP'
        path=Path(stage['best_checkpoint']);assert sha(path)==stage['best_checkpoint_sha256']
        for p in [recovery/'manifest.json',recovery/'decision.json',recovery/'diagnostic/result.json',path]:
            provenance[str(p)]=sha(p)
        arms.append(dict(variant='mdpf',checkpoint='author_start_tf_trained',
            fixture=read(constructor['fixtures']['mdpf']['path']),state=read(path)['model'],
            tf_checkpoint_sha256=sha(path)))
    assert len(arms)==(3 if intervention_training is not None else (5 if unit==1 else 4))
    write(out/'inputs.json',dict(schema='kdm_native_evaluation_inputs_v2',unit=unit,source_pin=PIN,
        evaluation_scope='unit02_mdpf_initialization' if intervention_training is not None else 'recent_checkpoints',
        provenance=provenance,dataset_sha256=sha(data_path),data={k:v.tolist() for k,v in data.items()},arms=arms,
        gpu_intentionally_hidden=True,role='INDEPENDENT_REFERENCE_EXPORT_ONLY',
        source_state_optimizer_note='Optimizer fields are unused copied metadata; fixed parameters only are evaluated'))
    print(json.dumps(dict(status='PREPARED',unit=unit,arms=len(arms),rows=256,horizon=150)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--unit',type=int,choices=(1,2),required=True)
    parser.add_argument('--intervention-training',type=Path)
    args=parser.parse_args();prepare(args.output,args.unit,args.intervention_training)
