"""Independent CPU reference preparation and metric replay for native KDM evaluation."""
import argparse,ast,copy,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path('/home/chakwong/BayesFilter')
BASE=ROOT/'docs/plans/artifacts/controlled-smc-kdm-replication-20260928'
PILOT=BASE/'repaired-training-20261011/training-pilot-02'
PIN='b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p,v): Path(p).write_text(json.dumps(v,allow_nan=False)+chr(10))
def load_reference():
    os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['OMP_NUM_THREADS']='1'
    if not sys.flags.isolated: raise RuntimeError('CPU reference requires isolated Python')
    spec=importlib.util.spec_from_file_location('native_eval_reference',ROOT/'docs/benchmarks/reference_kdm_neural_exact_replay.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    ref.torch.set_num_threads(1)
    subprocess.run(['git','-C',str(ref.AUTHOR),'diff','--quiet',PIN,'--','src','packages'],check=True)
    return ref

def prepare(out):
    ref=load_reference();torch=ref.torch
    manifest=read(PILOT/'manifest.json');audit=read(PILOT/'audit.json')
    report_path=PILOT/'evaluation/diagnostic/result.json';report=read(report_path)
    assert manifest['status']=='COMPLETED' and manifest['evaluation_sha256']==sha(report_path)
    assert audit['verdict']=='PASS' and not audit['failures']
    assert audit['run_manifest_sha256']==sha(PILOT/'training/manifest.json')
    assert report['decision']['status']=='PASS' and report['source_pin']==PIN
    data_path=report_path.parent/'evaluation-dataset.pt';assert sha(data_path)==report['dataset_sha256']
    raw=torch.load(data_path,map_location='cpu',weights_only=False)
    data={'states':raw['car_states'],'observations':raw['sensor_obs']}
    assert tuple(data['states'].shape)==(256,150,3) and tuple(data['observations'].shape)==(256,150,1)
    assert all(x.dtype==torch.float32 and bool(torch.isfinite(x).all()) for x in data.values())
    constructor=read(BASE/'kdm-fresh-training-cost-20261007-01/diagnostic/result.json');arms=[]
    for entry in report['checkpoint_provenance']:
        variant=entry['variant'];fixture=constructor['fixtures'][variant]
        assert sha(fixture['path'])==fixture['sha256']
        assert sha(entry['source_checkpoint'])==entry['source_sha256'] and sha(entry['tf_checkpoint'])==entry['tf_sha256']
        state=read(entry['tf_checkpoint'])['model']
        author=torch.load(entry['source_checkpoint'],map_location='cpu',weights_only=True)
        assert set(state['parameters'])<=set(author)
        source_state=copy.deepcopy(state)
        for name,value in state['parameters'].items():
            tensor=author[name]
            assert tensor.dtype==torch.float32 and tuple(tensor.shape)==tuple(torch.tensor(value).shape)
            assert bool(torch.isfinite(tensor).all())
            source_state['parameters'][name]=tensor.tolist()
        for label,saved in [('source_trained',source_state),('tf_trained',state)]:
            arms.append(dict(variant=variant,checkpoint=label,fixture=read(fixture['path']),state=saved,
                source_checkpoint_sha256=entry['source_sha256'],tf_checkpoint_sha256=entry['tf_sha256']))
    assert len(arms)==4
    write(out/'inputs.json',dict(schema='kdm_native_evaluation_inputs_v1',source_pin=PIN,
        pilot_manifest_sha256=sha(PILOT/'manifest.json'),common_report_sha256=sha(report_path),
        dataset_sha256=sha(data_path),data={k:v.tolist() for k,v in data.items()},arms=arms,
        gpu_intentionally_hidden=True,role='INDEPENDENT_REFERENCE_EXPORT_ONLY',
        source_state_optimizer_note='Optimizer fields are unused copied metadata; source parameters only are evaluated'))
    print(json.dumps(dict(status='PREPARED',arms=len(arms),rows=256,horizon=150)))

def metrics(out):
    ref=load_reference();torch=ref.torch
    from loss_functions.loss_functions import create_loss_function
    from models.particle_transformer import BearingsOnlyTransformer
    path=ref.AUTHOR/'src/evaluations/full_sequence_evaluations/full_sequence_evaluation_base.py'
    tree=ast.parse(ref.verify(path))
    nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('EvaluationMetric','NonSweepEvaluationMetric')]
    ns=dict(torch=torch,create_loss_function=create_loss_function,os=os)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    config_path=ref.EXPERIMENT/'configs/common_parameters_evaluation.yaml'
    cfg=ref.yaml.safe_load(ref.verify(config_path))['evaluation_params']['quantitative_evaluation_params']['metrics']
    for p in ('src/loss_functions/loss_functions.py','src/models/particle_transformer.py'): ref.verify(ref.AUTHOR/p)
    native=read(out/'native-result.json');assert native['status']=='PASS';records=[];failed=[]
    for fixture in native['metric_replays']:
        p=out/fixture['path'];assert sha(p)==fixture['sha256']
        v=read(p);truth=torch.tensor(v['states'],dtype=torch.float32)
        arrays={k:torch.tensor(v[k],dtype=torch.float32) for k in ('particles','particle_weights','bandwidths')}
        outputs=[{k:a[:,i] for k,a in arrays.items()} for i in range(150)]
        owner=SimpleNamespace(particle_transformer=BearingsOnlyTransformer())
        for name in ('nll','rmse'):
            metric=ns['NonSweepEvaluationMetric'](name,owner,cfg[name],str(out))
            metric.compute_and_add_metric(outputs,truth,torch.arange(2),SimpleNamespace(get_subsequence_length=lambda:150))
            actual=torch.cat(metric.sequence_level_metric_values)
            expected=torch.tensor(v['nll' if name=='nll' else 'squared_error'],dtype=torch.float32)
            delta=(actual-expected).abs();scaled=delta/(1+actual.abs())
            aggregate=float(metric.get_value());tf_aggregate=v['aggregate'][name]
            aggregate_error=abs(aggregate-tf_aggregate)/(1+abs(aggregate))
            valid=bool(torch.isfinite(actual).all()) and bool((scaled<=2e-4).all()) and aggregate_error<=2e-4
            row=dict(variant=v['variant'],checkpoint=v['checkpoint'],metric=name,
                max_absolute_error=float(delta.max()),max_scaled_error=float(scaled.max()),
                aggregate_source=aggregate,aggregate_tf=tf_aggregate,aggregate_scaled_error=aggregate_error,passed=valid)
            records.append(row)
            if not valid: failed.append(row)
    assert len(records)==2*len(native['metric_replays']) and len(native['metric_replays'])>0
    result=dict(status='FAIL' if failed else 'PASS',records=records,failures=failed,
        tolerance='2e-4*(1+abs(source value)); diagnostic only',source_pin=PIN,
        source_hashes=ref.HASHES,gpu_intentionally_hidden=True,torch_version=torch.__version__,
        native_result_sha256=sha(out/'native-result.json'))
    write(out/'metric-audit.json',result)
    print(json.dumps(dict(status=result['status'],checks=len(records),max_scaled_error=max(x['max_scaled_error'] for x in records))))
    return 1 if failed else 0
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('prepare','metrics'),required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    raise SystemExit(prepare(a.output) if a.mode=='prepare' else metrics(a.output))
