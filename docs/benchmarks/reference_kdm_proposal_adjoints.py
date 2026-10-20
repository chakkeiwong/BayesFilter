"""Independent pinned-author CPU reference: exact local activations and VJPs.

NumPy is used only to serialize diagnostic reference arrays, never by runtime.
"""
import argparse, importlib.util, json, os, sys
from pathlib import Path
from types import SimpleNamespace
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['OMP_NUM_THREADS']='1'
ROOT=Path('/home/chakwong/BayesFilter')
BASE=ROOT/'docs/plans/artifacts/controlled-smc-kdm-replication-20260928'

def main(output,reference):
    assert sys.flags.isolated
    spec=importlib.util.spec_from_file_location('reference_helpers',ROOT/'docs/benchmarks/reference_kdm_neural_exact_replay.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    torch=ref.torch
    import numpy as np
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    fixtures=json.loads(reference.read_text())
    configs={r['name']:r for r in json.loads((BASE/'kdm-complete-training-20261007-03/diagnostic/configurations.json').read_text())['executed_stages']}
    transformer=ref.source_member(ref.AUTHOR/'src/problems/bearings_only/bearings_only_problem.py','BearingsOnlyObservationTransformer')
    source_trainer=ref.AUTHOR/'src/trainers/specific_trainers/full_sequence_trainer.py'
    forward=ref.source_member(source_trainer,'FullSequenceTrainer','do_forward_pass')
    convert=ref.source_member(source_trainer,'FullSequenceTrainer','convert_to_model_dtype')
    output.mkdir(exist_ok=False)
    records=[]
    for case in fixtures['cases']:
        if case['dtype']!='float32': continue
        torch.set_default_dtype(torch.float32)
        cfg=configs[case['variant']]
        architecture={}
        directory={'mdpf':'experiment0002_importance','am_dpf':'experiment0003_importance_init'}[case['variant']]
        for path in (ref.EXPERIMENT/'configs/models.yaml',ref.EXPERIMENT/directory/'config.yaml'):
            architecture.update(ref.yaml.safe_load(ref.verify(path))['models'])
        assert json.loads(json.dumps(architecture))==cfg['architecture']
        model=ref.KDEParticleFilter(cfg['configuration']['model_params'],architecture)
        model.train()
        for name,p in model.named_parameters(): p.requires_grad_(name in case['trainable_names'])
        trainer=SimpleNamespace(main_model=model,device=torch.device('cpu'),
            problem=SimpleNamespace(observation_transformer=transformer()),number_of_particles=25,
            skip_first_step_for_loss=False,max_sequence_length=None,truncated_bptt_modulo=4,
            loss_shaper=None,loss_function=ref.KDE_NLL(dict(kde_params=architecture['dpf_full_training']['kde']),model),
            convert_to_model_dtype=lambda data:convert(trainer,data))
        for index,update in enumerate(case['updates']):
            parameters=dict(case['parameters'])
            if index: parameters.update(case['updates'][index-1]['parameters_after'])
            with torch.no_grad():
                for name,p in model.named_parameters(): p.copy_(torch.tensor(parameters[name]))
            model.zero_grad(set_to_none=True)
            data={k:torch.tensor(v,dtype=torch.bool if k=='ground_truth_mask' else torch.float32) for k,v in update['data'].items()}
            captured=update['captured'];cursor=dict(sample=0,noise=0)
            layers={i:[] for i in (0,2,4)};raw=[];normal=[]
            original_sample=ref.KernelDensityEstimator.sample
            original_init=model.create_initial_dpf_state
            original_norm=model.particle_transformer.apply_norm
            def initialize(*a,**k):
                return {key:torch.tensor(v) if isinstance(v,list) else v for key,v in captured['initial'].items()}
            def sample(*a,**k):
                j=cursor['sample'];cursor['sample']+=1
                return torch.tensor(captured['samples'][j])
            def inputs(module,values):
                j=cursor['noise'];cursor['noise']+=1
                return (*values,torch.tensor(captured['noise'][j]))
            def hook(i):
                def capture(module,inputs,result):
                    result.retain_grad();layers[i].append((inputs[0],result))
                return capture
            def proposal(module,inputs,result):
                result[0].retain_grad();raw.append(result[0])
            def norm(x):
                y=original_norm(x)
                if raw and x is raw[-1]:
                    y.retain_grad();normal.append(y)
                return y
            model.create_initial_dpf_state=initialize
            ref.KernelDensityEstimator.sample=sample
            model.particle_transformer.apply_norm=norm
            hooks=[model.proposal_model.register_forward_pre_hook(inputs),model.proposal_model.register_forward_hook(proposal)]
            hooks += [model.proposal_model.proposal_model[i].register_forward_hook(hook(i)) for i in layers]
            try:
                loss,effective=forward(trainer,data,None,0);loss.backward()
            finally:
                model.create_initial_dpf_state=original_init
                ref.KernelDensityEstimator.sample=original_sample
                model.particle_transformer.apply_norm=original_norm
                for h in hooks:h.remove()
            assert effective==64 and float(loss)==update['loss']
            assert len(raw)==len(normal)==16
            for name,p in model.named_parameters():
                if name in case['trainable_names']:
                    assert torch.equal(p.grad,torch.tensor(update['raw_gradients'][name])),name
            arrays={}
            for i,values in layers.items():
                layer=model.proposal_model.proposal_model[i]
                arrays[f'l{i}_w']=layer.weight.detach().numpy()
                arrays[f'l{i}_b']=layer.bias.detach().numpy()
                for t,(x,y) in enumerate(values):
                    g=y.grad if y.grad is not None else torch.zeros_like(y)
                    xx=x.detach().requires_grad_()
                    yy=torch.nn.functional.linear(xx,layer.weight,layer.bias)
                    gx,gw,gb=torch.autograd.grad(yy,(xx,layer.weight,layer.bias),g)
                    for key,val in dict(x=x,y=y,g=g,gx=gx,gw=gw,gb=gb).items():
                        arrays[f't{t}_l{i}_{key}']=val.detach().numpy()
            for t,(q,n) in enumerate(zip(raw,normal)):
                arrays[f't{t}_raw']=q.detach().numpy();arrays[f't{t}_norm']=n.detach().numpy()
                arrays[f't{t}_raw_g']=(q.grad if q.grad is not None else torch.zeros_like(q)).detach().numpy()
                arrays[f't{t}_norm_g']=(n.grad if n.grad is not None else torch.zeros_like(n)).detach().numpy()
            arrays['residual_scale']=np.array(case['constants']['residual_scale'],dtype=np.float32)
            path=output/(case['variant']+'-'+str(index)+'.npz')
            np.savez_compressed(path,**arrays)
            records.append(dict(variant=case['variant'],update=index,path=str(path),source_replay_exact=True))
            print(json.dumps(records[-1]),flush=True)
    (output/'summary.json').write_text(json.dumps(dict(records=records,cpu_only=True,gpu_intentionally_hidden=True,
        source_commit=ref.PIN,source_hashes=ref.HASHES,torch_version=torch.__version__),indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    args=parser.parse_args();main(args.output,args.reference)
