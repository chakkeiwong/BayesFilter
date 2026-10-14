"""CPU-only author endpoint fixture; PyTorch is an independent reference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import types
ROOT=Path('/home/chakwong/BayesFilter')
AUTHOR=ROOT/'.localresources/code/younis-mdpf-neurips-2023'
sys.path.insert(0,str(AUTHOR/'packages/kernel-density-estimator-bandwdidth-prediction/packages'))
import torch
from kernel_density_estimation.kernel_density_estimator import KernelDensityEstimator

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(20260929)
    path=AUTHOR/'src/models/kde_particle_filter/kde_particle_filter.py'
    tree=ast.parse(path.read_text());method=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='resample_particles')
    namespace=dict(torch=torch,KernelDensityEstimator=KernelDensityEstimator)
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
    # Only constructor-independent scalar settings are supplied. The actual
    # unchanged method and real KDE sampler/density execute below.
    source=types.SimpleNamespace(decouple_weights_for_resampling=False,decouple_bandwidths_for_resampling=False,
        use_differentiable_resampling=True,training=True,differentiable_resampling_method='ImportanceSampling',
        kde_params={'dims':{0:{'distribution_type':'Normal'}}},
        particle_transformer=types.SimpleNamespace(apply_norm=lambda x:x))
    locations=torch.tensor([[[-1.],[0.],[2.]],[[-2.],[.3],[1.7]]],dtype=torch.float64,requires_grad=True)
    weights=torch.tensor([[.2,.5,.3],[.6,.1,.3]],dtype=torch.float64,requires_grad=True)
    bandwidth=torch.tensor([[.4],[.8]],dtype=torch.float64,requires_grad=True)
    sample,iwsg,_=namespace['resample_particles'](source,dict(particles_downscaled=locations,
        particle_weights=weights,bandwidths_downscaled=bandwidth,do_resample=True))
    density=KernelDensityEstimator(source.kde_params,locations,weights,bandwidth).log_prob(sample)
    integrand=(sample*sample+.3*sample).sum(-1)
    loss=(iwsg*integrand).sum();normalized=(iwsg/iwsg.sum(-1,keepdim=True)*integrand).sum()
    inputs=[locations,weights,bandwidth]
    raw_gradient=torch.autograd.grad(loss,inputs,retain_graph=True)
    normalized_gradient=torch.autograd.grad(normalized,inputs)
    payload=dict(locations=locations.detach().tolist(),weights=weights.detach().tolist(),bandwidth=bandwidth.detach().tolist(),
        samples=sample.detach().tolist(),iwsg=iwsg.detach().tolist(),log_density=density.detach().tolist(),
        raw_loss=loss.item(),normalized_loss=normalized.item(),raw_gradient=[x.tolist() for x in raw_gradient],
        normalized_gradient=[x.tolist() for x in normalized_gradient],sample_gradient_stopped=not sample.requires_grad,
        source_entry=str(path)+':674 resample_particles -> KernelDensityEstimator.sample/log_prob',
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),torch=torch.__version__,seed=20260929,
        cpu_only=True,gpu_intentionally_hidden=True,fixture='constructed Gaussian mechanics; not Appendix B numeric settings')
    Path(args.output).write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
