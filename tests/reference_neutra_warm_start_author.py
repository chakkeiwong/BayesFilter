"""CPU-only reference fixture generator executing pinned flonaco functions.

Imports only selected unchanged function ASTs, avoiding upstream plotting and
GPU example launchers. NumPy/PyTorch are independent reference backends here.
"""
import argparse
import ast
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main():
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    source=Path(__file__).resolve().parents[1]/'.localresources/flonaco-author-20260929/upstream/flonaco/sampling.py'
    tree=ast.parse(source.read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)
               and n.name in ('run_MALA','run_metropolis','run_metromalangevin')]
    namespace={'torch':torch,'np':np}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),namespace)
    starts=[[-2.,.5],[1.,-1.],[0.,2.]]
    proposals=[[[.2,-.4],[-1.,2.],[2.,.2]], [[-2.,-1.],[1.,.3],[.5,-.3]]]
    noises=[[[.1,-.2],[2.,-.1],[-.3,.8]], [[-.5,.3],[.2,.7],[1.,-.6]]]
    uniforms=[[.1,.9,.5],[.99,.1,.8],[.8,.2,.7],[.3,.95,.1]]
    class Target:
        beta=1.
        def U(self,x):
            return .5*((x[:,0]-.7)**2+3*(x[:,1]+.2)**2)
        def grad_U(self,x):
            return torch.stack((x[:,0]-.7,3*(x[:,1]+.2)),dim=1)
    class Flow:
        def __init__(self): self.index=0
        def sample(self,n):
            x=torch.tensor(proposals[self.index]); self.index+=1
            return x
        def nll(self,x): return .5*(x*x).sum(1)+np.log(2*np.pi)
    noise_iter=iter(noises); uniform_iter=iter(uniforms)
    old_noise,old_uniform=torch.randn_like,torch.rand_like
    torch.randn_like=lambda x: torch.tensor(next(noise_iter))
    torch.rand_like=lambda x: torch.tensor(next(uniform_iter))
    try:
        positions,accepted=namespace['run_metromalangevin'](
            Flow(),Target(),torch.tensor(starts),2,.2)
    finally:
        torch.randn_like,torch.rand_like=old_noise,old_uniform
    result={'source_commit':'6b9286b4e58194aa65373200d7bfacde06a2d180',
        'source_function':'run_metromalangevin','backend':'torch CPU independent reference',
        'starts':starts,'proposals':proposals,'noises':noises,'uniforms':uniforms,
        'dt':.2,'positions':positions.tolist(),'global_accepted':accepted.tolist()}
    wiggle_source=source.with_name('croissant_utils.py')
    classes=[n for n in ast.parse(wiggle_source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Croissants']
    wiggle_namespace={'torch':torch,'nn':torch.nn,'np':np}
    exec(compile(ast.Module(body=classes,type_ignores=[]),str(wiggle_source),'exec'),wiggle_namespace)
    wiggle=wiggle_namespace['Croissants']([torch.tensor([6.,0.])],[torch.eye(2)],5.,8.,
        wiggle=True,dtype=torch.float64,device='cpu')
    points=torch.tensor([[.3,-.7],[3.,4.],[-2.,6.],[3.4,-3.6]])
    result['wiggle']={'points':points.tolist(),'log_density':(-wiggle.U(points)).tolist(),
        'score':(-wiggle.grad_U(points)).tolist(),'source_class':'Croissants'}
    with Path(args.output).open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps({'output':args.output,'source_executed':True,'torch':torch.__version__}))


if __name__=='__main__': main()
