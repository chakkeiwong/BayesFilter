"""Independent CPU diagnostic of the pinned author KDE sampling call chain."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse, hashlib, inspect, json, math, sys, time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import reference_kdm_neural as source

def host(x):
    return x.detach().cpu().tolist()

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    started=time.monotonic();cases=[]
    for dtype_name in ["float64","float32"]:
        dtype=getattr(torch,dtype_name);torch.set_default_dtype(dtype)
        kappas=torch.tensor([1e-6,1e-4,.2,3.74,3.76,50.,1000.])
        p=torch.tensor([[-2.,3.,-3.13],[0.,-2.,.2],[3.,1.,3.13]])[None].repeat(7,1,1)
        w=torch.tensor([[.2,.3,.5]]).repeat(7,1)
        w[0]=torch.tensor([0.,.25,.75]);w[-1]=torch.tensor([0.,0.,1.])
        bw=torch.stack([torch.full([7],.3),torch.full([7],1.2),1/kappas-1e-8],-1)
        config={"dims":{0:{"distribution_type":"Normal"},1:{"distribution_type":"Normal"},
                        2:{"distribution_type":"Von_Mises"}}}
        kde=source.KernelDensityEstimator(config,p,w,bw)
        captured={"uniforms":[],"normal_components":[]}
        original_rand=torch.rand
        original_cat=torch.distributions.Categorical.sample
        original_normal=torch.distributions.Normal.sample
        def rand(*a,**kw):
            value=original_rand(*a,**kw);captured["uniforms"].append(value);return value
        def categorical(self,*a,**kw):
            value=original_cat(self,*a,**kw);captured["ancestors"]=value;return value
        def normal(self,*a,**kw):
            value=original_normal(self,*a,**kw);captured["normal_components"].append(value);return value
        torch.manual_seed(2026100611)
        torch.rand=rand;torch.distributions.Categorical.sample=categorical
        torch.distributions.Normal.sample=normal
        try: replay=kde.sample((25,))
        finally:
            torch.rand=original_rand;torch.distributions.Categorical.sample=original_cat
            torch.distributions.Normal.sample=original_normal
        count=len(captured["uniforms"])
        assert 0<count<=128
        uniforms=torch.zeros([128,3,25,7,3],dtype=torch.float64)
        for i,value in enumerate(captured["uniforms"]): uniforms[i]=value
        assert len(captured["normal_components"])==2
        torch.manual_seed(2026100612)
        native_categories={}
        def native_categorical(self,*a,**kw):
            value=original_cat(self,*a,**kw)
            native_categories["counts"]=torch.nn.functional.one_hot(value,3).sum(0)
            return value
        torch.distributions.Categorical.sample=native_categorical
        try: native=kde.sample((32768,))
        finally: torch.distributions.Categorical.sample=original_cat
        cases.append(dict(dtype=dtype_name,locations=host(p),weights=host(w),bandwidth=host(bw),
            replay=host(replay),ancestors=host(captured["ancestors"].T),
            normal_components=host(torch.stack(captured["normal_components"],-1)),
            uniforms=host(uniforms),iterations=count,native=host(native),native_counts=host(native_categories["counts"])))
    dependency=source.AUTHOR/"packages/kernel-density-estimator-bandwdidth-prediction/packages/kernel_density_estimation"
    for filename in ["kernel_density_estimator.py","von_mises_full_dist.py"]:
        source.verify(dependency/filename)
    torch_source=Path(inspect.getfile(torch.distributions.VonMises))
    payload=dict(schema="kdm_sampling_source_v1",cases=cases,source_hashes=source.HASHES,
        torch_sampler_path=str(torch_source),torch_sampler_sha256=hashlib.sha256(torch_source.read_bytes()).hexdigest(),
        torch=torch.__version__,cpu_only=True,cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        wall_seconds=time.monotonic()-started,source_commit=source.PIN)
    args.output.write_text(json.dumps(payload,allow_nan=False)+"\n")
    print(json.dumps(dict(cases=len(cases),wall_seconds=payload["wall_seconds"],output=str(args.output))))
if __name__=="__main__": main()
