"""GPU/XLA MDPF stage-isolation diagnostic using the existing epoch consumer."""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"]="0"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["TF_NUM_INTEROP_THREADS"]="1"
os.environ["TF_NUM_INTRAOP_THREADS"]="1"
import argparse, hashlib, json, math, sys, time, traceback
from pathlib import Path
ROOT=Path("/home/chakwong/BayesFilter")
sys.path[:0]=[str(ROOT),str(ROOT/"docs/benchmarks")]
BASE=ROOT/"docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    tmp=p.with_suffix(p.suffix+".tmp");tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n");tmp.replace(p)
def main(args):
    unitname="unit%02d" % args.unit
    old=BASE/"paired-calibration-20261011/calibration-01"/unitname/"training/diagnostic"
    source_path=BASE/"kdm-source-variability-20261008-01"/(unitname+"-full/diagnostic")
    seed_base=1000000000+args.unit*100000000
    started=time.monotonic(); out=args.output; out.mkdir(parents=True,exist_ok=False)
    checks=[]
    result=dict(status="RUNNING",mode=args.mode,replication="INCOMPLETE",unit=args.unit,seed_base=seed_base,
        reference_sha256=sha(source_path/"result.json"),initialization="isolated MDPF state; original global stage index and seed schedule")
    try:
        source=read(source_path/"result.json")
        assert source["status"]=="PASS" and source["source_training_protocol_complete"]
        fixtures={}
        for name,info in source["fixtures"].items():
            p=Path(info["path"]);assert sha(p)==info["sha256"];fixtures[name]=read(p)
        for info in source["datasets"].values(): assert sha(Path(info["path"]))==info["sha256"]
        cfg=source["source_training_configs"]
        reference=dict(datasets=source["datasets"],pretraining_cases=[fixtures["weights"]],pretraining_config=cfg["weights"],
                       full_training_config=cfg["mdpf"],full_training_configs=cfg)
        baseline=dict(cases=[fixtures["mdpf"],fixtures["am_dpf"]])
        statepath=old/"mdpf-starting-state.json" if args.mode=="control" else args.author_start
        state=read(statepath)
        assert state["step"]==0.
        assert state["rates"]==fixtures["mdpf"]["learning_rates"]
        assert state["trainable_names"]==fixtures["mdpf"]["trainable_names"]
        result["starting_state"]=dict(path=str(statepath),sha256=sha(statepath))
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        tf.config.set_soft_device_placement(False)
        result.update(tensorflow_version=tf.__version__,memory_policy=memory,jit_compile=True,tf32=False,
            devices=[d.name for d in tf.config.list_logical_devices("GPU")],
            dtype="float32",random_backend="TF stateless Philox",datasets=source["datasets"],
            selected_stages=["mdpf"],original_stage_index=1)
        def boolean(name,value):
            checks.append(dict(name=name,passed=bool(value)))
            if not value:raise AssertionError(name)
        def run(label,function,*inputs):
            write(out/"progress.json",dict(stage=label,checks=len(checks),wall_seconds=time.monotonic()-started))
            value=function(*inputs);bool(value["valid"].numpy());return value
        from diagnose_kdm_source_epoch_tf import run_epochs
        run_epochs(tf,reference,baseline,args,out,result,boolean,run,write,require_same_reference=False,
            seed_base=seed_base,epoch_limit=2 if args.mode=="control" else None,
            selected_stages=("mdpf",),initial_states={"mdpf":state})
        boolean("explicit starting state consumed",read(out/"mdpf-starting-state.json")==state)
        if args.mode=="control":
            expected=[r for r in map(json.loads,(old/"batches.jsonl").open()) if r["stage"]=="mdpf" and r["epoch"]<2]
            actual=list(map(json.loads,(out/"batches.jsonl").open()))
            boolean("control batch count",len(expected)==len(actual)==166)
            maximum=0.
            for index,(a,b) in enumerate(zip(actual,expected)):
                for key in ("stage","epoch","split","batch_index","sequence_batch","effective_batch","row_ids","seed","valid","committed"):
                    boolean("control batch%d %s"%(index,key),a[key]==b[key])
                maximum=max(maximum,abs(a["loss"]-b["loss"]))
                boolean("control loss%d"%index,math.isclose(a["loss"],b["loss"],abs_tol=1e-5,rel_tol=1e-6))
            expected=[r for r in map(json.loads,(old/"epochs.jsonl").open()) if r["stage"]=="mdpf" and r["epoch"]<2]
            actual=list(map(json.loads,(out/"epochs.jsonl").open()))
            for index,(a,b) in enumerate(zip(actual,expected)):
                boolean("control controller%d"%index,a["event"]==b["event"])
                for split in ("training","validation"):
                    boolean("control row ordering",a["splits"][split]["row_order_sha256"]==b["splits"][split]["row_order_sha256"])
            result["control_max_batch_loss_difference"]=maximum
        result["status"]="PASS"
        result["selected_stages"]=["mdpf"]
    except Exception:
        result.update(status="FAIL",error=traceback.format_exc())
    result.update(checks_total=len(checks),checks_failed=sum(not x["passed"] for x in checks),
        wall_seconds=time.monotonic()-started)
    write(out/"checks.json",checks);write(out/"result.json",result)
    write(out/"progress.json",dict(terminal=True,status=result["status"],wall_seconds=result["wall_seconds"]))
    return 0 if result["status"]=="PASS" else 1
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=("control","train"),required=True)
    p.add_argument("--unit",type=int,choices=(1,2),default=1)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--author-start",type=Path)
    p.add_argument("--verification",type=Path,required=True)
    raise SystemExit(main(p.parse_args()))
