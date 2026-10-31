"""Read-only matched M13 score diagnostic; never an HMC runtime or tuner."""
from __future__ import annotations
import argparse, datetime, hashlib, io, json, math, os, pathlib, shutil
import subprocess, sys, tarfile, time, types

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PYTHON = "/home/chakwong/anaconda3/envs/tftwogpu/bin/python"
PLAN = "docs/plans/surrogate-hmc-m13-score-before-after-2026-10-04.md"
SYNC = ROOT/"docs/plans/artifacts/main-sync-20261003-01"
STASH = "2ca7b4f06b623cdfe694ed67c2fde89449cecf37"
DATA = ROOT/"docs/plans/artifacts/surrogate-hmc-m13-sequential-48h-03/m13-public-tuning-02/data.json"
SELECTION = ROOT/"docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-materiality-repair-01/m13-materiality-repair/selection.json"
ARCHIVE = ROOT/"docs/plans/artifacts/surrogate-hmc-m13-reclassified-20261002-02/all-postwarmup-samples.tfbin"
GEOMETRY = ROOT/"docs/plans/artifacts/surrogate-hmc-m13-whitening-check-20261002-02/report.json"
STEPS = [1e-3, 3e-4, 1e-4]

def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def read(path):
    return json.loads(pathlib.Path(path).read_text())
def write(path, value):
    pathlib.Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
def serial(x):
    if hasattr(x, "numpy"):
        return serial(x.numpy().tolist())
    if isinstance(x, dict):
        return {str(k):serial(v) for k,v in x.items()}
    if isinstance(x, (tuple, list)):
        return [serial(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x
def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)
def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
def progress(message):
    print(stamp(), message, flush=True)

def source_snapshots():
    records = {}
    before = HERE/"source_before"
    after = HERE/"source_after"
    before.mkdir()
    after.mkdir()
    paths = [p for p in git("ls-tree","-r","--name-only",STASH,"--","bayesfilter","experiments").decode().splitlines()
             if p.endswith(".py")]
    raw = git("archive",STASH,*paths)
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for item in archive:
            if item.isfile():
                assert item.name in paths
                dst = before/item.name
                dst.parent.mkdir(parents=True,exist_ok=True)
                dst.write_bytes(archive.extractfile(item).read())
    omitted = []
    for line in (SYNC/"untracked-inventory.jsonl").read_text().splitlines():
        row = json.loads(line)
        p = row["path"]
        if not p.endswith(".py") or not p.startswith(("bayesfilter/","experiments/")):
            continue
        if not (ROOT/p).is_file() or sha(ROOT/p) != row.get("sha256"):
            omitted.append(p)
            continue
        dst = before/p
        assert not dst.exists()
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/p,dst)
    for package in ("bayesfilter","experiments"):
        for p in (ROOT/package).rglob("*.py"):
            if "__pycache__" not in p.parts:
                dst = after/p.relative_to(ROOT)
                dst.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(p,dst)
    for name, base in (("before",before),("after",after)):
        records[name] = {str(p.relative_to(base)):sha(p) for p in sorted(base.rglob("*.py"))}
    write(HERE/"source-snapshots.json",dict(
        created_utc=stamp(),original_head=git("rev-parse",STASH+"^1").decode().strip(),
        stash=STASH,current_head=git("rev-parse","HEAD").decode().strip(),
        branch=git("branch","--show-current").decode().strip(),
        pre_sync_unrecoverable_untracked_modules_excluded=omitted,
        exclusion_rule="Excluded modules must not be imported by a measurement worker.",
        sources=records))
    progress(f"Source snapshots: {len(records['before'])} before, {len(records['after'])} after; omitted {omitted}")

def setup_tf(source, cpu):
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]="1"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
    os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
    os.environ["TF_NUM_INTEROP_THREADS"]="2"
    os.environ["TF_NUM_INTRAOP_THREADS"]="4"
    sys.path.insert(0,str(source))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    policy=configure_tensorflow_gpu_memory_growth(tf,require_gpu=not cpu)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    meta=dict(tensorflow=tf.__version__,python=sys.version,executable=sys.executable,
              gpu_intentionally_hidden=cpu,cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
              gpu_memory_policy=serial(policy),jit_compile=not cpu,
              tf32=bool(tf.config.experimental.tensor_float_32_execution_enabled()),
              physical_gpus=[str(x) for x in tf.config.list_physical_devices("GPU")])
    return tf,meta

def fixture_from_data(tf, d):
    attrs={k:tf.constant(d["model"][k],tf.float64) for k in ("A","Q","R","m0","P0")}
    attrs.update(observations=tf.constant(d["observations"],tf.float64),
                 horizon=d["model"]["horizon"],name=d["model"]["model_id"],
                 observation_checksum=d["observation_checksum"],fixture_generation_seed=d["data_seed"])
    attrs["model_definition"]=lambda:d["model"]
    return types.SimpleNamespace(**attrs)

def prepare():
    start=time.monotonic()
    source_snapshots()
    tf,env=setup_tf(HERE/"source_before",True)
    from bayesfilter.highdim.ledh_range_bearing_target_tf import make_range_bearing_target
    assert sha(ARCHIVE)=="1fef58b4e04aa4f69dbc8479161e8fafe8f856b4c4e44681e10d60d8e04dc9cc"
    d=read(DATA)
    fixture=fixture_from_data(tf,d)
    controls=read(SELECTION)["controls"]
    target=make_range_bearing_target(fixture,particle_count=4096,controls=controls,
        dtype=tf.float32,design_seed=20260923,design_partition=9)
    folder=HERE/"inputs";folder.mkdir()
    tensors=dict(target.inputs)|{"reset_design":target.controls["reset_design"]}
    tensor_records={}
    for key,value in tensors.items():
        path=folder/(key+".tfbin")
        tf.io.write_file(str(path),tf.io.serialize_tensor(value))
        tensor_records[key]=dict(path=str(path.relative_to(HERE)),sha256=sha(path),
                                  dtype=value.dtype.name,shape=value.shape.as_list())
    samples=tf.io.parse_tensor(tf.io.read_file(str(ARCHIVE)),tf.float64)
    tf.debugging.assert_equal(tf.shape(samples),[12750,4,2])
    tf.debugging.assert_all_finite(samples,"archive")
    flat=tf.reshape(samples,[-1,2])
    order=tf.argsort(flat[:,0],stable=True)
    probes=[]
    for pct in (1,5,25,50,75,95,99):
        rank=round((int(flat.shape[0])-1)*pct/100)
        row=int(order[rank])
        point=flat[row]
        probes.append(dict(label=f"visited_Q_p{pct:02}",draw=row//4,chain=row%4,
            original_theta=serial(point),theta=serial(tf.cast(point,tf.float32))))
    mean=tf.reduce_mean(flat,axis=0)
    probes.append(dict(label="visited_pooled_mean",draw=None,chain=None,
        original_theta=serial(mean),theta=serial(tf.cast(mean,tf.float32))))
    coefficients={k:serial(tf.cast(getattr(fixture,k),tf.float32)) for k in ("A","Q","R")}
    write(HERE/"inputs.json",dict(data=d,controls=controls,probes=probes,tensors=tensor_records,
        model_coefficients_rounded_fp32=coefficients,target_manifest=dict(target.manifest),
        geometry=read(GEOMETRY)["transform"],finite_difference_steps=STEPS,
        source_files={str(p):sha(p) for p in (DATA,SELECTION,ARCHIVE,GEOMETRY)},
        environment=env,wall_seconds=time.monotonic()-start))
    progress("Eight probes and original N4096 designs frozen")

def input_tensors(tf, record, dtype):
    result={}
    for k,v in record["tensors"].items():
        p=HERE/v["path"];assert sha(p)==v["sha256"]
        result[k]=tf.cast(tf.io.parse_tensor(tf.io.read_file(str(p)),tf.float32),dtype)
    return result

def worker(arm, dtype_name):
    start=time.monotonic()
    output=pathlib.Path(os.environ.get("M13_SCORE_OUTPUT_ROOT",str(HERE)))/(arm+"-"+dtype_name+".json")
    assert not output.exists()
    source=HERE/("source_"+arm)
    tf,env=setup_tf(source,False)
    import tensorflow_probability as tfp
    env["tensorflow_probability"]=tfp.__version__
    from bayesfilter.highdim.ledh_range_bearing_target_tf import M13LikelihoodTarget,make_range_bearing_model
    from bayesfilter.testing.surrogate_hmc_m13_four_chain_cost import native_force
    common=read(HERE/"inputs.json")
    dtype=tf.as_dtype(dtype_name)
    inputs=input_tensors(tf,common,dtype)
    controls=common["controls"]|{"reset_design":inputs.pop("reset_design")}
    coefficients=common["model_coefficients_rounded_fp32"]
    model=make_range_bearing_model(*[tf.constant(coefficients[k],dtype) for k in ("A","Q","R")],dtype=dtype)
    target=M13LikelihoodTarget(model,inputs,controls,common["target_manifest"],"diagnostic-frozen-inputs")
    evaluator=target.compile(4,filter_moments=False,all_correction_diagnostics=False)
    points=tf.constant([p["theta"] for p in common["probes"]],dtype)
    report=dict(schema="bayesfilter.m13.before_after_score_diagnostic.v1",arm=arm,dtype=dtype_name,
        environment=env,started_utc=stamp(),status="RUNNING",input_manifest_sha256=sha(HERE/"inputs.json"),
        plan=PLAN,source_snapshot_sha256=sha(HERE/"source-snapshots.json"),base=[],differences=[],
        command=sys.argv,steps=STEPS)
    write(output,report)
    def evaluate(batch):
        t=time.monotonic()
        value,score,status=evaluator(batch)
        result=dict(value=serial(value),score=serial(score),
            valid=serial(status["program_valid"]),value_valid=serial(status["value_valid"]),
            force_valid=serial(status["force_valid"]),wall_seconds=time.monotonic()-t)
        return result
    progress(f"{arm} {dtype_name} first batch compilation started")
    for begin in (0,4):
        row=evaluate(points[begin:begin+4])
        report["base"].append(row);write(output,report)
        progress(f"{arm} {dtype_name} base batch {begin//4}: {row['wall_seconds']:.2f}s valid={row['valid']}")
    if dtype==tf.float32:
        force=native_force(target,batch_size=4)
        actual=force(tf.cast(points[:4],tf.float64))
        expected=-tf.constant(report["base"][0]["score"],tf.float32)+points[:4]/tf.cast(
            tf.square(tf.constant(math.log(2.),tf.float64)),tf.float32)
        delta=tf.reduce_max(tf.abs(actual.gradient-tf.cast(expected,tf.float64))) if hasattr(actual,"gradient") else None
        # PositionOnlyForceEvaluation is a NamedTuple with a named force field.
        report["runtime_force_fields"]=getattr(actual,"_fields",None)
        report["runtime_force"]=serial(actual[0])
        report["runtime_force_valid"]=serial(actual[1])
        report["runtime_force_parity_max_abs"]=serial(tf.reduce_max(tf.abs(actual[0]-tf.cast(expected,tf.float64))))
        assert report["runtime_force_parity_max_abs"]==0.
    else:
        for h in STEPS:
            for dim in (0,1):
                pair={}
                for sign in (-1,1):
                    offset=tf.one_hot(dim,2,dtype=dtype)*tf.constant(sign*h,dtype)
                    rows=[evaluate(points[b:b+4]+offset) for b in (0,4)]
                    pair[str(sign)]=rows
                report["differences"].append(dict(step=h,parameter=dim,values=pair))
                write(output,report)
                progress(f"{arm} FP64 own-likelihood differences h={h} coordinate={dim} completed")
        from bayesfilter.testing.surrogate_hmc_m13_endpoint_mechanics import endpoint_components
        _,endpoint=endpoint_components(fixture_from_data(tf,common["data"]))
        endpoint_rows=[]
        for i,point in enumerate(tf.unstack(tf.cast(points,tf.float64))):
            value,valid,_=endpoint(point)
            row=dict(label=common["probes"][i]["label"],value=serial(value),valid=bool(valid),steps=[])
            for h in STEPS:
                row_h=dict(step=h,coordinates=[])
                for dim in (0,1):
                    offset=tf.one_hot(dim,2,dtype=tf.float64)*tf.constant(h,tf.float64)
                    vm,gm,_=endpoint(point-offset)
                    vp,gp,_=endpoint(point+offset)
                    row_h["coordinates"].append(dict(minus=serial(vm),plus=serial(vp),
                        minus_valid=bool(gm),plus_valid=bool(gp),
                        derivative=serial((vp-vm)/(2*h))))
                row["steps"].append(row_h)
            endpoint_rows.append(row)
            report["endpoint"]=endpoint_rows;write(output,report)
            progress(f"{arm} endpoint finite differences point {i+1}/8 completed")
    loaded={}
    for name,module in list(sys.modules.items()):
        if name.startswith(("bayesfilter","experiments")) and getattr(module,"__file__",None):
            path=pathlib.Path(module.__file__).resolve()
            if path.suffix==".py":
                assert path.is_relative_to(source),(name,str(path))
                rel=str(path.relative_to(source))
                expected=read(HERE/"source-snapshots.json")["sources"][arm][rel]
                assert sha(path)==expected
                loaded[rel]=expected
    report["imported_source_hashes"]=loaded
    report["gpu_memory"]=tf.config.experimental.get_memory_info("GPU:0")
    report["wall_seconds"]=time.monotonic()-start
    report["completed_utc"]=stamp()
    report["status"]="COMPLETED"
    write(output,report)
    progress(f"{arm} {dtype_name} complete in {report['wall_seconds']:.2f}s")

def run():
    start=time.monotonic()
    manifest=dict(schema="bayesfilter.m13.score_measurement_manifest.v1",started_utc=stamp(),
        git_commit=git("rev-parse","HEAD").decode().strip(),plan=PLAN,
        result=str(HERE/"result.md"),environment=PYTHON,aggregate_worker_budget_seconds=3600,
        workers=[],status="RUNNING",data_seed=1704,design_seed=20260923,design_partition=9,
        endpoint_seed=[2026092402,9101],new_hmc_transitions=0)
    for arm,dtype in (("before","float32"),("after","float32"),("before","float64"),("after","float64")):
        elapsed=time.monotonic()-start
        timeout=min(1200,3600-elapsed)
        if timeout<=0:
            manifest["status"]="BUDGET_EXHAUSTED";break
        command=[PYTHON,str(HERE/"measure.py"),"--worker",arm,"--dtype",dtype]
        log=HERE/(arm+"-"+dtype+".log")
        row=dict(command=command,log=str(log),started_utc=stamp(),timeout_seconds=timeout)
        manifest["workers"].append(row);write(HERE/"run-manifest.json",manifest)
        t=time.monotonic()
        with log.open("w") as stream:
            try:
                proc=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
                row["exit_code"]=proc.returncode
            except subprocess.TimeoutExpired:
                row["timed_out"]=True
                row["exit_code"]=124
        row["wall_seconds"]=time.monotonic()-t
        row["finished_utc"]=stamp()
        write(HERE/"run-manifest.json",manifest)
        progress(f"{arm} {dtype} exit {row['exit_code']} elapsed {row['wall_seconds']:.2f}s; log {log.name}")
        if row["exit_code"]:
            manifest["status"]="REPAIR_NEEDED";break
    else:
        manifest["status"]="COMPLETED"
    manifest["wall_seconds"]=time.monotonic()-start
    manifest["remaining_worker_budget_seconds"]=max(0.,3600-manifest["wall_seconds"])
    manifest["script_sha256"]=sha(HERE/"measure.py")
    write(HERE/"run-manifest.json",manifest)
    return 0 if manifest["status"]=="COMPLETED" else 1

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--worker",choices=["before","after"])
    parser.add_argument("--dtype",choices=["float32","float64"],default="float32")
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.run:sys.exit(run())
    elif args.worker:worker(args.worker,args.dtype)
    else:parser.error("choose a mode")
