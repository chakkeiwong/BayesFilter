"""Additional observability and repeatability at the same eight fixed probes."""
import argparse,json,pathlib,subprocess,time,sys
import measure as m
parser=argparse.ArgumentParser();parser.add_argument("--arm",choices=["before","after"])
args=parser.parse_args()
folder=m.HERE/"observability-01"
if args.arm is None:
    folder.mkdir(exist_ok=False)
    rows=[]
    for arm in ("before","after"):
        cmd=[m.PYTHON,str(pathlib.Path(__file__).resolve()),"--arm",arm]
        t=time.monotonic()
        with (folder/(arm+".log")).open("w") as log:
            proc=subprocess.run(cmd,cwd=m.ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        rows.append(dict(command=cmd,wall_seconds=time.monotonic()-t,exit_code=proc.returncode))
        m.write(folder/"manifest.json",dict(workers=rows,purpose="Same-probe FP32 repeatability, all-eight runtime force parity, and retained base status diagnostics"))
        print(arm,proc.returncode,rows[-1]["wall_seconds"],flush=True)
        if proc.returncode:sys.exit(proc.returncode)
else:
    started=time.monotonic()
    tf,env=m.setup_tf(m.HERE/("source_"+args.arm),False)
    from bayesfilter.highdim.ledh_range_bearing_target_tf import M13LikelihoodTarget,make_range_bearing_model
    from bayesfilter.testing.surrogate_hmc_m13_four_chain_cost import native_force
    common=m.read(m.HERE/"inputs.json")
    inp=m.input_tensors(tf,common,tf.float32)
    controls=common["controls"]|{"reset_design":inp.pop("reset_design")}
    coefficients=common["model_coefficients_rounded_fp32"]
    model=make_range_bearing_model(*[tf.constant(coefficients[k],tf.float32) for k in ("A","Q","R")],dtype=tf.float32)
    target=M13LikelihoodTarget(model,inp,controls,common["target_manifest"],"diagnostic-frozen-inputs")
    evaluate=target.compile(4,filter_moments=False,all_correction_diagnostics=False)
    force=native_force(target,batch_size=4)
    points=tf.constant([x["theta"] for x in common["probes"]],tf.float32)
    rows=[]
    for i in (0,4):
        value,score,status=evaluate(points[i:i+4])
        actual=force(tf.cast(points[i:i+4],tf.float64))
        rows.append(dict(value=m.serial(value),score=m.serial(score),status=m.serial(status),
                         runtime_force=m.serial(actual.force),runtime_force_valid=m.serial(actual.force_valid)))
    from bayesfilter.testing.surrogate_hmc_m13_endpoint_mechanics import endpoint_components
    _,endpoint=endpoint_components(m.fixture_from_data(tf,common["data"]))
    endpoints=[]
    for point in tf.unstack(tf.cast(points,tf.float64)):
        value,valid,diagnostics=endpoint(point)
        endpoints.append(dict(value=m.serial(value),valid=bool(valid),diagnostics=m.serial(diagnostics)))
    m.write(folder/(args.arm+".json"),dict(arm=args.arm,environment=env,base=rows,endpoint=endpoints,
        wall_seconds=time.monotonic()-started,gpu_memory=tf.config.experimental.get_memory_info("GPU:0")))
