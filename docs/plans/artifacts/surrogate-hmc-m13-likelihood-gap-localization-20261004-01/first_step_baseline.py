"""Diagnostic first-observation bootstrap comparator; no replacement LEDH lane."""
import pathlib, subprocess, sys, time
import diagnose as d
m, tf = d.m, None
HERE = d.HERE / "first-step-baseline"
def worker():
    started = time.monotonic()
    tf, env = m.setup_tf(d.PREVIOUS / "source_after", False)
    from bayesfilter.highdim.ledh_range_bearing_target_tf import make_range_bearing_model
    common = m.read(d.PREVIOUS / "inputs.json")
    inputs = m.input_tensors(tf, common, tf.float64)
    c = common["model_coefficients_rounded_fp32"]
    a, q, r = [tf.constant(c[k],tf.float64) for k in ("A","Q","R")]
    model = make_range_bearing_model(a,q,r,dtype=tf.float64)
    anchors = inputs["initial_states"] @ tf.transpose(a)
    innovation = inputs["noises"][0] @ tf.transpose(tf.linalg.cholesky(q))
    n = int(anchors.shape[0])
    observation = inputs["observations"][0]

    @tf.function(input_signature=[tf.TensorSpec([2],tf.float64)],jit_compile=True,autograph=False)
    def evaluate(theta):
        scaled = tf.exp(theta[0]) * innovation
        points = anchors + scaled
        theta_rows = tf.broadcast_to(theta[None],[n,2])
        logw = model.observation_log_density_fn(theta_rows,points,observation)
        logsum = tf.reduce_logsumexp(logw)
        weights = tf.exp(logw-logsum)
        scores = []
        for j in range(2):
            direction = tf.broadcast_to(tf.one_hot(j,2,dtype=tf.float64)[None],[n,2])
            dx = scaled if j == 0 else tf.zeros_like(scaled)
            tangents = model.observation_log_density_tangent_fn(
                theta_rows,points,observation,dx,direction)
            scores.append(tf.reduce_sum(weights*tangents))
        return dict(value=logsum-tf.math.log(tf.cast(n,tf.float64)),
                    score=tf.stack(scores),ess=1/tf.reduce_sum(weights**2),
                    maximum_weight=tf.reduce_max(weights))
    rows = []
    for p in common["probes"]:
        theta=tf.constant(p["theta"],tf.float64)
        evaluated=m.serial(evaluate(theta))
        fd=[]
        for j in range(2):
            step=1e-4*tf.one_hot(j,2,dtype=tf.float64)
            fd.append(float((evaluate(theta+step)["value"]-evaluate(theta-step)["value"])/2e-4))
        assert max(abs(x-y) for x,y in zip(fd,evaluated["score"])) < 1e-6
        rows.append(dict(probe=p,**evaluated,score_finite_difference=fd))
    source_hashes=m.read(d.PREVIOUS/"attempt-02/after-float64.json")["imported_source_hashes"]
    for rel,expected in source_hashes.items():
        assert m.sha(d.PREVIOUS/"source_after"/rel)==expected
    m.write(HERE/"report.json",dict(status="COMPLETED",rows=rows,environment=env,
        plan=d.PLAN,input_manifest_sha256=m.sha(d.PREVIOUS/"inputs.json"),
        checked_source_hashes=source_hashes,wall_seconds=time.monotonic()-started,
        gpu_memory=tf.config.experimental.get_memory_info("GPU:0"),
        scientific_role="T1 bootstrap diagnostic using identical frozen particles/noises; no runtime or default change",
        statistical_ranking_supported=False))
def supervisor():
    HERE.mkdir(exist_ok=False)
    cmd=[m.PYTHON,str(pathlib.Path(__file__).resolve()),"--worker"]
    previous=m.read(d.HERE/"run-manifest.json")
    manifest=dict(command=cmd,git_commit=previous["git_commit"],cwd=str(d.ROOT),
        plan=d.PLAN,plan_sha256=m.sha(d.ROOT/d.PLAN),
        source_snapshot=previous["source_snapshot"],executable=m.PYTHON,
        data=previous["data"],seeds=previous["seeds"],hardware=previous["hardware"],
        timeout_seconds=120,script_sha256=m.sha(pathlib.Path(__file__)),
        started_utc=m.stamp(),status="RUNNING",
        result=str(d.HERE/"result.md"),report=str(HERE/"report.json"),log=str(HERE/"worker.log"))
    m.write(HERE/"run-manifest.json",manifest)
    start=time.monotonic()
    with (HERE/"worker.log").open("x") as log:
        try:
            p=subprocess.run(cmd,cwd=d.ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=120)
            code=p.returncode
        except subprocess.TimeoutExpired:
            code=124
    wall=time.monotonic()-start
    manifest.update(status="COMPLETED" if code==0 else "FAILED",exit_code=code,
                    wall_seconds=wall,remaining_budget_seconds=previous["remaining_budget_seconds"]-wall,
                    completed_utc=m.stamp(),log_sha256=m.sha(HERE/"worker.log"))
    if (HERE/"report.json").exists():
        manifest["report_sha256"]=m.sha(HERE/"report.json")
    m.write(HERE/"run-manifest.json",manifest)
    print(code,wall,manifest["remaining_budget_seconds"])
    sys.exit(code)
if __name__=="__main__":
    worker() if "--worker" in sys.argv else supervisor()
