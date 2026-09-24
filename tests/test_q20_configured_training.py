"""Tiny CPU engineering checks; no evidence about learned q20 quality."""
from datetime import datetime, timedelta, timezone
import importlib.util
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import time

import tensorflow as tf

from bayesfilter.inference.neutra_transport import NeuTraTransport, NeuTraTransportConfig


def test_evaluation_graph_measures_changed_checkpoints_on_the_same_bank(tmp_path):
    path = Path(__file__).resolve().parents[1]/"docs/benchmarks/run_q20_configured_training_2026_09_24.py"
    spec=importlib.util.spec_from_file_location("configured_training_worker",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime=module.Runtime.__new__(module.Runtime)
    runtime.tf=tf
    runtime.request={"worker_seconds":120.,"evaluation_batch_size":8,
        "deadline_utc":(datetime.now(timezone.utc)+timedelta(minutes=2)).isoformat()}
    runtime.began=time.monotonic()
    runtime.noise_graphs={}
    runtime.measure_graphs={}
    runtime.target=lambda x: (-.5*tf.reduce_sum(x*x,axis=1),-x,tf.ones([tf.shape(x)[0]],tf.bool))
    flow=NeuTraTransport(NeuTraTransportConfig(4,"iaf",(4,4),1,"elu",(4,8),2.,dtype="float32"))
    before=runtime.measurement(flow,16,"heldout")
    flow.stages[0].biases[-1].assign_add(tf.ones_like(flow.stages[0].biases[-1])*.2)
    after=runtime.measurement(flow,16,"heldout")
    again=runtime.measurement(flow,16,"heldout")
    assert before["blocks"]["latent"]==after["blocks"]["latent"]
    assert before["map_parameters_sha256"]!=after["map_parameters_sha256"]
    assert before["mean_loss"]!=after["mean_loss"]
    assert after==again
    assert len(runtime.measure_graphs)==1
    assert next(iter(runtime.measure_graphs.values()))[1].experimental_get_tracing_count()==1


def test_final_report_preserves_paired_difference_and_checks_bank_identity(tmp_path):
    """Constant paired differences have an exact bootstrap interval."""
    runner=Path(__file__).resolve().parents[1]/"docs/benchmarks/summarize_q20_configured_training_2026_09_24.py"
    latent=[[0.,0.,0.,0.]]*2048
    baseline={"role":"untouched-final-bank","rows":2048,"blocks":{
        "latent":latent,"physical":[[0.,0.,-1.,0.]]*1024+[[0.,0.,1.,0.]]*1024,
        "loss":[2.]*2048,"residual":[[2.,0.,0.,0.]]*2048,"valid":[True]*2048}}
    parameters=[{"weights":[],"biases":[]}]
    candidate={"role":"untouched-final-bank","rows":2048,"transport_config":{},
        "map_parameters_sha256":hashlib.sha256(json.dumps(parameters,sort_keys=True).encode()).hexdigest(),
        "blocks":{"latent":latent,"physical":[[0.,0.,1.,0.]]*2048,
        "loss":[1.]*2048,"residual":[[1.,0.,0.,0.]]*2048,"valid":[True]*2048}}
    checkpoint={"transport_config":{},"parameters":parameters,"checkpoint_hash":"fixture-only"}
    finalized={"checkpoint":checkpoint,"post_training":{"rows":1000,"valid_rows":1000,
        "complete":True,"finite":True},"inverse_tail_check":{"passed":True}}
    for name,data in (("baseline",baseline),("candidate",candidate),("checkpoint",checkpoint),("finalized",finalized)):
        (tmp_path/f"{name}.json").write_text(json.dumps(data))
    request={"baseline":str(tmp_path/"baseline.json"),"candidates":[{"family":"fixture","root_seed":0,
        "measurement":str(tmp_path/"candidate.json"),"checkpoint":str(tmp_path/"checkpoint.json"),
        "finalized":str(tmp_path/"finalized.json")}],"scope":"synthetic reporting invariant only"}
    request_path=tmp_path/"request.json"
    request_path.write_text(json.dumps(request))
    subprocess.run([sys.executable,str(runner),"--request",str(request_path),"--output",str(tmp_path/"report")],check=True,capture_output=True)
    report=json.loads((tmp_path/"report/result.json").read_text())
    stats=report["comparisons"][0]["statistics"]
    assert stats["mean"]==[-1.,-3.]
    assert stats["bootstrap_interval"]==[[-1.,-3.],[-1.,-3.]]
    assert not report["families"]["fixture"]["replicated_local_improvement"]
    assert report["baseline_geometry"]["observation_weight_sign"]["positive_rows"]==1024
    coverage=report["comparisons"][0]["geometry"]["observation_weight_sign"]
    assert coverage["positive_rows"]==2048 and not coverage["posterior_probability_estimate"]
    candidate["blocks"]["latent"]=[[1.,0.,0.,0.]]*2048
    (tmp_path/"candidate.json").write_text(json.dumps(candidate))
    rejected=subprocess.run([sys.executable,str(runner),"--request",str(request_path),"--output",str(tmp_path/"mismatch")],capture_output=True,text=True)
    assert rejected.returncode!=0 and "do not share the untouched bank" in rejected.stderr
