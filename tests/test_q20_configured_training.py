"""Tiny CPU engineering checks; no evidence about learned q20 quality."""
from datetime import datetime, timedelta, timezone
import importlib.util
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
