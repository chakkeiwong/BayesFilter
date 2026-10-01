"""Inspect log-domain NAF validity at saved failed inputs; no target calls."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gpu",required=True)
    args=parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,TF_FORCE_GPU_ALLOW_GROWTH="true",
        TF_NUM_INTRAOP_THREADS="2",TF_NUM_INTEROP_THREADS="2",OMP_NUM_THREADS="2")
    args.output.mkdir(parents=True,exist_ok=False)
    began=time.monotonic()
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    from bayesfilter.inference.neutra_transport import NeuTraTransport,NeuTraTransportConfig
    result={"command":sys.argv,"memory_policy":memory,"rows":[],"purpose":"debug-only log-weight guard inspection"}
    for seed in (0,1):
        path=args.campaign/"training-queue-01"/f"naf16-seed{seed}-u4096"/"last-valid-checkpoint.json"
        checkpoint=json.loads(path.read_text())
        iteration=checkpoint["optimizer"][0]["value"]
        flow=NeuTraTransport(NeuTraTransportConfig(**checkpoint["transport_config"]))
        flow.restore_parameters(checkpoint["parameters"])
        raw=hashlib.sha256(f"q20-configured-training-20260924:training-{seed}:{iteration}".encode()).digest()
        key=[int.from_bytes(raw[i:i+4],"big") & 0x7fffffff for i in (0,4)]
        @tf.function(input_signature=[tf.TensorSpec([2],tf.int32)],jit_compile=True,autograph=False)
        def inspect(key):
            x=tf.random.stateless_normal([32,4],key,dtype=tf.float32)
            rows=[]
            for layer in flow.components:
                if getattr(layer,"kind",None)=="naf_dsf":
                    log_slope,offset,logits=layer.pseudo_parameters(x)
                    log_w=tf.nn.log_softmax(logits,axis=-1)
                    zeros=tf.exp(log_w)==0.
                    rows.append({"finite_log_weights":tf.reduce_all(tf.math.is_finite(log_w)),
                        "finite_slopes":tf.reduce_all(tf.math.is_finite(tf.exp(log_slope))),
                        "minimum_slope":tf.reduce_min(tf.exp(log_slope)),
                        "minimum_log_weight":tf.reduce_min(log_w),
                        "underflowed_weight_count":tf.reduce_sum(tf.cast(zeros,tf.int32)),
                        "rejected_row_coordinate":tf.where(tf.reduce_any(zeros,axis=-1))})
                x,_=layer.forward_and_logdet(x)
            return rows
        rows=[{k:v.numpy().tolist() for k,v in row.items()} for row in inspect(tf.constant(key))]
        result["rows"].append({"seed":seed,"failed_input_index":iteration,"seed_pair":key,
            "checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"layers":rows})
    result.update(wall_seconds=time.monotonic()-began,finished_at=datetime.now(timezone.utc).isoformat())
    (args.output/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"result":str(args.output/"result.json"),"wall_seconds":result["wall_seconds"]}))


if __name__=="__main__":
    main()
