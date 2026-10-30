"""CPU-only, non-XLA localization; no GPU/default-readiness evidence."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT = Path("/home/chakwong/BayesFilter")
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    import tensorflow as tf
    from bayesfilter.replication.kdm_normal_von_mises_tf import NormalVonMisesKDM
    base = ROOT / "docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
    refpath = base / "kdm-circular-20261006-02/author.json"
    reference = json.loads(refpath.read_text())
    kernels, checks, outputs = {}, [], []
    def compare(name, actual, expected, dtype):
        if isinstance(actual, (tuple, list)):
            assert len(actual) == len(expected)
            for i, (x, y) in enumerate(zip(actual, expected)):
                compare(name + "/" + str(i), x, y, dtype)
            return
        expected = tf.constant(expected, dtype=dtype)
        bound = tf.constant(128*2**-23 if dtype == tf.float32 else 1e-9, dtype) * (1 + tf.abs(expected))
        delta = tf.abs(actual - expected)
        checks.append(dict(name=name, pass_check=bool(tf.reduce_all(delta <= bound).numpy()),
                           max_absolute_error=float(tf.reduce_max(delta).numpy()),
                           max_scaled_error=float(tf.reduce_max(delta / bound).numpy())))
    def host(value):
        if isinstance(value, dict):
            return {k: host(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [host(v) for v in value]
        return value.numpy().tolist()
    for case in reference["cases"]:
        dtype = getattr(tf, case["dtype"])
        if case["dtype"] not in kernels:
            kernels[case["dtype"]] = NormalVonMisesKDM(
                batch_size=2, particles=25, circular_dimensions=[False, False, True], dtype=dtype, jit_compile=False)
        kernel = kernels[case["dtype"]]
        v = {k: tf.constant(x, dtype) for k, x in case["inputs"].items()}
        l,w,b,s = [v[k] for k in ("locations","weights","bandwidth","samples")]
        res = kernel.resampling_score(l,w,b,s)
        post = kernel.posterior_score(l,w,b,v["states"])
        consumer = kernel.consumer_score(l,w,b,s,v["posterior_locations"],v["model_weights"],v["posterior_bandwidth"],v["states"])
        for category, result in (("resampling",res),("posterior",post),("consumer",consumer)):
            assert bool(result["valid"].numpy())
            for k, expected in case[category].items():
                if k == "sample_gradient_stopped":
                    assert bool(result[k].numpy()) == expected
                else:
                    compare(case["id"]+"/"+category+"/"+k,result[k],expected,dtype)
        outputs.append(dict(id=case["id"], resampling=host(res), posterior=host(post), consumer=host(consumer)))
    summary=dict(status="PASS", scope="CPU_NON_XLA_EXPLANATORY_LOCALIZATION", replication="INCOMPLETE",
                 numerical_screen_passed=all(x["pass_check"] for x in checks), checks=checks,
                 failed=[x for x in checks if not x["pass_check"]], tensorflow=tf.__version__,
                 cpu_only=True,gpu_intentionally_hidden=True,jit_compile=False,
                 gpu_default_readiness_evidence=False,
                 traces={k:{n:getattr(v,n).experimental_get_tracing_count() for n in ("resampling_score","posterior_score","consumer_score")} for k,v in kernels.items()})
    summary["scientific_verdict"]="CPU_REFERENCE_MATCH" if summary["numerical_screen_passed"] else "CPU_REFERENCE_ALSO_DIFFERS"
    for name, obj in (("summary.json",summary),("outputs.json",outputs)):
        (args.output/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
    manifest=dict(command=[sys.executable]+sys.argv,plan="docs/plans/controlled-smc-kdm-circular-consumer-20261006.md",
                  result="docs/plans/controlled-smc-kdm-circular-consumer-result-20261006.md",
                  cpu_only=True,gpu_intentionally_hidden=True,jit_compile=False,
                  fixture=str(refpath),fixture_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest(),
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/"bayesfilter/replication/kdm_normal_von_mises_tf.py"]},
                  wall_seconds=time.monotonic()-started)
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({k:summary[k] for k in ("status","scientific_verdict","failed")}))
if __name__=="__main__":main()
