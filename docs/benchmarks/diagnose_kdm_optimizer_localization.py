"""Independent optimizer diagnostic; no training, admission or default change.

Feed identical saved gradients into the actual TF update method and installed
author PyTorch Adam, with fresh or saved mature optimizer moments. This isolates
optimizer arithmetic from model gradients, resampling and training randomness.
"""
import argparse
import ast
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
AUTHOR = BASE / "kdm-trained-update-20261008-01/author.json"
STATE = BASE / "kdm-complete-training-20261007-04/diagnostic/mdpf-best.json"
PLAN = ROOT / "docs/plans/controlled-smc-kdm-root-cause-investigation-20261008.md"
ENDPOINTS = (1, 2, 16)


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs():
    source = json.loads(AUTHOR.read_text())
    case = next(c for c in source["cases"] if c["fixture_id"] == "tf_trained"
                and c["variant"] == "mdpf" and c["dtype"] == "float32")
    state = json.loads(STATE.read_text())["model"]
    assert case["parameters"] == state["parameters"]
    assert case["trainable_names"] == state["trainable_names"]
    return case, state


def reference(out):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    case, saved = inputs()
    names = case["trainable_names"]
    rows = []
    for dtype_name in ("float32", "float64"):
        dtype = getattr(torch, dtype_name)
        for origin in ("fresh", "mature"):
            params = {n: torch.nn.Parameter(torch.tensor(case["parameters"][n], dtype=dtype)) for n in names}
            # Same frozen learning rates in both moment arms; they are explicitly
            # a mechanics probe, not a resumed training trajectory.
            optimizer = torch.optim.Adam([
                {"params": [params[n]], "lr": case["learning_rates"][n]} for n in names],
                weight_decay=0.0, foreach=False)
            for n, p in params.items():
                if origin == "mature":
                    optimizer.state[p] = dict(step=torch.tensor(saved["step"], dtype=torch.float32),
                        exp_avg=torch.tensor(saved["first_moments"][n], dtype=dtype),
                        exp_avg_sq=torch.tensor(saved["second_moments"][n], dtype=dtype))
                p.grad = torch.tensor(case["updates"][0]["clipped_gradients"][n], dtype=dtype)
            for step in range(1, 17):
                optimizer.step()
                if step in ENDPOINTS:
                    rows.append(dict(dtype=dtype_name, origin=origin, step=step,
                        parameters={n:p.detach().tolist() for n,p in params.items()},
                        first_moments={n:optimizer.state[p]["exp_avg"].tolist() for n,p in params.items()},
                        second_moments={n:optimizer.state[p]["exp_avg_sq"].tolist() for n,p in params.items()}))
    import torch.optim.adam as adam
    write(out / "reference.json", dict(rows=rows, torch_version=torch.__version__,
        cpu_only=True, gpu_intentionally_hidden=True, adam_file=adam.__file__,
        adam_sha256=sha(Path(adam.__file__)), optimizer="Adam foreach=False, no decay; identical prescribed gradients",
        steps=16, endpoints=ENDPOINTS, mature_step=saved["step"]))


def candidate(out):
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    sys.path.insert(0, str(ROOT))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.set_soft_device_placement(False)
    from bayesfilter.replication.kdm_neural_replay_tf import BearingsNeuralReplay
    case, saved = inputs()
    expected = json.loads((out / "reference.json").read_text())
    refs = {(r["dtype"], r["origin"], r["step"]): r for r in expected["rows"]}
    rows, traces = [], []
    # Diagnostic intervention changes ONLY bias-correction scalar precision.
    # Preserve the exact original method and its diff in the result directory.
    original = textwrap.dedent(inspect.getsource(BearingsNeuralReplay._apply_update))
    changed = original.replace("self.c(1)-tf.pow(self.c(.9),t)",
        "tf.cast(tf.constant(1.,tf.float64)-tf.pow(tf.constant(.9,tf.float64),tf.cast(t,tf.float64)),self.dtype)")
    changed = changed.replace("tf.sqrt(self.c(1)-tf.pow(self.c(.999),t))",
        "tf.cast(tf.sqrt(tf.constant(1.,tf.float64)-tf.pow(tf.constant(.999,tf.float64),tf.cast(t,tf.float64))),self.dtype)")
    assert changed != original
    env = {"tf": tf}
    exec(compile(ast.parse(changed), "diagnostic_bias_precision_intervention", "exec"), env)
    altered_update = env["_apply_update"]
    (out / "original_update.py").write_text(original)
    (out / "bias_precision_intervention.py").write_text(changed)
    modes = (("gpu_xla", "float32", "/GPU:0", True, False),
             ("gpu_xla", "float64", "/GPU:0", True, False),
             ("cpu_graph_reference", "float32", "/CPU:0", False, False),
             ("gpu_xla_bias64_diagnostic", "float32", "/GPU:0", True, True))
    for mode, dtype_name, device, jit, intervene in modes:
        fixture = copy.deepcopy(case)
        fixture["dtype"] = dtype_name
        with tf.device(device):
            model = BearingsNeuralReplay(fixture, jit_compile=jit)
            gradients = [tf.constant(case["updates"][0]["clipped_gradients"][n], model.dtype) for n in model.names]
            def sequence(*values):
                loss = tf.add_n([tf.reduce_sum(p*g) for p,g in zip(model.trainable, values)])
                return loss, tf.constant(True), ()
            update_method = altered_update.__get__(model) if intervene else model._apply_update
            def update(*values):
                return update_method(sequence, values, gradient_clip=None)
            function = tf.function(update, input_signature=[tf.TensorSpec(g.shape, g.dtype) for g in gradients],
                                   jit_compile=jit, autograph=False)
            for origin in ("fresh", "mature"):
                state = model.state_dict()
                state["parameters"] = copy.deepcopy(case["parameters"])
                if origin == "mature":
                    state["first_moments"] = saved["first_moments"]
                    state["second_moments"] = saved["second_moments"]
                    state["step"] = saved["step"]
                else:
                    state["step"] = 0.
                    for field in ("first_moments", "second_moments"):
                        state[field] = {n:tf.zeros_like(model.parameters[n]).numpy().tolist() for n in model.names}
                model.load_state_dict(state)
                for step in range(1, 17):
                    result = function(*gradients)
                    assert bool(result["committed"].numpy()) and bool(result["valid"].numpy())
                    if step not in ENDPOINTS:
                        continue
                    ref = refs[dtype_name, origin, step]
                    stats = {}
                    for field, tensors in (("parameters", model.trainable),
                                           ("first_moments", model.moments1),
                                           ("second_moments", model.moments2)):
                        errors = []
                        for n, actual in zip(model.names, tensors):
                            target = tf.constant(ref[field][n], dtype=model.dtype)
                            error = float(tf.reduce_max(tf.abs(actual-target)).numpy())
                            scale = float(tf.reduce_max(tf.abs(target)).numpy())
                            errors.append(dict(name=n, max_abs_error=error, max_abs_reference=scale,
                                               relative_to_tensor_max=error/max(scale,1e-30)))
                        stats[field] = dict(max_abs_error=max(e["max_abs_error"] for e in errors),
                            max_tensor_relative_error=max(e["relative_to_tensor_max"] for e in errors),
                            tensors=errors)
                    rows.append(dict(mode=mode,dtype=dtype_name,origin=origin,step=step,comparisons=stats))
            count = function.experimental_get_tracing_count()
            assert count == 1
            traces.append(dict(mode=mode,dtype=dtype_name,traces=count,device=model.trainable[0].device,jit_compile=jit))
    # Float64 primitive correspondence is a validity control, not a float32
    # acceptance rule and not a model-training claim.
    control = all(v["max_abs_error"] < 1e-10 for r in rows if r["dtype"] == "float64"
                  for v in r["comparisons"].values())
    write(out / "result.json", dict(status="PASS" if control else "FAIL_CONTROL", rows=rows,
        float64_control_passed=control, trace_counts=traces, tensorflow_version=tf.__version__,
        memory_policy=memory,tf32=False,allocator=tf.config.experimental.get_memory_info("GPU:0"),
        inference="optimizer mechanics only; neither training drift attribution nor qualification"))
    if not control:
        raise RuntimeError("Float64 optimizer correspondence control failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=("reference", "candidate"))
    args = parser.parse_args()
    out = args.output.resolve()
    if args.worker:
        return (reference if args.worker == "reference" else candidate)(out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(plan=str(PLAN), command=sys.argv, git_commit=subprocess.check_output(
        ["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(), status="RUNNING",
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in (AUTHOR,STATE)},
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),
            ROOT/"bayesfilter/replication/kdm_neural_replay_tf.py")},
        seeds="N/A: frozen prescribed gradients; no sampling", result=str(out/"result.json"), children=[])
    write(out/"manifest.json",manifest)
    for worker, interpreter in (("reference","/home/chakwong/anaconda3/envs/tf-gpu2/bin/python"),
                                ("candidate","/home/chakwong/anaconda3/envs/tftwogpu/bin/python")):
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1",TF_NUM_INTEROP_THREADS="1",TF_NUM_INTRAOP_THREADS="1")
        if worker == "reference":
            env["CUDA_VISIBLE_DEVICES"]="-1"
            env["LD_LIBRARY_PATH"]="/home/chakwong/anaconda3/envs/tf-gpu2/lib"
        else:
            env["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
            env["CUDA_VISIBLE_DEVICES"]=json.loads((BASE/"kdm-complete-training-20261007-04/manifest.json").read_text())["environment"]["CUDA_VISIBLE_DEVICES"]
        command=[interpreter,"-I","-B",str(Path(__file__).resolve()),"--worker",worker,"--output",str(out)]
        tick=time.monotonic()
        with (out/(worker+".log")).open("w") as log:
            completed=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
        manifest["children"].append(dict(worker=worker,command=command,returncode=completed.returncode,
            wall_seconds=time.monotonic()-tick,environment={k:env.get(k) for k in
                ("CUDA_VISIBLE_DEVICES","LD_LIBRARY_PATH","OMP_NUM_THREADS","TF_FORCE_GPU_ALLOW_GROWTH")}))
        write(out/"manifest.json",manifest)
        if completed.returncode:
            manifest["status"]="FAILED"
            write(out/"manifest.json",manifest)
            raise SystemExit(completed.returncode)
    manifest.update(status="COMPLETED", result_sha256=sha(out/"result.json"))
    write(out/"manifest.json",manifest)
    print(json.dumps(dict(status=manifest["status"],result=manifest["result"],children=manifest["children"])))


if __name__ == "__main__":
    main()
