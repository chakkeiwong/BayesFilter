"""GPU debugging of an exact failed fit; no proposal-quality comparison."""
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path.cwd()))
from bayesfilter.score_study.runtime import configure_runtime, memory_usage
from bayesfilter.score_study.contracts import DiagnosticFailure
runtime = configure_runtime(device="GPU", tf32=True, jit_compile=True)
import tensorflow as tf
from bayesfilter.score_study.gaussian_tf import make_data_kernel
from bayesfilter.score_study.fitted_twist_tf import make_fitted_twist_kernel
from bayesfilter.score_study.iapf_fit_tf import make_density_recursive_fit_kernel

root = Path(__file__).resolve().parent
study = json.loads((root / "iapf-integration-gpu-study.json").read_text())
saved = json.loads((root / "iapf-integration-gpu-01/attempts/iapf_0/attempt-001/failure_diagnostics.json").read_text())["details"]
baseline = json.loads((root / "iapf-integration-gpu-01/attempts/bootstrap_0/attempt-001/result.json").read_text())
config = saved["iapf_configuration"]
previous = saved["fit_iterations"][0]["density_fit_parameters"]
seeds = saved["fit_seed_records"]
N, T, d, o = 16, 2, 1, 1
start = time.monotonic()
with tf.device("/GPU:0"):
    theta = tf.constant(config["fit_theta"], tf.float32)
    observations = make_data_kernel(d, o, T, "float32")(
        tf.constant(study["settings"]["data_theta"], tf.float32),
        tf.constant(baseline["diagnostics"]["data_seed"], tf.int32))
    def draw(name, shape, normal):
        fn = tf.random.stateless_normal if normal else tf.random.stateless_uniform
        return fn(shape, seeds["iapf_fit1_" + name], dtype=tf.float32)
    value, score, clouds, *_ = make_fitted_twist_kernel(d, o, N, T, "float32", True)(
        theta, observations, draw("initial", [N,d], True), draw("process", [T,N,d], True),
        draw("ancestors", [T+1,N], False), draw("mixture", [T,N], False),
        *(tf.constant(previous[key], tf.float32) for key in ("centers", "covariances", "log_floors")))
    replay_error = abs(float(value) - saved["fit_iterations"][1]["log_value"])
    if replay_error != 0:
        raise RuntimeError(f"Exact GPU replay failed: {replay_error}")
    fits = {}
    arguments = [d,o,N,T,config["mean_bound"],config["sd_lower"],config["sd_upper"],
                 config["max_fit_steps"],config["max_backtracks"],config["fit_tolerance"],config["floor_ratio"]]
    for dtype_name in ("float32", "float64"):
        fit = make_density_recursive_fit_kernel(*arguments,dtype_name,True)
        out = fit(*(tf.cast(x, dtype_name) for x in (theta,observations,clouds)))
        fits[dtype_name] = {"valid":bool(out[3]),"converged":bool(out[4]),
            "centers":out[0].numpy().tolist(),"covariances":out[1].numpy().tolist(),
            "log_floors":out[2].numpy().tolist(),"diagnostics":out[5].numpy().tolist()}
result = {"classification":"exact GPU failed-cloud precision diagnostic; no quality claim",
    "plan":"docs/plans/younis-score-iapf-implementation-2026-09-15.md",
    "source_root":str(Path.cwd()),"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    "command":sys.argv,"runtime":runtime,"memory":memory_usage("GPU"),
    "replay_error":replay_error,"theta":theta.numpy().tolist(),"observations":observations.numpy().tolist(),
    "clouds":clouds.numpy().tolist(),"fit_arguments":arguments,"fits":fits,
    "seed_records":seeds,"data_seed":baseline["diagnostics"]["data_seed"],
    "wall_seconds":time.monotonic()-start}
with Path(sys.argv[1]).open("x") as f:
    json.dump(DiagnosticFailure("diagnostic",result).diagnostics,f,indent=2,allow_nan=False)
print(json.dumps({"replay_error":replay_error,"fits":{k:{n:v[n] for n in ("valid","converged","diagnostics")} for k,v in fits.items()},"wall_seconds":result["wall_seconds"]}))
