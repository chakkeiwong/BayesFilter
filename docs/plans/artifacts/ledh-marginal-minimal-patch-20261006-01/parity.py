"""Diagnostic only: matched finite-program comparison, not accuracy evidence."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES="-1", TF_NUM_INTRAOP_THREADS="2",
                  TF_NUM_INTEROP_THREADS="1", OMP_NUM_THREADS="2",
                  BAYESFILTER_PRELOAD_CUSTOM_OP="1", PYTHONDONTWRITEBYTECODE="1",
                  BAYESFILTER_OP_LIB_DIR="/home/chakwong/BayesFilter-SQMC/bayesfilter/ops")
import sys, json, time, hashlib, ast, importlib.util
from pathlib import Path
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--checkout", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--policies", nargs="+", default=["default"])
parser.add_argument("--models", nargs="+", default=["lg3", "ksc", "m13"])
parser.add_argument("--horizons", nargs="+", type=int, default=[1, 20])
parser.add_argument("--particles", type=int, default=64)
args = parser.parse_args()
sys.path.insert(0, args.checkout)
import tensorflow as tf
from bayesfilter.highdim.ledh_canonical_score_tf import NonlinearScoreModel, canonical_value_and_analytical_score
from experiments.dpf_implementation.tf_tfp.fixtures.range_bearing_tf import build_range_bearing_fixture_tf
HERE = Path(__file__).resolve().parent
SOURCE = Path(json.loads((HERE/"setup.json").read_text())["source_snapshot"])
DTYPE = tf.float64
Tensor = tf.Tensor
exec(compile((HERE/"ksc_model_baseline.py").read_text(), "ksc_model_baseline.py", "exec"))
spec = importlib.util.spec_from_file_location("pinned_wrapped", SOURCE/"bayesfilter/wrapped_gaussian_tf.py")
wrapped_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapped_module)
wrapped = wrapped_module.wrapped_gaussian_batch_kernel(dtype=DTYPE)
polar_path = SOURCE/"bayesfilter/highdim/ledh_range_bearing_target_tf.py"
tree = ast.parse(polar_path.read_text())
polar_functions = [x for x in tree.body if isinstance(x, ast.FunctionDef) and
                   x.name in {"_polar_geometry", "_checked", "polar_observation", "polar_observation_tangent", "polar_jacobian", "polar_jacobian_tangent"}]
exec(compile(ast.Module(body=polar_functions, type_ignores=[]), str(polar_path), "exec"))
fixture = build_range_bearing_fixture_tf()
controls = dict(flow_substeps=10, reset_policy="contract_e", reset_epsilon=2.,
    reset_sinkhorn_steps=8, reset_balance_steps=8, reset_ridge=1e-5,
    correction_steps=2, correction_strength=.2, correction_lm_damping=1e-3,
    correction_lm_scale_floor=1e-6, correction_trust_radius=.1,
    pairwise_steps=2, pairwise_strength=.02, pairwise_rms_cap=2.,
    coordinate_cap=16., coordinate_cap_power=8)


def model(kind, theta, direction):
    if kind == "ksc":
        result, set_direction = ksc_sv_canonical_model(theta)
        set_direction(direction)
        return result
    if kind == "lg3":
        a = tf.constant([[.6,.1,0.],[.1,.6,.1],[0.,.1,.6]], DTYPE)
        eye = tf.eye(3, dtype=DTYPE)
        q = tf.exp(2*theta[1])*eye
        return NonlinearScoreModel(
            transition_mean_fn=lambda t,x: t[0]*(x@tf.transpose(a)),
            transition_mean_tangent_fn=lambda t,x,dx: (direction[0]*x+t[0]*dx)@tf.transpose(a),
            observation_fn=lambda x:x,
            observation_tangent_fn=lambda x,dx:dx,
            observation_jacobian_fn=lambda x:tf.broadcast_to(eye,[x.shape[0],3,3]),
            process_covariance=q, observation_covariance=.5*eye,
            process_covariance_tangent_fn=lambda t:2*direction[1]*q)
    a, q, r = fixture.A, fixture.Q*tf.exp(2*theta[0]), fixture.R*tf.exp(2*theta[1])
    def density(t,x,y):
        rr = tf.broadcast_to(fixture.R*tf.exp(2*t[1]),[x.shape[0],2,2])
        res = y-polar_observation(x)
        return wrapped(res,rr,tf.zeros_like(res),tf.zeros_like(rr))["value"]
    def density_tangent(t,x,y,dx):
        rr = tf.broadcast_to(fixture.R*tf.exp(2*t[1]),[x.shape[0],2,2])
        return wrapped(y-polar_observation(x),rr,-polar_observation_tangent(x,dx),
                       2*direction[1]*rr)["tangent"]
    return NonlinearScoreModel(
        transition_mean_fn=lambda t,x:x@tf.transpose(a),
        transition_mean_tangent_fn=lambda t,x,dx:dx@tf.transpose(a),
        observation_fn=polar_observation, observation_tangent_fn=polar_observation_tangent,
        observation_jacobian_fn=polar_jacobian, observation_jacobian_tangent_fn=polar_jacobian_tangent,
        process_covariance=q, process_covariance_tangent_fn=lambda t:2*direction[0]*q,
        observation_covariance=r, observation_covariance_tangent_fn=lambda t:2*direction[1]*r,
        observation_log_density_fn=density, observation_log_density_tangent_fn=density_tangent)


started = time.monotonic()
result = dict(checkout=args.checkout, cpu_only=True, gpus_intentionally_hidden=True,
              dtype="float64", jit_compile=True, seed=20261006, particles=args.particles,
              controls=controls, tensorflow=tf.__version__, rows=[])
for kind in args.models:
    d = dict(lg3=3, ksc=1, m13=4)[kind]
    points = dict(lg3=[[1.,0.],[.8,-.5]], ksc=[[0.,-0.7],[.8,-1.2]],
                  m13=[[0.,0.],[-.73272085,-.01465508]])[kind]
    for horizon in args.horizons:
        n = args.particles
        z = tf.random.stateless_normal([horizon+2,n,d], [20261006,d*1000+horizon], dtype=DTYPE)
        base, noise, reset = z[0], z[1:-1], z[-1]
        reset -= tf.reduce_mean(reset, axis=0)
        reset = tf.transpose(tf.linalg.triangular_solve(tf.linalg.cholesky(tf.transpose(reset)@reset/n),tf.transpose(reset)))
        if kind == "m13":
            base = fixture.m0+base@tf.transpose(tf.linalg.cholesky(fixture.P0))
            cov = tf.broadcast_to(fixture.P0,[n,d,d])
            observed = fixture.observations[:horizon]
        else:
            cov = tf.broadcast_to(tf.eye(d,dtype=DTYPE),[n,d,d])
            observed = tf.random.stateless_normal([horizon,d],[20261006,4000+d*100+horizon],dtype=DTYPE)
            if kind == "ksc": observed -= 2.
        input_hash = hashlib.sha256(b"".join(bytes(tf.io.serialize_tensor(x).numpy()) for x in
            [base,cov,noise,reset,observed])).hexdigest()
        for policy in args.policies:
            options = dict(controls, reset_design=reset)
            if policy != "default": options["importance_weight_policy"] = policy
            @tf.function(input_signature=[tf.TensorSpec([2],DTYPE)], jit_compile=True, autograph=False)
            def run(theta):
                def direction_run(direction):
                    value, score = canonical_value_and_analytical_score(model(kind,theta,direction),theta,
                        base,cov,noise,observed,with_score=True,**options)
                    return value, score[0]
                v,s = tf.map_fn(direction_run,tf.eye(2,dtype=DTYPE),
                                fn_output_signature=(tf.TensorSpec([],DTYPE),tf.TensorSpec([],DTYPE)),parallel_iterations=1)
                return v[0],s
            for point in points:
                tic=time.monotonic()
                value,score=run(tf.constant(point,DTYPE))
                row=dict(model=kind,horizon=horizon,policy=policy,theta=point,input_sha256=input_hash,
                         value=float(value),score=score.numpy().tolist(),wall_seconds=time.monotonic()-tic)
                row["finite"]=bool(tf.math.is_finite(value)&tf.reduce_all(tf.math.is_finite(score)))
                result["rows"].append(row)
                result["wall_seconds"]=time.monotonic()-started
                Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
                print(json.dumps(row),flush=True)
result["source_hashes"]={str(p.relative_to(args.checkout)):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in Path(args.checkout,"bayesfilter/highdim").glob("ledh_*.py")}
Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
