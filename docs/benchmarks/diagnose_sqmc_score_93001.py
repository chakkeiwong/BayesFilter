"""CPU-only diagnostic of the saved SQMC score discrepancy, not an admitted route."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import tensorflow as tf
from bayesfilter.highdim import sqmc_campaign_tf as campaign
from bayesfilter.highdim.ledh_canonical_score_tf import canonical_value_and_analytical_score
from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec

DTYPE = tf.float64
SPEC = LGSSMSpec("p44", 3)
SAVED = ROOT / "docs/plans/artifacts/sqmc-repair-20260925/06-tuning-workflow"
CACHE = {}

def plain(value):
    if tf.is_tensor(value):
        return value.numpy().tolist()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value

def dump(path, value):
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")

def trace_kernel(route, controls, n):
    key = route, json.dumps(controls, sort_keys=True), n
    if key in CACHE:
        return CACHE[key]
    settings = campaign.numerical_settings(controls)
    settings.update(campaign.route_settings(route))
    design = campaign.reset_design(n, 3, DTYPE)
    signature = [tf.TensorSpec([4], DTYPE), tf.TensorSpec([n, 3], DTYPE),
                 tf.TensorSpec([2, n, 3], DTYPE), tf.TensorSpec([2, n], DTYPE),
                 tf.TensorSpec([2, 3], DTYPE)]
    @tf.function(input_signature=signature, jit_compile=False, autograph=False)
    def compute(theta, initial, noise, uniforms, observations):
        direction = tf.constant([1., 0., 0., 0.], DTYPE)
        model, _ = SPEC.model(theta, direction)
        states, covs, ds, dc = SPEC.initial_cloud(theta, initial, direction)
        return canonical_value_and_analytical_score(
            model, theta, states, covs, noise, observations, with_score=True,
            return_trace=True, initial_state_tangent=ds, initial_covariance_tangent=dc,
            reset_design=design, process_ancestor_uniforms=uniforms, **settings)
    CACHE[key] = compute
    return compute

def summarize(theta, inputs, observations, result):
    value, score, trace = result
    physical = SPEC.parts(theta, tf.constant([1., 0., 0., 0.], DTYPE))
    initial_states = SPEC.initial_cloud(theta, inputs[0])[0]
    rows = []
    previous_states = initial_states
    for t, record in enumerate(trace):
        indices = record["ancestor_indices"]
        parents = tf.gather(previous_states, indices)
        noise = inputs[1][t]
        centered_parents = parents - tf.reduce_mean(parents, axis=0)
        centered_noise = noise - tf.reduce_mean(noise, axis=0)
        cross = tf.matmul(centered_parents, centered_noise, transpose_a=True) / tf.cast(tf.shape(parents)[0], DTYPE)
        weights = record["posterior_weights"]
        row = {
            "time": t + 1, "ancestor_indices": indices,
            "unique_ancestors": tf.size(tf.unique(indices).y),
            "hilbert_ties": record["hilbert_tie_count"],
            "state_map_saturation": record["state_map_saturation_rate"],
            "value_increment": tf.reduce_logsumexp(record["posterior_logits"]),
            "score_increment": tf.reduce_sum(weights * record["d_posterior_logits"]),
            "score_prior_proposal_part": tf.reduce_sum(weights * record["d_prior_observation_logits"]),
            "score_observation_part": tf.reduce_sum(weights * record["d_observation_log_density"]),
            "ess": 1. / tf.reduce_sum(tf.square(weights)),
            "parent_mean": tf.reduce_mean(parents, axis=0),
            "process_normal_mean": tf.reduce_mean(noise, axis=0),
            "parent_process_cross_covariance": cross,
            "pre_flow_mean": tf.reduce_mean(record["pre_flow"], axis=0),
            "post_reset_mean": tf.reduce_mean(record["states_after_reset"], axis=0),
        }
        if t == 0:
            variance = physical["q"] + physical["r"]
            residual = observations[0] - parents * physical["phi"]
            logits = -.5 * tf.reduce_sum(tf.math.log(tf.constant(2.*3.141592653589793, DTYPE)*variance)
                                         + tf.square(residual)/variance, axis=1)
            tangent = tf.reduce_sum(residual * parents * physical["dphi"] / variance, axis=1)
            row["process_observation_integrated_value"] = tf.reduce_logsumexp(logits) - tf.math.log(tf.cast(tf.shape(parents)[0], DTYPE))
            row["process_observation_integrated_score"] = tf.reduce_sum(tf.nn.softmax(logits) * tangent)
        rows.append(row)
        previous_states = record["states_after_reset"]
    return plain({"value": value, "score1": score[0], "steps": rows})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    theta = SPEC.default_theta()
    summary = {"program": "saved_repair_cpu_diagnostic", "tuning": "replay saved scope-specific tiny diagnostic controls",
               "promotion": False, "cpu_only": "CUDA_VISIBLE_DEVICES=-1; GPUs intentionally hidden",
               "jit_compile": False, "dtype": "float64", "theta": plain(theta), "cases": []}
    saved = {r: json.loads((SAVED/r/"result.json").read_text()) for r in campaign.ROUTES}
    for seed in [93001, 93002]:
        obs = SPEC.simulate(theta, 2, seed, jit_compile=False)
        oracle_value, oracle_score = SPEC.reference_value_and_score(theta, obs)
        v1, s1 = SPEC.reference_value_and_score(theta, obs[:1])
        oracle = plain({"value": oracle_value, "score": oracle_score, "score1_increments": [s1[0], oracle_score[0]-s1[0]],
                        "value_increments": [v1, oracle_value-v1]})
        for route in campaign.ROUTES:
            controls = saved[route]["selected_controls"]
            inputs = campaign.random_inputs(route, seed, 12, 3, 2, DTYPE)
            compute = trace_kernel(route, controls, 12)
            baseline = compute(theta, *inputs, obs)
            case = dict(seed=seed, route=route, controls=controls, oracle=oracle,
                        **summarize(theta, inputs, obs, baseline))
            expected = next(row for row in saved[route]["untouched_results"] if row["seed"] == seed)
            case["replay_value_abs_difference"] = abs(case["value"]-expected["value"])
            case["replay_score1_abs_difference"] = abs(case["score1"]-expected["score"][0])
            assert max(case["replay_value_abs_difference"], case["replay_score1_abs_difference"]) < 1e-9, case
            dump(out/f"{seed}-{route}-trace.json",
                 {"theta": theta, "observations": obs, "inputs": inputs, "trace": baseline[2]})
            case["finite_differences"] = []
            if seed == 93001:
                for h in [1e-4, 1e-5]:
                    delta = tf.constant([h, 0., 0., 0.], DTYPE)
                    plus = compute(theta+delta, *inputs, obs)
                    minus = compute(theta-delta, *inputs, obs)
                    fd = float(((plus[0]-minus[0])/(2*h)).numpy())
                    stable = all(bool(tf.reduce_all(b["ancestor_indices"] == q["ancestor_indices"]).numpy())
                                 for b, p, m in zip(baseline[2], plus[2], minus[2]) for q in (p,m))
                    error = abs(fd-case["score1"])
                    case["finite_differences"].append(dict(h=h, score1=fd, absolute_difference=error,
                        ancestry_unchanged=stable, parity_pass=stable and error < 2e-6*(1+abs(case["score1"]))))
                    dump(out/f"{seed}-{route}-fd-{h}.json",
                         {"plus": {"value": plus[0], "score": plus[1], "trace": plus[2]},
                          "minus": {"value": minus[0], "score": minus[1], "trace": minus[2]}})
            summary["cases"].append(case)
            dump(out/"results.json", summary)
            print(json.dumps({"seed": seed, "route": route, "score1": case["score1"],
                              "increments": [r["score_increment"] for r in case["steps"]],
                              "conditional_t1": case["steps"][0]["process_observation_integrated_score"],
                              "finite_differences": case["finite_differences"]}), flush=True)
    summary["wall_seconds"] = time.perf_counter()-start
    dump(out/"results.json", summary)
    sources = ["bayesfilter/highdim/sqmc_campaign_tf.py", "bayesfilter/highdim/sqmc_lgssm_tf.py",
               "bayesfilter/highdim/sqmc_tf.py", "bayesfilter/highdim/ledh_canonical_score_tf.py",
               "bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py", "docs/benchmarks/diagnose_sqmc_score_93001.py"]
    dump(out/"manifest.json", {"git_commit": subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
         "command": [sys.executable, *sys.argv], "environment": sys.executable, "tensorflow": tf.__version__,
         "cpu_gpu_status": summary["cpu_only"], "jit_compile": False, "seeds": [93001,93002],
         "data_version": "saved P44 predict-first stateless LGSSM v2; observations preserved in trace files",
         "wall_seconds": summary["wall_seconds"], "output": str(out),
         "plan": "docs/plans/sqmc-score-93001-diagnostic-20260926.md",
         "result": str(out/"results.json"), "source_sha256": {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}})
    print(json.dumps({"complete": True, "wall_seconds":summary["wall_seconds"]}), flush=True)

if __name__ == "__main__":
    main()
