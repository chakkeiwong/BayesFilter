#!/usr/bin/env python3
"""Freeze LG2 data and independent posterior reference; explicit CPU exception."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["LG2"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    import tensorflow as tf
    from bayesfilter.highdim.ledh_program_fixtures_tf import fixture_seed, linear_fixture_spec
    from bayesfilter.testing.surrogate_hmc_linear_reference import stationary_covariance
    from bayesfilter.testing.surrogate_hmc_linear_posterior_reference import prepare_reference
    from bayesfilter.inference.surrogate_hmc_program import file_hash, write_json
    spec = linear_fixture_spec("LG2", tf.float64)
    transition, process = spec["base_f"], spec["base_q"]
    covariance = stationary_covariance(transition[None], process[None])[0]

    @tf.function(input_signature=[tf.TensorSpec([3,2],tf.int32)], jit_compile=True, autograph=False)
    def generate(seeds):
        initial = tf.linalg.cholesky(covariance) @ tf.random.stateless_normal([2,1],seeds[0],dtype=tf.float64)
        innovations = tf.random.stateless_normal([20,2,1],seeds[1],dtype=tf.float64)
        noise = tf.random.stateless_normal([20,2,1],seeds[2],dtype=tf.float64)
        process_chol, observation_chol = tf.linalg.cholesky(process), tf.linalg.cholesky(spec["r"])
        states = tf.TensorArray(tf.float64,size=20,element_shape=[2])
        observations = tf.TensorArray(tf.float32,size=20,element_shape=[2])
        def step(index, state, states, observations):
            state = transition @ state + process_chol @ innovations[index]
            value = spec["h"] @ state + observation_chol @ noise[index]
            return index+1,state,states.write(index,state[:,0]),observations.write(index,tf.cast(value[:,0],tf.float32))
        _,_,states,observations = tf.while_loop(lambda index,*_: index<20,step,
            (tf.constant(0),initial,states,observations),maximum_iterations=20)
        return initial[:,0],states.stack(),observations.stack()

    partitions = {}
    for partition in ("calibration","validation","claim"):
        directory = args.output / partition
        directory.mkdir()
        seeds = {str(p):fixture_seed("LG2",20,partition,1,0,p) for p in (1,2,3)}
        initial,states,observations = generate(tf.stack(list(seeds.values())))
        tf.debugging.assert_all_finite(observations,"generated observations")
        data = dict(model_id="LG2",attempt=1,replicate=0,partition=partition,horizon=20,
            observations=observations.numpy().tolist(),states=states.numpy().tolist(),initial_state=initial.numpy().tolist(),
            truth=[1.,0.],transition_before_observation=True,initial_law="stationary",
            seeds={key:value.numpy().tolist() for key,value in seeds.items()},
            observation_precision="FP32 rounded once, same values for every comparator",
            generation="CPU TensorFlow/XLA tiny reference-data exception; no training",
            classification="synthetic_fixture")
        write_json(directory/"data.json",data)
        partitions[partition] = dict(data=str(directory/"data.json"),data_sha256=file_hash(directory/"data.json"))
        if partition == "claim":
            report,tensors = prepare_reference(data["observations"])
            report["data_sha256"] = partitions[partition]["data_sha256"]
            report["tensors"] = {}
            for name,value in tensors.items():
                path=directory/(name+".tfbin")
                tf.io.write_file(str(path),tf.io.serialize_tensor(value))
                report["tensors"][name] = dict(path=path.name,sha256=file_hash(path),dtype=value.dtype.name)
            write_json(directory/"reference.json",report)
            partitions[partition].update(reference=str(directory/"reference.json"),qualified=report["qualified"])
    sources = [Path(__file__),Path("bayesfilter/testing/surrogate_hmc_linear_reference.py"),
               Path("bayesfilter/testing/surrogate_hmc_linear_posterior_reference.py"),
               Path("bayesfilter/highdim/ledh_program_fixtures_tf.py")]
    write_json(args.output/"report.json",dict(qualified=report["qualified"],partitions=partitions,
        model_id="LG2",posterior_comparisons="NOT_RUN",heuristic_dominance="NOT_RUN",default_readiness=False,
        inference_attempts_consumed=0))
    write_json(args.output/"run-manifest.json",dict(command=[sys.executable,*sys.argv],
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        environment=sys.executable,tensorflow=tf.__version__,cpu_only=True,gpu_intentionally_hidden=True,
        seeds={name:"partition data.json" for name in partitions},jit_compile=True,
        plan="docs/plans/surrogate-hmc-lg2-inference-preparation-2026-09-20.md",result="report.json",
        source_hashes={str(path):file_hash(path) for path in sources},partitions=partitions,
        wall_seconds=time.monotonic()-started))
    print({"qualified":report["qualified"],"output":str(args.output)},flush=True)
    return 0 if report["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
