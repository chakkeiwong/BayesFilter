#!/usr/bin/env python3
"""Bounded diagnostic of the failed M13 fresh-start trajectory; no tuning authority."""
from pathlib import Path
import argparse
import json
import os
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAN = "docs/plans/surrogate-hmc-m13-force-localization-2026-09-27.md"
PUBLIC = ROOT / "docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-public-tuning-03/m13-public-tuning/tuning"
RWM = ROOT / "docs/plans/artifacts/surrogate-hmc-m13-assessment-rwm-20260926-06/worker/rwm"

def worker(output):
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "2"
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf)
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("GPU required")
    from bayesfilter.testing.surrogate_hmc_m13_public_tuning import selected_components, read_tensor, PRIOR_SD
    from bayesfilter.testing.surrogate_hmc_m13_four_chain_cost import batched_endpoint
    from bayesfilter.testing import surrogate_hmc_m13_endpoint_mechanics as mechanics
    from bayesfilter.inference.surrogate_hmc_equivalence import finite_report
    from bayesfilter.inference.surrogate_hmc_program import write_json, file_hash
    adapter, binding, target, fixture, selection, selection_path = selected_components()
    manifest = json.loads((PUBLIC / "manifest.json").read_text())
    nomination = json.loads((PUBLIC / "nomination/manifest.json").read_text())
    arrays = {k: read_tensor(PUBLIC / "nomination", nomination["tensors"][k])
              for k in ("center", "factor", "step_size", "covariance")}
    starts = tf.constant(manifest["config"]["initial_chain_positions"], tf.float64)
    warm = tf.io.parse_tensor(tf.io.read_file(str(RWM / "metric-warmup/states.tfbin")), tf.float64)
    cloud = tf.reshape(warm[-256:], [-1, 2])
    center = tf.reduce_mean(cloud, 0)
    delta = cloud-center
    covariance = .9*tf.linalg.matmul(delta, delta, transpose_a=True)/tf.cast(tf.shape(cloud)[0]-1, tf.float64)+.1*PRIOR_SD**2*tf.eye(2, dtype=tf.float64)
    proposed_starts = center + tf.linalg.matmul(
        tf.constant([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]], tf.float64),
        tf.linalg.cholesky(covariance), transpose_b=True)
    evaluate = target.compile(4, filter_moments=False, all_correction_diagnostics=False)
    _, evaluator = mechanics.endpoint_components(fixture)
    endpoint = batched_endpoint(evaluator)
    report = dict(plan=PLAN, status="RUNNING", memory_policy=memory, device="gpu",
                  target_manifest=dict(target.manifest), selection_sha256=file_hash(selection_path),
                  public_manifest_sha256=file_hash(PUBLIC / "manifest.json"),
                  tuning_authority=False, posterior_qualified=False, rows=[])
    write_json(output / "report.json", finite_report(report))
    for name, positions in (("original", starts), ("warmup_covariance_starts", proposed_starts)):
        value, score, status = evaluate(tf.cast(positions, tf.float32))
        force = tf.cast(-score + tf.cast(positions, tf.float32)/tf.cast(PRIOR_SD**2, tf.float32), tf.float64)
        displacement = -.5*arrays["step_size"]**2*tf.linalg.matmul(force, arrays["covariance"], transpose_b=True)
        proposed = positions + displacement
        _, proposal_score, proposal_status = evaluate(tf.cast(proposed, tf.float32))
        initial_endpoint, proposed_endpoint = endpoint(positions), endpoint(proposed)
        row = dict(name=name, starts=positions.numpy().tolist(), value=value.numpy().tolist(),
            force=force.numpy().tolist(),
            status={k: v.numpy().tolist() for k,v in status.items() if v.shape.rank <= 1},
            zero_momentum_proposals=proposed.numpy().tolist(),
            proposal_status={k: v.numpy().tolist() for k,v in proposal_status.items() if v.shape.rank <= 1},
            initial_potential=initial_endpoint.potential.numpy().tolist(),
            proposal_potential=proposed_endpoint.potential.numpy().tolist())
        report["rows"].append(row)
        print(json.dumps(finite_report(row)), flush=True)
        write_json(output / "report.json", finite_report(report))
    report.update(status="LOCALIZATION_COMPLETE", metric={k:v.numpy().tolist() for k,v in arrays.items()},
                  cpu_exception=False, xla=True, tf32=tf.config.experimental.tensor_float_32_execution_enabled())
    write_json(output / "report.json", finite_report(report))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.output)
        return 0
    from bayesfilter.inference.surrogate_hmc_program import write_json, run_job, file_hash
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), "--output", str(out/"worker"), "--worker"]
    source = {str(Path(__file__).relative_to(ROOT)): file_hash(Path(__file__)), PLAN: file_hash(ROOT/PLAN)}
    ledger = dict(limits=dict(cpu_worker_seconds=600.,gpu_seconds=600.),
                  charged=dict(cpu_worker_seconds=0.,gpu_seconds=0.),jobs=[])
    before=resource.getrusage(resource.RUSAGE_SELF); start=time.monotonic()
    result = run_job(command, out/"worker", ledger, device="gpu", timeout=590., cpu_allowance=590.)
    after=resource.getrusage(resource.RUSAGE_SELF)
    ledger["charged"]["cpu_worker_seconds"] += after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
    write_json(out/"accounting.json",dict(charged_this_run=ledger["charged"],jobs=ledger["jobs"]))
    write_json(out/"run-manifest.json",dict(plan=PLAN,command=sys.argv,worker_command=command,
        commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        source_hashes=source,source_unchanged=all(file_hash(ROOT/k)==v for k,v in source.items()),
        python=sys.executable,device="gpu",data_seed=1704,random_seeds="N/A: deterministic diagnostic",
        wall_seconds=time.monotonic()-start,result_file=str(out/"worker/report.json")))
    write_json(out/"report.json",result)
    print(json.dumps(dict(status=result["status"],charged=ledger["charged"])),flush=True)
    return 0 if result["status"]=="COMPLETED" else 1

if __name__=="__main__":
    raise SystemExit(main())
