"""CPU-only independent-reference diagnostic; never a tuning or runtime path."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

PLAN = "docs/plans/bayesfilter-modern-hmc-c3-reference-continuation-2026-10-29.md"


def reference_seeds(local=False):
    if not local:
        return dict(parity=202610296,pricing=202610297,calibration=[202610290,202620290],
            reference_rung_bases={str(p):303610290+1000*p for p in (12,14,16)})
    seeds=dict(pricing=410000001,calibration=[420000000,420010000],
        reference_rung_bases={str(p):430000000+10000*p for p in (14,16,18,20,22)})
    inventory=[seeds['pricing']]
    for base in seeds['calibration']: inventory.extend(range(base,base+8))
    for base in seeds['reference_rung_bases'].values(): inventory.extend(range(base,base+64))
    if len(inventory)!=len(set(inventory)): raise ValueError('reference seed overlap')
    return seeds


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + chr(10))
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reference(path):
    spec = importlib.util.spec_from_file_location("independent_macrofinance_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_sources(campaign, reference):
    launch = read(campaign / "launch.json")
    expected = dict(launch["source_sha256"])
    expected.update({str(reference): digest(reference), str(Path(__file__).resolve()): digest(__file__)})
    helpers = Path(__file__).resolve().parent / "reference_helpers"
    if helpers.is_dir():
        expected.update({str(path): digest(path) for path in helpers.glob("*.py")})
    return expected


def verify_sources(sources):
    for path, checksum in sources.items():
        if digest(path) != checksum:
            raise ValueError("source or input drift: " + path)


def parity(args, reference, launch, inputs, output, progress):
    import numpy as np
    sys.path[:0] = [launch["consumer"], launch["candidate"]]
    from one_country_affine_macro_source_binding import Phase9SourceBinding
    import one_country_affine_macro_phase9_hmc as phase9
    binding = Phase9SourceBinding(Path(launch["candidate"]), launch["commit"])
    adapter = phase9.build_generic_target(inputs["observations"],
        dataset_id=inputs["design"]["dataset_id"], source_binding=binding,
        execution_device="CPU")
    import tensorflow as tf
    center = np.asarray(inputs["initial_position"])
    factor = np.linalg.cholesky(inputs["initial_covariance"])
    offsets = np.array([-.15,-.05,.05,.15])[:,None]*np.ones((1,9))
    points = np.concatenate((center[None], center+offsets@factor.T,
        np.zeros((1,9)), np.random.default_rng(202610296).normal(0,.35,(4,9)),
        center[None] + .10*np.concatenate((np.eye(9),-np.eye(9)))@factor.T))
    assert points.shape == (28,9)
    evaluate = tf.function(adapter.log_prob_and_grad,
        input_signature=[tf.TensorSpec([4,9],tf.float64)], autograph=False,jit_compile=True)
    values, scores = [], []
    progress("public_cpu_parity", points=len(points))
    for first in range(0,len(points),4):
        a,b = evaluate(tf.constant(points[first:first+4],tf.float64))
        values.extend(a.numpy().tolist()); scores.extend(b.numpy().tolist())
    actual = np.asarray(values); score=np.asarray(scores)
    expected = reference.log_density(points, inputs["observations"])
    np.testing.assert_allclose(actual,expected,atol=1.e-7,rtol=1.e-8)
    gradient_errors=[]
    for step in (1.e-5,5.e-6):
        directions=step*np.eye(9)
        plus=(points[:,None,:]+directions[None]).reshape(-1,9)
        minus=(points[:,None,:]-directions[None]).reshape(-1,9)
        gradient=(reference.log_density(plus,inputs["observations"])-reference.log_density(minus,inputs["observations"])).reshape(28,9)/(2*step)
        np.testing.assert_allclose(score,gradient,atol=1.e-7,rtol=1.e-8)
        gradient_errors.append(dict(step=step,max_absolute_error=float(np.max(np.abs(score-gradient)))))
    # Also compare the separately coded SciPy chart and four physical targets.
    f,q,r,p=reference.decode_chart(points)
    chart_errors=[]
    for i,point in enumerate(points):
        oracle=phase9.decode_chart(point)
        chart_errors.append(max(float(np.max(np.abs(a-b))) for a,b in zip(
            (f[i],q[i],r[i],p[i]),(oracle.transition,oracle.innovation_covariance,
                oracle.measurement_covariance,oracle.stationary_covariance))))
        np.testing.assert_allclose(reference.quantities_standardized(point[None]/.35)[0,-4:],
            list(phase9.four_targets(oracle.transition,oracle.stationary_covariance).values()),atol=1.e-10,rtol=1.e-10)
    np.testing.assert_allclose(chart_errors,0,atol=1.e-9,rtol=0)
    binding.audit();binding.check_imports()
    write(output/"parity-points.json",dict(points=points.tolist(),public_values=values,
        reference_values=expected.tolist(),public_scores=scores))
    return dict(passed=True,points=28,max_absolute_value_error=float(np.max(np.abs(actual-expected))),
        finite_difference_scores=gradient_errors,max_absolute_chart_error=max(chart_errors),
        four_physical_quantities_match=True,cpu_reference_only=True,
        target_signature=adapter.target_signature,adapter_signature=adapter.adapter_signature(),
        independent_target="scipy_expm_direct_lyapunov_covariance_kalman",
        public_target="analytic_factor_qr",jit_compile=True)


def reference_envelope(rungs):
    import numpy as np
    if len(rungs)<3:
        return dict(passed=False,status="two_refinements_required")
    latest=rungs[-1]; mean=np.asarray(latest["mean"]); sd=np.asarray(latest["sd"])
    if not np.isfinite(sd).all() or np.any(sd<=0):
        raise ValueError("reference quantity scale unavailable")
    shifts=[np.abs(np.asarray(rungs[i]["mean"])-np.asarray(rungs[i-1]["mean"])) for i in range(len(rungs)-2,len(rungs))]
    envelope=np.maximum(*shifts)+3*np.asarray(latest["randomization_mcse"])
    ratio=envelope/sd
    return dict(passed=bool(np.all(ratio<=.02)),status="reference_screen_passed" if np.all(ratio<=.02) else "reference_limited",
        uncertainty=envelope.tolist(),normalized_uncertainty=ratio.tolist(),
        max_normalized_uncertainty=float(ratio.max()),tolerance=.02,
        uncertainty_role="empirical_refinement_plus_randomization_mcse_not_rigorous_error_bound")


def integrate(args, reference, launch, inputs, output, progress, deadline):
    import numpy as np
    sys.path.insert(0,launch["candidate"])
    # Diagnostic helpers are a separately frozen package. The clean sampler
    # checkout intentionally does not include every reporting/reference module.
    helpers = Path(__file__).resolve().parent / "reference_helpers"
    if not helpers.is_dir():
        raise ValueError("frozen reference_helpers package required")
    sys.path.insert(0, str(helpers.parent))
    from reference_helpers import gaussian_mixture_reference as mixture
    local = args.proposal_method == 'independent-local'
    seeds = reference_seeds(local)
    from scipy.stats import qmc
    from scipy.special import ndtri
    parity_result=read(args.parity_result)
    if parity_result.get("status")!="completed" or not parity_result.get("checks",{}).get("passed"):
        raise ValueError("successful exact-source public parity required")
    parity_manifest=read(args.parity_result.parent/"manifest.json")
    if parity_manifest["campaign"]!=str(args.campaign.resolve()) or parity_manifest["source_sha256"][str(args.reference.resolve())]!=digest(args.reference):
        raise ValueError("parity source or data scope mismatch")
    def check():
        if time.monotonic()>=deadline-60:
            raise TimeoutError("independent reference allocation exhausted")
    def target(z):
        check()
        return reference.log_posterior_standardized(z,inputs["observations"])
    center=np.asarray(inputs["initial_position"])/reference.PRIOR_SCALE
    factor=np.linalg.cholesky(inputs["initial_covariance"])/reference.PRIOR_SCALE
    # Numerical workload price is a full-horizon batch, not a shorter target.
    price_points=ndtri(qmc.Sobol(9,scramble=True,seed=seeds['pricing']).random_base2(10))
    start=time.monotonic();target(price_points); price=time.monotonic()-start
    projected=(2*2**15+8*sum(2**p for p in (12,14,16)))*price/len(price_points)
    write(output/"price.json",dict(batch_points=1024,seconds=price,
        projected_likelihood_seconds=projected,role="operational estimate excludes proposal fitting and contention"))
    progress("calibration",projected_likelihood_seconds=projected)
    def on_round(row):
        check();write(output/("calibration-%02d.json"%row["round"]),row)
        progress("calibration_complete",round=row["round"],weight_ess=row["weight_ess"])
    if local:
        from reference_helpers.macrofinance_proposal import local_proposal, preserve_local_coverage
        def on_start(row):
            check();write(output/('optimizer-%02d.json'%row['index']),row)
            progress('independent_optimization',start=row['index'],max_gradient=row['max_gradient'])
        proposal, guards, search=local_proposal(target,center,on_start=on_start)
        write(output/'mode-search.json',search);write(output/'initial-proposal.json',proposal)
        calibration=[]
        for iteration,seed in enumerate(seeds['calibration']):
            q,weights,row=mixture.weighted_draws(target,proposal,power=16,seed=seed)
            proposal=preserve_local_coverage(mixture.fit_weighted_proposal(q,weights),guards)
            mixture.validate_proposal(proposal)
            row.update(round=iteration,proposal=proposal);calibration.append(row);on_round(row)
        del q,weights
    else:
        proposal, calibration=mixture.train_proposal(target,center,factor,seed=seeds['calibration'][0],on_round=on_round)
    write(output/"proposal.json",proposal)
    proposal_hash=digest(output/"proposal.json")
    rungs=[]
    powers=(14,16,18,20,22) if local else (12,14,16)
    for power in powers:
        check()
        if rungs:
            predicted=rungs[-1]['elapsed_seconds']*2**(power-rungs[-1]['power'])
            if time.monotonic()+1.5*predicted+60>=deadline:
                progress('reference_budget_limited',next_power=power,predicted_seconds=predicted)
                break
        # Lower and upper rungs use disjoint scrambled streams.
        seed=seeds['reference_rung_bases'][str(power)]
        def on_replication(row):
            check();write(output/("power-%d-replication-%02d.json"%(power,row["replication"])),row)
            progress("integration",power=power,replication=row["replication"],weight_ess=row["weight_ess"])
        rung_started=time.monotonic()
        rung=mixture.reference_rung(target,reference.quantities_standardized,proposal,
            power=power,seed=seed,replications=8,on_replication=on_replication)
        rung['elapsed_seconds']=time.monotonic()-rung_started
        write(output/("power-%d.json"%power),rung);rungs.append(rung)
        if len(rungs)>=3 and reference_envelope(rungs)['passed']: break
    screen=reference_envelope(rungs)
    if len(rungs)<3:
        raise TimeoutError('allocation cannot support two independent refinements')
    final=dict(target="one-country-affine-phase9-synthetic",dataset_id=inputs["design"]["dataset_id"],
        quantities=list(reference.QUANTITY_NAMES),mean=rungs[-1]["mean"],sd=rungs[-1]["sd"],
        proposal_sha256=proposal_hash,screen=screen,rungs=rungs,
        method="independent_covariance_kalman_defensive_mixture_scrambled_integration",
        prior_coordinate="z=theta/0.35",posterior_qualified=False,
        reference_role="independent diagnostic; cannot establish promotion without eligible TensorFlow assessment",
        exact_posterior=False,rigorous_error_bound=False)
    write(output/"reference.json",final)
    return dict(passed=screen["passed"],reference_status=screen["status"],
        max_normalized_uncertainty=screen["max_normalized_uncertainty"],rung_count=len(rungs),
        quantities=len(reference.QUANTITY_NAMES),proposal_sha256=proposal_hash)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stage",choices=("parity","reference"),required=True)
    parser.add_argument("--wall-seconds",type=float,required=True)
    parser.add_argument("--parity-result",type=Path)
    parser.add_argument('--proposal-method',choices=('inherited','independent-local'),default='inherited')
    args=parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="-1":
        raise ValueError("independent CPU reference must hide GPUs before import")
    if args.stage=="reference" and args.parity_result is None:
        raise ValueError("reference integration requires --parity-result")
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();sources=frozen_sources(args.campaign,args.reference.resolve())
    launch=read(args.campaign/"launch.json");inputs=read(launch["inputs"])
    manifest=dict(plan=PLAN,command=[sys.executable,*sys.argv],campaign=str(args.campaign.resolve()),
        git_commit=launch["commit"],started_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sources,inputs_sha256=digest(launch["inputs"]),wall_seconds=args.wall_seconds,
        environment={k:os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES","TF_FORCE_GPU_ALLOW_GROWTH","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS")},
        gpu_intentionally_hidden=True,evidence_role="independent_reference_only",stage=args.stage,
        proposal_method=args.proposal_method,seeds=reference_seeds(args.proposal_method=='independent-local'))
    write(args.output/"manifest.json",manifest)
    result=dict(status="running",plan=PLAN,stage=args.stage,posterior_qualified=False)
    def progress(stage,**extra):
        write(args.output/"progress.json",dict(stage=stage,elapsed_seconds=time.monotonic()-started,
            updated_utc=datetime.now(timezone.utc).isoformat(),**extra))
    try:
        verify_sources(sources);reference=load_reference(args.reference)
        result["checks"]=(parity(args,reference,launch,inputs,args.output,progress)
            if args.stage=="parity" else integrate(args,reference,launch,inputs,args.output,progress,started+args.wall_seconds))
        verify_sources(sources)
        result.update(status="completed",source_unchanged=True)
    except TimeoutError as exc:
        result.update(status="resource_stopped",error=str(exc))
    except Exception as exc:
        result.update(status="error",error=repr(exc),traceback=traceback.format_exc())
    result.update(elapsed_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
    write(args.output/"result.json",result);progress(result["status"])
    print(json.dumps(result))
    return 0 if result["status"]=="completed" else 1


if __name__=="__main__":
    raise SystemExit(main())
