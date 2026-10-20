"""Independent CPU-only calibration/reference for the bounded Lorenz covariance repair.

NumPy/SciPy are confined to this independent reference. No runtime consumer
imports this module.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import json
from pathlib import Path
import time
import numpy as np
import tensorflow as tf
import reference_lorenz_stable_fit as ref
import validate_lorenz_covariance_reference as fullref

FLOORS = [float(np.sqrt(np.finfo(float).eps)), 1e-6, 1e-4, 1e-3]


def load(folder, key):
    return tf.io.parse_tensor(tf.io.read_file(str(folder / (key + ".tensor"))), tf.float64).numpy()


def project(case, rho, activation_margin=None):
    fitted = {k: np.asarray(v).copy() for k, v in case["fit"].items()}
    raw = np.asarray(case["base_precision"]) + 2 * (np.asarray(case["old_A"]) + fitted["A"])
    values, vectors = np.linalg.eigh(raw)
    margin = values[0] / max(abs(values))
    unsafe = bool(margin < rho if activation_margin is None else margin <= activation_margin)
    if unsafe:
        if np.any(values == 0):
            raise ValueError("Singular raw precision")
        inverse = 1 / values
        precision = (vectors * (1 / np.maximum(inverse, rho * max(abs(inverse))))) @ vectors.T
        precision = .5 * (precision + precision.T)
        added = .5 * (precision - raw)
        fitted["A"] += added
        mu, scale = fitted["center"], fitted["scale"]
        fitted["centered_A"] += added * scale[:, None] * scale[None, :]
        fitted["centered_b"] += 2 * scale * (mu @ added)
        fitted["centered_c"] += mu @ added @ mu
    old_mean, old_cov, _, _ = ref.twist(
        np.asarray(case["mean"]), np.asarray(case["base_covariance"]),
        np.asarray(case["old_A"]), np.asarray(case["old_b"]), case["old_c"])
    zmean = (old_mean - fitted["center"]) / fitted["scale"]
    zcov = old_cov / (fitted["scale"][:, None] * fitted["scale"][None, :])
    integral = ref.twist(zmean, zcov, fitted["centered_A"],
                         fitted["centered_b"], fitted["centered_c"])[2]
    precision = np.asarray(case["base_precision"]) + 2 * (np.asarray(case["old_A"]) + fitted["A"])
    eig = np.linalg.eigvalsh(precision)
    return dict(fit={k: v.tolist() for k, v in fitted.items()},
                integral=integral.tolist(), protected=unsafe,
                margin=float(eig[0]/max(abs(eig))), max_A=float(np.max(abs(fitted["A"]))))


def prepare(args):
    fixture = json.loads((args.reference / "fixture.json").read_text())["metadata"]
    history = {k: load(args.design / "rejected_batch", k)
               for k in ("A", "b", "c", "x", "log_incremental_weight")}
    covariance = np.asarray(fixture["transition_covariance"])
    prior = np.asarray(fixture["initial_covariance"])
    cases, seen, singular_ratios = [], [], []
    # Fixed calibration only: two valid refits and the first invalid third refit.
    for stage in (1, 2, 3):
        old = [history[k][stage-1, :, 0] for k in ("A", "b", "c")]
        x = history["x"][stage-1, :, 0]
        weights = history["log_incremental_weight"][stage-1, :, 0]
        increment = None
        for t in reversed(range(len(x))):
            correction = 0.
            if t < len(x)-1:
                old_mean, old_cov, _, _ = ref.twist(ref.rk4(x[t], 2.5), covariance,
                                                   *(v[t+1] for v in old))
                zm = (old_mean-increment["center"])/increment["scale"]
                zc = old_cov/(increment["scale"][:, None]*increment["scale"][None, :])
                correction = ref.twist(zm, zc, increment["A"], increment["b"], increment["c"])[2]
            a, b, c, _, ratio, increment = ref.regress(x[t], -weights[t]-correction)
            singular_ratios.append(ratio)
            base = prior if t == 0 else covariance
            precision = np.linalg.inv(base)
            eig = np.linalg.eigvalsh(precision+2*(old[0][t]+a))
            invalid = bool(eig[0] <= 0)
            if (stage < 3 and t in (100, 99, 50, 0)) or invalid:
                fit = dict(A=a, b=b, c=np.asarray(c), center=increment["center"],
                           scale=increment["scale"], centered_A=increment["A"],
                           centered_b=increment["b"], centered_c=np.asarray(increment["c"]))
                case = dict(label="refit%d_time%d" % (stage, t), stage=stage, time=t,
                            invalid=invalid, raw_margin=float(eig[0]/max(abs(eig))),
                            base_covariance=base.tolist(), base_precision=precision.tolist(),
                            old_A=old[0][t].tolist(), old_b=old[1][t].tolist(), old_c=float(old[2][t]),
                            x=x[t, :16].tolist(),
                            mean=(np.zeros((16, 8)) if t == 0 else ref.rk4(x[t-1, :16], 2.5)).tolist(),
                            fit={k: v.tolist() for k, v in fit.items()})
                case["expected"] = [dict(floor=rho, **project(case, rho)) for rho in FLOORS]
                cases.append(case)
            if invalid:
                seen.append([stage, t])
                break
    if seen != [[3, 12]] or len(cases) != 9:
        raise ValueError("Frozen failure/healthy fixtures differ from the audited source")
    payload = dict(floors=FLOORS, cases=cases)
    (args.output/"calibration.json").write_text(json.dumps(payload, allow_nan=False)+"\n")
    return dict(status="PASS", case_count=len(cases), first_invalid=seen,
                minimum_singular_ratio=min(singular_ratios),
                checks=[dict(name="frozen_first_invalid_update", passed=True),
                        dict(name="calibration_design_full_rank",
                             passed=bool(min(singular_ratios)>np.sqrt(64*np.finfo(float).eps)))])


def integral_reference_one(args,rho):
    import mpmath as mp
    mp.mp.dps=80
    payload=json.loads((args.calibration/"calibration.json").read_text())
    rows=[]
    for case in payload["cases"]:
        expected=dict(floor=rho,**project(case,rho,getattr(args,"activation_margin",None)))
        matrix=lambda value: mp.matrix(value)
        p=matrix(case["base_precision"])
        old_precision=p+2*matrix(case["old_A"])
        old_cov=old_precision**-1
        covariance=(old_precision+2*matrix(expected["fit"]["A"]))**-1
        integrals=[]
        for mean in case["mean"]:
            old_mean=old_cov*(p*matrix(mean)-matrix(case["old_b"]))
            shift=old_precision*old_mean-matrix(expected["fit"]["b"])
            value=(mp.log(mp.det(covariance)/mp.det(old_cov))/2
                   -mp.mpf(expected["fit"]["c"])
                   +(shift.T*covariance*shift)[0]/2
                   -(old_mean.T*old_precision*old_mean)[0]/2)
            integrals.append(float(value))
        difference=np.abs(np.asarray(expected["integral"])-integrals)
        fitted={k:np.asarray(v) for k,v in case["fit"].items()}
        added=np.asarray(expected["fit"]["A"])-fitted["A"]
        old_mean,old_covariance,_,_=ref.twist(np.asarray(case["mean"]),
            np.asarray(case["base_covariance"]),np.asarray(case["old_A"]),
            np.asarray(case["old_b"]),case["old_c"])
        increment=dict(center=fitted["center"],scale=fitted["scale"],
            unprotected_A=fitted["centered_A"],unprotected_b=fitted["centered_b"],
            unprotected_c=fitted["centered_c"],projection_added_A=added)
        stable=fullref.projected_integral(old_mean,old_covariance,increment)
        scaled=np.abs(stable-integrals)/(1e-7+1e-9*np.abs(integrals))
        rows.append(dict(label=case["label"],passed=bool(np.all(scaled<=1)),
                         old_formula_max_abs=float(np.max(difference)),
                         stable_max_scaled=float(np.max(scaled))))
        expected["integral"]=integrals
        case["expected"]=[expected]
    payload.update(floors=[rho],activation_margin=getattr(args,"activation_margin",None),oracle="mpmath80_raw_gaussian_completing_square")
    (args.output/"calibration.json").write_text(json.dumps(payload,allow_nan=False)+"\n")
    return dict(status="PASS" if all(row["passed"] for row in rows) else "FAIL",
                reference_precision_decimal=80,stable_integral=True,checks=rows)

def integral_reference(args):
    if not args.condition_grid:
        return integral_reference_one(args,args.repair_floor)
    outputs=[];checks=[]
    for rho in (0.01,0.1):
        folder=args.output/("floor_"+str(rho));folder.mkdir()
        options=argparse.Namespace(**vars(args));options.output=folder
        result=integral_reference_one(options,rho)
        outputs.append(json.loads((folder/"calibration.json").read_text()))
        checks.extend(dict(floor=rho,**row) for row in result["checks"])
    payload=outputs[0]
    payload["floors"]=[0.01,0.1]
    for left,right in zip(payload["cases"],outputs[1]["cases"]):
        left["expected"]+=right["expected"]
    (args.output/"calibration.json").write_text(json.dumps(payload,allow_nan=False)+"\n")
    return dict(status="PASS" if all(row["passed"] for row in checks) else "FAIL",
                reference_precision_decimal=80,stable_integral=True,checks=checks)

def validate(args):
    source = json.loads((args.candidate/"summary.json").read_text())
    outcomes = []
    for index, candidate in enumerate(source["candidates"]):
        if not candidate["eligible"]:
            continue
        output = args.output / candidate["folder"]
        output.mkdir()
        argv = ["--output", str(output), "--reference", str(args.reference),
                "--candidate", str(args.candidate), "--design", str(args.design),
                "--relative-floor", str(candidate["floor"]), "--candidate-index", str(index),
                "--candidate-folder", candidate["folder"]]
        if source.get("stable_integral",False):argv+=["--stable-integral"]
        if source.get("activation_margin") is not None:argv+=["--activation-margin",str(source["activation_margin"])]
        try:
            fullref.main(argv)
        except SystemExit:
            pass
        decision = json.loads((output/"summary.json").read_text())
        outcomes.append(dict(floor=candidate["floor"], folder=candidate["folder"],
                             status=decision["status"], error=decision.get("error"),
                             checks=len(decision["checks"]),
                             failures=[row for row in decision["checks"] if not row["passed"]]))
    passing = [row["floor"] for row in outcomes if row["status"] == "PASS"]
    return dict(status="PASS", activation_margin=source.get("activation_margin"), validation_checks=sum(row["checks"] for row in outcomes), candidates=outcomes, validated_floors=passing,
                frozen_floor=min(passing) if passing else None,
                verdict="NUMERICALLY_VALIDATED_OPTIONAL_REPAIR" if passing else "NO_VALIDATED_REPAIR",
                checks=[dict(name="every_safety_survivor_checked",
                             passed=len(outcomes)==sum(row["eligible"] for row in source["candidates"]))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("output", "reference", "design"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--mode", choices=("reference", "validation", "integral_reference"), required=True)
    parser.add_argument("--condition-grid",action="store_true")
    parser.add_argument("--activation-margin",type=float)
    parser.add_argument("--repair-floor",type=float,default=0.001)
    parser.add_argument("--calibration",type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    summary = dict(status="FAIL", device="CPU", gpu_intentionally_hidden=True,
                   diagnostic_numpy_reference=True, activation_margin=args.activation_margin, replication="INCOMPLETE",
                   heuristic_dominance_verdict="NOT_ASSESSED_NUMERICAL_REPAIR_ONLY")
    try:
        summary.update(integral_reference(args) if args.mode=="integral_reference" else prepare(args) if args.mode == "reference" else validate(args))
        if not all(row["passed"] for row in summary["checks"]):
            summary["status"] = "FAIL"
    except Exception as error:
        summary["error"] = repr(error)
    summary["wall_seconds"] = time.monotonic()-start
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: summary.get(k) for k in
                     ("status", "case_count", "verdict", "validated_floors", "frozen_floor", "error", "wall_seconds")}))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
