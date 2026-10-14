"""Independent CPU-only local-input numerical diagnostic; no runtime consumer."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse,json,time
from pathlib import Path
import numpy as np
import mpmath as mp
import tensorflow as tf
import reference_lorenz_stable_fit as ref
from validate_lorenz_covariance_reference import projected_integral
def load(folder,key):
    return tf.io.parse_tensor(tf.io.read_file(str(folder/(key+".tensor"))),tf.float64).numpy()
def comparison(actual,expected):
    scaled=np.abs(actual-expected)/(1e-7+1e-9*np.abs(expected))
    return dict(passed=bool(np.all(scaled<=1)),max_scaled=float(np.max(scaled)),
                max_absolute=float(np.max(np.abs(actual-expected))))
def main():
    p=argparse.ArgumentParser()
    for key in ("output","reference","candidate","capture"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--stable-all",action="store_true")
    p.add_argument("--design",type=Path)
    args=p.parse_args();start=time.monotonic()
    summary=dict(status="FAIL",device="CPU",gpu_intentionally_hidden=True,
                 diagnostic_numpy_reference=True,replication="INCOMPLETE",rows=[],high_precision=[])
    try:
        folder=args.candidate/"covariance_floor_0.001"
        history={k:load(folder,k)[3,:,0] for k in ("A","b","c","x","log_incremental_weight")}
        keys=("A","b","c","correction_log_integral","increment_center","increment_scale",
              "increment_centered_A","increment_centered_b","increment_centered_c","projection_added_A")
        captured={k:load(args.capture,k)[:,0] for k in keys}
        fixture=json.loads((args.reference/"fixture.json").read_text())["metadata"]
        covariance=np.asarray(fixture["transition_covariance"])
        for t in reversed(range(len(history["x"]))):
            target=-history["log_incremental_weight"][t]-captured["correction_log_integral"][t]
            a,b,c,rss,ratio,fitted=ref.regress(history["x"][t],target)
            fit_checks={k:comparison(captured["increment_centered_"+k][t],fitted[k]) for k in ("A","b","c")}
            raw_checks={k:comparison(captured[k][t],history[k][t]+value)
                        for k,value in zip(("A","b","c"),(a+captured["projection_added_A"][t],b,c))}
            row=dict(time=t,singular_ratio=ratio,fixed_target_centered_fit=fit_checks,
                     fixed_target_raw_fit=raw_checks)
            if t<len(history["x"])-1:
                j=t+1
                mean=ref.rk4(history["x"][t],2.5)
                old_mean,old_cov,_,_=ref.twist(mean,covariance,*(history[k][j] for k in ("A","b","c")))
                inc=dict(center=captured["increment_center"][j],scale=captured["increment_scale"][j],
                         projection_added_A=captured["projection_added_A"][j],
                         **{"unprotected_"+k:captured["increment_centered_"+k][j] for k in ("A","b","c")})
                if args.stable_all or np.any(inc["projection_added_A"]):
                    correction=projected_integral(old_mean,old_cov,inc)
                else:
                    zmean=(old_mean-inc["center"])/inc["scale"]
                    zcov=old_cov/(inc["scale"][:,None]*inc["scale"][None,:])
                    correction=ref.twist(zmean,zcov,*(inc["unprotected_"+k] for k in ("A","b","c")))[2]
                row["fixed_increment_integral"]=comparison(captured["correction_log_integral"][t],correction)
                if t in (12,11,10):
                    mp.mp.dps=80;m=mp.matrix
                    base=m(covariance.tolist())**-1
                    old_p=base+2*m(history["A"][j].tolist());old_cov_mp=old_p**-1
                    mu=m(inc["center"].tolist());scale=inc["scale"]
                    ac=m([[mp.mpf(inc["unprotected_A"][i,k])/mp.mpf(scale[i])/mp.mpf(scale[k])
                           for k in range(8)] for i in range(8)])
                    linear=m([mp.mpf(inc["unprotected_b"][i])/mp.mpf(scale[i]) for i in range(8)])
                    raw_b=linear-2*ac*mu
                    raw_c=mp.mpf(inc["unprotected_c"])-(linear.T*mu)[0]+(mu.T*ac*mu)[0]
                    new_p=old_p+2*(ac+m(inc["projection_added_A"].tolist()));new_cov=new_p**-1
                    answers=[]
                    for v in mean[:32]:
                        om=old_cov_mp*(base*m(v.tolist())-m(history["b"][j].tolist()))
                        shift=old_p*om-raw_b
                        ans=(mp.log(mp.det(new_cov)/mp.det(old_cov_mp))/2-raw_c
                             +(shift.T*new_cov*shift)[0]/2-(om.T*old_p*om)[0]/2)
                        answers.append(float(ans))
                    exact=np.asarray(answers)
                    summary["high_precision"].append(dict(time=t,decimal_digits=80,points=32,
                        gpu_integral=comparison(captured["correction_log_integral"][t,:32],exact),
                        numpy_integral=comparison(correction[:32],exact)))
            summary["rows"].append(row)
        if args.stable_all:
            import reference_lorenz_bounded_spd as full
            nested=args.output/"full_validation";nested.mkdir()
            arguments=argparse.Namespace(output=nested,reference=args.reference,
                design=args.design,candidate=args.candidate)
            summary["full_validation"]=full.validate(arguments)
            summary["oracle_pass"]=all(v[k]["passed"] for v in summary["high_precision"]
                for k in ("gpu_integral","numpy_integral"))
        summary["status"]="PASS"
        summary["interpretation"]="Local arithmetic and propagation diagnosis only; no candidate admission."
    except Exception as e:
        summary["error"]=repr(e)
        import traceback;traceback.print_exc()
    summary["wall_seconds"]=time.monotonic()-start
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(status=summary["status"],error=summary.get("error"),high_precision=summary["high_precision"],
        first_local_failures=[row for row in summary["rows"] if
            any(not c["passed"] for field in ("fixed_target_centered_fit","fixed_target_raw_fit")
                for c in row.get(field,{}).values()) or not row.get("fixed_increment_integral",{"passed":True})["passed"]][:3])))
    if summary["status"]!="PASS":raise SystemExit(1)
if __name__=="__main__":main()
