"""Reassess the preserved missed-mode counterexample without resampling."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started=time.monotonic()
    import tensorflow as tf
    from bayesfilter.testing.inference_validation.storage import read_json,read_tensor,write_json,file_hash
    from bayesfilter.testing.inference_validation.execution import source_state
    from bayesfilter.inference.hmc_convergence import rank_normalized_split_rhat_summary
    from bayesfilter.inference.hmc_posterior_assessment import HMCPosteriorAssessmentPolicy,assess_posterior
    from bayesfilter.inference.hmc_precision import HMCPrecisionPolicy,HMCPrecisionTarget
    rows=[]
    for member in read_json(args.pipeline)["members"]:
        if member.get("status")!="assessed":
            continue
        values=read_tensor(member["draws_path"])
        if values.shape[0]<4:
            continue
        rhat=rank_normalized_split_rhat_summary(values,rhat_max=1.01)
        policy=HMCPosteriorAssessmentPolicy(quantities_id="validation.mixture.left_of_zero.v1",
            precision=HMCPrecisionPolicy((HMCPrecisionTarget("left_mode_probability",mcse_absolute_max=.1),),
                                        method="lugsail",jit_compile=False))
        result=assess_posterior(values,("x","y"),policy=policy,stage="retained",rhat=rhat,
            quantities_fn=lambda x:{"left_mode_probability":tf.cast(x[...,0]<0.,tf.float64)})
        rows.append({"candidate_id":member["candidate_id"],"original_passed":member["posterior"]["passed"],
            "local_rhat":rhat,"mode_assessment":result,"draws_path":member["draws_path"],
            "draws_sha256":file_hash(member["draws_path"])})
    write_json(args.output,{"rows":rows,"source":source_state(),"command":sys.argv,
        "elapsed_seconds":time.monotonic()-started,"device":"CPU diagnostic; GPU deliberately hidden",
        "plan_file":"docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md",
        "sampler_executed":False,"role":"exploratory reassessment of preserved counterexample"})
    print([{ "original_passed":r["original_passed"],"mode_passed":r["mode_assessment"]["passed"],
             "precision":r["mode_assessment"]["precision"]["targets"]} for r in rows])


if __name__=="__main__":
    main()
