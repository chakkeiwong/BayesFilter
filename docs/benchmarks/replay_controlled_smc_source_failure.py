"""Independent CPU-only replay of one frozen author-source covariance failure."""
import argparse
import difflib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import time

os.environ["CUDA_VISIBLE_DEVICES"]="-1"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--source-lock",type=Path,required=True)
    p.add_argument("--source-runner",type=Path,required=True)
    p.add_argument("--failure",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    start=time.monotonic()
    spec=importlib.util.spec_from_file_location("source_runner",args.source_runner)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    import numpy as np  # Independent post-run reference matrix diagnostics only.
    from scipy.io import loadmat
    failure=json.loads((args.failure/"failure.json").read_text())
    assert failure["classification"]=="ORIGINAL_SOURCE_NUMERICAL_FAILURE_IN_OCTAVE"
    assert (failure["failing_case"]["grid_index"],failure["failing_case"]["repetition"])==(36,38)
    for name,digest in failure["files_sha256"].items():
        assert runner.sha(args.failure/name)==digest,"frozen failure evidence changed"
    lock=json.loads((args.source_lock/"source-lock.json").read_text())
    assert lock["status"]=="PASS"
    for entry in lock["verified_snapshot_files"]:
        relative=Path(entry["path"]).relative_to(".localresources/code/heng-controlled-smc")
        assert runner.sha(args.source_lock/"author-source"/relative)==entry["sha256"]
    for entry in lock["runtime_dependencies"]:
        assert runner.sha(args.root/entry["path"])==entry["sha256"]
    stats=args.root/".localresources/code/octave-statistics-1.5.0/inst"
    source=args.source_lock/"author-source/StateSpaceModels/Lorenz"
    assert runner.sha(source/"cSMC_Resample.m")==failure["source_algorithm_sha256"]
    commands=[];arms=[]
    for arm in ("unchanged","observed"):
        dest=args.output/arm
        dest.mkdir()
        for item in source.iterdir():
            if item.is_file():shutil.copyfile(item,dest/item.name)
        shutil.copyfile(args.failure/"lorenz_csmc-checkpoint.mat",dest/"lorenz_csmc-checkpoint.mat")
        text=runner.instrument((source/"cSMC_VaryF.m").read_text(),"lorenz_csmc","Lorenz",
                               "cSMC_LogNormConst",[1,20,40],"full")
        text=runner.add_case_checkpoint(text,"lorenz_csmc","cSMC_LogNormConst",isolated=True)
        driver=dest/"cSMC_VaryF_failure_replay.m";driver.write_text(text)
        if arm=="observed":
            algorithm=dest/"cSMC_Resample.m"
            original=algorithm.read_text()
            target="X = mvnrnd(TwistedTransitionMean, TwistedG);"
            capture="""try
                X = mvnrnd(TwistedTransitionMean, TwistedG);
            catch source_failure
                failed_t = t;
                failed_G = TwistedG;
                failed_A = squeeze(Twisting.A(t+1,:,:));
                failed_transition_precision = Transition.Ginv;
                failed_Q = Transition.Ginv + 2 * failed_A;
                failed_policy_A = Twisting.A;
                failed_policy_b = Twisting.b;
                failed_policy_c = Twisting.c;
                save('-mat7-binary','source-failure-matrices.mat','failed_t','failed_G','failed_A','failed_transition_precision','failed_Q','failed_policy_A','failed_policy_b','failed_policy_c');
                rethrow(source_failure);
            end"""
            assert original.count(target)==1
            observed=original.replace(target,capture)
            algorithm.write_text(observed)
            (args.output/"observation-only.diff").write_text("".join(difflib.unified_diff(
                original.splitlines(True),observed.splitlines(True),fromfile="original",tofile="observed")))
        command=runner.octave_command(stats,dest,driver,202610131)
        commands.append(dict(arm=arm,command=command,cwd=str(dest),source_algorithm_sha256=runner.sha(dest/"cSMC_Resample.m")))
        (args.output/"commands.json").write_text(json.dumps(commands,indent=2)+"\n")
        outcome=runner.run_isolated_cases(command,dest,"lorenz_csmc",[(36,38)],
            50-(time.monotonic()-start),args.output/(arm+".log"),args.output/(arm+"-processes.jsonl"))
        log=(args.output/(arm+".log")).read_text()
        error=failure["error"]
        assert outcome["returncode"]==1 and error in log,(arm,outcome,"unexpected source behavior")
        assert runner.sha(dest/"lorenz_csmc-checkpoint.mat")==failure["files_sha256"]["lorenz_csmc-checkpoint.mat"]
        arms.append(dict(arm=arm,expected_failure_reproduced=True,returncode=1,error=error))
    captured=loadmat(args.output/"observed/source-failure-matrices.mat")
    covariance=captured["failed_G"];precision=captured["failed_Q"]
    def matrix_info(matrix):
        norm=float(np.linalg.norm(matrix,2))
        return dict(symmetric_part_eigenvalues=np.linalg.eigvalsh((matrix+matrix.T)/2).tolist(),
            relative_asymmetry=float(np.linalg.norm(matrix-matrix.T,2)/norm),
            condition_number=float(np.linalg.cond(matrix)),spectral_norm=norm)
    assert precision.shape==covariance.shape==(8,8)
    assert np.isrealobj(precision) and np.isrealobj(covariance)
    assert np.isfinite(precision).all() and np.isfinite(covariance).all()
    # Reference serialization only; never discard imaginary source components.
    policy_real_finite=all(np.isrealobj(captured['failed_policy_'+k]) and
                          np.isfinite(captured['failed_policy_'+k]).all() for k in ('A','b','c'))
    fixture=dict(schema='lorenz_failed_policy_v1',arrays={},policy_real_finite=bool(policy_real_finite),
                 source_failure_reproduced=True,time_index=int(captured['failed_t'].item()))
    fields={'precision':precision,'covariance':covariance}
    if policy_real_finite:
        fields.update({k:captured['failed_policy_'+k] for k in ('A','b','c')})
    for name,value in fields.items():
        array=np.ascontiguousarray(value,dtype='<f8')
        if name=='c':array=array.reshape(-1)
        path=args.output/('failed-policy-'+name+'.bin');path.write_bytes(array.tobytes())
        fixture['arrays'][name]=dict(file=path.name,shape=list(array.shape),sha256=runner.sha(path))
    (args.output/'failed-policy.json').write_text(json.dumps(fixture,indent=2,allow_nan=False)+'\n')
    q,g=matrix_info(precision),matrix_info(covariance)
    eps=np.finfo(np.float64).eps
    qmargin=64*eps*q["spectral_norm"]
    gmargin=64*eps*g["spectral_norm"]
    qbad=q["symmetric_part_eigenvalues"][0] < -qmargin
    gbad=g["symmetric_part_eigenvalues"][0] < -gmargin
    diagnosis=("SOURCE_PRECISION_AND_COVARIANCE_INDEFINITE" if qbad and gbad else
               "COVARIANCE_INDEFINITE_PRECISION_NOT_RESOLVED" if gbad else
               "RUNTIME_DOMAIN_FAILURE_REQUIRES_FURTHER_ANALYSIS")
    residual=float(np.linalg.norm(precision@covariance-np.eye(8),2)/
        max(1.,q["spectral_norm"]*g["spectral_norm"]))
    summary=dict(status="PASS",scope="FROZEN_SOURCE_FAILURE_REPLAY",replication="INCOMPLETE",
        scientific_verdict=diagnosis,cpu_only=True,gpu_intentionally_hidden=True,
        source_attempt=failure["source_attempt"],grid_index=36,repetition=38,seed=202610131,
        source_failure_reproduced=True,reference_failure_is_replication_veto=False,author_method_repair_in_scope=False,next_step="matched_source_to_tensorflow_consumer_comparison",arms=arms,time_index=int(captured["failed_t"].item()),
        precision=q,covariance=g,scaled_inverse_residual=residual,
        eigenvalue_roundoff_screen=dict(formula="64*float64_epsilon*spectral_norm; explanatory only",
            precision_margin=qmargin,covariance_margin=gmargin),
        upstream_fit_root_cause="NOT_CHECKED_IN_THIS_REPLAY",
        matlab_runtime_equivalence="NOT_CHECKED",numerical_repair_selected=False,
        wall_seconds=time.monotonic()-start,
        checks=[dict(name="exact_failing_source_consumer_reproduced_in_both_arms",passed=True)])
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    note=f"""# Frozen source failure replay

Verdict: {diagnosis}. The unchanged source and observation-only copy both
reproduce the same covariance error from the saved pre-case RNG state.

| Decision | Primary criterion | Veto | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Source failure reproduced | Both arms match expected error | No input or replay mismatch; intrinsic author failure is accepted baseline behavior | Matched port behavior and MATLAB compatibility | Compare the TensorFlow consumer on these exact reference inputs | Author-method repair, paper refutation or numerical repair acceptance |

| Inference status | Finding |
|---|---|
| Hard veto screen | Reference fails its Gaussian covariance domain; this is an accepted reference outcome, not a replication veto |
| Statistically supported ranking | None; deterministic replay |
| Descriptive differences | Matrix diagnostics in summary.json |
| Default readiness | No change or promotion |
| Next evidence needed | Matched TensorFlow comparison on valid reference cases and this intrinsic failure |

The source full-quadratic branch forms Q=Transition.Ginv+2*A and G=inv(Q).
Gaussian proposal precision must be positive definite. Saved matrices are the
actual failing consumer inputs; eigenvalues use the symmetric part and
asymmetry is reported separately. The matrices were never substituted back into
the source. Matrix spectra identify where invalidity is visible; they do not
by themselves establish why the preceding least-squares fit produced it.

The strongest alternative explanation remains Octave/MATLAB solver semantics.
This replay contains no TensorFlow or XLA path, adds no covariance protection,
and supplies no stochastic ranking or full replication result.
"""
    (args.output/"result.md").write_text(note)
    print(json.dumps({"status":"PASS","diagnosis":diagnosis,"scope":summary["scope"]}))


if __name__=="__main__":main()
