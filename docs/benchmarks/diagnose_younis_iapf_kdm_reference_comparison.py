"""CPU-only independent-reference diagnostics, not a BayesFilter runtime route.

Executes pinned public source bodies with restricted dependency loaders. Torch
autodiff is only the independent author-reference derivative authority. All
BayesFilter computations use the existing TensorFlow analytical kernels.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-reference-comparison-mpl")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
AUTHOR_ROOT = ROOT / ".localresources/code/younis-mdpf-neurips-2023"
KDE_SOURCE = AUTHOR_ROOT / ("packages/kernel-density-estimator-bandwdidth-prediction/"
                           "packages/kernel_density_estimation/kernel_density_estimator.py")
RESAMPLE_SOURCE = AUTHOR_ROOT / "src/models/kde_particle_filter/kde_particle_filter.py"
IAPF_SOURCE = ROOT / ".localresources/code/sempreteamo-iapf-a8811439/iapf.R"
PLAN = "docs/plans/younis-iapf-kdm-reference-comparison-2026-09-18.md"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_output(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()


def flat(value):
    if hasattr(value, "numpy"):
        value = value.numpy().tolist()
    if isinstance(value, (list, tuple)):
        return [number for part in value for number in flat(part)]
    return [float(value)]


def serial(value):
    if hasattr(value, "numpy"):
        return value.numpy().tolist()
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


class Checks:
    def __init__(self):
        self.rows = []

    def close(self, name, actual, expected, atol=2e-8, rtol=2e-8):
        a, b = flat(actual), flat(expected)
        if len(a) != len(b) or not a:
            raise ValueError(f"{name}: incompatible comparison lengths")
        finite = all(math.isfinite(x) for x in a + b)
        error = max(abs(x-y) for x, y in zip(a, b)) if finite else None
        scaled = max(abs(x-y)/(atol+rtol*abs(y)) for x, y in zip(a, b)) if finite else None
        self.rows.append(dict(name=name, passed=finite and scaled <= 1,
                              maximum_absolute_error=error, tolerance_ratio=scaled,
                              atol=atol, rtol=rtol, compared_scalars=len(a)))

    def condition(self, name, condition, detail=None):
        self.rows.append(dict(name=name, passed=bool(condition), detail=serial(detail)))


def load_author(torch):
    namespace = {"torch": torch, "D": torch.distributions}
    tree = ast.parse(KDE_SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and
               n.name == "KernelDensityEstimator")
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(KDE_SOURCE), "exec"), namespace)
    tree = ast.parse(RESAMPLE_SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KDEParticleFilter")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "resample_particles")
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(RESAMPLE_SOURCE), "exec"), namespace)
    return namespace["KernelDensityEstimator"], namespace["resample_particles"]


def run_kdm(tf, torch, checks):
    from bayesfilter.highdim.ledh_younis_kdm_tf import (
        make_gaussian_kdm_kernel, make_full_mixture_iwsg_resampling_kernel)
    AuthorKDE, author_resample = load_author(torch)
    torch.set_default_dtype(torch.float64)
    rows = []
    epsilon = 1e-8
    for d in (1, 2):
        for regime in ("overlap", "separated", "tiny_weight"):
            name = f"kdm/d{d}/{regime}"
            n, directions = 3, 3
            centers = [-.6, .1, .7] if regime == "overlap" else [-4., 0., 4.]
            means = [[v] if d == 1 else [v, -.7*v+.1*i] for i, v in enumerate(centers)]
            weights = [1e-12, .4, .6-1e-12] if regime == "tiny_weight" else [.2, .35, .45]
            bandwidth = [.7] if d == 1 else [.7, 1.1]
            points = means + [[-5.5]*d, [5.5]*d]
            dw = [weights[i]*[-.3,.2,0.][i] for i in range(n)]
            total = sum(dw)
            dw = [v-w*total for v, w in zip(dw, weights)]
            dm = [[.12*(i+1)*(-1 if j else 1) for j in range(d)] for i in range(n)]
            db = [.09*(j+1) for j in range(d)]
            tw, tm, tb = [torch.tensor(v) for v in (weights, means, bandwidth)]
            tdw, tdm, tdb = [torch.tensor(v) for v in (dw, dm, db)]
            delta = torch.zeros(directions, requires_grad=True)
            w = tw + delta[0]*tdw
            m = tm + delta[1]*tdm
            b = tb + delta[2]*tdb
            params = {"dims": {j: {"distribution_type": "Normal"} for j in range(d)}}
            author = AuthorKDE(params, m[None], w[None], b[None])
            author_log = author.log_prob(torch.tensor(points)[None])[0]

            def derivatives(values):
                return torch.stack([torch.autograd.grad(v, delta, retain_graph=True)[0]
                                    for v in values.reshape(-1)]).T.detach().tolist()

            def constant(value):
                return tf.constant(value, tf.float64)

            wtf, mtf, btf = [constant(v) for v in (weights, means, bandwidth)]
            cov = tf.repeat(tf.linalg.diag(btf**2)[None], n, axis=0)
            dweights = tf.stack([constant(dw), tf.zeros([n],tf.float64), tf.zeros([n],tf.float64)])
            dmeans = tf.stack([tf.zeros([n,d],tf.float64), constant(dm), tf.zeros([n,d],tf.float64)])
            dcov = tf.stack([tf.zeros_like(cov), tf.zeros_like(cov),
                            tf.repeat(tf.linalg.diag(2*btf*constant(db))[None],n,axis=0)])
            denominator = tf.reduce_sum(wtf) + n*epsilon
            aligned_w = (wtf+epsilon)/denominator
            aligned_dw = (dweights*denominator - (wtf+epsilon)[None]*
                          tf.reduce_sum(dweights,axis=1)[:,None])/denominator**2
            kernel = make_gaussian_kdm_kernel(evaluation_count=len(points), component_count=n,
                                              dimension=d, direction_count=directions,
                                              dtype=tf.float64, jit_compile=True)
            common = (mtf,cov,tf.zeros([directions,len(points),d],tf.float64))
            raw = kernel(constant(points),wtf,*common,dweights,dmeans,dcov)
            aligned = kernel(constant(points),aligned_w,*common,aligned_dw,dmeans,dcov)
            checks.condition(name+"/density_valid",tf.reduce_all(aligned["valid"]))
            checks.close(name+"/aligned_log_density",aligned["log_density"],author_log.detach().tolist())
            checks.close(name+"/analytical_score",aligned["d_log_density"],derivatives(author_log))
            direct = []
            for x in points:
                terms = [math.log((weights[i]+epsilon)/(1+n*epsilon))-
                         sum(math.log(bandwidth[j])+.5*math.log(2*math.pi)+
                             .5*((x[j]-means[i][j])/bandwidth[j])**2 for j in range(d))
                         for i in range(n)]
                largest = max(terms)
                direct.append(largest+math.log(sum(math.exp(v-largest) for v in terms)))
            checks.close(name+"/direct_gaussian_sum",aligned["log_density"],direct)
            fd = []
            step = 1e-5
            for k in range(directions):
                sides = []
                for sign in (-1,1):
                    ww = tw + (sign*step*tdw if k == 0 else 0)
                    mm = tm + (sign*step*tdm if k == 1 else 0)
                    bb = tb + (sign*step*tdb if k == 2 else 0)
                    ref = AuthorKDE(params, mm[None], ww[None], bb[None])
                    sides.append(ref.log_prob(torch.tensor(points)[None])[0])
                fd.append(((sides[1]-sides[0])/(2*step)).tolist())
            checks.close(name+"/central_difference",aligned["d_log_density"],fd,atol=2e-5,rtol=2e-5)

            seed = 91801+len(rows)
            torch.manual_seed(seed)
            consumer = SimpleNamespace(decouple_weights_for_resampling=False,
                decouple_bandwidths_for_resampling=False, use_differentiable_resampling=True,
                training=True, differentiable_resampling_method="ImportanceSampling",
                kde_params=params, particle_transformer=SimpleNamespace(apply_norm=lambda x:x))
            samples, injected, _ = author_resample(consumer, dict(particles_downscaled=m[None],
                particle_weights=w[None], bandwidths_downscaled=b[None],do_resample=True))
            checks.condition(name+"/source_samples_detached",not samples.requires_grad)
            sample_list = samples[0].tolist()
            anchor = author.log_prob(samples)[0].detach().tolist()
            resample = make_full_mixture_iwsg_resampling_kernel(particle_count=n, dimension=d,
                direction_count=directions,dtype=tf.float64,jit_compile=True)
            result = resample(constant(sample_list),aligned_w,mtf,cov,cov,constant(anchor),
                              aligned_dw,dmeans,dcov,tf.zeros_like(dcov))
            checks.condition(name+"/resampling_valid",tf.reduce_all(result["valid"]))
            checks.close(name+"/importance_weights",result["importance_weights"],
                         ((injected[0]-epsilon)/n).detach().tolist())
            checks.close(name+"/importance_weight_derivative",result["d_importance_weights"],
                         [[v/n for v in row] for row in derivatives(injected[0])])
            likelihood = torch.exp(-.25*torch.sum(samples[0]**2,dim=1))+.2
            source_posterior = injected[0]*likelihood
            source_posterior = source_posterior/source_posterior.sum()
            ltf = constant(likelihood.tolist())
            unnormalized = result["importance_weights"]*ltf
            dunnormalized = result["d_importance_weights"]*ltf[None]
            mass = tf.reduce_sum(unnormalized)
            posterior = unnormalized/mass
            dposterior = (dunnormalized-posterior[None]*
                          tf.reduce_sum(dunnormalized,axis=1)[:,None])/mass
            checks.close(name+"/next_normalized_weights",posterior,source_posterior.detach().tolist())
            checks.close(name+"/next_weight_derivative_epsilon_accounted",
                         dposterior/(1+epsilon), derivatives(source_posterior))
            rows.append(dict(dimension=d,regime=regime,seed=seed,
                raw_maximum_log_density_difference=max(abs(a-b) for a,b in
                    zip(flat(raw["log_density"]),author_log.detach().tolist())),
                raw_maximum_score_difference=max(abs(a-b) for a,b in
                    zip(flat(raw["d_log_density"]),flat(derivatives(author_log)))),
                post_resampling_epsilon=epsilon,
                samples=sample_list,aligned_log_density=serial(aligned["log_density"])))
    return rows


def prepare_iapf_reference(output):
    """Admit only the source operations whose paper identities were checked."""
    from docs.benchmarks import diagnose_iapf_paper_conformance as conformance
    reference = conformance.run_reference(output, source=IAPF_SOURCE)
    report = conformance.assess_reference(reference, source=IAPF_SOURCE)
    (output / "iapf-paper-conformance.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)+"\n")
    conformance.require_reference(report, scope="matched_gaussian")
    return reference, report


def run_iapf(tf, checks, output):
    reference, conformance = prepare_iapf_reference(output)
    from bayesfilter.score_study.iapf_fit_tf import _density_profile, bounded_density_fit
    from bayesfilter.score_study.fitted_twist_tf import (
        normalizer, twisted_transition, gaussian_floor_log, make_fitted_twist_kernel)
    from bayesfilter.score_study.iapf_adapter import iteration_decision
    from bayesfilter.score_study.gaussian_tf import gaussian_log_density_and_tangent
    const = lambda value: tf.constant(value,tf.float64)
    points = tf.reshape(const(reference["fit"]["points"]),[-1,2])
    n = points.shape[0]
    targets = const(reference["objective"]["targets"])

    @tf.function(input_signature=[tf.TensorSpec([4],tf.float64)],jit_compile=True)
    def objective(parameters):
        return _density_profile(points,tf.math.log(targets),parameters,"relative_shape")[:2]

    objectives = []
    for i in (1,2,3):
        source = reference[f"objective{i}"]
        relative, gradient = objective(const(source["parameters"]))
        c = tf.reduce_sum(targets**2)
        expected = c*relative/(1-relative)
        expected_gradient = c*gradient/(1-relative)**2
        checks.close(f"iapf/objective{i}/monotone_transform",expected,source["loss"])
        checks.close(f"iapf/objective{i}/transformed_gradient",expected_gradient,source["gradient"],
                     atol=2e-5,rtol=2e-5)
        objectives.append(dict(relative_shape=float(relative),source_loss=source["loss"][0],
                               transformed_loss=float(expected)))

    @tf.function(input_signature=[tf.TensorSpec([n],tf.float64)],jit_compile=True)
    def fit(log_target):
        return bounded_density_fit(points,log_target,mean_bound=4.,sd_lower=.2,sd_upper=4.,
            max_steps=10000,max_backtracks=30,tolerance=1e-7,floor_ratio=1e-8,
            objective="relative_shape")

    observations = tf.reshape(const(reference["fit"]["observations"]),[3,2])
    r_fit = tf.reshape(const(reference["fit"]["parameters"]),[3,4])
    tight_fit = tf.reshape(const(reference["fit"]["tight_parameters"]),[3,4])
    fits = []
    for t in range(3):
        log_target = -.5*tf.reduce_sum((points-observations[t])**2,axis=1)-math.log(2*math.pi)
        center,covariance,log_floor,info = fit(log_target)
        # The R backward recursion visits terminal time first.
        r_optimizer = reference["fit"][f"optimizer{3-t}"]
        eligible = bool(info["valid"]) and bool(info["converged"]) and not bool(info["boundary_active"])
        checks.condition(f"iapf/fit{t}/local_convergence",eligible,info)
        checks.condition(f"iapf/fit{t}/source_convergence",r_optimizer[0] == 0,r_optimizer)
        if eligible and r_optimizer[0] == 0:
            checks.close(f"iapf/fit{t}/mean_source",center,r_fit[t,:2],atol=2e-3,rtol=0.)
            checks.close(f"iapf/fit{t}/variance_source",tf.linalg.diag_part(covariance),r_fit[t,2:],atol=2e-3,rtol=0.)
            checks.close(f"iapf/fit{t}/mean_exact",center,observations[t],atol=2e-3,rtol=0.)
            checks.close(f"iapf/fit{t}/variance_exact",tf.linalg.diag_part(covariance),[1.,1.],atol=2e-3,rtol=0.)
        tight_optimizer = reference["fit"][f"tight_optimizer{3-t}"]
        checks.condition(f"iapf/fit{t}/tight_source_convergence",tight_optimizer[0] == 0,tight_optimizer)
        checks.close(f"iapf/fit{t}/tight_source_mean_exact",tight_fit[t,:2],observations[t],atol=2e-3,rtol=0.)
        checks.close(f"iapf/fit{t}/tight_source_variance_exact",tight_fit[t,2:],[1.,1.],atol=2e-3,rtol=0.)
        if eligible and tight_optimizer[0] == 0:
            checks.close(f"iapf/fit{t}/tight_mean_source",center,tight_fit[t,:2],atol=2e-3,rtol=0.)
            checks.close(f"iapf/fit{t}/tight_variance_source",tf.linalg.diag_part(covariance),tight_fit[t,2:],atol=2e-3,rtol=0.)
        fits.append(dict(time=t,source_parameters=serial(r_fit[t]),local_center=serial(center),
                         local_covariance=serial(covariance),local_info=serial(info),
                         positive_log_floor=float(log_floor),source_optimizer=r_optimizer,
                         tight_source_parameters=serial(tight_fit[t]),tight_source_optimizer=tight_optimizer))

    psi = tf.reshape(const(reference["components"]["psi"]),[3,4])
    transition = tf.reshape(const(reference["components"]["A"]),[2,2])
    x = const(reference["components"]["x"])[None]
    identity = tf.eye(2,dtype=tf.float64)
    zero_q = tf.zeros([6,2,2],tf.float64)
    no_floor = const(-math.inf)

    @tf.function(input_signature=[tf.TensorSpec([2],tf.float64),tf.TensorSpec([2],tf.float64),
                                  tf.TensorSpec([2],tf.float64)],jit_compile=True)
    def proposal(mean,center,variance):
        noise = tf.concat([tf.zeros([1,2],tf.float64),identity],axis=0)
        drawn,_ = twisted_transition(tf.repeat(mean[None],3,axis=0),tf.zeros([6,3,2],tf.float64),
            identity,zero_q,center,tf.linalg.diag(variance),tf.ones([3],tf.float64),noise,
            tf.fill([3],const(.5)))
        factor = tf.transpose(drawn[1:]-drawn[0])
        return drawn[0],factor@tf.transpose(factor)

    for label,mean,t in (("initial",tf.zeros([2],tf.float64),0),
                         ("transition",tf.linalg.matvec(transition,x[0]),1)):
        mean_out,cov = proposal(mean,psi[t,:2],psi[t,2:])
        checks.close(f"iapf/{label}/proposal_mean",mean_out,reference[label]["mean"])
        checks.close(f"iapf/{label}/proposal_covariance",cov,reference[label]["covariance"])
        exact_var = 1/(1+1/psi[t,2:])
        checks.close(f"iapf/{label}/closed_form_mean",mean_out,exact_var*(mean+psi[t,:2]/psi[t,2:]))
        checks.close(f"iapf/{label}/closed_form_covariance",cov,tf.linalg.diag(exact_var))

    @tf.function(input_signature=[tf.TensorSpec([2],tf.float64),tf.TensorSpec([2],tf.float64),
                                  tf.TensorSpec([2],tf.float64)],jit_compile=True)
    def log_normalizer(mean,center,variance):
        return normalizer(mean[None],tf.zeros([6,1,2],tf.float64),identity,zero_q,
                          center,tf.linalg.diag(variance),no_floor)[0][0]

    @tf.function(input_signature=[tf.TensorSpec([2],tf.float64),tf.TensorSpec([2],tf.float64)],
                 jit_compile=True)
    def log_twist(center,variance):
        return gaussian_floor_log(x,tf.zeros([6,1,2],tf.float64),center,
                                  tf.linalg.diag(variance),no_floor)[0][0]

    for t in range(3):
        lg,_ = gaussian_log_density_and_tangent(observations[t][None]-x,
            tf.zeros([6,1,2],tf.float64),identity,zero_q[:,None])
        lp = log_twist(psi[t,:2],psi[t,2:])
        future = (log_normalizer(tf.linalg.matvec(transition,x[0]),psi[t+1,:2],psi[t+1,2:])
                  if t < 2 else const(0.))
        weight = lg[0]+future-lp
        if t == 0:
            weight += log_normalizer(tf.zeros([2],tf.float64),psi[0,:2],psi[0,2:])
        source = reference[f"weight{t+1}"]
        checks.close(f"iapf/weight{t}/twist",lp,source["log_psi"])
        checks.close(f"iapf/weight{t}/future",future,source["log_future"])
        checks.close(f"iapf/weight{t}/increment",weight,source["log_weight"])

    controllers = []
    for name in ("partial","first_complete","next_complete","oscillating","shifted"):
        history = reference[name]["history"]
        source_cv,source_fit,source_particles = reference[name]["controller"]
        local = iteration_decision(history,[8]*len(history),k=2,tau=.1,max_particles=64)
        if local["cv"] is not None:
            checks.close(f"iapf/controller/{name}/sample_cv",local["cv"],source_cv,atol=2e-12,rtol=2e-12)
        paper_final = len(history)-1 > 2 and source_cv < .1
        checks.condition(f"iapf/controller/{name}/paper_stopping",(local["action"] == "final") == paper_final)
        expected_difference = name == "first_complete"
        different = (local["action"] == "fit") != bool(source_fit)
        checks.condition(f"iapf/controller/{name}/source_difference_classified",different == expected_difference)
        if not expected_difference:
            checks.close(f"iapf/controller/{name}/particle_count",local["next_particles"],source_particles,atol=0.,rtol=1e-12)
        controllers.append(dict(case=name,source_cv=source_cv,source_action="fit" if source_fit else "final",
                                local=local,expected_stopping_difference=expected_difference))

    N,T = 32,3
    h = 1+float(tf.cast(.15,tf.float64))
    theta = const([0.,0.,0.,1/h,0.,0.])
    obs = tf.reshape(const(reference["full_filter"]["observations"]),[T,1])
    kernel = make_fitted_twist_kernel(1,1,N,T,dtype_name="float64",jit_compile=True)
    value,score,_ = kernel(theta,obs,tf.random.stateless_normal([N,1],[91801,0],dtype=tf.float64),
        tf.random.stateless_normal([T,N,1],[91801,1],dtype=tf.float64),
        tf.random.stateless_uniform([T+1,N],[91801,2],dtype=tf.float64),
        tf.random.stateless_uniform([T,N],[91801,3],dtype=tf.float64),obs,
        tf.ones([T,1,1],tf.float64),tf.fill([T],const(-math.inf)))
    exact = sum(-.5*math.log(4*math.pi)-y*y/4 for y in flat(obs))
    checks.close("iapf/full_filter/local_vs_source",value,reference["full_filter"]["log_likelihood"])
    checks.close("iapf/full_filter/local_vs_exact",value,exact)
    checks.close("iapf/full_filter/source_vs_exact",reference["full_filter"]["log_likelihood"],exact)
    checks.condition("iapf/full_filter/finite_frozen_program_score",tf.reduce_all(tf.math.is_finite(score)))
    return dict(reference_scope="matched_gaussian_operations_only",
                paper_reference_eligible=conformance["paper_reference_eligible"],
                paper_conformance=conformance,
                objectives=objectives,fits=fits,controllers=controllers,
                full_filter=dict(local=float(value),source=reference["full_filter"]["log_likelihood"][0],
                                 exact=exact,score_interpretation="frozen finite program only"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--methods",choices=("all","iapf","kdm"),default="all")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    checks = Checks()
    result = dict(plan=PLAN,command=[sys.executable,*sys.argv],git_commit=command_output("git","rev-parse","HEAD"),
        git_changes=command_output("git","status","--short"),cpu_only=True,gpu_intentionally_hidden=True,
        cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],jit_compile=True,dtype="float64",
        environment=sys.executable,data_version="fresh_reference_fixtures_20260918_v1",
        seeds=list(range(91801,91807)),result_file=str(output/"results.json"),
        methods=args.methods,sources={str(p.relative_to(ROOT)):sha(p) for p in
                 (KDE_SOURCE,RESAMPLE_SOURCE,IAPF_SOURCE,Path(__file__),
                  ROOT/"docs/benchmarks/diagnose_iapf_public_reference.R",
                  ROOT/"docs/benchmarks/diagnose_iapf_paper_conformance.py",
                  ROOT/"bayesfilter/highdim/ledh_younis_kdm_tf.py",
                  ROOT/"bayesfilter/score_study/iapf_fit_tf.py",
                  ROOT/"bayesfilter/score_study/iapf_adapter.py",
                  ROOT/"bayesfilter/score_study/fitted_twist_tf.py",
                  ROOT/"bayesfilter/score_study/gaussian_tf.py")},
        author_kdm_commit=command_output("git","-C",str(AUTHOR_ROOT),"rev-parse","HEAD"),
        independent_iapf_commit="a88114395f6c11075fedc653db9480791b43391a",
        reference_loader=dict(kdm="unchanged AST class and resampling method; Normal/IWSG only",
            iapf="unchanged parsed functions and controller conditions; no top-level experiment",
            substitutions="base-R Gaussian density and Cholesky sampler; optim wrapper retains status"),
        claim_scope="matched reference operations only; no ranking, default, LEDH or HMC promotion")
    try:
        import tensorflow as tf
        import torch
        torch.set_num_threads(2)
        result["versions"] = dict(python=sys.version,tensorflow=tf.__version__,torch=torch.__version__,
                                  R=command_output("Rscript","--version"))
        for name,run in (("kdm",lambda:run_kdm(tf,torch,checks)),
                         ("iapf",lambda:run_iapf(tf,checks,output))):
            if args.methods not in ("all",name):
                continue
            try:
                result[name] = run()
            except Exception:
                result[name+"_error"] = traceback.format_exc()
                checks.condition(name+"/execution",False,result[name+"_error"])
    except Exception:
        result["environment_error"] = traceback.format_exc()
        checks.condition("environment",False,result["environment_error"])
    result["checks"] = checks.rows
    result["wall_seconds"] = time.monotonic()-started
    result["passed"] = bool(checks.rows) and all(row["passed"] for row in checks.rows)
    result["passed_checks"] = sum(row["passed"] for row in checks.rows)
    result["failed_checks"] = [row["name"] for row in checks.rows if not row["passed"]]
    with (output/"results.json").open("w") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({k:result[k] for k in ("passed","passed_checks","failed_checks","wall_seconds")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
