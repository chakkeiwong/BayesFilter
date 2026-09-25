"""Independent CPU audit fixtures; never imported by an algorithmic route."""

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
REPO = Path("/home/chakwong/BayesFilterZhaoCui")
sys.path.insert(0, str(REPO))

import numpy as np
from scipy.integrate import quad
from scipy.special import eval_hermitenorm, logsumexp
import tensorflow as tf

import bayesfilter.highdim as hd
import bayesfilter.highdim.zhao_cui_algorithm2_preparation_tf as prep
from bayesfilter.highdim.hermite_gram_tf import normalized_hermite_incomplete_gram
from bayesfilter.highdim.squared_tt_engine_v0_tf import (
    EngineConfig, _fixed_als_fit, _initial_tt_cores,
)
from bayesfilter.highdim.zhao_cui_algorithm3_tf import (
    AffineTTProposal, FrozenAlgorithm3Program, compile_algorithm3,
)

DT = tf.float64
spec = importlib.util.spec_from_file_location(
    "existing_algorithm3_reference", REPO / "tests/highdim/test_zhao_cui_algorithm3_tf.py"
)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
NormalModel = reference.NormalModel


def rank_activation():
    grid = np.linspace(-1.5, 1.5, 9)
    points = np.array([(x, y) for x in grid for y in grid])
    target = 1.0 + 0.25 * points[:, 0] * points[:, 1]
    config = EngineConfig(1, 2, len(points), 4, 1e-10, 1e-6, 3.0, 1701)
    basis = prep._hermite_product_basis(2, 1)
    rows = tf.constant(points, DT)
    weights = tf.ones([len(points)], DT) / len(points)
    initials = _initial_tt_cores(2, 2, 2)
    fitted, diagnostics = _fixed_als_fit(basis, rows, tf.constant(target, DT), weights, initials, config)
    coefficient = fitted[0].values.numpy()[0] @ fitted[1].values.numpy()[:, :, 0]
    exact = (
        hd.TTCore(tf.constant([[[1., 0.], [0., 1.]]], DT)),
        hd.TTCore(tf.constant([[[1.], [0.]], [[0.], [.25]]], DT)),
    )
    warm, warm_diagnostics = _fixed_als_fit(basis, rows, tf.constant(target, DT), weights, exact, config)
    warm_coefficient = warm[0].values.numpy()[0] @ warm[1].values.numpy()[:, :, 0]
    expected = np.array([[1., 0.], [0., .25]])
    return {
        "target": "h(x,y)=1+0.25*x*y; positive on all diagnostic rows",
        "config": dataclasses.asdict(config),
        "inherited_initialization_coefficients": coefficient.tolist(),
        "inherited_effective_rank": int(np.linalg.matrix_rank(coefficient, tol=1e-8)),
        "inherited_fit": dict(diagnostics),
        "exact_rank_two_warm_start_fit": dict(warm_diagnostics),
        "exact_warm_start_max_coefficient_error": float(np.max(np.abs(warm_coefficient-expected))),
        "rank_activation_defect_reproduced": bool(
            np.linalg.matrix_rank(coefficient, tol=1e-8) == 1
            and diagnostics["weighted_fit_rms"] > .1
            and np.max(np.abs(warm_coefficient-expected)) < 1e-7
        ),
    }


def gram_quadrature():
    degree, worst, smallest_eigenvalue = 5, 0., 1.
    for z in [-2., .1, 1.4]:
        actual = normalized_hermite_incomplete_gram(tf.constant(z, DT), degree).numpy()
        for a in range(degree+1):
            for b in range(degree+1):
                expected = quad(
                    lambda x: eval_hermitenorm(a, x)*eval_hermitenorm(b, x)
                    * math.exp(-x*x/2)/math.sqrt(2*math.pi*math.factorial(a)*math.factorial(b)),
                    -np.inf, z, epsabs=1e-11,
                )[0]
                worst = max(worst, abs(actual[a,b]-expected))
        smallest_eigenvalue = min(smallest_eigenvalue, float(np.linalg.eigvalsh(actual).min()))
    return {"degree": degree, "max_absolute_quadrature_error": worst,
            "minimum_eigenvalue": smallest_eigenvalue,
            "pass": worst < 2e-10 and smallest_eigenvalue > -2e-12}


def bounded_initial_density():
    tr = hd.FixedTTSIRTTransport(reference.bounded_density(), hd.KRCDFConfig(9,48,1e-12,1e-12,1e-12,0))
    proposal = AffineTTProposal(tr, tf.zeros([2],DT), tf.eye(2,dtype=DT),
                                tf.zeros([0],DT), tf.zeros([0,0],DT))
    uniforms = tf.constant([[.17,.28],[.38,.61],[.72,.84]],DT)
    branch = compile_algorithm3(proposal, [], tf.zeros([1,1],DT), uniforms,
                                tf.zeros([0,3,2],DT), jit_compile=False)
    points = branch.states[0].numpy()
    numerical = []
    for point in points:
        jacobian = np.zeros([2,2])
        for j in range(2):
            delta = np.eye(2)[j]*1e-6
            plus = tr.forward_transport(tf.constant((point+delta)[:,None],DT)).numpy()[:,0]
            minus = tr.forward_transport(tf.constant((point-delta)[:,None],DT)).numpy()[:,0]
            jacobian[:,j] = (plus-minus)/(2e-6)
        numerical.append(np.linalg.slogdet(jacobian)[1])
    error = float(np.max(np.abs(branch.proposal_log_densities[0].numpy()-numerical)))
    return {"maximum_log_density_jacobian_error": error, "pass": error < 1e-7}


def joint_reference(transport, coordinates):
    value = np.ones([1])
    for core, x in zip(transport.cores, coordinates):
        c = core.numpy()
        basis = np.array([eval_hermitenorm(k,x)/math.sqrt(math.factorial(k)) for k in range(c.shape[1])])
        value = value @ np.einsum("akb,k->ab",c,basis)
    return (float(value[0])**2+transport.tau)*math.exp(-.5*sum(x*x for x in coordinates))/(2*math.pi)**(len(coordinates)/2)


def prepare_compile_score():
    config = EngineConfig(2,2,128,2,1e-10,1e-6,3.,1701)
    model, theta = NormalModel(), np.array([.15,-.2])
    observations = np.array([[.2],[.5],[-.4]])
    charts = [(tf.constant([m],DT),tf.constant([[s]],DT)) for m,s in [(-.3,1.3),(.7,.8),(.2,1.1)]]
    captures = []
    original = prep._fixed_als_fit
    def spy(basis, rows, target, weights, cores, cfg):
        captures.append((rows.numpy(),target.numpy()))
        return original(basis,rows,target,weights,cores,cfg)
    with patch.object(prep,"_fixed_als_fit",spy):
        proposals, diagnostics = prep.prepare_algorithm2_proposals(
            model,theta,tf.constant(observations,DT),charts,config,defensive_relative_mass=.01)
    target_errors = []
    for t in [1,2]:
        rows, target = captures[t]
        previous = proposals[t-1]
        tr = previous.transport
        if len(tr.cores)==1:
            mass = quad(lambda z:joint_reference(tr,[z]),-np.inf,np.inf,epsabs=1e-10)[0]
        else:
            nodes, weights = np.polynomial.hermite_e.hermegauss(5)
            # Exact Gaussian quadrature for this degree-four squared polynomial.
            mass = sum(wx*wy*joint_reference(tr,[x,y])/math.exp(-.5*(x*x+y*y))
                       for x,wx in zip(nodes,weights) for y,wy in zip(nodes,weights))
        for i in [0,7]:
            zold = rows[i,1]
            marginal = (joint_reference(tr,[zold]) if len(tr.cores)==1 else
                        quad(lambda e:joint_reference(tr,[zold,e]),-np.inf,np.inf,epsabs=1e-10)[0])/mass
            physical_old = float(previous.offset[0])+float(previous.matrix[0,0])*zold
            current = float(proposals[t].offset[0])+float(proposals[t].matrix[0,0])*rows[i,0]
            log_previous_physical = math.log(marginal)-math.log(abs(float(previous.matrix[0,0])))
            log_jac = math.log(abs(float(previous.matrix[0,0]*proposals[t].matrix[0,0])))
            log_f = -.5*(math.log(2*math.pi)+(current-.7*physical_old-theta[0])**2)
            log_g = -theta[1]-.5*(math.log(2*math.pi)+(observations[t,0]-current)**2*np.exp(-2*theta[1]))
            log_ref = -math.log(2*math.pi)-.5*np.sum(rows[i]**2)
            expected = log_previous_physical+log_f+log_g+log_jac-log_ref
            target_errors.append(abs(2*math.log(target[i])+diagnostics[t]["target_log_shift"]-expected))
    rng = np.random.default_rng(1907)
    u = rng.uniform(.05,.95,size=(3,8,1))
    seen = []
    original_sample = AffineTTProposal.sample
    def sample_spy(self,condition,uniforms,**kwargs):
        seen.append(condition.numpy().copy())
        return original_sample(self,condition,uniforms,**kwargs)
    with patch.object(AffineTTProposal,"sample",sample_spy):
        branch = compile_algorithm3(proposals[0],proposals[1:],tf.constant(observations,DT),
                                    tf.constant(u[0],DT),tf.constant(u[1:],DT),jit_compile=False)
    result = branch.evaluate(model,theta,jit_compile=False)
    x, logq = branch.states.numpy()[:,:,0], branch.proposal_log_densities.numpy()
    residual = np.concatenate([(x[0]-theta[0])[None,:],x[1:]-.7*x[:-1]-theta[0]],axis=0)
    obs = observations[:,0,None]-x
    factors = -.5*(math.log(2*math.pi)+residual**2)-theta[1]-.5*(math.log(2*math.pi)+obs**2*np.exp(-2*theta[1]))-logq
    direct = logsumexp(factors.sum(axis=0))-math.log(x.shape[1])
    weights = np.exp(factors.sum(axis=0)-logsumexp(factors.sum(axis=0)))
    direct_score = (weights[:,None]*np.stack([residual.sum(axis=0),(obs**2*np.exp(-2*theta[1])-1).sum(axis=0)],axis=1)).sum(axis=0)
    fd = []
    for j in range(2):
        delta=np.eye(2)[j]*1e-5
        fd.append((float(branch.evaluate(model,theta+delta,jit_compile=False)["log_likelihood"])
                   -float(branch.evaluate(model,theta-delta,jit_compile=False)["log_likelihood"]))/2e-5)
    result_score = result["score"].numpy()
    errors = {"preparation_target_max_log_error":float(max(target_errors)),
              "path_sum_value_error":abs(float(result["log_likelihood"])-direct),
              "path_sum_score_max_error":float(np.max(np.abs(result_score-direct_score))),
              "finite_difference_score_max_error":float(np.max(np.abs(result_score-fd)))}
    identity = all(np.array_equal(seen[t],branch.states[t-1].numpy()) for t in [1,2])
    return {**errors,"identity_ancestry_wiring":identity,"config":dataclasses.asdict(config),
            "pass":identity and errors["preparation_target_max_log_error"]<1e-8
            and errors["path_sum_value_error"]<1e-9 and errors["path_sum_score_max_error"]<1e-9
            and errors["finite_difference_score_max_error"]<2e-7}


def guide_checks():
    class Flat:
        def state_dim(self):return 2
        def observation_log_density(self,theta,x,y,t):return tf.zeros([tf.shape(x)[0]],DT)
    mean=tf.constant([.3,-.4],DT)
    covariance=tf.constant([[2.,.7],[.7,1.]],DT)
    m,l,diag=prep.likelihood_weighted_sigma_point_chart(Flat(),[],[],0,mean,covariance)
    flat_error=float(tf.reduce_max(tf.abs(tf.linalg.matmul(l,l,transpose_b=True)-covariance)))
    changes=[]
    for y in [0.,1.,3.]:
        m,l,diag=prep.likelihood_weighted_sigma_point_chart(NormalModel(),[0.,0.],[y],0,[0.],[[1.]])
        changes.append({"observation":y,"mean":float(m[0]),"variance":float(l[0,0]**2)})
    expected_mean=math.tanh(1.)
    errors=max(abs(changes[1]["mean"]-expected_mean),abs(changes[1]["variance"]-(1-expected_mean**2)))
    rejected=[]
    for bad in [[[1.,.3],[.1,1.]],[[1.,0.],[0.,0.]]]:
        try:prep.likelihood_weighted_sigma_point_chart(Flat(),[],[],0,mean,bad)
        except ValueError as e:rejected.append(str(e))
    return {"flat_likelihood_covariance_max_error":flat_error,"two_point_exact_algebra_error":errors,
            "observation_response":changes,"invalid_covariances_rejected":rejected,
            "linear_gaussian_exact_posterior_at_y1":{"mean":.5,"variance":.5},
            "pass":flat_error<1e-12 and errors<1e-12 and len(rejected)==2,
            "nonclaim":"positive sigma-point projection is not exact Gaussian posterior updating"}


def finite_output_guard():
    branch=FrozenAlgorithm3Program(tf.zeros([4,1],DT),tf.zeros([4,2,1],DT),tf.zeros([4,2],DT),{})
    try:
        result=branch.evaluate(NormalModel(),[1e154,0.],jit_compile=False)
    except ValueError as e:
        return {"accepted":False,"error":str(e),"guard_defect_reproduced":False}
    value=float(result["log_likelihood"])
    return {"accepted":True,"valid":bool(result["valid"]),"log_likelihood":str(value),
            "weight_sums":tf.reduce_sum(result["weights"],axis=1).numpy().tolist(),
            "guard_defect_reproduced":not math.isfinite(value)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    started=time.monotonic()
    checks={}
    for name,function in [("rank_activation",rank_activation),("gram_quadrature",gram_quadrature),
                          ("bounded_initial_density",bounded_initial_density),
                          ("prepare_compile_score",prepare_compile_score),("guide_checks",guide_checks),
                          ("finite_output_guard",finite_output_guard)]:
        try:checks[name]=function()
        except Exception as exc:checks[name]={"diagnostic_error":type(exc).__name__+": "+str(exc)}
    files=["bayesfilter/highdim/"+name for name in ["zhao_cui_algorithm3_tf.py",
        "zhao_cui_algorithm2_preparation_tf.py","gaussian_hermite_tt_transport_tf.py",
        "hermite_gram_tf.py","transport.py","squared_tt_engine_v0_tf.py"]]
    result={"cpu_only":True,"gpu_devices_intentionally_hidden":True,"jit_compile":False,
            "role":"independent diagnostic; no promotion or stochastic ranking",
            "tensorflow":tf.__version__,"command":sys.argv,"checkout":str(REPO),
            "wall_seconds":time.monotonic()-started,
            "source_sha256":{p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in files},
            "checks":checks}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"artifact":str(args.output),"wall_seconds":result["wall_seconds"],"checks":checks},indent=2))


if __name__=="__main__":main()
