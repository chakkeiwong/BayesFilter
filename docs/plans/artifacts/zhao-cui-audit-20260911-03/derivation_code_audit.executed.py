"""Independent CPU/reference checks; never imported by a runtime route."""

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
REPO = Path("/home/chakwong/BayesFilterZhaoCui")
sys.path.insert(0, str(REPO))

import numpy as np
from scipy.integrate import quad
from scipy.linalg import solve_discrete_lyapunov
from scipy.special import logsumexp, ndtr
import tensorflow as tf

import bayesfilter.highdim as hd
from bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf import C2StochasticVolatilityFrozenAPFModel
from bayesfilter.highdim.gaussian_hermite_tt_transport_tf import GaussianHermiteTTTransport
from bayesfilter.highdim.zhao_cui_algorithm3_tf import AffineTTProposal, compile_algorithm3

DT = tf.float64
spec = importlib.util.spec_from_file_location(
    "algorithm3_reference", REPO / "tests/highdim/test_zhao_cui_algorithm3_tf.py"
)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def coupled_upper_law():
    # h(x,y,c)=1+0.4*x*c+0.2*x*y couples BOTH generated coordinates.
    a, b, tau = .4, .2, .3
    tr = GaussianHermiteTTTransport(
        (tf.constant([[[1., 0.], [0., 1.]]], DT),
         tf.constant([[[1., 0.], [0., 0.]], [[0., a], [b, 0.]]], DT),
         tf.constant([[[1.], [0.]], [[0.], [1.]]], DT)), (0, 1), (2,), tau)
    c = np.array([-1., -.2, .6, 1.1])
    u = np.array([[.17, .28], [.38, .61], [.72, .84], [.46, .09]])
    result = tr.sample(tf.constant(c[:, None], DT), tf.constant(u, DT), jit_compile=False)
    points = result["points"].numpy()
    x, y = points.T
    mass = 1 + a*a*c*c + b*b + tau
    pdf = lambda z: np.exp(-z*z/2)/math.sqrt(2*math.pi)
    fy = ndtr(y) - pdf(y)*(2*a*b*c+b*b*y)/mass
    slope = a*c+b*y
    fx = ndtr(x) - pdf(x)*(2*slope+slope*slope*x)/(1+slope*slope+tau)
    expected = (np.log(((1+a*x*c+b*x*y)**2+tau)/mass)
                -.5*(x*x+y*y+2*math.log(2*math.pi)))
    cdf_error = float(np.max(np.abs(np.column_stack([fx, fy])-u)))
    density_error = float(np.max(np.abs(result["log_density"].numpy()-expected)))
    quadrature = max(abs(quad(
        lambda yy: (1+(a*cc+b*yy)**2+tau)*math.exp(-yy*yy/2)/math.sqrt(2*math.pi),
        -np.inf, np.inf, epsabs=1e-11)[0]-mm) for cc, mm in zip(c, mass))
    offset, matrix = np.array([.7, -.3]), np.array([[1.5, .4], [-.2, .9]])
    proposal = AffineTTProposal(tr, offset, matrix, [-.3], [[1.7]])
    physical = proposal.sample(tf.constant((-.3+1.7*c)[:, None], DT), tf.constant(u, DT), jit_compile=False)
    point_error = float(np.max(np.abs(physical["points"].numpy()-(offset+points@matrix.T))))
    jacobian_error = float(np.max(np.abs(physical["log_density"].numpy()
                                          -(expected-np.linalg.slogdet(matrix)[1]))))
    marginal = tr.marginal_log_density((0, 1), tf.constant(points, DT)).numpy()
    expected_marginal = (np.log(((1+b*x*y)**2+a*a*x*x+tau)/(1+a*a+b*b+tau))
                         -.5*(x*x+y*y+2*math.log(2*math.pi)))
    marginal_error = float(np.max(np.abs(marginal-expected_marginal)))
    return {"amplitude": "1+0.4*x*c+0.2*x*y", "conditional_mass_quadrature_error": quadrature,
            "independent_upper_cdf_error": cdf_error, "log_density_error": density_error,
            "physical_point_error": point_error, "physical_jacobian_error": jacobian_error,
            "marginal_log_density_error": marginal_error,
            "pass": bool(max(quadrature, cdf_error, density_error, point_error, jacobian_error, marginal_error)<2e-10)}


def numerical_vs_smooth_law():
    tr = hd.FixedTTSIRTTransport(reference.bounded_density(), hd.KRCDFConfig(9, 48, 1e-12, 1e-12, 1e-12, 0))
    c, x = tf.constant([[.3, -.5]], DT), tf.constant([[.13, -.38]], DT)
    numerical = tf.exp(tr.conditional_proposal_log_density_suffix(conditioning_points=c, generated_points=x)).numpy()
    derivative = ((tr.conditional_forward_transport_suffix(c, x+1e-6)
                   -tr.conditional_forward_transport_suffix(c, x-1e-6))/(2e-6)).numpy()[0]
    joint = tf.transpose(tf.concat([x, c], axis=0))
    smooth = (tf.exp(tr.density.log_density(joint))
              / tr.density.normalized_marginal_density_values((1,), tf.transpose(c))/2).numpy()
    return {"numerical_density": numerical.tolist(), "smooth_TT_density": smooth.tolist(),
            "numerical_jacobian_error": float(np.max(np.abs(numerical-derivative))),
            "smooth_TT_density_gap": float(np.max(np.abs(numerical-smooth))),
            "different_law_reproduced": bool(np.max(np.abs(numerical-smooth))>1e-4),
            "pass": bool(np.max(np.abs(numerical-derivative))<2e-9)}


def c2_model_and_full_frozen_score():
    coupling = np.array([[0., .12], [-.04, 0.]])
    sigma, theta = .8, np.array([.45, -.2])
    model = C2StochasticVolatilityFrozenAPFModel(coupling, sigma)
    observations = np.array([[.2, -.3], [.6, .1], [-.4, .7]])
    plain = lambda axes, cond: GaussianHermiteTTTransport(
        tuple(tf.ones([1, 1, 1], DT) for _ in range(axes)), (0, 1), cond, .1)
    initial = AffineTTProposal(plain(2, ()), np.zeros(2), np.eye(2), np.zeros(0), np.zeros((0, 0)))
    transition = AffineTTProposal(plain(4, (2, 3)), np.zeros(2), np.eye(2), np.zeros(2), np.eye(2))
    uniforms = np.linspace(.1, .9, 42).reshape(3, 7, 2)
    program = compile_algorithm3(initial, [transition, transition], observations, uniforms[0], uniforms[1:], jit_compile=False)
    result = program.evaluate(model, theta, jit_compile=False)
    x, logq = program.states.numpy(), program.proposal_log_densities.numpy()

    def reference_value(parameters):
        gamma, xi = parameters
        transition_matrix = coupling+gamma*np.eye(2)
        covariance = solve_discrete_lyapunov(transition_matrix, sigma*sigma*np.eye(2))
        prior = -.5*(2*math.log(2*math.pi)+np.linalg.slogdet(covariance)[1]
                     +np.einsum("ni,ij,nj->n", x[0], np.linalg.inv(covariance), x[0]))
        residual = x[1:]-x[:-1]@transition_matrix.T
        transitions = -.5*(2*math.log(2*math.pi*sigma*sigma)+np.sum(residual**2, axis=2)/(sigma*sigma))
        observation = np.sum(-.5*math.log(2*math.pi)-xi-.5*x
                             -.5*observations[:, None, :]**2*np.exp(-x-2*xi), axis=2)
        factors = np.concatenate([prior[None, :], transitions])+observation-logq
        return float(logsumexp(factors.sum(axis=0))-math.log(7))

    fd = np.array([(reference_value(theta+np.eye(2)[j]*1e-5)
                    -reference_value(theta-np.eye(2)[j]*1e-5))/2e-5 for j in range(2)])
    covariance, derivative = model.stationary_covariance_and_derivative(tf.constant(theta, DT))
    a = coupling+theta[0]*np.eye(2)
    cov_reference = solve_discrete_lyapunov(a, sigma*sigma*np.eye(2))
    delta = np.eye(2)*1e-5
    dot_reference = (solve_discrete_lyapunov(a+delta, sigma*sigma*np.eye(2))
                     -solve_discrete_lyapunov(a-delta, sigma*sigma*np.eye(2)))/2e-5
    errors = {"covariance_error": float(np.max(np.abs(covariance.numpy()-cov_reference))),
              "covariance_derivative_error": float(np.max(np.abs(derivative.numpy()-dot_reference))),
              "value_error": abs(float(result["log_likelihood"])-reference_value(theta)),
              "score_error": float(np.max(np.abs(result["score"].numpy()-fd)))}
    return {**errors, "score": result["score"].numpy().tolist(), "independent_finite_difference": fd.tolist(),
            "covariance_derivative_max": float(np.max(np.abs(derivative.numpy()))),
            "parameters": theta.tolist(), "sigma": sigma, "coupling": coupling.tolist(),
            "pass": max(errors.values())<2e-7,
            "scope": "real C2 model wired to generic compiler in this diagnostic; no C2 campaign runner or fit-quality claim"}


def consumer_inventory():
    names = {"compile_algorithm3", "prepare_algorithm2_proposals", "likelihood_weighted_sigma_point_chart"}
    search = subprocess.run(["rg", "--no-ignore", "-l", "--glob", "*.py", "--glob", "!**/artifacts/**",
                             "|".join(sorted(names)), "bayesfilter", "tests", "docs/benchmarks"],
                            cwd=REPO, check=True, capture_output=True, text=True)
    calls = {name: [] for name in sorted(names)}
    for path in search.stdout.splitlines():
        tree = ast.parse((REPO/path).read_text())
        aliases = {name: name for name in names}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in names:
                        aliases[alias.asname or alias.name] = alias.name
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                called = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
                if called in aliases:
                    calls[aliases[called]].append(f"{path}:{node.lineno}")
    return {"call_sites": calls, "classification": "static named-consumer inventory; diagnostic files excluded",
            "guide_has_runtime_consumer": any(not p.startswith("tests/") for p in calls["likelihood_weighted_sigma_point_chart"]),
            "algorithm3_has_benchmark_consumer": any(p.startswith("docs/benchmarks/") for p in calls["compile_algorithm3"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    checks = {}
    for name, function in [("coupled_upper_law", coupled_upper_law),
                           ("numerical_vs_smooth_law", numerical_vs_smooth_law),
                           ("c2_model_and_full_frozen_score", c2_model_and_full_frozen_score),
                           ("consumer_inventory", consumer_inventory)]:
        try:
            checks[name] = function()
        except Exception as exc:
            checks[name] = {"diagnostic_error": type(exc).__name__+": "+str(exc)}
    files = ["bayesfilter/highdim/"+name for name in ["zhao_cui_algorithm3_tf.py",
             "gaussian_hermite_tt_transport_tf.py", "zhao_cui_algorithm2_preparation_tf.py",
             "transport.py", "c2_sv_frozen_proposal_apf_tf.py", "squared_tt_engine_v0_tf.py"]]
    result = {"role": "independent CPU/reference audit; no scientific promotion",
              "cpu_only": True, "gpu_devices_intentionally_hidden": True, "jit_compile": False,
              "tensorflow": tf.__version__, "command": sys.argv, "checkout": str(REPO),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
              "wall_seconds": time.monotonic()-started,
              "source_sha256": {p: hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in files},
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "checks": checks}
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"artifact": str(args.output), "wall_seconds": result["wall_seconds"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
