"""Deterministic document audit using MathDevMCP's public obligation checker.

SymPy constructs derivative and matrix expressions. MathDevMCP independently
compares the resulting scalar sides. These are algebraic reductions of the
cited chapter equations, not proofs of the probabilistic assumptions.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sympy as sp
from mathdevmcp.proof_obligations import check_proof_obligation

ROOT = Path(__file__).resolve().parents[6]
SOURCE = ROOT / "docs/chapters/ch26g_modern_hmc_methods.tex"
parser = argparse.ArgumentParser()
parser.add_argument("--output-tag", required=True)
args = parser.parse_args()
assert re.fullmatch(r"[A-Za-z0-9_-]+", args.output_tag)
OUTPUT = Path(__file__).parent / args.output_tag
OUTPUT.mkdir(exist_ok=True)
OUT = OUTPUT / "symbolic-obligations.json"
assert not OUT.exists(), "Use a fresh output tag"
rows = []


def check(name, label, lhs, rhs, domain):
    left, right = str(lhs), str(rhs)
    if max(len(left), len(right)) > 500:
        raise ValueError(f"Bounded MathDevMCP grammar exceeded: {name}")
    result = check_proof_obligation(left, right, assumptions=[domain], backend="sympy")
    rows.append({"name": name, "chapter_label": label, "domain": domain,
                 "result": result})
    print(name, result["status"], flush=True)


t, c, k, h, y, b, r, a, v, p, q, gx, gy, z, C, D, tau, kap = sp.symbols(
    "t c k h y b r a v p q gx gy z C D tau kap")
den = ((1+c)*sp.exp(t)+(1-c)*sp.exp(-t))/2
num = ((1+c)*sp.exp(t)-(1-c)*sp.exp(-t))/2
check("sphere norm", "eq:bf-modern-force-norm-identity",
      1-c*c+num*num, den*den, "t real; -1 <= c <= 1")
check("force projection ODE", "eq:bf-modern-force-projection",
      sp.diff(num/den,t), 1-(num/den)**2, "t real; -1 <= c <= 1; denominator > 0")
check("surface Jacobian integral", "eq:bf-modern-force-jacobian",
      sp.diff(den,t)/den, num/den, "same domain; t=k*h")
check("force inverse projection", "eq:bf-modern-force-projection",
      ((-sp.sinh(t)+(num/den)*sp.cosh(t))/
       (sp.cosh(t)-(num/den)*sp.sinh(t))).rewrite(sp.exp),
      c, "real t; -1 <= c <= 1")
check("stable force denominator", "eq:bf-modern-stable-force-denominator",
      ((sp.exp(t)+sp.exp(-t))/2+c*(sp.exp(t)-sp.exp(-t))/2), den, "real t,c")
check("OU Gaussian exponent balance", "eq:bf-modern-ou-balance",
      p*p+(q-a*p)**2/(1-a*a),
      q*q+(p-a*q)**2/(1-a*a), "unit mass; real p,q; abs(a)<1")
check("MALT local energy error", "eq:bf-modern-malt-local-error",
      ((z/h-h*gy/2)**2-(z/h+h*gx/2)**2)/2,
      -z*(gy+gx)/2+h*h*(gy*gy-gx*gx)/8, "h nonzero; scalar reduction per coordinate")

A = sp.Matrix([[1-t*t/2,t],[-t*(1-t*t/4),1-t*t/2]])
Q = A.T*A-sp.eye(2)
check("leapfrog determinant", "eq:bf-modern-gaussian-leapfrog-matrix",
      A.det(), 1, "dimensionless q=x/sigma; t=h/sigma real")
check("energy error Q11", "eq:bf-modern-gaussian-error-entries",
      Q[0,0], -t**4*(1-t*t/4)/4, "dimensionless q")
check("energy error Q12", "eq:bf-modern-gaussian-error-entries",
      Q[0,1], t**3*(1-t*t/2)/4, "dimensionless q")
check("energy error Q22", "eq:bf-modern-gaussian-error-entries",
      Q[1,1], t**4/4, "dimensionless q")
check("unadjusted stationary variance", "eq:bf-modern-gaussian-biased-variance",
      t*t/(1-(1-t*t/2)**2), 1/(1-t*t/4), "0<t*t<4")
check("energy error variance", "eq:bf-modern-gaussian-eevpd",
      y**4/16+y**3*(1-y/2)**2/(16*(1-y/4)),
      y**3/(16*(1-y/4)), "0<y<4")
check("bias/error conversion", "eq:bf-modern-gaussian-eevpd",
      (4*b/(1+b))**3/(16*(1-b/(1+b))),
      4*b**3/(1+b)**2, "b>0; y=4*b/(1+b)")
# Use r=sqrt(D)>0 so the domain of fractional powers is explicit.
F = 4*r**3/(1+r)**2
fp = sp.diff(F,r)/(2*r)
check("F first derivative", "eq:bf-modern-bias-function-derivatives",
      fp, 2*r*(3+r)/(1+r)**3, "r=sqrt(D)>0")
check("F second derivative", "eq:bf-modern-bias-function-derivatives",
      sp.diff(2*r*(3+r)/(1+r)**3,r)/(2*r),
      (3-4*r-r*r)/(r*(1+r)**4), "r=sqrt(D)>0")
check("convexity boundary", "eq:bf-modern-bias-function-derivatives",
      (sp.sqrt(7)-2)**2, 11-4*sp.sqrt(7), "positive root in r is sqrt(7)-2")
check("LAPS rate derivative", "eq:bf-modern-laps-rate",
      sp.diff(h*(c*D-(2-c)*C*h**kap)/tau,h),
      (c*D-(kap+1)*(2-c)*C*h**kap)/tau,
      "h,tau,C,kap,D>0 and 0<c<1")
check("LAPS contraction after optimum", "eq:bf-modern-laps-ideal-schedule",
      (1-c)*D+(2-c)*c*D/((2-c)*(kap+1)),
      (1-c*kap/(kap+1))*D, "kap>0; 0<c<1")
check("equal-variance mixture fourth moment", "eq:bf-modern-equipartition-gaussian",
      a**4+6*a*a*(1-a*a)+3*(1-a*a)**2,
      3-2*a**4, "0<a<1; symmetric mixture means +/-a, variance 1-a*a")
check("funnel density cancellation", "eq:bf-modern-funnel-whitened",
      -v*v/2 - k*v/2-sp.exp(-v)*(sp.exp(v/2)*z)**2/2+k*v/2,
      -v*v/2-z*z/2, "one child contribution; k children in log determinant")
check("partial funnel scale", "eq:bf-modern-funnel-partial",
      sp.exp(-v)*sp.exp(v/2-D)**2,
      sp.exp(-2*D), "real v,D; D=delta(v)")
check("spectral Jensen convexity", "eq:bf-modern-esjd-bound",
      sp.diff((1+t)/(1-t),t,2), 4/(1-t)**3, "-1<t<1")
check("Gaussian square IAT", "eq:bf-modern-gaussian-iat",
      1+2*r*r/(1-r*r), (1+r*r)/(1-r*r), "abs(r)<1")
check("random length cosine integral", "eq:bf-modern-random-length-correlations",
      sp.integrate(sp.cos(t),(t,0,2*a))/(2*a),
      sp.sin(2*a)/(2*a), "a>0")
check("random length square integral", "eq:bf-modern-random-length-correlations",
      sp.integrate(sp.cos(t)**2,(t,0,2*a))/(2*a),
      sp.Rational(1,2)+sp.sin(4*a)/(8*a), "a>0")
J, H, R, S, x, m, u, G = sp.symbols("J H R S x m u G")
check("Gaussian Schur square expansion", "eq:bf-modern-gaussian-schur",
      -R*x*x/2+H*x+(u-J*x)**2/(2*S),
      -(R-J*J/S)*x*x/2+(H-J*u/S)*x+u*u/(2*S),
      "S>0; scalar reduction of symmetric quadratic form")
check("Kalman covariance differential", "eq:bf-modern-kalman-score-differential",
      sp.diff(-(sp.log(S)+v*v/S)/2,S),
      -1/(2*S)+v*v/(2*S*S), "S>0; scalar innovation")
check("GRAD conditional covariance", "eq:bf-modern-grad-gaussian-product",
      1/(1/q+1/r), r*q/(q+r), "q>0,r>0")
check("GRAD conditional mean", "eq:bf-modern-grad-gaussian-product",
      (m/q+u/r)/(1/q+1/r), r*m/(q+r)+q*u/(q+r), "q>0,r>0")
check("GRAD log auxiliary weight", "eq:bf-modern-grad-conditional-weight",
      -((u-x-r*G)**2-(u-x)**2)/(2*r),
      G*(u-x)-r*G*G/2, "r>0; scalar coordinate reduction")
check("lugsail leading bias", "eq:bf-modern-lugsail-bias",
      (G/b-c*G/(b/r))/(1-c),
      (1-r*c)/(1-c)*G/b, "b>0, r>1, 0<=c<1")
check("AR1 first weighted sum", "eq:bf-modern-ar1-batch-bias",
      -2*G*r*sp.diff(1/(1-r),r),
      -2*G*r/(1-r)**2, "0<r<1; G=gamma0")
check("nested independent variance", "eq:bf-modern-nested-decomposition",
      k*v/k**2, v/k, "k=M positive; conditional independence is separately assumed")

negative_control = check_proof_obligation("1+1", "3", backend="sympy")
payload = {
    "source": str(SOURCE),
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "method": "MathDevMCP proof_obligations.check_proof_obligation, SymPy backend",
    "limits": [
        "Scalar algebra and finite-dimensional reductions only.",
        "Domain assumptions are recorded and manually checked; not formally discharged.",
        "No proof of convergence, mixing, finite-sample coverage or implementation speed.",
        "Probability and matrix arguments also receive manual source/derivation review."
    ],
    "checks": rows,
    "negative_control": negative_control,
}
OUT.write_text(json.dumps(payload,indent=2)+"\n")
assert negative_control["status"] == "mismatch"
assert all(x["result"]["status"] == "equivalent" for x in rows)
print(f"Saved {len(rows)} successful obligations to {OUT}")
