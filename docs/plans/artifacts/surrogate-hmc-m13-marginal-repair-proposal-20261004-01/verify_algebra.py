"""Independent CPU-only symbolic diagnostics; no runtime algorithm imports."""
import hashlib, json, pathlib, subprocess, time
import sympy as s
from fractions import Fraction as F

out = pathlib.Path(__file__).resolve().parent
start = time.monotonic()
a,b,c,da,db,dc,ea,eb,dea,deb = s.symbols("a b c da db dc ea eb dea deb", real=True)
Q = s.Matrix([[a,b],[b,c]])
DQ = s.Matrix([[da,db],[db,dc]])
e = s.Matrix([ea,eb])
De = s.Matrix([dea,deb])
logf = -s.log(Q.det())/2 - (e.T*Q.inv()*e)[0]/2
direct = sum(s.diff(logf,v)*dv for v,dv in
             [(a,da),(b,db),(c,dc),(ea,dea),(eb,deb)])
u = Q.inv()*e
analytic = -(u.T*De)[0] + (u.T*DQ*u)[0]/2 - s.trace(Q.inv()*DQ)/2
gaussian_zero = s.factor(direct-analytic) == 0

t = s.symbols("t", real=True)
M1 = s.Matrix([[1+t/3,s.Rational(1,5)],[-s.Rational(1,7),1-t/4]])
M2 = s.Matrix([[1-t/6,-s.Rational(1,3)],[s.Rational(1,2),1+t/8]])
d1 = s.Matrix([t/7,s.Rational(1,4)])
d2 = s.Matrix([-t/5,s.Rational(1,9)+t/6])
X = s.Matrix([1+t/5,-1+t/8])
B = M2*M1
offset = M2*d1+d2
DX = X.diff(t)
v = B.inv()*(X-offset)
inverse_diff = s.simplify(v.diff(t)-B.inv()*(DX-offset.diff(t)-B.diff(t)*v))
checks = {
 "gaussian_symmetric_2x2_total_derivative": gaussian_zero,
 "noncommuting_test_matrices": (M1*M2-M2*M1).subs(t,0) != s.zeros(2),
 "affine_composition": s.simplify(M2*(M1*X+d1)+d2-(B*X+offset)) == s.zeros(2,1),
 "affine_matrix_tangent": s.simplify(B.diff(t)-M2.diff(t)*M1-M2*M1.diff(t)) == s.zeros(2),
 "affine_offset_tangent": s.simplify(offset.diff(t)-M2.diff(t)*d1-M2*d1.diff(t)-d2.diff(t)) == s.zeros(2,1),
 "inverse_map_total_tangent": inverse_diff == s.zeros(2,1),
}
qa1, qb1, qa2, qb2 = F(9,11), F(2,11), F(1,19), F(18,19)
wa, wb = F(209,182), F(209,118)
mean = ((qa1+qa2)*wa+(qb1+qb2)*wb)/2
variance = (qa1*qb1+qa2*qb2)*(wa-wb)**2/4
checks["counterexample_mean"] = mean == F(3,2)
checks["counterexample_variance"] = variance == F(555264,28826161)
checks["counterexample_positive_variance"] = variance > 0
counterexample = {
 "state_space":["A","B"],"g":["1","2"],"f1":["9/10","1/10"],
 "f2":["1/10","9/10"],"component_weights":["1/2","1/2"],
 "q1":["9/11","2/11"],"q2":["1/19","18/19"],
 "sampling_law":"one independent draw per component, average the two weights",
 "original_estimator":"3/2 identically", "original_variance":"0",
 "marginal_estimator_mean":str(mean),"marginal_estimator_variance":str(variance),
 "marginal_estimator_variance_decimal":float(variance),
 "conclusion":"Universal variance non-deterioration for stratified sampling is false.",
}
(out/"stratified-counterexample.json").write_text(json.dumps(counterexample,indent=2)+"\n")
payload = {"lhs":str(direct),"rhs":str(analytic),
 "assumptions":["a > 0","a*c-b**2 > 0",
 "a,b,c,da,db,dc,ea,eb,dea,deb are real"],
 "source":"Exact symbolic differentiation of the 2x2 Gaussian log density; constants omitted."}
(out/"gaussian-2x2-mathdev-input.json").write_text(json.dumps(payload,indent=2)+"\n")
report = {"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,
 "backend":"SymPy "+s.__version__+" and Python fractions",
 "execution":"CPU-only symbolic diagnostic; no TensorFlow/CUDA imported",
 "wall_seconds":time.monotonic()-start}
(out/"local-algebra-report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report))
if not all(checks.values()): raise SystemExit(1)
