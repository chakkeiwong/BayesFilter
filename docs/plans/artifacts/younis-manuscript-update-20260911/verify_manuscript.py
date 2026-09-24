"""Independent document/reference checks; no filter run or GPU initialization."""
from pathlib import Path
from fractions import Fraction as F
import ast
import hashlib
import json
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DOC = ROOT / 'docs/papers/ledh_younis_kdm_score'
source = (DOC / 'ledh_younis_kdm_score.tex').read_text()
baseline = (HERE / 'baseline/ledh_younis_kdm_score.tex').read_text()

def math_blocks(s):
    return [m.group(0) for m in re.finditer(
        r'\\begin\{(equation\*?|align\*?|gather\*?|multline\*?)\}.*?\\end\{\1\}', s, re.S)]

normalize = lambda s: re.sub(r'\s+', '', s)
labels = lambda s: re.findall(r'\\label\{([^}]+)\}', s)
cite_keys = lambda s: {k.strip() for group in re.findall(
    r'\\cite\w*\*?(?:\[[^]]*\]){0,2}\{([^}]+)\}', s) for k in group.split(',')}
old_math, new_math = math_blocks(baseline), math_blocks(source)
unchanged = {normalize(s) for s in new_math}
preservation = {
    'baseline_labels': len(labels(baseline)), 'revised_labels': len(labels(source)),
    'missing_labels': sorted(set(labels(baseline))-set(labels(source))),
    'duplicate_labels': sorted({x for x in labels(source) if labels(source).count(x)>1}),
    'baseline_math_environments': len(old_math), 'revised_math_environments': len(new_math),
    'missing_or_changed_math_environments': [s for s in old_math if normalize(s) not in unchanged],
    'baseline_citations': sorted(cite_keys(baseline)),
    'new_citations': sorted(cite_keys(source)-cite_keys(baseline)),
    'missing_citations': sorted(cite_keys(baseline)-cite_keys(source)),
}
assert not preservation['missing_labels']
assert not preservation['duplicate_labels']
assert not preservation['missing_or_changed_math_environments']
assert not preservation['missing_citations']
bib = (DOC / 'ledh_younis_kdm_score.bib').read_text()
assert cite_keys(source) <= set(re.findall(r'@\w+\{([^,]+)', bib))
(HERE / 'preservation-inventory.json').write_text(json.dumps(preservation, indent=2)+'\n')

# Scalar Gaussian completion of the square, checked with exact fractions.
Q, R, H, m, y = map(F, [2, 3, 4, 1, 2])
V = H*Q*H+R
K = Q*H/V
mt = m+K*(y-H*m)
Qt = Q-Q*H*H*Q/V
assert 1/Q+H*H/R == 1/Qt
assert m/Q+H*y/R == mt/Qt
assert m*m/Q+y*y/R == (y-H*m)**2/V+mt*mt/Qt
assert Q*R == V*Qt

# Two-component hybrid with changing weights, component means, and integrand.
# w=(theta,1-theta); mu=(theta,2theta); phi_theta(z)=theta*z^2+z.
theta, h, c = F(1,3), F(1,2), F(7,5)
w, dw, slopes = [theta, 1-theta], [F(1), F(-1)], [F(1), F(2)]
hybrid = F(0)
for wi, dwi, slope in zip(w, dw, slopes):
    mu = slope*theta
    mean_phi = theta*(mu*mu+h*h)+mu
    mean_pathwise = 3*slope*slope*theta*theta+h*h+slope
    hybrid += wi*mean_pathwise+dwi*(mean_phi-c)
closed_derivative = -12*theta**3+12*theta**2-2*theta+h*h+2
assert hybrid == closed_derivative
assert sum(dw)*c == 0

# Re-run the earlier exact unbiased-normalizer, variance, and finite-state
# Fisher examples from a copied, preserved script in this output directory.
prior = ROOT/'docs/plans/artifacts/younis-score-recovery-20260911/verify_analysis.py'
copy = HERE/'verify_prior_examples.py'
copy.write_bytes(prior.read_bytes())
run = subprocess.run([sys.executable, str(copy)], cwd=HERE, text=True, capture_output=True, check=True)
(HERE/'prior-example-checks.log').write_text(run.stdout+run.stderr)
checks = {'role': 'independent exact-arithmetic reference, no framework import',
    'GPU': 'not initialized; no TensorFlow/JAX/PyTorch imports',
    'Gaussian_product': {'V':str(V),'conditional_mean':str(mt),'conditional_variance':str(Qt),'status':'PASS'},
    'two_component_hybrid': {'analytic_derivative':str(closed_derivative),'expected_estimator':str(hybrid),'status':'PASS'},
    'prior_examples_exit_code':run.returncode}
(HERE/'exact-reference-checks.json').write_text(json.dumps(checks,indent=2)+'\n')

# Executable static wiring assertions. They do not establish runtime parity.
paths = ['ledh_canonical_batch_fused_tf.py','ledh_canonical_batch_tf.py',
    'ledh_canonical_score_tf.py','ledh_unified_correction_tf.py',
    'ledh_younis_kdm_integrated_tf.py','ledh_younis_kdm_resampling_tf.py']
snapshot = '804616e320d940a5ea288ba25cc43bd21d267abb'
files = {}
trees = {}
for name in paths:
    rel = 'bayesfilter/highdim/'+name
    data = (ROOT/rel).read_bytes()
    trees[name] = ast.parse(data)
    old = subprocess.run(['git','show',f'{snapshot}:{rel}'],cwd=ROOT,capture_output=True)
    files[rel] = {'sha256':hashlib.sha256(data).hexdigest(),
        'identical_to_research_snapshot':old.returncode==0 and old.stdout==data}
fused = trees[paths[0]]
maps = [n for n in ast.walk(fused) if isinstance(n,ast.Call) and ast.unparse(n.func)=='tf.map_fn']
assert len(maps)==2
calls = [n for n in ast.walk(fused) if isinstance(n,ast.Call) and ast.unparse(n.func)=='canonical_value_and_analytical_score']
assert len(calls)==1 and any(k.arg is None and ast.unparse(k.value)=='common_kwargs' for k in calls[0].keywords)
kwargs = next(n.value for n in ast.walk(fused) if isinstance(n,ast.Assign)
    and any(isinstance(t,ast.Name) and t.id=='common_kwargs' for t in n.targets))
required = {'reset_policy','correction_steps','correction_trust_radius','pairwise_steps','pairwise_rms_cap','coordinate_cap'}
assert required <= {k.arg for k in kwargs.keywords}
assert any(isinstance(n,ast.ImportFrom) and n.module=='bayesfilter.highdim.ledh_canonical_score_tf'
    and any(a.name=='canonical_value_and_analytical_score' for a in n.names) for n in ast.walk(fused))
score = trees['ledh_canonical_score_tf.py']
assert any(isinstance(n,ast.Call) and ast.unparse(n.func)=='batched_higher_moment_shape_jvp' for n in ast.walk(score))
assert any(isinstance(n,ast.ImportFrom) and 'ledh_unified_correction_tf' in str(n.module)
    and any(a.name=='batched_higher_moment_shape_jvp' for a in n.names) for n in ast.walk(score))
legacy = trees['ledh_canonical_batch_tf.py']
legacy_calls = [n for n in ast.walk(legacy) if isinstance(n,ast.Call) and ast.unparse(n.func)=='canonical_value_and_analytical_score']
assert len(legacy_calls)==1 and not any(k.arg=='reset_policy' or k.arg is None for k in legacy_calls[0].keywords)
audit = {'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    'research_commit':snapshot, 'files':files,
    'static_wiring':'PASS', 'fused_tf_map_fn_calls':len(maps),
    'fused_reset_controls_forwarded':sorted(required),
    'runtime_call_chain_and_numerical_parity':'NOT CHECKED in this document revision',
    'interpretation':'Fused wrapper forwards controls but remains row-mapped; later shared correction differs from research snapshot.'}
(HERE/'implementation-wiring.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({'preserved_math_blocks':len(old_math),'preserved_labels':len(labels(baseline)),
    'exact_reference_checks':'PASS','static_wiring':'PASS','runtime_parity':'NOT CHECKED'}))
