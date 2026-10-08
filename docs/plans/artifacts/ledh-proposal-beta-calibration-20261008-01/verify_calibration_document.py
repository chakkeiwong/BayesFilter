"""Diagnostic-only checks for the beta-selection documentation, not a filter."""
from pathlib import Path
import hashlib
import json
import re
import sympy as sp

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
body = (REPO / "docs/chapters/ledh_covariance_proposal_body.tex").read_text()
baseline = (ROOT / "baseline/ledh_covariance_proposal_body.tex").read_text()

def labelled_equations(text):
    matches = re.findall(r"\\begin\{(equation|align)\*?\}(.*?)\\end\{\1\*?\}", text, re.S)
    return {label: block for _, block in matches
            for label in re.findall(r"\\label\{([^}]+)\}", block)}

old, new = labelled_equations(baseline), labelled_equations(body)
assert len(old) == 49
assert all(new.get(label) == value for label, value in old.items())
assert len(new) == 57
old_algorithms = re.findall(r"\\begin\{lstlisting\}(.*?)\\end\{lstlisting\}", baseline, re.S)
new_algorithms = re.findall(r"\\begin\{lstlisting\}(.*?)\\end\{lstlisting\}", body, re.S)
assert len(old_algorithms) == 3 and len(new_algorithms) == 4
assert all(block in new_algorithms for block in old_algorithms)
labels = re.findall(r"\\label\{([^}]+)\}", body)
assert len(labels) == len(set(labels))

p, q, r, x, y, z, k, t = sp.symbols("p q r x y z k t", real=True)
den = p*x + q*y + r*z
symbolic = {
    "gradient": sp.simplify(sp.diff(k/den, p) + k*x/den**2) == 0,
    "hessian_direction": sp.simplify(sp.diff(k/(x+t*y), t, 2) - 2*k*y**2/(x+t*y)**3) == 0,
}
assert all(symbolic.values())

documents = {}
for directory, stem in [("monograph-build", "main"), ("standalone-build", "ledh_covariance_proposal_20261008")]:
    build = ROOT / directory
    log = (build / (stem + ".log")).read_text(errors="replace")
    aux = (build / (stem + ".aux")).read_text(errors="replace")
    fls = (build / (stem + ".fls")).read_text(errors="replace")
    bbl = (build / (stem + ".bbl")).read_text(errors="replace")
    assert "ledh_covariance_proposal_body.tex" in fls
    assert "HeOwen2014mixture" in bbl
    assert all("\\newlabel{" + label + "}" in aux for label in labels)
    assert "undefined references" not in log and "undefined citations" not in log
    assert "Rerun to get cross-references right" not in log
    assert "! LaTeX Error" not in log
    result = json.loads((build / "build-4-result.json").read_text())
    documents[directory] = {
        "shared_body_compiled": True, "all_shared_labels_resolved": True,
        "new_citation_resolved": True, "pending_reference_rerun": False,
        "pdf_sha256": hashlib.sha256((build / (stem + ".pdf")).read_bytes()).hexdigest(),
        "build_warning_count": len(result["warnings"]),
        "overfull_hboxes": result["overfull_hboxes"],
    }

result = {
    "role": "documentation and symbolic diagnostic only",
    "preserved_equations": len(old), "added_equations": len(new)-len(old),
    "preserved_algorithms": len(old_algorithms), "added_algorithms": 1,
    "all_shared_labels_unique": True,
    "direct_sympy_checks": symbolic,
    "symbolic_limit": "Rational identities on their nonzero-denominator domain; not proof of statistical assumptions or performance.",
    "documents": documents,
    "frameworks_imported": False, "gpu_used": False,
}
(ROOT / "document-verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
