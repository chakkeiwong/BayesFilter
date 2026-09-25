"""Exact reference and manuscript-preservation checks; no runtime admission."""

from collections import Counter
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import re


root = Path(__file__).resolve().parent
repo = root.parents[3]
name = "ledh_younis_kdm_score"
source = repo / "docs" / "papers" / name
before = (root / "baseline" / f"{name}.tex").read_text()
after = (source / f"{name}.tex").read_text()

phi1, phi2, sigma = F(3, 4), F(-1, 5), F(1, 2)
parents = [(F(-2), F(1)), (F(0), F(-1)), (F(3), F(2))]
innovations = [F(1, 3), F(-1, 2), F(2, 5)]
children = [
    (phi1 * a + phi2 * b + sigma * e, a)
    for (a, b), e in zip(parents, innovations)
]
support_mask = [[child[1] == parent[0] for parent in parents] for child in children]
assert support_mask == [[i == j for j in range(3)] for i in range(3)]
duplicate_parents = [parents[0], (parents[0][0], F(7)), parents[2]]
duplicate_mask = [children[0][1] == p[0] for p in duplicate_parents]
assert duplicate_mask == [True, True, False]

q_one = [[sigma**2, F(0)], [F(0), F(0)]]
q_two = [
    [sigma**2 * (1 + phi1**2), phi1 * sigma**2],
    [phi1 * sigma**2, sigma**2],
]
det = lambda a: a[0][0] * a[1][1] - a[0][1] * a[1][0]
assert det(q_one) == 0
assert det(q_two) == sigma**4 > 0

display_pattern = re.compile(
    r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}"
    r".*?\\end\{\1\}|\\\[.*?\\\]",
    re.S,
)
displays = lambda text: Counter(m.group() for m in display_pattern.finditer(text))
old_math, new_math = displays(before), displays(after)
assert not (old_math - new_math), "An existing displayed derivation changed or disappeared."
labels = lambda text: set(re.findall(r"\\label\{([^}]+)\}", text))
old_labels, new_labels = labels(before), labels(after)
assert old_labels <= new_labels


def citations(text):
    matches = re.findall(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}", text)
    return {key.strip() for match in matches for key in match.split(",")}


old_citations, new_citations = citations(before), citations(after)
assert old_citations <= new_citations
assert (root / "baseline" / f"{name}.bib").read_bytes() == (source / f"{name}.bib").read_bytes()
assert not [c for c in after if ord(c) < 32 and c not in "\n\r\t"]

result = {
    "status": "PASS",
    "role": "exact mathematical reference and document preservation, not empirical score validation",
    "backend": "Python standard library, rational arithmetic",
    "gpu": "not initialized; no numerical framework imported",
    "parameters": {"phi1": str(phi1), "phi2": str(phi2), "sigma": str(sigma)},
    "children": [[str(x) for x in row] for row in children],
    "compatible_ancestors_rows_are_children": support_mask,
    "duplicate_coordinate_compatibility": duplicate_mask,
    "one_step_covariance_determinant": str(det(q_one)),
    "two_step_covariance": [[str(x) for x in row] for row in q_two],
    "two_step_covariance_determinant": str(det(q_two)),
    "scope": "Distinct lag coordinates force single ancestry; duplicate coordinates need not. "
             "A singular one-step conditional does not imply a singular exact predictive marginal.",
    "preservation": {
        "old_display_blocks": sum(old_math.values()),
        "new_display_blocks": sum(new_math.values()),
        "all_old_displays_unchanged": True,
        "old_labels": len(old_labels),
        "new_labels": len(new_labels),
        "all_old_labels_retained": True,
        "old_citation_keys": len(old_citations),
        "new_citation_keys": len(new_citations),
        "all_old_citations_retained": True,
        "bibliography_unchanged": True,
    },
    "source_sha256": hashlib.sha256((source / f"{name}.tex").read_bytes()).hexdigest(),
}
(root / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"status": result["status"], "preservation": result["preservation"],
                  "artifact": str(root / "verification.json")}, indent=2))
