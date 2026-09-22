"""Bounded document and exact-arithmetic checks for the master amendment.

This is an independent reference check.  It does not import BayesFilter,
TensorFlow, or NumPy and does not establish filter-runtime readiness.
"""

from fractions import Fraction as F
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
MASTER = ROOT / "docs/plans/younis-kdm-score-master-program-2026-09-14.md"
ARTIFACT = Path(__file__).resolve().parent


def require(text, needle, label):
    if needle not in text:
        raise AssertionError(f"missing {label}: {needle}")


def matrix_transpose_times(v, s):
    return [sum(v[i][j] * s[i] for i in range(len(v))) for j in range(len(v[0]))]


def exact_math_checks():
    nodes = range(-5, 6)
    s2 = sum(F(j) ** 2 for j in nodes)
    s4 = sum(F(j) ** 4 for j in nodes)
    s6 = sum(F(j) ** 6 for j in nodes)
    det = s2 * s6 - s4 * s4
    cubic = {j: (s6 * j - s4 * j**3) / det for j in nodes}
    stencils = {
        "D2": ({-1: -F(1, 2), 0: F(0), 1: F(1, 2)}, 2, F(1, 6)),
        "D4": ({-2: F(1, 12), -1: -F(2, 3), 0: F(0), 1: F(2, 3), 2: -F(1, 12)}, 4, -F(1, 30)),
        "cubic": (cubic, 4, -F(143, 90)),
    }
    stencil_results = {}
    for name, (weights, order, leading) in stencils.items():
        moments = {k: sum(c * F(j) ** k for j, c in weights.items()) for k in range(order + 2)}
        assert moments[1] == 1
        assert all(moments[k] == 0 for k in range(order + 1) if k != 1)
        factorial = 1
        for k in range(1, order + 2):
            factorial *= k
        assert moments[order + 1] / factorial == leading
        assert sum(weights.values()) == 0
        stencil_results[name] = {"leading": str(leading), "moments": {str(k): str(v) for k, v in moments.items()}}

    # The MSE expansion retains the truncation/particle-bias cross term.
    t, b, variance = F(2), F(-1), F(3)
    assert (t + b) ** 2 + variance == 4
    assert t * t + b * b + variance == 8

    # Exact vector-combination coefficient, checked against direct quadratic expansion.
    a = [F(3), F(-1)]
    bb = [F(1), F(2)]
    score = [F(2), F(0)]
    delta = [a[i] - bb[i] for i in range(2)]
    numerator = sum(delta[i] * (bb[i] - score[i]) for i in range(2))
    denominator = sum(x * x for x in delta)
    alpha = -numerator / denominator
    direct = sum((alpha * a[i] + (1 - alpha) * bb[i] - score[i]) ** 2 for i in range(2))
    assert direct == min(
        sum((x * a[i] + (1 - x) * bb[i] - score[i]) ** 2 for i in range(2))
        for x in (alpha, alpha - F(1, 1000), alpha + F(1, 1000))
    )

    # Rectangular reconstruction uses V V^T; V^T V is singular here.
    v = [[F(1), F(0), F(1)], [F(0), F(1), F(2)]]
    s = [F(3), F(-2)]
    rhs_data = matrix_transpose_times(v, s)
    gram = [[sum(v[i][k] * v[j][k] for k in range(3)) for j in range(2)] for i in range(2)]
    rhs = [sum(v[i][j] * rhs_data[j] for j in range(3)) for i in range(2)]
    gram_det = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0]
    reconstructed = [
        (gram[1][1] * rhs[0] - gram[0][1] * rhs[1]) / gram_det,
        (gram[0][0] * rhs[1] - gram[1][0] * rhs[0]) / gram_det,
    ]
    assert reconstructed == s
    null = [F(-1), F(-2), F(1)]
    assert all(sum(v[i][j] * null[j] for j in range(3)) == 0 for i in range(2))

    return {
        "status": "pass",
        "stencils": stencil_results,
        "mse_cross_term": {"correct": "4", "separate_squares": "8"},
        "vector_combination_alpha": str(alpha),
        "rectangular_reconstruction": {"score": [str(x) for x in reconstructed], "V_shape": [2, 3], "VtV_singular_witness": [str(x) for x in null]},
    }


def document_checks(text):
    required = [
        "Active execution scope, amended 2026-09-14",
        "### KDM expectation gradient",
        "### UKF, KDM, and the covariance question",
        "## Phase 4C: fixed symmetric directional finite-difference ladder",
        "\\operatorname{MSE}(\\widehat D_h)",
        "There is no universal stochastic slope or U-shaped MSE curve.",
        "f_v\\in C^5",
        "(VV^\\mathsf T)^{-1}Vb",
        "An unrestricted normal draw for \\(h\\) is excluded",
        "## Phase 5 (deferred): regular-transition smoothing",
        "## Phase 6 (deferred): degenerate-transition support program",
        "## Open prerequisites and amendment review",
    ]
    for needle in required:
        require(text, needle, needle)
    phase4c = text[text.index("## Phase 4C"):text.index("## Phase 5")]
    forbidden_active = [
        "four continuous derivatives are sufficient",
        "a universal U-shaped",
        "V^\\mathsf T V)^{-1}V",
        ",qquad",
    ]
    for needle in forbidden_active:
        if needle in phase4c:
            raise AssertionError(f"stale active Phase 4C assertion: {needle}")
    for gate in "ABCDEFGH":
        require(text, f"**Gate {gate}", f"Gate {gate}")
    required_headings = [
        "## Phase 0:", "## Phase 1:", "## Phase 2:", "## Phase 3:",
        "## Phase 4:", "## Phase 4B:", "## Phase 4C:", "## Phase 7:",
        "## Phase 8:", "## Phase 9:",
    ]
    for heading in required_headings:
        require(text, heading, heading)
    return {
        "status": "pass",
        "line_count": len(text.splitlines()),
        "phase4c_checked": True,
        "required_anchors": len(required) + 8 + len(required_headings),
    }


def main():
    text = MASTER.read_text()
    result = {"document": document_checks(text), "exact_math": exact_math_checks()}
    result["scope"] = {
        "filter_runtime_checked": False,
        "gpu_used": False,
        "mathdevmcp_audit_run": False,
        "role": "document_and_independent_reference_check_only",
    }
    destination = ARTIFACT / "validation.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"PASS: document anchors and exact arithmetic checks; {destination}")


if __name__ == "__main__":
    main()
