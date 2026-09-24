"""Independent exact-arithmetic reference for a document review; no filter run."""

from fractions import Fraction as F
import json
from pathlib import Path


def check():
    nodes = range(-5, 6)
    s2, s4, s6 = [sum(F(j) ** k for j in nodes) for k in (2, 4, 6)]
    determinant = s2 * s6 - s4 * s4
    cubic = {j: (s6 * j - s4 * j ** 3) / determinant for j in nodes}
    stencils = {
        "central_three_point": ({-1: -F(1, 2), 0: F(0), 1: F(1, 2)}, 2, F(1, 6)),
        "central_five_point": ({-2: F(1, 12), -1: -F(2, 3), 0: F(0), 1: F(2, 3), 2: -F(1, 12)}, 4, -F(1, 30)),
        "eleven_point_cubic_fit": (cubic, 4, -F(143, 90)),
    }
    results = {}
    for name, (weights, order, leading) in stencils.items():
        moments = {k: sum(c * F(j) ** k for j, c in weights.items()) for k in range(order + 2)}
        assert moments[1] == 1
        assert all(moments[k] == 0 for k in range(order + 1) if k != 1)
        factorial = 1
        for k in range(1, order + 2):
            factorial *= k
        assert moments[order + 1] / factorial == leading
        # Perfectly shared additive noise cancels because all weights sum to zero.
        assert sum(weights.values()) ** 2 == 0
        results[name] = {
            "weights": {str(j): str(c) for j, c in weights.items()},
            "moments": {str(k): str(v) for k, v in moments.items()},
            "leading_taylor_coefficient": str(leading),
            "shared_additive_noise_variance": "0",
        }
    assert all(cubic[j] > 0 for j in range(1, 5)) and cubic[5] < 0
    # Existing ratio-bias witness: theta=2, a=1 gives 2/3 instead of 1/2.
    ratio_mean = (F(1, 3) + F(1, 1)) / 2
    assert ratio_mean == F(2, 3) and ratio_mean != F(1, 2)
    # MSE = (truncation + finite-N bias)^2 + noise variance, including cross term.
    truncation, log_bias, noise_variance = F(2), F(-1), F(3)
    correct_mse = (truncation + log_bias) ** 2 + noise_variance
    separate_squares = truncation ** 2 + log_bias ** 2 + noise_variance
    assert correct_mse == 4 and separate_squares == 8
    # V is 2 x 3, with nonorthogonal columns. Solve V^T s = directional.
    directions = [[F(1), F(0), F(1)], [F(0), F(1), F(2)]]
    score = [F(3), F(-2)]
    directional = [sum(directions[i][j] * score[i] for i in range(2)) for j in range(3)]
    gram = [[sum(directions[i][k] * directions[j][k] for k in range(3)) for j in range(2)] for i in range(2)]
    rhs = [sum(directions[i][j] * directional[j] for j in range(3)) for i in range(2)]
    det = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0]
    reconstructed = [(gram[1][1] * rhs[0] - gram[0][1] * rhs[1]) / det,
                     (gram[0][0] * rhs[1] - gram[1][0] * rhs[0]) / det]
    assert reconstructed == score
    # V^T V is singular: (-1,-2,1) is a nonzero null vector of V.
    null_vector = [F(-1), F(-2), F(1)]
    assert all(sum(directions[i][j] * null_vector[j] for j in range(3)) == 0 for i in range(2))
    return {
        "status": "pass",
        "role": "independent_document_derivation_reference_only",
        "backend": "Python standard library Fraction",
        "gpu_used": False,
        "filter_runtime_checked": False,
        "mathdevmcp_audit_run": False,
        "stencils": results,
        "ratio_bias_witness": {"expected_estimated_score": str(ratio_mean), "model_score": "1/2"},
        "mse_cross_term_witness": {"correct_mse": str(correct_mse), "incorrect_separate_squares": str(separate_squares)},
        "rectangular_reconstruction": {"direction_matrix_shape": [2, 3], "score": [str(v) for v in reconstructed], "incorrect_gram_matrix_is_singular": True},
    }


if __name__ == "__main__":
    destination = Path(__file__).with_name("derivation-checks.json")
    destination.write_text(json.dumps(check(), indent=2) + "\n")
    print("PASS: exact stencil moments and algebraic counterexamples; " + str(destination))
