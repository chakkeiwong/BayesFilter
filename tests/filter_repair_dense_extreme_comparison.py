"""Independent reference/comparator for only the approved huge D3/D5 fixtures.

NumPy and Decimal are diagnostic references, never runtime implementation.
The allowance belongs to raw_precision only; every field is still checked.
"""

import decimal

import numpy as np

FIELDS = frozenset(("raw_precision", "raw_eigenvalues", "design_condition", "design_rank",
    "minimum_eigenvalue", "maximum_eigenvalue", "precision_condition", "raw_spd",
    "selection_relative_rmse"))
MATRIX_RELATIVE_LIMIT = 1e-12
COMPARISON_ID = "huge_dense_d3_d5_matrix_relative_v1"


def exact_symmetric_fit(offsets, response, digits):
    """Solve tiny well-conditioned designs with exact input floats in Decimal."""
    with decimal.localcontext() as context:
        context.prec = digits
        x = [[decimal.Decimal.from_float(float(v)) for v in row] for row in offsets]
        y = [[decimal.Decimal.from_float(float(v)) for v in row] for row in response]
        n = len(x[0])
        a = [[sum(row[i] * row[j] for row in x) for j in range(n)] for i in range(n)]
        b = [[sum(x[k][i] * y[k][j] for k in range(len(x))) for j in range(n)] for i in range(n)]
        for i in range(n):
            pivot = max(range(i, n), key=lambda k: abs(a[k][i]))
            a[i], a[pivot] = a[pivot], a[i]
            b[i], b[pivot] = b[pivot], b[i]
            diagonal = a[i][i]
            assert diagonal != 0
            a[i] = [v / diagonal for v in a[i]]
            b[i] = [v / diagonal for v in b[i]]
            for k in range(n):
                if k == i:
                    continue
                multiplier = a[k][i]
                a[k] = [left - multiplier * right for left, right in zip(a[k], a[i], strict=True)]
                b[k] = [left - multiplier * right for left, right in zip(b[k], b[i], strict=True)]
        return np.array([[float((b[i][j] + b[j][i]) / 2) for j in range(n)] for i in range(n)])


def compare_huge_dense_records(actual, expected, *, dimension, case, offsets, response):
    """Check complete records plus independent fit accuracy under the owner rule."""
    assert case == "huge" and dimension in (3, 5), "unapproved comparison scope"
    assert actual.keys() == expected.keys() == FIELDS, "changed field set"
    x, y = np.asarray(offsets), np.asarray(response)
    assert x.shape == y.shape == (32, dimension)
    assert x.dtype == y.dtype == np.dtype("float64")
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert np.linalg.cond(x) < 2., "outside approved well-conditioned stress design"
    # Confirm this is the declared huge diagonal fixture, not an arbitrary
    # runtime precision record receiving a broad comparison exception.
    declared = np.diag(np.linspace(.7, 3., dimension)) * 1e150
    np.testing.assert_array_equal(y, x @ declared)
    exact = exact_symmetric_fit(x, y, 100)
    higher = exact_symmetric_fit(x, y, 160)
    np.testing.assert_array_equal(exact, higher)
    arrays = {}
    for name, record in (("original", expected), ("candidate", actual)):
        arrays[name] = {key: np.asarray(value) for key, value in record.items()}
        for key, value in arrays[name].items():
            shape = (dimension, dimension) if key == "raw_precision" else (dimension,) if key == "raw_eigenvalues" else ()
            assert value.shape == shape, (name, key, value.shape)
            kind = "i" if key == "design_rank" else "b" if key == "raw_spd" else "f"
            assert value.dtype.kind == kind, (name, key, value.dtype)
            assert np.isfinite(value).all(), (name, key, "nonfinite")
    left, right = arrays["candidate"], arrays["original"]
    assert left["raw_precision"].dtype == right["raw_precision"].dtype
    errors = {}
    scale = np.max(np.abs(right["raw_precision"]))
    assert scale > 0.
    relative = float(np.max(np.abs(left["raw_precision"] - right["raw_precision"])) / scale)
    assert relative <= MATRIX_RELATIVE_LIMIT, ("raw_precision", relative)
    for key in FIELDS - {"raw_precision"}:
        assert left[key].dtype == right[key].dtype, (key, "changed dtype")
        if key in ("design_rank", "raw_spd"):
            np.testing.assert_array_equal(left[key], right[key], err_msg=key)
        else:
            np.testing.assert_allclose(left[key], right[key], atol=1e-10, rtol=1e-10, err_msg=key)
    reference_scale = np.max(np.abs(exact))
    response_scale = np.max(np.abs(y))
    for name, values in arrays.items():
        reference_error = float(np.max(np.abs(values["raw_precision"] - exact)) / reference_scale)
        residual = float(np.linalg.norm((x @ values["raw_precision"] - y) / response_scale)
                         / np.linalg.norm(y / response_scale))
        assert reference_error <= MATRIX_RELATIVE_LIMIT, (name, "reference", reference_error)
        assert residual <= MATRIX_RELATIVE_LIMIT, (name, "response residual", residual)
        errors[name] = {"max_error_over_reference_scale": reference_error, "relative_response_residual": residual}
    return {"comparison_id": COMPARISON_ID, "raw_precision_matrix_relative_error": relative, "limit": MATRIX_RELATIVE_LIMIT,
            "precision_digits": [100, 160], "errors": errors}
