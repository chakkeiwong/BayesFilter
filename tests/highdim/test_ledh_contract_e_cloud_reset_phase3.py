from __future__ import annotations

import importlib.util
import hashlib
import inspect
import json
import os
import re
import struct
from fractions import Fraction
from pathlib import Path
from typing import Any


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_reset_tf as reset


ROOT = Path(__file__).resolve().parents[2]
CERTIFICATE_PATH = ROOT / "docs/plans" / (
    "bayesfilter-contract-e-canonical-gradient-migration-"
    "phase3-exact-certificate-2026-07-13.json"
)
NONCOMMUTING_CERTIFICATE_PATH = ROOT / "docs/plans" / (
    "bayesfilter-contract-e-canonical-gradient-migration-"
    "phase3-noncommuting-certificate-2026-07-13.json"
)
PARITY_ARTIFACT_PATH = ROOT / "docs/plans" / (
    "bayesfilter-contract-e-canonical-gradient-migration-"
    "phase3-parity-diagnostics-2026-07-13.json"
)
RUN_MANIFEST_PATH = ROOT / "docs/plans/logs" / (
    "contract-e-canonical-gradient-migration-2026-07-13/phase3/run-manifest.json"
)
REFERENCE_PATH = ROOT / "docs/benchmarks/contract_e_reset_tf.py"
DTYPE = tf.float64
EXACT_CERTIFICATE_SHA256 = (
    "e81bb163c0952bdf0fb09402c4be75034b52ab801decfd18a5a356063f507a95"
)
NONCOMMUTING_CERTIFICATE_SHA256 = (
    "f08efae945e6aed1787d0fd3c1470a8502d45d40d2cfec74f3f88af3a3a6f138"
)
_RATIONAL = re.compile(r"^[+-]?\d+(?:/\d+)?$")


def _certificate() -> dict[str, Any]:
    return json.loads(CERTIFICATE_PATH.read_text(encoding="utf-8"))


def _noncommuting_certificate() -> dict[str, Any]:
    return json.loads(NONCOMMUTING_CERTIFICATE_PATH.read_text(encoding="utf-8"))


def _parity_artifact() -> dict[str, Any]:
    return json.loads(PARITY_ARTIFACT_PATH.read_text(encoding="utf-8"))


def _run_manifest() -> dict[str, Any]:
    return json.loads(RUN_MANIFEST_PATH.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fraction(value: str) -> float:
    return float(Fraction(value))


def _matrix(rows: list[list[str]]) -> tf.Tensor:
    return tf.constant(
        [[[ _fraction(value) for value in row] for row in rows]],
        dtype=DTYPE,
    )


def _weights(values: list[str]) -> tf.Tensor:
    return tf.constant([[_fraction(value) for value in values]], dtype=DTYPE)


def _batch_vector(values: list[str]) -> tf.Tensor:
    return tf.constant([_fraction(value) for value in values], dtype=DTYPE)


def _positive_zero(value: tf.Tensor) -> tf.Tensor:
    return tf.where(tf.equal(value, 0), tf.zeros_like(value), value)


def _assert_bitwise_equal(actual: tf.Tensor, expected: tf.Tensor) -> None:
    actual = _positive_zero(tf.convert_to_tensor(actual))
    expected = _positive_zero(tf.convert_to_tensor(expected))
    tf.debugging.assert_equal(tf.shape(actual), tf.shape(expected))
    integer_dtype = tf.int64 if actual.dtype == tf.float64 else tf.int32
    tf.debugging.assert_equal(
        tf.bitcast(actual, integer_dtype), tf.bitcast(expected, integer_dtype)
    )


def _tensor_sha256(values: dict[str, tf.Tensor]) -> str:
    payload = {
        name: {
            "dtype": value.dtype.name,
            "shape": value.shape.as_list(),
            "float_hex": [float(item).hex() for item in tf.reshape(value, [-1]).numpy()],
        }
        for name, value in sorted(values.items())
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _max_abs(value: tf.Tensor) -> float:
    return float(tf.reduce_max(tf.abs(value)).numpy())


def _ordered_binary64(value: float) -> int:
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    if bits == 1 << 63:
        bits = 0
    if bits & (1 << 63):
        return (~bits) & ((1 << 64) - 1)
    return bits | (1 << 63)


def _max_ulp_distance(left: tf.Tensor, right: tf.Tensor) -> int:
    left_values = tf.reshape(left, [-1]).numpy()
    right_values = tf.reshape(right, [-1]).numpy()
    return max(
        abs(_ordered_binary64(float(a)) - _ordered_binary64(float(b)))
        for a, b in zip(left_values, right_values, strict=True)
    )


def _assert_dyadic_certificate_literals(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _assert_dyadic_certificate_literals(item)
        return
    if isinstance(value, list):
        for item in value:
            _assert_dyadic_certificate_literals(item)
        return
    if not isinstance(value, str) or _RATIONAL.fullmatch(value) is None:
        return
    rational = Fraction(value)
    assert rational.denominator & (rational.denominator - 1) == 0
    assert abs(rational.numerator).bit_length() <= 53
    assert Fraction(float(rational)) == rational


def _float_diagnostic(value: float) -> dict[str, Any]:
    return {"float64_hex": float(value).hex(), "descriptive_decimal": float(value)}


def test_frozen_certificates_have_expected_identity_and_exact_literals() -> None:
    assert hashlib.sha256(CERTIFICATE_PATH.read_bytes()).hexdigest() == (
        EXACT_CERTIFICATE_SHA256
    )
    assert hashlib.sha256(NONCOMMUTING_CERTIFICATE_PATH.read_bytes()).hexdigest() == (
        NONCOMMUTING_CERTIFICATE_SHA256
    )
    certificate = _certificate()
    noncommuting = _noncommuting_certificate()
    assert certificate["schema_version"] == (
        "bayesfilter.contract_e_chol_cloud_exact_certificate.v1"
    )
    assert certificate["status"] == "frozen_before_implementation_output"
    assert noncommuting["schema_version"] == (
        "bayesfilter.contract_e_chol_cloud_noncommuting_certificate.v1"
    )
    assert noncommuting["status"] == (
        "frozen_before_noncommuting_repair_execution"
    )
    _assert_dyadic_certificate_literals(certificate)
    _assert_dyadic_certificate_literals(noncommuting)


def _primary_inputs() -> tuple[tf.Tensor, ...]:
    certificate = _certificate()
    rows = _matrix(certificate["base_rows_r"])
    fixture = certificate["primary_fixture"]
    return (
        rows,
        _weights(fixture["normalized_weights"]),
        tf.zeros_like(rows),
        tf.identity(rows),
        _batch_vector(fixture["ridge"]),
    )


def _primary_tangents() -> dict[str, tuple[tf.Tensor, ...]]:
    certificate = _certificate()
    rows, weights, _, _, ridge = _primary_inputs()
    tangent = certificate["input_tangents"]
    zero_rows = tf.zeros_like(rows)
    zero_weights = tf.zeros_like(weights)
    zero_ridge = tf.zeros_like(ridge)
    return {
        "source_particles": (
            _matrix(tangent["source_particles"]),
            zero_weights,
            zero_rows,
            zero_rows,
            zero_ridge,
        ),
        "normalized_weights": (
            zero_rows,
            _weights(tangent["normalized_weights"]),
            zero_rows,
            zero_rows,
            zero_ridge,
        ),
        "transported_particles": (
            zero_rows,
            zero_weights,
            rows / tf.constant(32, DTYPE),
            zero_rows,
            zero_ridge,
        ),
        "residual_design": (
            zero_rows,
            zero_weights,
            zero_rows,
            rows / tf.constant(16, DTYPE),
            zero_ridge,
        ),
        "ridge": (
            zero_rows,
            zero_weights,
            zero_rows,
            zero_rows,
            _batch_vector(tangent["ridge"]),
        ),
    }


def _load_reference_module():
    specification = importlib.util.spec_from_file_location(
        "phase3_contract_e_dense_reference", REFERENCE_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load reference {REFERENCE_PATH}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _expected_forward(name: str, value: Any) -> tf.Tensor:
    if value == "R":
        return _matrix(_certificate()["base_rows_r"])
    if name == "target_mean":
        return _weights(value[0])
    return tf.constant(
        [[[_fraction(item) for item in row] for row in value[0]]], dtype=DTYPE
    )


def test_exact_certificate_forward_and_secondary_fixtures() -> None:
    certificate = _certificate()
    source, weights, transported, residual, ridge = _primary_inputs()
    actual = reset._contract_e_chol_cloud_forward_core(
        source, weights, transported, residual, ridge
    )
    for name, expected in certificate["primary_fixture"]["expected_forward"].items():
        _assert_bitwise_equal(actual[name], _expected_forward(name, expected))
    _assert_bitwise_equal(actual["residual_design_sum"], tf.zeros([1, 2], DTYPE))
    _assert_bitwise_equal(actual["ridged_identity_residual"], tf.zeros([1, 2, 2], DTYPE))
    _assert_bitwise_equal(
        actual["ridged_identity_absolute_scale"],
        2 * tf.eye(2, batch_shape=[1], dtype=DTYPE),
    )
    _assert_bitwise_equal(actual["raw_covariance_residual"], tf.zeros([1, 2, 2], DTYPE))
    assert bool(tf.reduce_all(actual["finite"]).numpy())
    assert bool(tf.reduce_all(actual["factor_diagonal_positive"]).numpy())

    one_dimensional = reset._contract_e_chol_cloud_forward_core(
        source[:, :, :1],
        weights,
        transported[:, :, :1],
        residual[:, :, :1],
        ridge,
    )
    _assert_bitwise_equal(one_dimensional["particles"], source[:, :, :1])

    scaled = reset._contract_e_chol_cloud_forward_core(
        2 * source,
        weights,
        transported,
        residual,
        tf.constant([1.0], DTYPE),
    )
    _assert_bitwise_equal(scaled["particles"], 2 * source)
    for factor in ("gap_chol", "target_chol", "injected_chol"):
        _assert_bitwise_equal(scaled[factor], 2 * tf.eye(2, batch_shape=[1], dtype=DTYPE))
    _assert_bitwise_equal(scaled["affine"], tf.eye(2, batch_shape=[1], dtype=DTYPE))


def test_exact_certificate_manual_jvp_and_intermediate_paths() -> None:
    certificate = _certificate()
    inputs = _primary_inputs()
    selected = certificate["selected_nonzero_intermediate_tangents"]
    for name, tangents in _primary_tangents().items():
        actual = reset._contract_e_chol_cloud_jvp_core(*inputs, *tangents)
        expected = _matrix(
            certificate["expected_jvp_particles_by_single_input"][name]
        )
        _assert_bitwise_equal(actual["particles"], expected)
        assert bool(tf.reduce_any(tf.not_equal(actual["particles"], 0)).numpy())
        for field, values in selected.get(name, {}).items():
            if field == "target_mean":
                expected_field = tf.constant(
                    [[_fraction(item) for item in values[0]]], DTYPE
                )
            else:
                expected_field = tf.constant(
                    [[[_fraction(item) for item in row] for row in values]], DTYPE
                )
            _assert_bitwise_equal(actual[field], expected_field)


def test_exact_certificate_manual_jvp_matches_forward_accumulator() -> None:
    inputs = _primary_inputs()
    for tangents in _primary_tangents().values():
        with tf.autodiff.ForwardAccumulator(inputs, tangents) as accumulator:
            particles = reset._contract_e_chol_cloud_forward_core(*inputs)["particles"]
        automatic = accumulator.jvp(particles)
        manual = reset._contract_e_chol_cloud_jvp_core(
            *inputs, *tangents
        )["particles"]
        _assert_bitwise_equal(manual, automatic)


def test_exact_certificate_manual_vjp_and_internal_reverse_paths() -> None:
    certificate = _certificate()
    inputs = _primary_inputs()
    upstream = _matrix(certificate["upstream_particles"])
    actual = reset._contract_e_chol_cloud_vjp_core(*inputs, upstream)
    for name in (
        "source_particles",
        "normalized_weights",
        "transported_particles",
        "residual_design",
        "ridge",
    ):
        values = certificate["expected_vjp"][name]
        if name == "normalized_weights":
            expected = _weights(values)
        elif name == "ridge":
            expected = _batch_vector(values)
        else:
            expected = _matrix(values)
        _assert_bitwise_equal(actual[name], expected)
        assert bool(tf.reduce_any(tf.not_equal(actual[name], 0)).numpy())

    for name, values in certificate["selected_nonzero_intermediate_vjp"].items():
        if name == "target_mean_bar":
            expected = _weights(values[0])
        elif len(values) == 8:
            expected = _matrix(values)
        else:
            expected = tf.constant(
                [[[_fraction(item) for item in row] for row in values]], DTYPE
            )
        _assert_bitwise_equal(actual["intermediates"][name], expected)


def test_exact_certificate_manual_vjp_matches_gradient_tape_and_duality() -> None:
    certificate = _certificate()
    inputs = _primary_inputs()
    upstream = _matrix(certificate["upstream_particles"])
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        particles = reset._contract_e_chol_cloud_forward_core(*inputs)["particles"]
        objective = tf.reduce_sum(particles * upstream)
    automatic = tape.gradient(objective, inputs)
    manual = reset._contract_e_chol_cloud_vjp_core(*inputs, upstream)
    for expected, name in zip(
        automatic,
        (
            "source_particles",
            "normalized_weights",
            "transported_particles",
            "residual_design",
            "ridge",
        ),
        strict=True,
    ):
        _assert_bitwise_equal(manual[name], expected)

    combined_tangent = tuple(
        sum((values[index] for values in _primary_tangents().values()), tf.zeros_like(inputs[index]))
        for index in range(5)
    )
    jvp = reset._contract_e_chol_cloud_jvp_core(
        *inputs, *combined_tangent
    )["particles"]
    primal_pairing = tf.reduce_sum(upstream * jvp)
    adjoint_pairing = sum(
        (
            tf.reduce_sum(manual[name] * tangent)
            for name, tangent in zip(
                (
                    "source_particles",
                    "normalized_weights",
                    "transported_particles",
                    "residual_design",
                    "ridge",
                ),
                combined_tangent,
                strict=True,
            )
        ),
        tf.constant(0.0, DTYPE),
    )
    expected_pairing = tf.constant(
        _fraction(certificate["expected_combined_jvp_vjp_pairing"]), DTYPE
    )
    _assert_bitwise_equal(primal_pairing, expected_pairing)
    _assert_bitwise_equal(adjoint_pairing, expected_pairing)


def test_exact_certificate_dense_reference_composition() -> None:
    certificate = _certificate()
    reference = _load_reference_module()
    source, weights, transported, residual, ridge = _primary_inputs()
    del transported
    count = tf.cast(tf.shape(source)[1], DTYPE)
    scale = tf.sqrt(count / (count - 1))
    matrix = tf.zeros([1, 8, 8], DTYPE)
    residual_noise = residual / scale
    replay = scale * (
        residual_noise - tf.reduce_mean(residual_noise, axis=1, keepdims=True)
    )
    _assert_bitwise_equal(replay, residual)

    cloud = reset._contract_e_chol_cloud_forward_core(
        source, weights, tf.zeros_like(source), residual, ridge
    )
    dense = reference.contract_e_cholesky_ridge_reset_fixed_ridge(
        tf,
        post_flow=source,
        weights=weights,
        matrix=matrix,
        residual_noise=residual_noise,
        rho=tf.constant(1.0, DTYPE),
        ridge=ridge,
        return_aux=True,
    )["aux"]
    shared_forward_fields = {
        "target_mean": "target_mean",
        "target_cov": "target_cov",
        "transported_particles": "y_plus",
        "plus_cov": "plus_cov",
        "gap": "gap",
        "gap_chol": "residual_chol",
        "residual_design": "xi",
        "injected_particles": "y_tilde",
        "injected_mean": "tilde_mean",
        "injected_cov": "tilde_cov",
        "centered_injected": "centered_tilde",
        "target_chol": "target_chol",
        "injected_chol": "tilde_chol",
        "affine": "affine",
        "particles": "y_star",
    }
    cloud_values = {**cloud, "transported_particles": tf.zeros_like(source), "residual_design": residual}
    for cloud_name, dense_name in shared_forward_fields.items():
        _assert_bitwise_equal(cloud_values[cloud_name], dense[dense_name])

    matrix_tangent = tf.eye(8, batch_shape=[1], dtype=DTYPE) / 32
    noise_tangent = (residual / 16) / scale
    replay_tangent = scale * (
        noise_tangent - tf.reduce_mean(noise_tangent, axis=1, keepdims=True)
    )
    _assert_bitwise_equal(replay_tangent, residual / 16)
    cloud_tangents = _primary_tangents()
    for name, tangents in cloud_tangents.items():
        dense_tangents = [
            tangents[0],
            tangents[1],
            matrix_tangent if name == "transported_particles" else tf.zeros_like(matrix),
            noise_tangent if name == "residual_design" else tf.zeros_like(residual_noise),
            tangents[4],
        ]
        dense_inputs = [source, weights, matrix, residual_noise, ridge]
        with tf.autodiff.ForwardAccumulator(dense_inputs, dense_tangents) as accumulator:
            dense_particles = reference.contract_e_cholesky_ridge_reset_fixed_ridge(
                tf,
                post_flow=dense_inputs[0],
                weights=dense_inputs[1],
                matrix=dense_inputs[2],
                residual_noise=dense_inputs[3],
                rho=tf.constant(1.0, DTYPE),
                ridge=dense_inputs[4],
            )["particles"]
        dense_jvp = accumulator.jvp(dense_particles)
        cloud_jvp = reset._contract_e_chol_cloud_jvp_core(
            source,
            weights,
            tf.zeros_like(source),
            residual,
            ridge,
            *tangents,
        )["particles"]
        _assert_bitwise_equal(cloud_jvp, dense_jvp)

    upstream = _matrix(certificate["upstream_particles"])
    cloud_vjp = reset._contract_e_chol_cloud_vjp_core(
        source, weights, tf.zeros_like(source), residual, ridge, upstream
    )
    dense_vjp = reference.contract_e_cholesky_ridge_reset_fixed_ridge_vjp(
        tf,
        post_flow=source,
        weights=weights,
        matrix=matrix,
        residual_noise=residual_noise,
        rho=tf.constant(1.0, DTYPE),
        ridge=ridge,
        upstream_particles=upstream,
    )
    composed_source = cloud_vjp["source_particles"] + tf.linalg.matmul(
        matrix, cloud_vjp["transported_particles"], transpose_a=True
    )
    composed_matrix = tf.linalg.matmul(
        cloud_vjp["transported_particles"], source, transpose_b=True
    )
    composed_noise = scale * (
        cloud_vjp["residual_design"]
        - tf.reduce_mean(cloud_vjp["residual_design"], axis=1, keepdims=True)
    )
    _assert_bitwise_equal(dense_vjp["post_flow"], composed_source)
    _assert_bitwise_equal(dense_vjp["weights"], cloud_vjp["normalized_weights"])
    _assert_bitwise_equal(dense_vjp["matrix"], composed_matrix)
    _assert_bitwise_equal(dense_vjp["residual_noise"], composed_noise)
    _assert_bitwise_equal(dense_vjp["ridge"], cloud_vjp["ridge"])
    _assert_bitwise_equal(
        dense_vjp["matrix"],
        tf.constant(
            [
                [
                    [_fraction(item) for item in row]
                    for row in certificate["dense_reference_replay"]["expected_matrix_vjp"]
                ]
            ],
            DTYPE,
        ),
    )


def test_exact_two_batch_forward_jvp_and_vjp_match_independent_batches() -> None:
    first = _primary_inputs()
    second = (
        2 * first[0],
        first[1],
        first[2],
        first[3],
        tf.constant([1.0], DTYPE),
    )
    batched = tuple(tf.concat([a, b], axis=0) for a, b in zip(first, second, strict=True))
    forward = reset._contract_e_chol_cloud_forward_core(*batched)
    expected_forward = [
        reset._contract_e_chol_cloud_forward_core(*values)
        for values in (first, second)
    ]
    for name, value in forward.items():
        if value.dtype == tf.bool:
            tf.debugging.assert_equal(
                value, tf.concat([item[name] for item in expected_forward], axis=0)
            )
        else:
            _assert_bitwise_equal(
                value, tf.concat([item[name] for item in expected_forward], axis=0)
            )

    first_tangents = tuple(
        sum((value[index] for value in _primary_tangents().values()), tf.zeros_like(first[index]))
        for index in range(5)
    )
    second_tangents = (
        2 * first_tangents[0],
        first_tangents[1],
        first_tangents[2],
        first_tangents[3],
        tf.constant([0.25], DTYPE),
    )
    batched_tangents = tuple(
        tf.concat([a, b], axis=0)
        for a, b in zip(first_tangents, second_tangents, strict=True)
    )
    batched_jvp = reset._contract_e_chol_cloud_jvp_core(
        *batched, *batched_tangents
    )
    separate_jvp = [
        reset._contract_e_chol_cloud_jvp_core(*values, *tangents)
        for values, tangents in (
            (first, first_tangents),
            (second, second_tangents),
        )
    ]
    for name, value in batched_jvp.items():
        _assert_bitwise_equal(
            value, tf.concat([item[name] for item in separate_jvp], axis=0)
        )

    upstream = _matrix(_certificate()["upstream_particles"])
    batched_upstream = tf.concat([upstream, 2 * upstream], axis=0)
    batched_vjp = reset._contract_e_chol_cloud_vjp_core(*batched, batched_upstream)
    separate_vjp = [
        reset._contract_e_chol_cloud_vjp_core(*first, upstream),
        reset._contract_e_chol_cloud_vjp_core(*second, 2 * upstream),
    ]
    for name in (
        "source_particles",
        "normalized_weights",
        "transported_particles",
        "residual_design",
        "ridge",
    ):
        _assert_bitwise_equal(
            batched_vjp[name],
            tf.concat([item[name] for item in separate_vjp], axis=0),
        )


def _frozen_nontrivial_chart_diagnostics() -> dict[str, Any]:
    fixture = _certificate()["nontrivial_diagnostic_fixture"]
    inputs = (
        _matrix(fixture["source_particles"]),
        _weights(fixture["normalized_weights"]),
        _matrix(fixture["transported_particles"]),
        _matrix(fixture["residual_design"]),
        _batch_vector(fixture["ridge"]),
    )
    tangent = fixture["input_tangents"]
    tangents = (
        _matrix(tangent["source_particles"]),
        _weights(tangent["normalized_weights"]),
        _matrix(tangent["transported_particles"]),
        _matrix(tangent["residual_design"]),
        _batch_vector(tangent["ridge"]),
    )
    upstream = _matrix(fixture["upstream_particles"])
    forward = reset._contract_e_chol_cloud_forward_core(*inputs)
    manual_jvp = reset._contract_e_chol_cloud_jvp_core(
        *inputs, *tangents
    )["particles"]
    manual_vjp = reset._contract_e_chol_cloud_vjp_core(*inputs, upstream)
    with tf.autodiff.ForwardAccumulator(inputs, tangents) as accumulator:
        automatic_particles = reset._contract_e_chol_cloud_forward_core(
            *inputs
        )["particles"]
    automatic_jvp = accumulator.jvp(automatic_particles)
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        objective = tf.reduce_sum(
            reset._contract_e_chol_cloud_forward_core(*inputs)["particles"]
            * upstream
        )
    automatic_vjp = tape.gradient(objective, inputs)
    names = (
        "source_particles",
        "normalized_weights",
        "transported_particles",
        "residual_design",
        "ridge",
    )

    reference = _load_reference_module()
    source, weights, transported, residual, ridge = inputs
    source_tangent, weights_tangent, transported_tangent, residual_tangent, ridge_tangent = tangents
    particle_count = tf.shape(source)[1]
    state_dimension = tf.shape(source)[2]
    basis = source[:, :state_dimension, :]
    coefficients = tf.linalg.matrix_transpose(
        tf.linalg.solve(
            tf.linalg.matrix_transpose(basis),
            tf.linalg.matrix_transpose(transported),
        )
    )
    matrix = tf.pad(
        coefficients,
        [[0, 0], [0, 0], [0, particle_count - state_dimension]],
    )
    matrix_tangent_target = transported_tangent - tf.linalg.matmul(
        matrix, source_tangent
    )
    matrix_tangent_coefficients = tf.linalg.matrix_transpose(
        tf.linalg.solve(
            tf.linalg.matrix_transpose(basis),
            tf.linalg.matrix_transpose(matrix_tangent_target),
        )
    )
    matrix_tangent = tf.pad(
        matrix_tangent_coefficients,
        [[0, 0], [0, 0], [0, particle_count - state_dimension]],
    )
    count = tf.cast(particle_count, DTYPE)
    residual_scale = tf.sqrt(count / (count - 1))
    residual_noise = residual / residual_scale
    residual_noise_tangent = residual_tangent / residual_scale
    dense_inputs = (source, weights, matrix, residual_noise, ridge)
    dense_tangents = (
        source_tangent,
        weights_tangent,
        matrix_tangent,
        residual_noise_tangent,
        ridge_tangent,
    )
    dense_forward = reference.contract_e_cholesky_ridge_reset_fixed_ridge(
        tf,
        post_flow=source,
        weights=weights,
        matrix=matrix,
        residual_noise=residual_noise,
        rho=tf.constant(1.0, DTYPE),
        ridge=ridge,
        return_aux=True,
    )["aux"]
    with tf.autodiff.ForwardAccumulator(
        dense_inputs, dense_tangents
    ) as accumulator:
        dense_particles = reference.contract_e_cholesky_ridge_reset_fixed_ridge(
            tf,
            post_flow=dense_inputs[0],
            weights=dense_inputs[1],
            matrix=dense_inputs[2],
            residual_noise=dense_inputs[3],
            rho=tf.constant(1.0, DTYPE),
            ridge=dense_inputs[4],
        )["particles"]
    dense_jvp = accumulator.jvp(dense_particles)
    dense_vjp = reference.contract_e_cholesky_ridge_reset_fixed_ridge_vjp(
        tf,
        post_flow=source,
        weights=weights,
        matrix=matrix,
        residual_noise=residual_noise,
        rho=tf.constant(1.0, DTYPE),
        ridge=ridge,
        upstream_particles=upstream,
    )
    composed_cloud_vjp = {
        "post_flow": manual_vjp["source_particles"]
        + tf.linalg.matmul(
            matrix, manual_vjp["transported_particles"], transpose_a=True
        ),
        "weights": manual_vjp["normalized_weights"],
        "matrix": tf.linalg.matmul(
            manual_vjp["transported_particles"], source, transpose_b=True
        ),
        "residual_noise": residual_scale
        * (
            manual_vjp["residual_design"]
            - tf.reduce_mean(
                manual_vjp["residual_design"], axis=1, keepdims=True
            )
        ),
        "ridge": manual_vjp["ridge"],
    }
    dense_forward_pairs = {
        "target_mean": (forward["target_mean"], dense_forward["target_mean"]),
        "target_cov": (forward["target_cov"], dense_forward["target_cov"]),
        "transported_particles": (
            transported,
            dense_forward["y_plus"],
        ),
        "plus_cov": (forward["plus_cov"], dense_forward["plus_cov"]),
        "gap": (forward["gap"], dense_forward["gap"]),
        "gap_chol": (forward["gap_chol"], dense_forward["residual_chol"]),
        "residual_design": (residual, dense_forward["xi"]),
        "injected_particles": (
            forward["injected_particles"],
            dense_forward["y_tilde"],
        ),
        "injected_mean": (forward["injected_mean"], dense_forward["tilde_mean"]),
        "injected_cov": (forward["injected_cov"], dense_forward["tilde_cov"]),
        "centered_injected": (
            forward["centered_injected"],
            dense_forward["centered_tilde"],
        ),
        "target_chol": (forward["target_chol"], dense_forward["target_chol"]),
        "injected_chol": (
            forward["injected_chol"],
            dense_forward["tilde_chol"],
        ),
        "affine": (forward["affine"], dense_forward["affine"]),
        "particles": (forward["particles"], dense_forward["y_star"]),
    }
    primal_pairing = tf.reduce_sum(upstream * manual_jvp)
    adjoint_pairing = sum(
        (
            tf.reduce_sum(manual_vjp[name] * tangent)
            for name, tangent in zip(names, tangents, strict=True)
        ),
        tf.constant(0.0, DTYPE),
    )
    tiny = tf.constant(tf.experimental.numpy.finfo(DTYPE.as_numpy_dtype).tiny, DTYPE)
    residual_relative_sum = tf.abs(forward["residual_design_sum"]) / tf.maximum(
        forward["residual_design_absolute_scale"], tiny
    )
    output_fields = {
        name: value
        for name, value in forward.items()
        if value.dtype != tf.bool
    }
    return {
        "input_tensor_sha256": _tensor_sha256(
            dict(zip(names, inputs, strict=True))
        ),
        "output_tensor_sha256": _tensor_sha256(output_fields),
        "finite": bool(tf.reduce_all(forward["finite"]).numpy()),
        "factor_diagonal_positive": bool(
            tf.reduce_all(forward["factor_diagonal_positive"]).numpy()
        ),
        "manual_vs_autodiff": {
            "jvp_max_abs": _float_diagnostic(
                _max_abs(manual_jvp - automatic_jvp)
            ),
            "jvp_max_ulp": _max_ulp_distance(manual_jvp, automatic_jvp),
            "vjp_max_abs_by_input": {
                name: _float_diagnostic(_max_abs(manual_vjp[name] - value))
                for name, value in zip(names, automatic_vjp, strict=True)
            },
            "vjp_max_ulp_by_input": {
                name: _max_ulp_distance(manual_vjp[name], value)
                for name, value in zip(names, automatic_vjp, strict=True)
            },
        },
        "cloud_vs_dense": {
            "forward_max_abs_by_field": {
                name: _float_diagnostic(_max_abs(left - right))
                for name, (left, right) in dense_forward_pairs.items()
            },
            "forward_max_ulp_by_field": {
                name: _max_ulp_distance(left, right)
                for name, (left, right) in dense_forward_pairs.items()
            },
            "jvp_max_abs": _float_diagnostic(_max_abs(manual_jvp - dense_jvp)),
            "jvp_max_ulp": _max_ulp_distance(manual_jvp, dense_jvp),
            "vjp_max_abs_by_input": {
                name: _float_diagnostic(
                    _max_abs(composed_cloud_vjp[name] - dense_vjp[name])
                )
                for name in composed_cloud_vjp
            },
            "vjp_max_ulp_by_input": {
                name: _max_ulp_distance(composed_cloud_vjp[name], dense_vjp[name])
                for name in composed_cloud_vjp
            },
        },
        "defining_equation_diagnostics": {
            "residual_design_relative_column_sum": [
                _float_diagnostic(float(value))
                for value in residual_relative_sum.numpy()[0]
            ],
            "mean_residual_fro": _float_diagnostic(
                float(tf.linalg.norm(forward["mean_residual"]).numpy())
            ),
            "ridged_identity_residual_fro": _float_diagnostic(
                float(tf.linalg.norm(forward["ridged_identity_residual"]).numpy())
            ),
            "raw_covariance_residual_fro": _float_diagnostic(
                float(tf.linalg.norm(forward["raw_covariance_residual"]).numpy())
            ),
            "raw_covariance_prediction_difference_fro": _float_diagnostic(
                float(
                    tf.linalg.norm(
                        forward["raw_covariance_residual"]
                        - forward["predicted_raw_covariance_residual"]
                    ).numpy()
                )
            ),
            "jvp_vjp_duality_abs_difference": _float_diagnostic(
                abs(float(primal_pairing.numpy()) - float(adjoint_pairing.numpy()))
            ),
        },
        "factor_condition_proxy": {
            name: [_float_diagnostic(float(value)) for value in forward[name].numpy()]
            for name in (
                "gap_condition_proxy",
                "target_condition_proxy",
                "injected_condition_proxy",
            )
        },
    }


def test_frozen_nontrivial_chart_is_finite_valid_and_unranked() -> None:
    fixture = _certificate()["nontrivial_diagnostic_fixture"]
    diagnostics = _frozen_nontrivial_chart_diagnostics()
    assert diagnostics["finite"]
    assert diagnostics["factor_diagonal_positive"]
    for group in ("manual_vs_autodiff", "cloud_vs_dense"):
        serialized = json.dumps(diagnostics[group], allow_nan=False)
        assert "NaN" not in serialized and "Infinity" not in serialized
    assert (
        fixture["agreement_classification"]
        == "INCONCLUSIVE_GENERAL_CHART_NO_JUSTIFIED_FORWARD_ERROR_BOUND"
    )
    assert "reported_without_threshold" in fixture


def test_persisted_parity_artifact_matches_recomputed_diagnostics() -> None:
    artifact = _parity_artifact()
    assert artifact["schema_version"] == (
        "bayesfilter.contract_e_canonical_gradient_migration.phase3_parity.v1"
    )
    assert artifact["status"] == (
        "EXACT_ENGINEERING_CERTIFICATE_PASSED_"
        "GENERAL_PARITY_AND_PROMOTION_BLOCKED"
    )
    assert artifact["general_chart"]["classification"] == (
        "INCONCLUSIVE_GENERAL_CHART_NO_JUSTIFIED_FORWARD_ERROR_BOUND"
    )
    manifest = _run_manifest()
    recorded_source = manifest["source_sha256"][
        "bayesfilter/highdim/ledh_contract_e_reset_tf.py"
    ]
    current_source = _file_sha256(ROOT / "bayesfilter/highdim/ledh_contract_e_reset_tf.py")
    if current_source == recorded_source:
        assert artifact["general_chart"]["diagnostics"] == (
            _frozen_nontrivial_chart_diagnostics()
        )
    else:
        # The artifact certifies the manifest-bound historical source. Recomputing
        # it against a later implementation is a source-closure audit, not parity.
        assert artifact["general_chart"]["diagnostics"] != (
            _frozen_nontrivial_chart_diagnostics()
        )
        assert current_source != recorded_source
        assert manifest["source_sha256"][
            "docs/plans/bayesfilter-contract-e-canonical-gradient-migration-"
            "phase3-parity-diagnostics-2026-07-13.json"
        ] == _file_sha256(PARITY_ARTIFACT_PATH)
    assert all(artifact["exact_certificates"][name]["passed"] for name in (
        "identity_and_scalar_factor_charts",
        "noncommuting_factor_chart",
        "transported_covariance_branch_chart",
        "two_batch_chart",
        "dense_composition_chart",
    ))
    assert len(artifact["promotion_blockers"]) == 6
    assert all(
        blocker["status"] == "UNRESOLVED_PROMOTION_BLOCKER"
        for blocker in artifact["promotion_blockers"]
    )


def _noncommuting_inputs() -> tuple[tf.Tensor, ...]:
    certificate = _noncommuting_certificate()
    return (
        _matrix(certificate["source_particles"]),
        _weights(certificate["normalized_weights"]),
        _matrix(certificate["transported_particles"]),
        _matrix(certificate["residual_design"]),
        _batch_vector(certificate["ridge"]),
    )


def _noncommuting_tangents() -> dict[str, tuple[tf.Tensor, ...]]:
    certificate = _noncommuting_certificate()
    source, weights, transported, residual, ridge = _noncommuting_inputs()
    tangent = certificate["input_tangents"]
    zero_source = tf.zeros_like(source)
    zero_weights = tf.zeros_like(weights)
    zero_transported = tf.zeros_like(transported)
    zero_residual = tf.zeros_like(residual)
    zero_ridge = tf.zeros_like(ridge)
    return {
        "source_particles": (
            _matrix(tangent["source_particles"]),
            zero_weights,
            zero_transported,
            zero_residual,
            zero_ridge,
        ),
        "normalized_weights": (
            zero_source,
            _weights(tangent["normalized_weights"]),
            zero_transported,
            zero_residual,
            zero_ridge,
        ),
        "transported_particles": (
            zero_source,
            zero_weights,
            _matrix(tangent["transported_particles"]),
            zero_residual,
            zero_ridge,
        ),
        "residual_design": (
            zero_source,
            zero_weights,
            zero_transported,
            _matrix(tangent["residual_design"]),
            zero_ridge,
        ),
        "ridge": (
            zero_source,
            zero_weights,
            zero_transported,
            zero_residual,
            _batch_vector(tangent["ridge"]),
        ),
    }


def test_noncommuting_exact_forward_certificate() -> None:
    certificate = _noncommuting_certificate()
    actual = reset._contract_e_chol_cloud_forward_core(*_noncommuting_inputs())
    for name, values in certificate["expected_forward"].items():
        if name == "target_mean":
            expected = _weights(values[0])
        elif name in {"injected_particles", "particles"}:
            expected = _matrix(values)
        elif name == "factor_commutator":
            expected = tf.constant(
                [[[_fraction(item) for item in row] for row in values]], DTYPE
            )
            commutator = tf.linalg.matmul(
                actual["target_chol"], actual["injected_chol"]
            ) - tf.linalg.matmul(actual["injected_chol"], actual["target_chol"])
            _assert_bitwise_equal(commutator, expected)
            assert bool(tf.reduce_any(tf.not_equal(commutator, 0)).numpy())
            continue
        else:
            expected = tf.constant(
                [[[_fraction(item) for item in row] for row in values]], DTYPE
            )
        _assert_bitwise_equal(actual[name], expected)
    _assert_bitwise_equal(
        actual["ridged_identity_residual"], tf.zeros([1, 2, 2], DTYPE)
    )
    assert not bool(
        tf.reduce_all(tf.equal(actual["affine"], tf.eye(2, batch_shape=[1], dtype=DTYPE))).numpy()
    )
    assert not bool(
        tf.reduce_all(tf.equal(actual["affine"], tf.linalg.matrix_transpose(actual["affine"]))).numpy()
    )


def test_noncommuting_exact_jvp_matches_rational_and_forward_accumulator() -> None:
    certificate = _noncommuting_certificate()
    inputs = _noncommuting_inputs()
    for name, tangents in _noncommuting_tangents().items():
        manual = reset._contract_e_chol_cloud_jvp_core(
            *inputs, *tangents
        )["particles"]
        expected = _matrix(
            certificate["expected_jvp_particles_by_single_input"][name]
        )
        _assert_bitwise_equal(manual, expected)
        with tf.autodiff.ForwardAccumulator(inputs, tangents) as accumulator:
            particles = reset._contract_e_chol_cloud_forward_core(*inputs)["particles"]
        automatic = accumulator.jvp(particles)
        _assert_bitwise_equal(manual, automatic)


def test_noncommuting_exact_vjp_matches_rational_autodiff_and_duality() -> None:
    certificate = _noncommuting_certificate()
    inputs = _noncommuting_inputs()
    tangents_by_input = _noncommuting_tangents()
    upstream = _matrix(certificate["upstream_particles"])
    manual = reset._contract_e_chol_cloud_vjp_core(*inputs, upstream)
    names = (
        "source_particles",
        "normalized_weights",
        "transported_particles",
        "residual_design",
        "ridge",
    )
    for name in names:
        values = certificate["expected_vjp"][name]
        if name == "normalized_weights":
            expected = _weights(values)
        elif name == "ridge":
            expected = _batch_vector(values)
        else:
            expected = _matrix(values)
        _assert_bitwise_equal(manual[name], expected)

    with tf.GradientTape() as tape:
        tape.watch(inputs)
        particles = reset._contract_e_chol_cloud_forward_core(*inputs)["particles"]
        objective = tf.reduce_sum(particles * upstream)
    automatic = tape.gradient(objective, inputs)
    for name, value in zip(names, automatic, strict=True):
        _assert_bitwise_equal(manual[name], value)

    combined_tangent = tuple(
        sum(
            (values[index] for values in tangents_by_input.values()),
            tf.zeros_like(inputs[index]),
        )
        for index in range(5)
    )
    jvp = reset._contract_e_chol_cloud_jvp_core(
        *inputs, *combined_tangent
    )["particles"]
    primal_pairing = tf.reduce_sum(upstream * jvp)
    adjoint_pairing = sum(
        (
            tf.reduce_sum(manual[name] * tangent)
            for name, tangent in zip(names, combined_tangent, strict=True)
        ),
        tf.constant(0.0, DTYPE),
    )
    expected = tf.constant(
        _fraction(certificate["expected_combined_jvp_vjp_pairing"]), DTYPE
    )
    _assert_bitwise_equal(primal_pairing, expected)
    _assert_bitwise_equal(adjoint_pairing, expected)


def test_exact_nonzero_transported_covariance_branch_jvp_vjp() -> None:
    certificate = _noncommuting_certificate()[
        "transported_covariance_branch_certificate"
    ]
    source = tf.constant(
        [[[_fraction(value)] for value in certificate["source_particles"]]], DTYPE
    )
    weights = _weights(certificate["normalized_weights"])
    transported = tf.constant(
        [[[_fraction(value)] for value in certificate["transported_particles"]]],
        DTYPE,
    )
    residual = tf.constant(
        [[[_fraction(value)] for value in certificate["residual_design"]]], DTYPE
    )
    ridge = _batch_vector(certificate["ridge"])
    inputs = (source, weights, transported, residual, ridge)
    forward = reset._contract_e_chol_cloud_forward_core(*inputs)
    for name, value in certificate["expected_forward"].items():
        if name == "particles":
            expected = tf.constant(
                [[[_fraction(item)] for item in value]], DTYPE
            )
        else:
            expected = tf.constant([[[_fraction(value)]]], DTYPE)
        _assert_bitwise_equal(forward[name], expected)

    transported_tangent = tf.constant(
        [
            [
                [_fraction(value)]
                for value in certificate["transported_particles_tangent"]
            ]
        ],
        DTYPE,
    )
    tangents = (
        tf.zeros_like(source),
        tf.zeros_like(weights),
        transported_tangent,
        tf.zeros_like(residual),
        tf.zeros_like(ridge),
    )
    manual_jvp = reset._contract_e_chol_cloud_jvp_core(*inputs, *tangents)
    jvp_expected = certificate["expected_transport_jvp_intermediates"]
    for name in ("plus_cov", "gap", "gap_chol"):
        _assert_bitwise_equal(
            manual_jvp[name], tf.constant([[[_fraction(jvp_expected[name])]]], DTYPE)
        )
    for name in ("injected_particles", "particles"):
        expected = tf.constant(
            [[[_fraction(value)] for value in jvp_expected[name]]], DTYPE
        )
        _assert_bitwise_equal(manual_jvp[name], expected)
    with tf.autodiff.ForwardAccumulator(inputs, tangents) as accumulator:
        particles = reset._contract_e_chol_cloud_forward_core(*inputs)["particles"]
    _assert_bitwise_equal(manual_jvp["particles"], accumulator.jvp(particles))

    upstream = tf.constant(
        [[[_fraction(value)] for value in certificate["upstream_particles"]]],
        DTYPE,
    )
    manual_vjp = reset._contract_e_chol_cloud_vjp_core(*inputs, upstream)
    vjp_expected = certificate["expected_transport_vjp"]
    _assert_bitwise_equal(
        manual_vjp["intermediates"]["plus_cov_bar"],
        tf.constant([[[_fraction(vjp_expected["plus_cov_bar"])]]], DTYPE),
    )
    _assert_bitwise_equal(
        manual_vjp["intermediates"]["gap_bar"],
        tf.constant([[[_fraction(vjp_expected["gap_bar"])]]], DTYPE),
    )
    for name in ("transported_particles", "residual_design"):
        expected = tf.constant(
            [[[_fraction(value)] for value in vjp_expected[name]]], DTYPE
        )
        _assert_bitwise_equal(manual_vjp[name], expected)
    _assert_bitwise_equal(
        manual_vjp["ridge"],
        tf.constant([_fraction(vjp_expected["ridge"])], DTYPE),
    )
    with tf.GradientTape() as tape:
        tape.watch(transported)
        objective = tf.reduce_sum(
            reset._contract_e_chol_cloud_forward_core(
                source, weights, transported, residual, ridge
            )["particles"]
            * upstream
        )
    _assert_bitwise_equal(
        manual_vjp["transported_particles"], tape.gradient(objective, transported)
    )
    primal_pairing = tf.reduce_sum(upstream * manual_jvp["particles"])
    adjoint_pairing = tf.reduce_sum(
        manual_vjp["transported_particles"] * transported_tangent
    )
    expected_pairing = tf.constant(
        _fraction(certificate["expected_transport_jvp_vjp_pairing"]), DTYPE
    )
    _assert_bitwise_equal(primal_pairing, expected_pairing)
    _assert_bitwise_equal(adjoint_pairing, expected_pairing)


def test_owned_module_boundaries_and_public_xla_defaults() -> None:
    source = inspect.getsource(reset)
    for forbidden in (
        "import numpy",
        "GradientTape",
        "ForwardAccumulator",
        "tf.linalg.inv",
        "tf.linalg.eigh",
        "rho",
        "ridge_escalation",
        "contract_e_reset_tf",
    ):
        assert forbidden not in source
    signatures = {
        "contract_e_chol_cloud_forward_tf": 5,
        "contract_e_chol_cloud_jvp_tf": 10,
        "contract_e_chol_cloud_vjp_tf": 6,
    }
    for name, expected_parameters in signatures.items():
        function = getattr(reset, name)
        assert function._jit_compile is True
        assert len(inspect.signature(function.python_function).parameters) == expected_parameters


def test_public_wrappers_execute_under_deliberate_cpu_xla() -> None:
    inputs = _primary_inputs()
    tangents = _primary_tangents()["ridge"]
    upstream = _matrix(_certificate()["upstream_particles"])
    eager_forward = reset._contract_e_chol_cloud_forward_core(*inputs)
    compiled_forward = reset.contract_e_chol_cloud_forward_tf(*inputs)
    _assert_bitwise_equal(compiled_forward["particles"], eager_forward["particles"])
    compiled_jvp = reset.contract_e_chol_cloud_jvp_tf(*inputs, *tangents)
    eager_jvp = reset._contract_e_chol_cloud_jvp_core(*inputs, *tangents)["particles"]
    _assert_bitwise_equal(compiled_jvp, eager_jvp)
    compiled_vjp = reset.contract_e_chol_cloud_vjp_tf(*inputs, upstream)
    eager_vjp = reset._contract_e_chol_cloud_vjp_core(*inputs, upstream)
    for name in compiled_vjp:
        _assert_bitwise_equal(compiled_vjp[name], eager_vjp[name])
