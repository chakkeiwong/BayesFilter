from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import struct
from pathlib import Path

import pytest
import tensorflow as tf

import bayesfilter.nonlinear.ssl_lstm_predictive_tf as predictive
from bayesfilter.nonlinear.ssl_lstm_sgqf_ukf_adapters import (
    make_ssl_lstm_svd_ukf_components,
    ssl_lstm_observation,
    ssl_lstm_transition,
)


POINT_NAMES = (
    "truth_free",
    "phase2s_center",
    "shell_0_minus",
    "shell_0_plus",
    "shell_1_minus",
    "shell_1_plus",
    "shell_2_minus",
    "shell_2_plus",
    "shell_3_minus",
    "shell_3_plus",
)
POINTS_HEX = (
    ("0x1.6666666666666p-2", "-0x1.47ae147ae147bp-4", "0x1.4cccccccccccdp-1", "0x1.999999999999ap-5"),
    ("0x1.2410a2e2543f1p-1", "-0x1.fcd3132f8ba11p-4", "0x1.52631979a086cp-1", "0x1.1557ab4d560a3p-3"),
    ("0x1.ee87ac2b0ee48p-2", "-0x1.fcd3132f8ba11p-4", "0x1.52631979a086cp-1", "0x1.1557ab4d560a3p-3"),
    ("0x1.50dd6faf210bep-1", "-0x1.fcd3132f8ba11p-4", "0x1.52631979a086cp-1", "0x1.1557ab4d560a3p-3"),
    ("0x1.2410a2e2543f1p-1", "-0x1.b19cbccaf903cp-3", "0x1.52631979a086cp-1", "0x1.1557ab4d560a3p-3"),
    ("0x1.2410a2e2543f1p-1", "-0x1.2cd959924a756p-5", "0x1.52631979a086cp-1", "0x1.1557ab4d560a3p-3"),
    ("0x1.2410a2e2543f1p-1", "-0x1.fcd3132f8ba11p-4", "0x1.25964cacd3b9fp-1", "0x1.1557ab4d560a3p-3"),
    ("0x1.2410a2e2543f1p-1", "-0x1.fcd3132f8ba11p-4", "0x1.7f2fe6466d539p-1", "0x1.1557ab4d560a3p-3"),
    ("0x1.2410a2e2543f1p-1", "-0x1.fcd3132f8ba11p-4", "0x1.52631979a086cp-1", "0x1.8891e0688b5c0p-5"),
    ("0x1.2410a2e2543f1p-1", "-0x1.fcd3132f8ba11p-4", "0x1.52631979a086cp-1", "0x1.c88ade80893d6p-3"),
)
POINT_MATRIX_SHA256 = "d6ba48e5a64897f87caeece4de776c139d8fc62d00fc118d89b4d88da468829a"


def _points() -> tf.Tensor:
    return tf.constant(
        [[float.fromhex(value) for value in row] for row in POINTS_HEX],
        tf.float64,
    )


def _raw_hash(tensor: tf.Tensor) -> str:
    values = tf.unstack(tf.reshape(tensor, [-1]))
    return hashlib.sha256(
        b"".join(struct.pack("<d", float(value)) for value in values)
    ).hexdigest()


def _max_residual(left: tf.Tensor, right: tf.Tensor) -> float:
    return float(tf.reduce_max(tf.abs(left - right)))


def _tolerance(multiplier: int, left: tf.Tensor, right: tf.Tensor) -> float:
    scale = max(
        1.0,
        float(tf.reduce_max(tf.abs(left))),
        float(tf.reduce_max(tf.abs(right))),
    )
    return multiplier * predictive.FLOAT64_EPSILON * scale


def _path_tensors(paths: predictive.SSLLSTMForecastPaths) -> tuple[tf.Tensor, ...]:
    return (
        paths.terminal_states,
        paths.states,
        paths.deterministic_transition_means,
        paths.process_innovations,
        paths.observation_means,
        paths.observation_innovations,
        paths.observations,
    )


def _replace_bank(
    bank: predictive.SSLLSTMInnovationBank,
    *,
    terminal: tf.Tensor,
    process: tf.Tensor,
    observation: tf.Tensor,
) -> predictive.SSLLSTMInnovationBank:
    provisional = dataclasses.replace(
        bank,
        terminal_standard_normal=terminal,
        process_standard_normal=process,
        observation_standard_normal=observation,
        content_signature="",
    )
    return dataclasses.replace(
        provisional,
        content_signature=predictive._innovation_bank_signature(provisional),
    )


def _slice_bank(
    bank: predictive.SSLLSTMInnovationBank,
    index: int,
) -> predictive.SSLLSTMInnovationBank:
    return _replace_bank(
        bank,
        terminal=bank.terminal_standard_normal[index : index + 1],
        process=bank.process_standard_normal[index : index + 1],
        observation=bank.observation_standard_normal[index : index + 1],
    )


@pytest.fixture(scope="module")
def compiled_bundle():
    config = predictive.SSLLSTMForecastConfig()
    points = _points()
    terminal = predictive.extract_ssl_lstm_terminal_states(points, config)
    bank = predictive.make_ssl_lstm_innovation_bank(
        config,
        2,
        tf.constant([20260712, 1202], tf.int32),
        "paired_diagnostic_shared",
        0,
    )
    paths = predictive.forecast_ssl_lstm_paths(points[:2], bank, config)
    return config, points, terminal, bank, paths


def test_frozen_matrix_and_config_contract() -> None:
    points = _points()
    assert POINT_NAMES[0:2] == ("truth_free", "phase2s_center")
    assert tuple(points.shape) == (10, 4)
    assert _raw_hash(points) == POINT_MATRIX_SHA256
    config = predictive.SSLLSTMForecastConfig()
    config.assert_evidence_config()
    assert config.signature() == predictive.A2_EVIDENCE_FORECAST_CONFIG_SIGNATURE

    with pytest.raises((TypeError, ValueError)):
        predictive.SSLLSTMForecastConfig(forecast_horizon=9)
    with pytest.raises((TypeError, ValueError)):
        predictive.SSLLSTMForecastConfig(replication_count=0)
    with pytest.raises((TypeError, ValueError)):
        predictive.SSLLSTMForecastConfig(replication_count=True)
    with pytest.raises((TypeError, ValueError)):
        predictive.SSLLSTMForecastConfig(jit_compile=False)
    eager = predictive.SSLLSTMForecastConfig(
        jit_compile=False,
        execution_role="eager_debug_reference",
    )
    assert eager.execution_role == "eager_debug_reference"


def test_innovation_bank_philox_replay_and_role_separation() -> None:
    config = predictive.SSLLSTMForecastConfig()
    seed = tf.constant([20260712, 1202], tf.int32)
    left = predictive.make_ssl_lstm_innovation_bank(
        config, 2, seed, "paired_diagnostic_shared", 0
    )
    replay = predictive.make_ssl_lstm_innovation_bank(
        config, 2, seed, "paired_diagnostic_shared", 0
    )
    arm_one = predictive.make_ssl_lstm_innovation_bank(
        config, 2, seed, "independent_arm", 1
    )
    arm_two = predictive.make_ssl_lstm_innovation_bank(
        config, 2, seed, "independent_arm", 2
    )
    changed = predictive.make_ssl_lstm_innovation_bank(
        config,
        2,
        tf.constant([20260712, 1203], tf.int32),
        "paired_diagnostic_shared",
        0,
    )
    assert left.algorithm == "philox"
    assert left.content_signature == replay.content_signature
    assert left.tensor_hashes() == replay.tensor_hashes()
    for name in ("terminal", "process", "observation"):
        assert arm_one.tensor_hashes()[name] != arm_two.tensor_hashes()[name]
        assert left.tensor_hashes()[name] != changed.tensor_hashes()[name]
    assert len(set(left.tensor_hashes().values())) == 3

    with pytest.raises((TypeError, ValueError)):
        predictive.make_ssl_lstm_innovation_bank(
            config, 2, tf.constant([1, 2], tf.int64), "paired_diagnostic_shared", 0
        )
    with pytest.raises((TypeError, ValueError)):
        predictive.make_ssl_lstm_innovation_bank(
            config, 2, seed, "paired_diagnostic_shared", 1
        )
    with pytest.raises((TypeError, ValueError)):
        predictive.make_ssl_lstm_innovation_bank(
            config, 2, seed, "independent_arm", 0
        )


def test_terminal_covariance_fail_closed_policy() -> None:
    exact_singular = tf.linalg.diag(tf.constant([2.0, 0.5, 0.0], tf.float64))
    singular = predictive._audit_terminal_covariance(exact_singular)
    assert int(singular[-1]) == predictive.STATUS_VALID
    tf.debugging.assert_near(singular[2], singular[3] @ tf.transpose(singular[3]))

    tau = 64.0 * predictive.FLOAT64_EPSILON * 2.0
    roundoff_negative = tf.linalg.diag(
        tf.constant([2.0, 0.5, -0.5 * tau], tf.float64)
    )
    clipped = predictive._audit_terminal_covariance(roundoff_negative)
    assert int(clipped[-1]) == predictive.STATUS_VALID
    assert float(clipped[5][0]) == 0.0

    material_negative = tf.linalg.diag(tf.constant([2.0, 0.5, -1.0e-6], tf.float64))
    rejected = predictive._audit_terminal_covariance(material_negative)
    assert int(rejected[-1]) & predictive.STATUS_MATERIALLY_INDEFINITE

    asymmetric = tf.constant(
        [[1.0, 1.0e-4, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        tf.float64,
    )
    rejected_asymmetry = predictive._audit_terminal_covariance(asymmetric)
    assert int(rejected_asymmetry[-1]) & predictive.STATUS_ASYMMETRIC

    nonfinite = predictive._audit_terminal_covariance(
        tf.constant([[float("nan"), 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], tf.float64)
    )
    assert int(nonfinite[-1]) & predictive.STATUS_NONFINITE
    assert all(bool(tf.math.is_finite(value)) for value in nonfinite[6:11])


def test_batched_terminal_covariance_audit_matches_scalar_eager_fixture() -> None:
    raw = tf.stack(
        (
            tf.linalg.diag(tf.constant((2.0, 0.5, 0.1), tf.float64)),
            tf.constant(
                ((1.0, 0.2, 0.0), (0.2, 0.8, -0.1), (0.0, -0.1, 0.6)),
                tf.float64,
            ),
        )
    )
    batched = predictive._audit_terminal_covariance_batch_core(raw)
    scalar_rows = [
        predictive._audit_terminal_covariance(item) for item in tf.unstack(raw)
    ]
    scalar = tuple(
        tf.stack([row[index] for row in scalar_rows], axis=0)
        for index in range(len(scalar_rows[0]))
    )
    for left, right in zip(batched, scalar, strict=True):
        if left.dtype.is_integer:
            tf.debugging.assert_equal(left, right)
        else:
            tf.debugging.assert_near(left, right, atol=1.0e-15, rtol=0.0)


def test_staged_covariance_audit_replaces_only_covariance_status_bits() -> None:
    tensors = tuple(tf.constant(index, tf.int32) for index in range(22))
    tensors = (
        *tensors[:21],
        tf.constant(predictive.STATUS_PROJECTION | predictive.STATUS_FILTER_PARITY),
    )
    covariance = tuple(tf.constant(index + 100, tf.int32) for index in range(12))
    covariance = (*covariance[:11], tf.constant(predictive.STATUS_ASYMMETRIC))
    replaced = predictive._replace_terminal_covariance_audit(tensors, covariance)
    assert len(replaced) == 22
    for index in range(1, 12):
        tf.debugging.assert_equal(replaced[index], covariance[index - 1])
    for index in range(12, 21):
        tf.debugging.assert_equal(replaced[index], tensors[index])
    assert int(replaced[21]) == (
        predictive.STATUS_FILTER_PARITY | predictive.STATUS_ASYMMETRIC
    )


def test_terminal_extraction_all_frozen_rows(compiled_bundle) -> None:
    config, points, terminal, _bank, _paths = compiled_bundle
    assert tuple(terminal.mean.shape) == (10, 3)
    assert tuple(terminal.raw_covariance.shape) == (10, 3, 3)
    assert tuple(terminal.full_parameters.shape) == (10, 24)
    assert tuple(int(value) for value in tf.unstack(terminal.status)) == (0,) * 10
    assert bool(tf.reduce_all(terminal.filter_parity_residual <= terminal.filter_parity_tolerance))
    assert bool(tf.reduce_all(terminal.total_parity_residual <= terminal.total_parity_tolerance))
    for index in range(10):
        tf.debugging.assert_equal(
            config.posterior_config.parameter_mask.extract(terminal.full_parameters[index]),
            points[index],
        )
        tf.debugging.assert_near(
            terminal.factor[index] @ tf.transpose(terminal.factor[index]),
            terminal.implemented_covariance[index],
            atol=float(16.0 * terminal.psd_tolerance[index]),
            rtol=0.0,
        )
    covariance_program = predictive.ssl_lstm_terminal_covariance_audit_compiled_program(
        10
    )
    assert covariance_program.experimental_get_tracing_count() == 1


def test_forecast_recursion_noise_placement_and_replay(compiled_bundle) -> None:
    config, points, _terminal, bank, paths = compiled_bundle
    expected_shapes = (
        (2, 2, 3),
        (2, 2, 10, 3),
        (2, 2, 10, 3),
        (2, 2, 10, 1),
        (2, 2, 10, 1),
        (2, 2, 10, 1),
        (2, 2, 10, 1),
    )
    tensors = (
        paths.terminal_states,
        paths.states,
        paths.deterministic_transition_means,
        paths.process_innovations,
        paths.observation_means,
        paths.observation_innovations,
        paths.observations,
    )
    assert tuple(tuple(tensor.shape) for tensor in tensors) == expected_shapes
    tf.debugging.assert_equal(
        paths.states[..., 1:],
        paths.deterministic_transition_means[..., 1:],
    )
    latent_residual = _max_residual(
        paths.states[..., :1] - paths.deterministic_transition_means[..., :1],
        paths.process_innovations,
    )
    assert latent_residual <= _tolerance(
        128,
        paths.states[..., :1] - paths.deterministic_transition_means[..., :1],
        paths.process_innovations,
    )
    observation_residual = _max_residual(
        paths.observations - paths.observation_means,
        paths.observation_innovations,
    )
    assert observation_residual <= _tolerance(
        128,
        paths.observations - paths.observation_means,
        paths.observation_innovations,
    )
    for draw in range(2):
        components = make_ssl_lstm_svd_ukf_components(
            config.posterior_config.parameter_mask.embed(points[draw]),
            config.posterior_config.static_config,
            evidence_path=predictive.A2_RESULT_PATH,
            std_floor=config.posterior_config.std_floor,
        )
        expected_process = (
            bank.process_standard_normal[draw]
            * components.parameters.process_std[tf.newaxis, tf.newaxis, :]
        )
        assert _max_residual(
            paths.process_innovations[draw], expected_process
        ) <= _tolerance(128, paths.process_innovations[draw], expected_process)
        expected_observation = (
            bank.observation_standard_normal[draw]
            * components.parameters.observation_std[tf.newaxis, tf.newaxis, :]
        )
        assert _max_residual(
            paths.observation_innovations[draw], expected_observation
        ) <= _tolerance(
            128,
            paths.observation_innovations[draw],
            expected_observation,
        )
    replay = predictive.forecast_ssl_lstm_paths(points[:2], bank, config)
    for left, right in zip(tensors, _path_tensors(replay), strict=True):
        tf.debugging.assert_equal(left, right)


def test_zero_bank_matches_direct_recursion(compiled_bundle) -> None:
    config, points, _terminal, bank, _paths = compiled_bundle
    zero_bank = _replace_bank(
        bank,
        terminal=tf.zeros_like(bank.terminal_standard_normal),
        process=tf.zeros_like(bank.process_standard_normal),
        observation=tf.zeros_like(bank.observation_standard_normal),
    )
    paths = predictive.forecast_ssl_lstm_paths(points[:2], zero_bank, config)
    tf.debugging.assert_equal(paths.terminal_states, paths.terminal.mean[:, tf.newaxis, :])
    tf.debugging.assert_equal(paths.process_innovations, tf.zeros_like(paths.process_innovations))
    tf.debugging.assert_equal(
        paths.observation_innovations,
        tf.zeros_like(paths.observation_innovations),
    )
    for draw in range(2):
        components = make_ssl_lstm_svd_ukf_components(
            config.posterior_config.parameter_mask.embed(points[draw]),
            config.posterior_config.static_config,
            evidence_path=predictive.A2_RESULT_PATH,
            std_floor=config.posterior_config.std_floor,
        )
        previous = paths.terminal_states[draw]
        for horizon in range(10):
            expected_state = ssl_lstm_transition(components.parameters, previous)
            expected_observation = ssl_lstm_observation(
                components.parameters, expected_state
            )
            assert _max_residual(paths.states[draw, :, horizon], expected_state) <= _tolerance(
                128, paths.states[draw, :, horizon], expected_state
            )
            assert _max_residual(
                paths.observations[draw, :, horizon], expected_observation
            ) <= _tolerance(
                128, paths.observations[draw, :, horizon], expected_observation
            )
            previous = expected_state


def test_scalar_batch_order_and_eager_xla_parity(compiled_bundle) -> None:
    config, points, _terminal, bank, paths = compiled_bundle
    for index in range(2):
        scalar = predictive.forecast_ssl_lstm_path(
            points[index],
            _slice_bank(bank, index),
            config,
        )
        for scalar_tensor, batch_tensor in zip(
            _path_tensors(scalar),
            _path_tensors(paths),
            strict=True,
        ):
            reference = batch_tensor[index]
            assert _max_residual(scalar_tensor, reference) <= _tolerance(
                128, scalar_tensor, reference
            )

    eager = predictive.eager_debug_ssl_lstm_forecast_paths(
        points[:2], bank, config
    )
    for eager_tensor, xla_tensor in zip(
        _path_tensors(eager), _path_tensors(paths), strict=True
    ):
        assert _max_residual(eager_tensor, xla_tensor) <= _tolerance(
            512, eager_tensor, xla_tensor
        )


def test_chunked_forecast_matches_unchunked_draw_order_and_provenance(
    compiled_bundle,
) -> None:
    config, points, _terminal, bank, unchunked = compiled_bundle
    chunked = predictive.forecast_ssl_lstm_paths(
        points[:2], bank, config, draw_chunk_size=1
    )
    for chunked_tensor, unchunked_tensor in zip(
        _path_tensors(chunked), _path_tensors(unchunked), strict=True
    ):
        tf.debugging.assert_equal(chunked_tensor, unchunked_tensor)
    assert chunked.provenance.draw_count == 2
    assert chunked.provenance.draw_chunk_size == 1
    assert unchunked.provenance.draw_chunk_size == 2
    assert chunked.provenance.innovation_bank_signature == bank.content_signature
    assert chunked.provenance.innovation_tensor_hashes == (
        unchunked.provenance.innovation_tensor_hashes
    )


def test_compiled_trace_hlo_and_provenance(compiled_bundle) -> None:
    config, points, terminal, bank, paths = compiled_bundle
    terminal_program = predictive.ssl_lstm_terminal_compiled_program(config, 10)
    forecast_program = predictive.ssl_lstm_forecast_compiled_program(config, 2)
    forecast_inputs = (
        points[:2],
        paths.terminal.mean,
        paths.terminal.factor,
        bank.terminal_standard_normal,
        bank.process_standard_normal,
        bank.observation_standard_normal,
    )
    terminal_program(points)
    forecast_program(*forecast_inputs)
    assert terminal_program.experimental_get_tracing_count() == 1
    assert forecast_program.experimental_get_tracing_count() == 1
    terminal_hlo = terminal_program.experimental_get_compiler_ir(points)(stage="hlo")
    forecast_hlo = forecast_program.experimental_get_compiler_ir(*forecast_inputs)(stage="hlo")
    assert terminal_hlo and "ENTRY" in terminal_hlo
    assert forecast_hlo and "ENTRY" in forecast_hlo
    assert tuple(int(value) for value in tf.unstack(terminal.status)) == (0,) * 10
    assert paths.provenance.a2_contract_signature == predictive.A2_CONTRACT_SIGNATURE
    assert (
        paths.provenance.a1_adapter_signature
        == predictive.SSLLSTMPosteriorTarget(config.posterior_config).adapter_signature()
    )
    assert paths.provenance.forecast_config_signature == config.signature()
    assert paths.provenance.innovation_bank_signature == bank.content_signature
    assert paths.provenance.innovation_replay_authority == "materialized_tensor_hashes"
    assert paths.provenance.innovation_seed_qualification == (
        "generation_metadata_not_cross_backend_bitwise_regeneration_evidence"
    )
    assert paths.provenance.nonclaims == predictive.NONCLAIMS
    assert paths.provenance.approximation_qualification.endswith(
        "not_exact_nonlinear_filter"
    )
    assert paths.provenance.cluster_unit == "complete_ten_step_path_per_draw_replication"


def test_invalid_shapes_dtype_and_bank_tampering() -> None:
    config = predictive.SSLLSTMForecastConfig()
    seed = tf.constant([7, 11], tf.int32)
    bank = predictive.make_ssl_lstm_innovation_bank(
        config, 1, seed, "paired_diagnostic_shared", 0
    )
    with pytest.raises((TypeError, ValueError)):
        predictive.extract_ssl_lstm_terminal_states(tf.zeros([1, 4], tf.float32), config)
    with pytest.raises((TypeError, ValueError)):
        predictive.extract_ssl_lstm_terminal_states(tf.zeros([4], tf.float64), config)
    with pytest.raises((TypeError, ValueError)):
        predictive.forecast_ssl_lstm_paths(tf.zeros([2, 4], tf.float64), bank, config)
    with pytest.raises(ValueError, match="draw_chunk_size"):
        predictive.forecast_ssl_lstm_paths(
            tf.zeros([1, 4], tf.float64), bank, config, draw_chunk_size=2
        )
    tampered = dataclasses.replace(bank, role_code=999)
    with pytest.raises((TypeError, ValueError)):
        predictive.forecast_ssl_lstm_paths(tf.zeros([1, 4], tf.float64), tampered, config)


@pytest.mark.parametrize(
    ("family", "bad_value"),
    (("terminal", float("nan")), ("process", float("inf")), ("observation", float("-inf"))),
)
def test_self_consistently_rehashed_nonfinite_innovation_bank_is_rejected(
    family: str,
    bad_value: float,
) -> None:
    config = predictive.SSLLSTMForecastConfig(
        jit_compile=False,
        execution_role="eager_debug_reference",
    )
    bank = predictive.make_ssl_lstm_innovation_bank(
        config,
        1,
        tf.constant([7, 11], tf.int32),
        "paired_diagnostic_shared",
        0,
    )
    tensors = {
        "terminal": bank.terminal_standard_normal,
        "process": bank.process_standard_normal,
        "observation": bank.observation_standard_normal,
    }
    tensors[family] = tf.tensor_scatter_nd_update(
        tensors[family],
        tf.zeros([1, tensors[family].shape.rank], tf.int32),
        tf.constant([bad_value], tf.float64),
    )
    nonfinite_bank = _replace_bank(
        bank,
        terminal=tensors["terminal"],
        process=tensors["process"],
        observation=tensors["observation"],
    )

    with pytest.raises(ValueError, match=f"{family}_standard_normal.*finite"):
        predictive.forecast_ssl_lstm_paths(
            tf.zeros([1, 4], tf.float64),
            nonfinite_bank,
            config,
        )


@pytest.mark.parametrize("bad_value", (float("nan"), float("inf"), float("-inf")))
def test_nonfinite_free_draws_are_rejected(bad_value: float) -> None:
    config = predictive.SSLLSTMForecastConfig(
        jit_compile=False,
        execution_role="eager_debug_reference",
    )
    free_draws = tf.constant([[bad_value, 0.0, 0.0, 0.0]], tf.float64)
    bank = predictive.make_ssl_lstm_innovation_bank(
        config,
        1,
        tf.constant([7, 11], tf.int32),
        "paired_diagnostic_shared",
        0,
    )

    with pytest.raises(ValueError, match="free_draws.*finite"):
        predictive.extract_ssl_lstm_terminal_states(free_draws, config)
    with pytest.raises(ValueError, match="free_draws.*finite"):
        predictive.forecast_ssl_lstm_paths(free_draws, bank, config)


def test_nonfinite_forecast_outputs_fail_closed() -> None:
    finite = tuple(tf.zeros([1], tf.float64) for _ in range(7))
    nonfinite = finite[:4] + (tf.constant([float("nan")], tf.float64),) + finite[5:]

    predictive._require_finite_forecast_outputs(finite)
    with pytest.raises(ValueError, match="forecast output observation_means.*finite"):
        predictive._require_finite_forecast_outputs(nonfinite)


@pytest.fixture()
def a2_terminal_verifier():
    verifier_path = (
        Path(__file__).resolve().parents[1]
        / "docs/benchmarks/verify_ssl_lstm_completion_phase_a2_terminal_forecast_2026_07_12.py"
    )
    spec = importlib.util.spec_from_file_location("a2_terminal_verifier", verifier_path)
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    return verifier


def test_a2_trace_parser_distinguishes_readlink_from_link(
    tmp_path: Path,
    a2_terminal_verifier,
) -> None:
    verifier = a2_terminal_verifier

    trace = tmp_path / "trace.log"
    trace.write_text(
        """10 readlink(\"/proc/self/exe\", \"/usr/bin/python\", 4096) = 15
10 readlinkat(AT_FDCWD, \"/tmp/source\", \"target\", 4096) = 6
10 openat(AT_FDCWD, \"/tmp/read-only\", O_RDONLY|O_CLOEXEC) = 3
10 link(\"/tmp/source\", \"/tmp/target\") = 0
10 linkat(AT_FDCWD, \"/tmp/source\", AT_FDCWD, \"/tmp/target2\", 0) = 0
10 openat(AT_FDCWD, \"/tmp/written\", O_WRONLY|O_CREAT, 0666) = 4
""",
        encoding="utf-8",
    )

    assert verifier._parse_strace_mutations(trace) == [
        '10 link("/tmp/source", "/tmp/target") = 0',
        '10 linkat(AT_FDCWD, "/tmp/source", AT_FDCWD, "/tmp/target2", 0) = 0',
        '10 openat(AT_FDCWD, "/tmp/written", O_WRONLY|O_CREAT, 0666) = 4',
    ]


def test_a2_terminal_trace_audit_accepts_one_resolved_write_open(
    tmp_path: Path,
    a2_terminal_verifier,
    capsys: pytest.CaptureFixture[str],
) -> None:
    verifier = a2_terminal_verifier
    trace = tmp_path / "allowed-trace.log"
    trace.write_text(
        """10 chdir("/home/ubuntu/python/BayesFilter") = 0
10 readlink("/proc/self/exe", "/usr/bin/python", 4096) = 15
10 openat(AT_FDCWD, "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/relative.log", O_WRONLY|O_CREAT, 0666) = 3</home/ubuntu/python/BayesFilter/docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/relative.log>
""",
        encoding="utf-8",
    )

    verifier.audit_terminal_trace(trace)
    assert "A2_TERMINAL_WRITE_TRACE_AUDIT_PASSED" in capsys.readouterr().out


@pytest.mark.parametrize(
    "line",
    (
        '10 openat(AT_FDCWD, "docs/plans/outside.log", O_WRONLY|O_CREAT, 0666) = 3',
        '10 openat(AT_FDCWD, "/tmp/bayesfilter-a2-tmp-escape/x", O_WRONLY|O_CREAT, 0666) = 3</tmp/bayesfilter-a2-tmp-escape/x>',
        '10 openat(4, "unresolved.log", O_WRONLY|O_CREAT, 0666) = 3',
        '10 ftruncate(5, 0) = 0',
        '10 mknod("/tmp/outside-node", S_IFREG|0600, makedev(0, 0)) = 0',
        '10 setxattr("/tmp/outside", "user.test", "x", 1, 0) = 0',
        '10 copy_file_range(6</tmp/bayesfilter-a2-tmp/source>, NULL, 7</tmp/outside>, NULL, 8, 0) = 8',
    ),
)
def test_a2_terminal_trace_audit_rejects_boundary_bypasses(
    tmp_path: Path,
    a2_terminal_verifier,
    line: str,
) -> None:
    verifier = a2_terminal_verifier
    trace = tmp_path / "rejected-trace.log"
    trace.write_text(line + "\n", encoding="utf-8")

    with pytest.raises(verifier.ContractError):
        verifier.audit_terminal_trace(trace)


def test_a2_trace_open_flag_parser_ignores_path_text(
    tmp_path: Path,
    a2_terminal_verifier,
) -> None:
    verifier = a2_terminal_verifier
    trace = tmp_path / "read-trace.log"
    trace.write_text(
        '10 openat(AT_FDCWD, "/tmp/file-O_WRONLY", O_RDONLY|O_CLOEXEC) = 3\n',
        encoding="utf-8",
    )
    assert verifier._parse_strace_mutations(trace) == []


@pytest.mark.parametrize(
    "trace_text",
    (
        "",
        "not a syscall\n",
        '10 openat(AT_FDCWD, "/tmp/allowed", O_WRONLY <unfinished ...>\n'
        '<... openat resumed>) = 3\n',
        '10 clone(child_stack=NULL, flags=SIGCHLD) = 11\n'
        '12 openat(AT_FDCWD, "unknown-child.log", O_WRONLY) = 3</tmp/bayesfilter-a2-tmp/unknown-child.log>\n',
        '10 openat(AT_FDCWD, "/tmp/bayesfilter-a2-tmp/allowed-prefix"..., O_WRONLY) = 3</tmp/bayesfilter-a2-tmp/allowed-prefix...>\n',
        '10 openat(AT_FDCWD, "/tmp/bayesfilter-a2-tmp/allowed", O_WRONLY) = 3\n',
        '10 openat(AT_FDCWD, "/tmp/bayesfilter-a2-tmp/read", O_RDONLY) = 3</tmp/bayesfilter-a2-tmp/read>\n',
        '10 fchmodat2(AT_FDCWD, "/tmp/bayesfilter-a2-tmp/allowed", 0600, 0) = 0\n',
    ),
)
def test_a2_terminal_trace_audit_fails_closed_on_invalid_trace_contract(
    tmp_path: Path,
    a2_terminal_verifier,
    trace_text: str,
) -> None:
    verifier = a2_terminal_verifier
    trace = tmp_path / "unknown-state.log"
    trace.write_text(trace_text, encoding="utf-8")

    with pytest.raises(verifier.ContractError):
        verifier.audit_terminal_trace(trace)


def test_only_five_predictive_dataclasses_are_lazy_package_exports() -> None:
    import bayesfilter.nonlinear as nonlinear

    expected = {
        "SSLLSTMTerminalState",
        "SSLLSTMForecastConfig",
        "SSLLSTMInnovationBank",
        "SSLLSTMForecastPaths",
        "SSLLSTMForecastProvenance",
    }
    assert expected <= set(nonlinear.__all__)
    for name in expected:
        assert getattr(nonlinear, name) is getattr(predictive, name)
