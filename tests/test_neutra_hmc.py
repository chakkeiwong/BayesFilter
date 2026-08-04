from __future__ import annotations

import ast
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference import neutra_hmc


def _fake_programs(monkeypatch):
    builds = []
    calls = []

    def build(**kwargs):
        draws = int(kwargs["num_results"])
        builds.append(draws)

        def compiled(current_state, seed):
            state = tf.convert_to_tensor(current_state, tf.float64)
            seed_tensor = tf.convert_to_tensor(seed, tf.int32)
            offset = tf.cast(
                tf.range(1, draws + 1), tf.float64
            )[:, tf.newaxis, tf.newaxis]
            samples = state[tf.newaxis, :, :] + 0.01 * offset
            calls.append(tuple(int(item) for item in seed_tensor.numpy().tolist()))
            return samples, {
                "is_accepted": tf.ones((draws, state.shape[0]), tf.bool),
                "log_accept_ratio": tf.zeros((draws, state.shape[0]), tf.float64),
                "target_log_prob": -tf.reduce_sum(tf.square(samples), axis=-1),
            }

        return compiled

    monkeypatch.setattr(neutra_hmc, "_build_batched_hmc_program", build)
    return builds, calls


def _script_rhat(monkeypatch, passed_values):
    values = iter(bool(item) for item in passed_values)

    def summary(draws, *, rhat_max):
        passed = next(values)
        return {
            "passed": passed,
            "rhat_threshold": float(rhat_max),
            "draw_count_per_chain": int(tf.convert_to_tensor(draws).shape[0]),
        }

    monkeypatch.setattr(neutra_hmc, "rank_normalized_split_rhat_summary", summary)


def _config(**overrides):
    values = {
        "step_size": 0.4,
        "num_leapfrog_steps": 2,
        "warmup_seed": (20260715, 101),
        "retained_seed": (20260715, 201),
        "warmup_chunk_results": 4,
        "warmup_min_results": 4,
        "warmup_check_window_results": 4,
        "warmup_max_results": 8,
        "retained_chunk_results": 4,
        "retained_min_results": 4,
        "retained_max_results": 8,
    }
    values.update(overrides)
    return neutra_hmc.SequentialNeuTraHMCConfig(**values)


def test_sequential_controller_retains_warmup_excludes_it_and_reuses_programs(
    monkeypatch,
) -> None:
    builds, calls = _fake_programs(monkeypatch)
    _script_rhat(monkeypatch, (False, True, False, True))
    archives = []

    def archive(**kwargs):
        archives.append(kwargs)
        return {
            "stage": kwargs["stage"],
            "cumulative": kwargs["cumulative"],
            "shape": tuple(kwargs["model_samples"].shape),
        }

    result = neutra_hmc.run_sequential_neutra_hmc(
        adapter=object(),
        initial_state=tf.zeros((4, 2), tf.float64),
        model_transform=lambda values: values + 5.0,
        parameter_names=("x", "y"),
        config=_config(),
        archive_callback=archive,
    )

    assert result["passed"] is True
    assert result["warmup_results_per_chain"] == 8
    assert result["retained_results_per_chain"] == 8
    assert result["warmup_excluded_from_posterior"] is True
    assert result["warmup_samples_retained"] is True
    assert result["private_warmup_z"].shape == (8, 4, 2)
    assert result["private_retained_raw"].shape == (8, 4, 2)
    assert bool(
        tf.reduce_all(result["private_warmup_raw"] >= 5.0).numpy()
    ) is True
    assert builds == [4]
    assert calls == [
        (20260715, 1110),
        (20260715, 2119),
        (20260715, 1210),
        (20260715, 2219),
    ]
    assert len([row for row in archives if not row["cumulative"]]) == 4
    assert len([row for row in archives if row["cumulative"]]) == 2


def test_sequential_controller_extends_for_full_diagnostic(monkeypatch) -> None:
    _fake_programs(monkeypatch)
    _script_rhat(monkeypatch, (True,))
    checks = iter(({"passed": False}, {"passed": True}))

    result = neutra_hmc.run_sequential_neutra_hmc(
        adapter=object(),
        initial_state=tf.zeros((5, 3), tf.float64),
        parameter_names=("a", "b", "c"),
        config=_config(),
        retained_diagnostic_fn=lambda _draws: next(checks),
    )

    assert result["passed"] is True
    assert result["config"]["chain_count"] == 5
    assert result["retained_results_per_chain"] == 8
    assert result["retained_checks"][0]["diagnostic_role"] == "full_convergence"


def test_sequential_controller_caps_and_chain_policy(monkeypatch) -> None:
    _fake_programs(monkeypatch)
    _script_rhat(monkeypatch, (False, False))
    result = neutra_hmc.run_sequential_neutra_hmc(
        adapter=object(),
        initial_state=tf.zeros((4, 2), tf.float64),
        parameter_names=("x", "y"),
        config=_config(),
    )
    assert result["passed"] is False
    assert result["warmup_cap_hit"] is True
    assert result["retained_results_per_chain"] == 0
    with pytest.raises(ValueError, match="warmup_max_results"):
        _config(warmup_max_results=10001)
    with pytest.raises(ValueError, match="retained_max_results"):
        _config(retained_max_results=10001)
    with pytest.raises(ValueError, match="must be distinct"):
        _config(retained_seed=(20260715, 101))
    with pytest.raises(neutra_hmc.NeuTraHMCError, match="at least 4 chains"):
        neutra_hmc.run_sequential_neutra_hmc(
            adapter=object(),
            initial_state=tf.zeros((3, 2), tf.float64),
            parameter_names=("x", "y"),
            config=_config(),
        )


def test_sequential_controller_health_veto_stops_before_retained(monkeypatch) -> None:
    _fake_programs(monkeypatch)

    original = neutra_hmc._summarize_batched_hmc_output

    def unhealthy(**kwargs):
        result = original(**kwargs)
        result["diagnostics"]["health_passed"] = False
        return result

    monkeypatch.setattr(neutra_hmc, "_summarize_batched_hmc_output", unhealthy)
    result = neutra_hmc.run_sequential_neutra_hmc(
        adapter=object(),
        initial_state=tf.zeros((4, 2), tf.float64),
        parameter_names=("x", "y"),
        config=_config(),
    )
    assert result["hard_vetoes"] == ("warmup_chunk_health_failed",)
    assert result["retained_results_per_chain"] == 0


def test_retained_continuation_uses_real_chunk_config_and_checkpoints(
    monkeypatch,
) -> None:
    builds, calls = _fake_programs(monkeypatch)
    diagnostics = iter(
        (
            {"passed": False, "hard_vetoes": ()},
            {"passed": True, "hard_vetoes": ()},
        )
    )
    archives = []
    checkpoints = []

    def archive(**kwargs):
        archives.append(kwargs)
        return {
            "stage": kwargs["stage"],
            "chunk_index": kwargs["chunk_index"],
            "cumulative": kwargs["cumulative"],
            "shape": tuple(kwargs["model_samples"].shape),
        }

    result = neutra_hmc.run_retained_neutra_hmc_continuation(
        adapter=object(),
        prefix_latent=tf.zeros((4, 4, 2), tf.float64),
        prefix_model=tf.zeros((4, 4, 2), tf.float64),
        model_transform=lambda values: values + 2.0,
        parameter_names=("x", "y"),
        config=_config(retained_max_results=12),
        next_chunk_index=1,
        retained_diagnostic_fn=lambda _draws: next(diagnostics),
        archive_callback=archive,
        checkpoint_callback=checkpoints.append,
    )

    assert result["passed"] is True
    assert result["completion_status"] == "passed"
    assert result["retained_results_per_chain"] == 12
    assert result["retained_checks"][0]["health"]["elapsed_seconds"] >= 0.0
    assert result["retained_checks"][0]["chunk_index"] == 1
    assert result["retained_checks"][1]["chunk_index"] == 2
    assert builds == [4]
    assert calls == [(20260715, 2219), (20260715, 3228)]
    assert len(checkpoints) == 2
    assert checkpoints[0]["terminal"] is False
    assert checkpoints[1]["terminal"] is True
    assert len([row for row in archives if row["cumulative"]]) == 1


def test_retained_continuation_propagates_diagnostic_veto(monkeypatch) -> None:
    _fake_programs(monkeypatch)
    result = neutra_hmc.run_retained_neutra_hmc_continuation(
        adapter=object(),
        prefix_latent=tf.zeros((4, 4, 2), tf.float64),
        prefix_model=tf.zeros((4, 4, 2), tf.float64),
        parameter_names=("x", "y"),
        config=_config(retained_max_results=8),
        next_chunk_index=1,
        retained_diagnostic_fn=lambda _draws: {
            "passed": False,
            "hard_vetoes": ("nonfinite_convergence_diagnostic",),
        },
    )

    assert result["passed"] is False
    assert result["completion_status"] == "hard_veto"
    assert result["hard_vetoes"] == ("nonfinite_convergence_diagnostic",)


def test_retained_continuation_stop_is_incomplete_not_rejected(monkeypatch) -> None:
    builds, _ = _fake_programs(monkeypatch)
    result = neutra_hmc.run_retained_neutra_hmc_continuation(
        adapter=object(),
        prefix_latent=tf.zeros((4, 4, 2), tf.float64),
        prefix_model=tf.zeros((4, 4, 2), tf.float64),
        parameter_names=("x", "y"),
        config=_config(retained_max_results=8),
        next_chunk_index=1,
        retained_diagnostic_fn=lambda _draws: {"passed": False},
        stop_requested_fn=lambda: True,
    )

    assert builds == []
    assert result["completion_status"] == "stopped_before_chunk"
    assert result["decision"] == "INCOMPLETE_SEQUENTIAL_FIXED_NEUTRA_HMC_KERNEL"
    assert result["retained_results_per_chain"] == 4
    assert result["hard_vetoes"] == ()


def test_retained_continuation_cap_checkpoint_is_terminal(monkeypatch) -> None:
    _fake_programs(monkeypatch)
    checkpoints = []
    result = neutra_hmc.run_retained_neutra_hmc_continuation(
        adapter=object(),
        prefix_latent=tf.zeros((4, 4, 2), tf.float64),
        prefix_model=tf.zeros((4, 4, 2), tf.float64),
        parameter_names=("x", "y"),
        config=_config(retained_max_results=8),
        next_chunk_index=1,
        retained_diagnostic_fn=lambda _draws: {
            "passed": False,
            "hard_vetoes": (),
        },
        checkpoint_callback=checkpoints.append,
    )

    assert result["completion_status"] == "retained_cap_reached"
    assert result["retained_cap_hit"] is True
    assert checkpoints[-1]["terminal"] is True
    assert checkpoints[-1]["completion_status"] == "retained_cap_reached"


def test_retained_continuation_nonfinite_model_transform_is_veto(monkeypatch) -> None:
    _fake_programs(monkeypatch)
    diagnostic_called = False

    def diagnostic(_draws):
        nonlocal diagnostic_called
        diagnostic_called = True
        return {"passed": True}

    result = neutra_hmc.run_retained_neutra_hmc_continuation(
        adapter=object(),
        prefix_latent=tf.zeros((4, 4, 2), tf.float64),
        prefix_model=tf.zeros((4, 4, 2), tf.float64),
        model_transform=lambda values: tf.fill(tf.shape(values), tf.constant(float("nan"), tf.float64)),
        parameter_names=("x", "y"),
        config=_config(retained_max_results=8),
        next_chunk_index=1,
        retained_diagnostic_fn=diagnostic,
    )

    assert diagnostic_called is False
    assert result["completion_status"] == "hard_veto"
    assert result["hard_vetoes"] == (
        "retained_continuation_model_samples_nonfinite",
    )


def test_finite_extreme_log_acceptance_is_explanatory_not_health_veto() -> None:
    initial = tf.zeros((4, 2), tf.float64)
    samples = tf.ones((3, 4, 2), tf.float64)
    trace = {
        "is_accepted": tf.zeros((3, 4), tf.bool),
        "log_accept_ratio": tf.fill((3, 4), tf.constant(-2000.0, tf.float64)),
        "target_log_prob": tf.zeros((3, 4), tf.float64),
    }
    result = neutra_hmc._summarize_batched_hmc_output(
        initial_state=initial,
        samples=samples,
        trace=trace,
        config=neutra_hmc.BatchedHMCConfig(
            num_results=3,
            num_burnin_steps=0,
            step_size=0.4,
            num_leapfrog_steps=2,
            seed=(1, 2),
        ),
        chain_count=4,
        elapsed_seconds=1.0,
    )

    diagnostics = result["diagnostics"]
    assert diagnostics["extreme_log_accept_count"] == 12
    assert diagnostics["extreme_log_accept_role"] == (
        "explanatory_only_not_a_veto_or_divergence"
    )
    assert diagnostics["health_passed"] is True


def test_core_source_has_no_forbidden_backend_or_model_tokens() -> None:
    path = Path(neutra_hmc.__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "numpy" not in imported_roots
    assert "tf.numpy_function" not in source
    assert "tf.py_function" not in source
    assert "lgssm" not in source.lower()
    assert "docs/plans" not in source


def test_batched_gaussian_cpu_xla_smoke() -> None:
    class GaussianAdapter:
        @staticmethod
        def log_prob_and_grad(theta):
            values = tf.convert_to_tensor(theta, tf.float64)
            return -0.5 * tf.reduce_sum(tf.square(values), axis=-1), -values

    run = neutra_hmc.run_batched_hmc(
        adapter=GaussianAdapter(),
        initial_state=tf.constant(
            [[-1.0, 0.5], [-0.5, -1.0], [0.5, 1.0], [1.0, -0.5]],
            tf.float64,
        ),
        config=neutra_hmc.BatchedHMCConfig(
            num_results=16,
            num_burnin_steps=8,
            step_size=0.4,
            num_leapfrog_steps=2,
            seed=(20260715, 20),
        ),
    )
    assert run["samples"].shape == (16, 4, 2)
    assert run["diagnostics"]["health_passed"] is True
    assert run["config"]["chain_count"] == 4
