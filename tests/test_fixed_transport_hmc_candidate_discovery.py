from __future__ import annotations

import ast
import hashlib
import inspect
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import tensorflow as tf

import bayesfilter
import bayesfilter.inference as inference
from bayesfilter.inference import fixed_transport_hmc_mechanics_tf as mechanics
from bayesfilter.inference.hmc import FullChainHMCRunResult
from bayesfilter.inference.posterior_adapter import ValueScoreCapability
from bayesfilter.inference import fixed_transport_hmc_candidate_discovery_tf as module
from bayesfilter.inference import fixed_transport_hmc_tuning_tf as legacy


class GaussianAdapter:
    parameter_dim = 2

    def adapter_signature(self) -> str:
        return "gaussian-candidate-discovery-v1"

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=False,
            runtime_backend="candidate_fixture",
            evidence_path="fixture-target-xla.json",
            target_scope="candidate_fixture",
        )

    def log_prob_and_grad(self, theta):
        value = tf.convert_to_tensor(theta, tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(value), axis=-1), -value


class IdentityTransport:
    parameter_dim = 2

    def manifest_payload(self):
        return {"schema": "candidate-identity.v1", "parameter_dim": 2}

    def forward(self, value):
        return tf.convert_to_tensor(value, tf.float64)

    forward_batch = forward

    def log_abs_det_jacobian(self, value):
        del value
        return tf.constant(0.0, tf.float64)

    def log_abs_det_jacobian_batch(self, value):
        return tf.zeros(tf.shape(value)[:-1], tf.float64)

    def pullback_score(self, value, score):
        del value
        return score

    pullback_score_batch = pullback_score

    def log_abs_det_jacobian_score(self, value):
        return tf.zeros_like(value)

    log_abs_det_jacobian_score_batch = log_abs_det_jacobian_score


START = tf.constant(
    [[-1.0, -0.5], [-0.2, 0.4], [0.3, -0.7], [0.8, 1.1]], tf.float64
)


def _receipt(*, target_status_trace_policy="none"):
    base = GaussianAdapter()
    transport = IdentityTransport()
    adapter = module._adapter(
        base_adapter=base,
        fixed_transport=transport,
        target_scope="candidate_fixture:fixed_transport_candidate_discovery",
    )
    code_hash = module._qualification_code_hash()
    payload = {
        "schema": "bayesfilter.fixed_transport_hmc_xla_qualification_evidence.v1",
        "status": "passed",
        "route": module.FIXED_TRANSPORT_CANDIDATE_DISCOVERY_ROUTE,
        "base_adapter_signature": base.adapter_signature(),
        "fixed_transport_manifest_hash": adapter.transport_manifest_hash,
        "transformed_adapter_signature": adapter.adapter_signature(),
        "initial_state_shape": [4, 2],
        "target_scope": adapter.target_scope,
        "qualification_code_hash": code_hash,
        "config": {
            "initial_step_size": 0.2,
            "adaptation_steps": 3,
            "tune_num_results": 2,
            "screen_num_results": 4,
            "screen_num_burnin_steps": 1,
            "primary_l_grid": list(module.PRIMARY_L_GRID),
            "target_accept_prob": 0.70,
            "target_status_trace_policy": target_status_trace_policy,
        },
    }
    path = Path(tempfile.mkdtemp()) / "fixture-full-chain-xla.json"
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(encoded)
    return module.FixedTransportHMCXLAQualificationReceipt(
        base_adapter_signature=base.adapter_signature(),
        fixed_transport_manifest_hash=adapter.transport_manifest_hash,
        transformed_adapter_signature=adapter.adapter_signature(),
        initial_state_shape=(4, 2),
        target_scope=adapter.target_scope,
        evidence_path=str(path),
        evidence_sha256=hashlib.sha256(encoded).hexdigest(),
        qualification_code_hash=code_hash,
    )


def _config(**overrides):
    values = dict(
        initial_step_size=0.2,
        adaptation_steps=3,
        tune_num_results=2,
        screen_num_results=4,
        screen_num_burnin_steps=1,
        xla_qualification=_receipt(),
        require_distinct_starts=True,
        target_scope="candidate_fixture:fixed_transport_candidate_discovery",
        target_status_trace_policy="none",
    )
    values.update(overrides)
    if values["target_status_trace_policy"] != "none" and "xla_qualification" not in overrides:
        values["xla_qualification"] = _receipt(
            target_status_trace_policy=values["target_status_trace_policy"]
        )
    return module.FixedTransportHMCCandidateDiscoveryConfig(**values)


class FakeRunner:
    def __init__(self, acceptance_by_l=None, *, move=True):
        self.acceptance_by_l = acceptance_by_l or {}
        self.move = move
        self.calls = []
        self.replication_index = {}

    def __call__(self, adapter, initial_state, config):
        del adapter
        state = tf.convert_to_tensor(initial_state, tf.float64)
        self.calls.append((tf.identity(state), config))
        adaptive = config.tuning_policy.uses_dual_averaging
        if adaptive:
            acceptance = 0.70
        else:
            index = self.replication_index.get(config.num_leapfrog_steps, 0)
            self.replication_index[config.num_leapfrog_steps] = index + 1
            acceptance = self.acceptance_by_l.get(
                config.num_leapfrog_steps, (0.68, 0.72)
            )[index]
        samples = tf.broadcast_to(
            state[tf.newaxis, :, :] + (0.1 if self.move else 0.0),
            (config.num_results, 4, 2),
        )
        log_accept = tf.fill(
            (config.num_results, 4),
            tf.math.log(tf.constant(acceptance, tf.float64)),
        )
        trace = {
            "is_accepted": tf.ones((config.num_results, 4), tf.bool),
            "log_accept_ratio": log_accept,
            "target_log_prob": tf.zeros((config.num_results, 4), tf.float64),
            "proposed_target_log_prob": tf.zeros(
                (config.num_results, 4), tf.float64
            ),
            "target_score": tf.zeros((config.num_results, 4, 2), tf.float64),
        }
        return FullChainHMCRunResult(
            samples=samples,
            trace=trace,
            diagnostics={"final_step_size": tf.constant(0.15, tf.float64)},
            metadata={"jit_compile": True},
        )


class QualificationRunner:
    def __init__(self, *, disagree: bool = False):
        self.disagree = disagree
        self.calls = []

    def __call__(self, adapter, initial_state, config):
        del adapter
        state = tf.convert_to_tensor(initial_state, tf.float64)
        self.calls.append(config)
        offset = 0.1
        if self.disagree and config.use_xla and config.num_leapfrog_steps == 9:
            offset = 0.2
        samples = tf.broadcast_to(
            state[tf.newaxis, :, :] + offset,
            (config.num_results, 4, 2),
        )
        trace = {
            "is_accepted": tf.ones((config.num_results, 4), tf.bool),
            "log_accept_ratio": tf.zeros((config.num_results, 4), tf.float64),
            "target_log_prob": tf.zeros((config.num_results, 4), tf.float64),
            "proposed_target_log_prob": tf.zeros(
                (config.num_results, 4), tf.float64
            ),
            "target_score": tf.zeros((config.num_results, 4, 2), tf.float64),
        }
        if config.tuning_policy.uses_dual_averaging:
            trace["step_size"] = tf.fill(
                (config.num_results, 4), tf.constant(0.15, tf.float64)
            )
        return FullChainHMCRunResult(
            samples=samples,
            trace=trace,
            diagnostics={"final_step_size": tf.constant(0.15, tf.float64)},
            metadata={"jit_compile": config.use_xla},
        )


class InvalidAdaptationStatusRunner(FakeRunner):
    def __call__(self, adapter, initial_state, config):
        result = super().__call__(adapter, initial_state, config)
        if not config.tuning_policy.uses_dual_averaging:
            return result
        trace = dict(result.trace)
        trace["target_status_telemetry"] = {
            "status_code": tf.ones((config.num_results, 4), tf.int32),
            "valid_pre_regularized_score": tf.zeros(
                (config.num_results, 4), tf.bool
            ),
        }
        return FullChainHMCRunResult(
            samples=result.samples,
            trace=trace,
            diagnostics=result.diagnostics,
            metadata=result.metadata,
        )


class CampaignRunner:
    def __init__(self, config, calls):
        self.config = config
        self.calls = calls
        self.program_signature = (
            f"fixture:{config.adaptation_policy}:xla={config.use_xla}"
        )
        self.call_count = 0
        self.tracing_count = 1

    def run(self, *, current_state, seed, step_size, num_leapfrog_steps):
        state = tf.convert_to_tensor(current_state, tf.float64)
        step = float(tf.convert_to_tensor(step_size, tf.float64).numpy())
        leapfrog = int(tf.convert_to_tensor(num_leapfrog_steps, tf.int32).numpy())
        seed_value = tuple(int(item) for item in tf.convert_to_tensor(seed).numpy())
        adaptive = self.config.tuning_policy.uses_dual_averaging
        adapted = 0.01 * leapfrog
        self.call_count += 1
        self.calls.append(
            {
                "adaptive": adaptive,
                "use_xla": self.config.use_xla,
                "seed": seed_value,
                "input_step_size": step,
                "num_leapfrog_steps": leapfrog,
            }
        )
        samples = tf.broadcast_to(
            state[tf.newaxis, :, :] + 0.1,
            (self.config.num_results, 4, 2),
        )
        trace = {
            # Binary decisions deliberately contradict the continuous target.
            "is_accepted": tf.zeros((self.config.num_results, 4), tf.bool),
            "log_accept_ratio": tf.fill(
                (self.config.num_results, 4),
                tf.math.log(tf.constant(0.70, tf.float64)),
            ),
            "target_log_prob": tf.zeros(
                (self.config.num_results, 4), tf.float64
            ),
            "proposed_target_log_prob": tf.zeros(
                (self.config.num_results, 4), tf.float64
            ),
            "target_score": tf.zeros(
                (self.config.num_results, 4, 2), tf.float64
            ),
        }
        if adaptive:
            trace["step_size"] = tf.fill(
                (self.config.num_results, 4),
                tf.constant(adapted, tf.float64),
            )
        diagnostics = mechanics.fixed_transport_tensor_diagnostics(samples, trace)
        if adaptive:
            diagnostics["final_step_size"] = tf.constant(adapted, tf.float64)
        return FullChainHMCRunResult(
            samples=samples,
            trace=trace,
            diagnostics=diagnostics,
            metadata={
                "jit_compile": self.config.use_xla,
                "runner_trace_count": 1,
            },
        )


class CampaignRunnerFactory:
    def __init__(self):
        self.calls = []
        self.runners = []

    def __call__(self, _adapter, _state, config):
        runner = CampaignRunner(config, self.calls)
        self.runners.append(runner)
        return runner


class FirstFreshScreenImmobileRunner(CampaignRunner):
    def run(self, *, current_state, seed, step_size, num_leapfrog_steps):
        result = super().run(
            current_state=current_state,
            seed=seed,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
        )
        adaptive = self.config.tuning_policy.uses_dual_averaging
        seed_value = tuple(int(item) for item in tf.convert_to_tensor(seed).numpy())
        if adaptive or seed_value[1] < 2000:
            return result
        state = tf.convert_to_tensor(current_state, tf.float64)
        samples = tf.broadcast_to(
            state[tf.newaxis, :, :],
            (self.config.num_results, 4, 2),
        )
        diagnostics = mechanics.fixed_transport_tensor_diagnostics(samples, result.trace)
        return FullChainHMCRunResult(
            samples=samples,
            trace=result.trace,
            diagnostics=diagnostics,
            metadata=result.metadata,
        )


class FirstFreshScreenImmobileFactory(CampaignRunnerFactory):
    def __call__(self, _adapter, _state, config):
        runner = FirstFreshScreenImmobileRunner(config, self.calls)
        self.runners.append(runner)
        return runner


class RefinementRunner:
    def __init__(self, config, calls, *, acceptance_fn):
        self.config = config
        self.calls = calls
        self.acceptance_fn = acceptance_fn
        self.program_signature = "fixture-program"

    def run(self, *, current_state, seed, step_size, num_leapfrog_steps):
        state = tf.convert_to_tensor(current_state, tf.float64)
        epsilon = float(tf.convert_to_tensor(step_size, tf.float64).numpy())
        leapfrog = int(tf.convert_to_tensor(num_leapfrog_steps, tf.int32).numpy())
        self.calls.append(
            {
                "num_results": self.config.num_results,
                "num_burnin_steps": self.config.num_burnin_steps,
                "epsilon": epsilon,
                "leapfrog": leapfrog,
                "seed": tuple(int(item) for item in tf.convert_to_tensor(seed).numpy()),
            }
        )
        draws = tf.cast(tf.range(self.config.num_results), tf.float64)[:, None, None]
        chain = tf.cast(tf.range(4), tf.float64)[None, :, None]
        parameter = tf.cast(tf.range(2), tf.float64)[None, None, :]
        # Distinct, nondegenerate chain-major values make the rank diagnostic finite.
        samples = state[None, :, :] + 0.01 * draws + 0.02 * chain + 0.01 * parameter
        acceptance = float(self.acceptance_fn(leapfrog, epsilon))
        log_accept = tf.fill(
            (self.config.num_results, 4),
            tf.math.log(tf.constant(acceptance, tf.float64)),
        )
        trace = {
            "is_accepted": tf.ones((self.config.num_results, 4), tf.bool),
            "log_accept_ratio": log_accept,
            "target_log_prob": tf.zeros((self.config.num_results, 4), tf.float64),
            "proposed_target_log_prob": tf.zeros((self.config.num_results, 4), tf.float64),
            "target_score": tf.zeros((self.config.num_results, 4, 2), tf.float64),
        }
        diagnostics = mechanics.fixed_transport_tensor_diagnostics(samples, trace)
        return FullChainHMCRunResult(
            samples=samples,
            trace=trace,
            diagnostics=diagnostics,
            metadata={"jit_compile": True},
        )


class RefinementRunnerFactory:
    def __init__(self, acceptance_fn):
        self.acceptance_fn = acceptance_fn
        self.calls = []

    def __call__(self, _adapter, _state, config):
        return RefinementRunner(config, self.calls, acceptance_fn=self.acceptance_fn)


class InterruptingRefinementRunner(RefinementRunner):
    def __init__(self, config, calls, *, acceptance_fn, fail_on_call):
        super().__init__(config, calls, acceptance_fn=acceptance_fn)
        self.fail_on_call = fail_on_call

    def run(self, **kwargs):
        if len(self.calls) + 1 == self.fail_on_call:
            raise KeyboardInterrupt("simulated candidate process interruption")
        return super().run(**kwargs)


class InterruptingRefinementRunnerFactory(RefinementRunnerFactory):
    def __init__(self, acceptance_fn, *, fail_on_call):
        super().__init__(acceptance_fn)
        self.fail_on_call = fail_on_call

    def __call__(self, _adapter, _state, config):
        return InterruptingRefinementRunner(
            config,
            self.calls,
            acceptance_fn=self.acceptance_fn,
            fail_on_call=self.fail_on_call,
        )


def test_discovers_unranked_six_l_union_and_preserves_start_bank():
    runner = FakeRunner()
    result = module.discover_fixed_transport_hmc_candidates(
        base_adapter=GaussianAdapter(),
        fixed_transport=IdentityTransport(),
        initial_state=START,
        config=_config(),
        run_full_chain=runner,
    )

    assert tuple(item.num_leapfrog_steps for item in result.candidates) == (
        3,
        5,
        9,
        13,
        18,
        25,
    )
    assert len(runner.calls) == 18
    assert all(tf.reduce_all(state == START) for state, _config in runner.calls)
    assert all(item.nominated for item in result.candidates)
    assert result.payload()["selected_candidate_index"] is None
    assert result.payload()["final_kernel_payload"] is None
    assert result.payload()["confirmation_performed"] is False
    assert all(config.use_xla for _state, config in runner.calls)


def test_two_replication_sample_sd_rule_retains_noisy_candidate():
    runner = FakeRunner({3: (0.58, 0.72)})
    result = module.discover_fixed_transport_hmc_candidates(
        base_adapter=GaussianAdapter(),
        fixed_transport=IdentityTransport(),
        initial_state=START,
        config=_config(),
        run_full_chain=runner,
    )
    evidence = result.candidates[0]
    assert evidence.grand_mean == pytest.approx(0.65)
    assert evidence.sample_standard_deviation == pytest.approx(
        ((0.58 - 0.65) ** 2 + (0.72 - 0.65) ** 2) ** 0.5
    )
    assert evidence.nomination_interval[1] > 0.65
    assert evidence.nominated


def test_statistically_incompatible_arm_is_not_a_hard_rejection():
    runner = FakeRunner({3: (0.30, 0.32)})
    result = module.discover_fixed_transport_hmc_candidates(
        base_adapter=GaussianAdapter(),
        fixed_transport=IdentityTransport(),
        initial_state=START,
        config=_config(),
        run_full_chain=runner,
    )
    evidence = result.candidates[0]
    assert evidence.disposition == "statistically_incompatible"
    assert evidence.hard_rejection_reasons == ()


def test_no_movement_is_a_hard_rejection():
    result = module.discover_fixed_transport_hmc_candidates(
        base_adapter=GaussianAdapter(),
        fixed_transport=IdentityTransport(),
        initial_state=START,
        config=_config(),
        run_full_chain=FakeRunner(move=False),
    )
    assert all(item.disposition == "hard_rejected" for item in result.candidates)
    assert all(
        "chain_without_movement" in item.hard_rejection_reasons
        for item in result.candidates
    )


def test_invalid_adaptation_target_status_fails_each_arm():
    result = module.discover_fixed_transport_hmc_candidates(
        base_adapter=GaussianAdapter(),
        fixed_transport=IdentityTransport(),
        initial_state=START,
        config=_config(target_status_trace_policy="per_chain_step"),
        run_full_chain=InvalidAdaptationStatusRunner(),
    )
    assert all(item.disposition == "adaptation_failed" for item in result.candidates)
    assert all(
        "adaptation target status telemetry failed" in item.hard_rejection_reasons[0]
        for item in result.candidates
    )


def test_distinct_start_policy_and_receipt_are_fail_closed():
    duplicate = tf.tensor_scatter_nd_update(START, ((1,),), (START[0],))
    with pytest.raises(ValueError, match="pairwise distinct"):
        module.discover_fixed_transport_hmc_candidates(
            base_adapter=GaussianAdapter(),
            fixed_transport=IdentityTransport(),
            initial_state=duplicate,
            config=_config(),
            run_full_chain=FakeRunner(),
        )


def test_bayesfilter_issues_route_bound_xla_receipt(tmp_path, monkeypatch):
    runner = QualificationRunner()
    monkeypatch.setattr(module, "_run_full_chain_tfp_hmc", runner)
    receipt = module.qualify_fixed_transport_hmc_candidate_discovery_xla(
        base_adapter=GaussianAdapter(),
        fixed_transport=IdentityTransport(),
        initial_state=START,
        config=module.FixedTransportHMCXLAQualificationConfig(
            initial_step_size=0.2,
            adaptation_steps=3,
            tune_num_results=2,
            screen_num_results=4,
            screen_num_burnin_steps=1,
            seed=(20260803, 7),
            value_atol=0.0,
            value_rtol=0.0,
            score_atol=0.0,
            score_rtol=0.0,
            transition_atol=0.0,
            transition_rtol=0.0,
            target_status_trace_policy="none",
        ),
        evidence_path=tmp_path / "qualification.json",
        target_scope="candidate_fixture:fixed_transport_candidate_discovery",
    )

    assert len(runner.calls) == 24
    assert {config.num_leapfrog_steps for config in runner.calls} == set(
        module.PRIMARY_L_GRID
    )
    assert {config.use_xla for config in runner.calls} == {False, True}
    assert receipt.evidence_sha256 == hashlib.sha256(
        Path(receipt.evidence_path).read_bytes()
    ).hexdigest()
    payload = json.loads(Path(receipt.evidence_path).read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert len(payload["runs"]) == 12

    result = module.discover_fixed_transport_hmc_candidates(
        base_adapter=GaussianAdapter(),
        fixed_transport=IdentityTransport(),
        initial_state=START,
        config=_config(xla_qualification=receipt),
        run_full_chain=FakeRunner(),
    )
    assert len(result.candidates) == 6


def test_xla_qualification_fails_on_route_disagreement(tmp_path, monkeypatch):
    monkeypatch.setattr(
        module, "_run_full_chain_tfp_hmc", QualificationRunner(disagree=True)
    )
    receipt = module.qualify_fixed_transport_hmc_candidate_discovery_xla(
        base_adapter=GaussianAdapter(),
        fixed_transport=IdentityTransport(),
        initial_state=START,
        config=module.FixedTransportHMCXLAQualificationConfig(
            initial_step_size=0.2,
            adaptation_steps=3,
            tune_num_results=2,
            screen_num_results=4,
            screen_num_burnin_steps=1,
            seed=(20260803, 7),
            value_atol=0.0,
            value_rtol=0.0,
            score_atol=0.0,
            score_rtol=0.0,
            transition_atol=0.0,
            transition_rtol=0.0,
            target_status_trace_policy="none",
        ),
        evidence_path=tmp_path / "qualification.json",
        target_scope="candidate_fixture:fixed_transport_candidate_discovery",
    )
    assert receipt.status == "passed"
    payload = json.loads((tmp_path / "qualification.json").read_text())
    assert any(
        row.get("diagnostic_role") == "explanatory_stochastic_route_telemetry"
        and row.get("hard_veto") is False
        for row in payload["comparisons"]
    )


def test_discovery_rejects_tampered_xla_evidence():
    receipt = _receipt()
    Path(receipt.evidence_path).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        module.discover_fixed_transport_hmc_candidates(
            base_adapter=GaussianAdapter(),
            fixed_transport=IdentityTransport(),
            initial_state=START,
            config=_config(xla_qualification=receipt),
            run_full_chain=FakeRunner(),
        )
    wrong = module.FixedTransportHMCXLAQualificationReceipt(
        **{**_receipt().__dict__, "transformed_adapter_signature": "wrong"}
    )
    with pytest.raises(ValueError, match="receipt mismatch"):
        module.discover_fixed_transport_hmc_candidates(
            base_adapter=GaussianAdapter(),
            fixed_transport=IdentityTransport(),
            initial_state=START,
            config=_config(xla_qualification=wrong),
            run_full_chain=FakeRunner(),
        )


def test_candidate_and_legacy_default_runners_share_numerical_mechanics():
    assert module._run_full_chain_tfp_hmc is mechanics.run_fixed_transport_full_chain_tfp_hmc
    assert legacy._run_full_chain_tfp_hmc is mechanics.run_fixed_transport_full_chain_tfp_hmc
    assert module._FullChainHMCConfig is mechanics.FixedTransportFullChainConfig
    assert legacy._FullChainHMCConfig is mechanics.FixedTransportFullChainConfig


def test_xla_qualification_and_discovery_are_publicly_exported():
    names = (
        "FixedTransportHMCXLAQualificationConfig",
        "FixedTransportHMCXLAQualificationReceipt",
        "FixedTransportHMCCandidateDiscoveryConfig",
        "discover_fixed_transport_hmc_candidates",
        "qualify_fixed_transport_hmc_candidate_discovery_xla",
    )
    for name in names:
        assert hasattr(inference, name)
        assert hasattr(bayesfilter, name)
        assert name in inference.__all__
        assert name in bayesfilter.__all__


def test_shared_mechanics_owns_no_workflow_decision_symbols():
    tree = ast.parse(inspect.getsource(mechanics))
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    forbidden_fragments = (
        "grid",
        "nomination",
        "selection",
        "refinement",
        "confirmation",
        "convergence",
    )
    assert not {
        name
        for name in defined
        if any(fragment in name.lower() for fragment in forbidden_fragments)
    }
    assert not hasattr(mechanics, "PRIMARY_L_GRID")
    assert not hasattr(mechanics, "FixedTransportHMCCandidateEvidence")
    assert not hasattr(mechanics, "FixedTransportHMCKernelTuningResult")


def test_terminal_step_size_requires_scalar_or_equal_terminal_replicas():
    scalar = mechanics.fixed_transport_terminal_step_size(
        {"step_size": tf.constant(0.2, tf.float64)}
    )
    replicated = mechanics.fixed_transport_terminal_step_size(
        {
            "step_size": tf.constant(
                [[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2]], tf.float64
            )
        }
    )
    assert float(scalar.numpy()) == pytest.approx(0.2)
    assert float(replicated.numpy()) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="replicas disagree"):
        mechanics.fixed_transport_terminal_step_size(
            {
                "step_size": tf.constant(
                    [[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.3, 0.2]],
                    tf.float64,
                )
            }
        )


def test_reusable_runner_changes_tensor_epsilon_and_l_without_retracing():
    adapter = module._adapter(
        base_adapter=GaussianAdapter(),
        fixed_transport=IdentityTransport(),
        target_scope="candidate_fixture:reusable_runner",
    )
    config = mechanics.FixedTransportFullChainConfig(
        num_results=2,
        num_burnin_steps=1,
        step_size=0.05,
        num_leapfrog_steps=3,
        seed=(20260804, 1),
        use_xla=False,
        trace_policy="standard",
        target_status_trace_policy="none",
        tuning_policy=mechanics.FixedTransportHMCPolicy.fixed(source="test"),
        target_scope=adapter.target_scope,
        chain_execution_mode="tf_function",
    )
    runner = mechanics.build_fixed_transport_reusable_runner(adapter, START, config)
    first = runner.run(
        current_state=START,
        seed=(20260804, 2),
        step_size=tf.constant(0.03, tf.float64),
        num_leapfrog_steps=tf.constant(3, tf.int32),
    )
    second = runner.run(
        current_state=START,
        seed=(20260804, 2),
        step_size=tf.constant(0.09, tf.float64),
        num_leapfrog_steps=tf.constant(25, tf.int32),
    )
    assert runner.call_count == 2
    assert runner.tracing_count == 1
    assert first.metadata["runner_program_signature"] == second.metadata[
        "runner_program_signature"
    ]
    assert not bool(tf.reduce_all(tf.equal(first.samples, second.samples)).numpy())


def test_one_process_campaign_reuses_four_runners_and_exact_adapted_epsilon():
    factory = CampaignRunnerFactory()
    config = module.FixedTransportHMCCandidateCampaignConfig(
        initial_step_size=0.0001,
        adaptation_steps=3,
        tune_num_results=2,
        screen_num_results=4,
        screen_num_burnin_steps=1,
        value_atol=0.0,
        value_rtol=0.0,
        score_atol=0.0,
        score_rtol=0.0,
        transition_atol=0.0,
        transition_rtol=0.0,
        wall_cap_seconds=60.0,
        target_scope="candidate_fixture:fixed_transport_candidate_campaign",
        target_status_trace_policy="none",
        gaussian_pilot_grid_size=33,
        gaussian_pilot_draw_count=128,
    )
    result = module.run_fixed_transport_hmc_candidate_campaign(
        base_adapter=GaussianAdapter(),
        fixed_transport=IdentityTransport(),
        initial_state=START,
        config=config,
        runner_factory=factory,
    )

    assert len(factory.runners) == 4
    assert result.hmc_call_count == 36
    assert tuple(item.num_leapfrog_steps for item in result.candidates) == (
        3,
        5,
        9,
        13,
        18,
        25,
    )
    assert all(item.nominated for item in result.candidates)
    assert result.target_qualification["tracing_count"] == 1
    assert set(result.epsilon_warm_starts) == {3, 5, 9, 13, 18, 25}
    assert all(
        row["route"] == module.GAUSSIAN_EPSILON_WARM_START_ROUTE
        for row in result.epsilon_warm_starts.values()
    )
    assert {receipt["tracing_count"] for receipt in result.runner_evidence.values()} == {1}
    assert {receipt["call_count"] for receipt in result.runner_evidence.values()} == {
        6,
        18,
    }
    adaptive_calls = [call for call in factory.calls if call["adaptive"]]
    for call in adaptive_calls:
        expected = result.epsilon_warm_starts[call["num_leapfrog_steps"]]["epsilon"]
        assert call["input_step_size"] == pytest.approx(expected)
    fixed_calls = [call for call in factory.calls if not call["adaptive"]]
    for call in fixed_calls:
        assert call["input_step_size"] == pytest.approx(
            0.01 * call["num_leapfrog_steps"]
        )
    assert all(item.replication_means == pytest.approx((0.70, 0.70)) for item in result.candidates)
    assert result.payload()["selected_candidate_index"] is None
    assert result.payload()["confirmation_performed"] is False
    assert result.payload()["retained_sampling_authorized"] is False
    assert all(
        row["binary_acceptance_rate_explanatory_only"] == 0.0
        for arm in result.arm_evidence
        for row in arm["fresh_replications"]
    )
    assert all(
        row["acceptance_probability_by_chain"] == pytest.approx([0.70] * 4)
        for arm in result.arm_evidence
        for row in arm["fresh_replications"]
    )


def test_campaign_preserves_partial_arm_evidence_after_movement_veto():
    factory = FirstFreshScreenImmobileFactory()
    config = module.FixedTransportHMCCandidateCampaignConfig(
        initial_step_size=0.5,
        adaptation_steps=3,
        tune_num_results=2,
        screen_num_results=4,
        screen_num_burnin_steps=1,
        value_atol=0.0,
        value_rtol=0.0,
        score_atol=0.0,
        score_rtol=0.0,
        transition_atol=0.0,
        transition_rtol=0.0,
        wall_cap_seconds=60.0,
        target_scope="candidate_fixture:fixed_transport_candidate_campaign",
        target_status_trace_policy="none",
        gaussian_pilot_grid_size=17,
        gaussian_pilot_draw_count=64,
    )
    result = module.run_fixed_transport_hmc_candidate_campaign(
        base_adapter=GaussianAdapter(),
        fixed_transport=IdentityTransport(),
        initial_state=START,
        config=config,
        runner_factory=factory,
    )

    assert result.hmc_call_count == 30
    assert all(item.disposition == "hard_rejected" for item in result.candidates)
    assert all(item.tuned_step_size is not None for item in result.candidates)
    assert all(item.replication_means == pytest.approx((0.70,)) for item in result.candidates)
    for arm in result.arm_evidence:
        assert arm["adapted_epsilon"] == pytest.approx(0.01 * arm["num_leapfrog_steps"])
        assert set(arm["adaptation_checks"]) == {"adaptive_control", "adaptive_xla"}
        assert set(arm["exact_epsilon_checks"]) == {"fixed_control", "fixed_xla"}
        assert len(arm["fresh_replications"]) == 1
        row = arm["fresh_replications"][0]
        assert row["acceptance_mean"] == pytest.approx(0.70)
        assert row["acceptance_probability_by_chain"] == pytest.approx([0.70] * 4)
        assert row["diagnostics"]["moved_by_chain"] == [False] * 4
        assert row["hard_rejection_reasons"] == ("chain_without_movement",)


def test_campaign_route_disagreement_is_explanatory_and_fixed_screens_continue():
    factory = CampaignRunnerFactory()
    config = module.FixedTransportHMCCandidateCampaignConfig(
        initial_step_size=0.0001,
        adaptation_steps=3,
        tune_num_results=2,
        screen_num_results=4,
        screen_num_burnin_steps=1,
        value_atol=0.0,
        value_rtol=0.0,
        score_atol=0.0,
        score_rtol=0.0,
        transition_atol=0.0,
        transition_rtol=0.0,
        wall_cap_seconds=60.0,
        target_scope="candidate_fixture:fixed_transport_candidate_campaign",
        target_status_trace_policy="none",
        gaussian_pilot_grid_size=17,
        gaussian_pilot_draw_count=64,
    )
    result = module.run_fixed_transport_hmc_candidate_campaign(
        base_adapter=GaussianAdapter(),
        fixed_transport=IdentityTransport(),
        initial_state=START,
        config=config,
        runner_factory=factory,
    )
    assert result.final_status == "candidate_set_found"
    assert result.hmc_call_count == 36
    assert all(
        all(
            row["diagnostic_role"] == "explanatory_stochastic_route_telemetry"
            and row["hard_veto"] is False
            for row in arm["adaptation_comparisons"]
        )
        for arm in result.arm_evidence
    )
    assert {row["call_count"] for row in result.runner_evidence.values()} == {6, 18}


def test_candidate_refinement_runs_500_then_continues_500_and_selects_lowest_rhat():
    factory = RefinementRunnerFactory(lambda leapfrog, epsilon: 0.70)
    result = module.refine_fixed_transport_hmc_candidates(
        base_adapter=GaussianAdapter(),
        fixed_transport=IdentityTransport(),
        initial_state=START,
        candidates=(
            {"num_leapfrog_steps": 3, "epsilon": 0.20},
            {"num_leapfrog_steps": 5, "epsilon": 0.15},
        ),
        config=module.FixedTransportCandidateRefinementConfig(
            target_scope="candidate_fixture:refinement",
            target_status_trace_policy="none",
        ),
        runner_factory=factory,
    )
    assert [stage["num_results"] for stage in result.stages] == [500, 500]
    assert [stage["num_burnin_steps"] for stage in result.stages] == [500, 0]
    assert result.stages[1]["stage_role"] == "continued_candidate_diagnostic"
    assert all(stage["survivor_count"] == 2 for stage in result.stages)
    assert result.selected_candidate["num_leapfrog_steps"] in {3, 5}
    assert result.selected_candidate["rhat"]["maximum_over_parameters"] == pytest.approx(
        min(
            row["rhat"]["maximum_over_parameters"]
            for row in result.stages[-1]["attempts"][-1]["candidates"]
        )
    )
    assert {call["num_results"] for call in factory.calls} == {500}
    assert {call["num_burnin_steps"] for call in factory.calls} == {0, 500}
    for row in result.stages[1]["attempts"][-1]["candidates"]:
        assert row["continued_from_previous_stage"] is True
        prior = next(
            item
            for item in result.stages[0]["attempts"][-1]["candidates"]
            if item["num_leapfrog_steps"] == row["num_leapfrog_steps"]
        )
        assert row["initial_state_hash"] == prior["final_state_hash"]
    assert result.payload()["retained_sampling_authorized"] is False


def test_required_xla_refinement_runs_exactly_warmup_then_diagnostic(monkeypatch):
    receipt = module.SequentialNeuTraHMCXLAQualificationReceipt(
        status="passed",
        policy_id=module.NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
        adapter_signature="fixture",
        initial_state_shape=(4, 2),
        chunk_results=500,
        program_signature="fixture-program",
        tracing_count=1,
        target_value_max_abs_residual=0.0,
        target_score_max_abs_residual=0.0,
        all_chains_moved=True,
        final_state_equals_last_sample=True,
        sequential_handoff_verified=True,
        target_status_passed=True,
        evidence_path="fixture.json",
        evidence_sha256="fixture",
        qualification_code_hash="fixture-code",
    )
    validations = []
    monkeypatch.setattr(
        module,
        "validate_sequential_neutra_hmc_xla_receipt",
        lambda *args, **kwargs: validations.append((args, kwargs)),
    )
    factory = RefinementRunnerFactory(lambda leapfrog, epsilon: 0.70)
    result = module.refine_fixed_transport_hmc_candidates(
        base_adapter=GaussianAdapter(),
        fixed_transport=IdentityTransport(),
        initial_state=START,
        candidates=(
            {"num_leapfrog_steps": 3, "epsilon": 0.20},
            {"num_leapfrog_steps": 5, "epsilon": 0.15},
        ),
        config=module.FixedTransportCandidateRefinementConfig(
            target_scope="candidate_fixture:refinement",
            target_status_trace_policy="none",
            xla_qualification=receipt,
            xla_qualification_required=True,
        ),
        runner_factory=factory,
    )
    assert validations
    assert len(factory.calls) == 6
    assert [call["num_burnin_steps"] for call in factory.calls] == [0] * 6
    assert result.stages[0]["survivor_count"] == 2
    first_stage = result.stages[0]["attempts"][0]["candidates"]
    second_stage = result.stages[1]["attempts"][0]["candidates"]
    for row in second_stage:
        prior = next(
            item
            for item in first_stage
            if item["num_leapfrog_steps"] == row["num_leapfrog_steps"]
        )
        assert row["initial_state_hash"] == prior["final_state_hash"]


def test_candidate_refinement_resumes_after_warmup_checkpoint(tmp_path, monkeypatch):
    receipt = module.SequentialNeuTraHMCXLAQualificationReceipt(
        status="passed",
        policy_id=module.NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
        adapter_signature="fixture",
        initial_state_shape=(4, 2),
        chunk_results=500,
        program_signature="fixture-program",
        tracing_count=1,
        target_value_max_abs_residual=0.0,
        target_score_max_abs_residual=0.0,
        all_chains_moved=True,
        final_state_equals_last_sample=True,
        sequential_handoff_verified=True,
        target_status_passed=True,
        evidence_path="fixture.json",
        evidence_sha256="fixture",
        qualification_code_hash="fixture-code",
    )
    monkeypatch.setattr(
        module, "validate_sequential_neutra_hmc_xla_receipt", lambda *a, **k: None
    )
    config = module.FixedTransportCandidateRefinementConfig(
        target_scope="candidate_fixture:refinement",
        target_status_trace_policy="none",
        xla_qualification=receipt,
        xla_qualification_required=True,
    )
    candidates = ({"num_leapfrog_steps": 3, "epsilon": 0.20},)
    interrupted = InterruptingRefinementRunnerFactory(
        lambda leapfrog, epsilon: 0.70, fail_on_call=2
    )
    with pytest.raises(KeyboardInterrupt, match="simulated"):
        module.refine_fixed_transport_hmc_candidates(
            base_adapter=GaussianAdapter(),
            fixed_transport=IdentityTransport(),
            initial_state=START,
            candidates=candidates,
            config=config,
            runner_factory=interrupted,
            checkpoint_dir=tmp_path / "checkpoint",
        )
    assert len(interrupted.calls) == 1
    resumed = RefinementRunnerFactory(lambda leapfrog, epsilon: 0.70)
    result = module.refine_fixed_transport_hmc_candidates(
        base_adapter=GaussianAdapter(),
        fixed_transport=IdentityTransport(),
        initial_state=START,
        candidates=candidates,
        config=config,
        runner_factory=resumed,
        checkpoint_dir=tmp_path / "checkpoint",
        resume=True,
    )
    assert result.final_status == "candidate_selected"
    assert len(resumed.calls) == 1


def test_candidate_refinement_checkpoint_rejects_mismatch_corruption_orphan_and_terminal(
    tmp_path, monkeypatch
):
    receipt = module.SequentialNeuTraHMCXLAQualificationReceipt(
        status="passed", policy_id=module.NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
        adapter_signature="fixture", initial_state_shape=(4, 2), chunk_results=500,
        program_signature="fixture-program", tracing_count=1,
        target_value_max_abs_residual=0.0, target_score_max_abs_residual=0.0,
        all_chains_moved=True, final_state_equals_last_sample=True,
        sequential_handoff_verified=True, target_status_passed=True,
        evidence_path="fixture.json", evidence_sha256="fixture",
        qualification_code_hash="fixture-code",
    )
    monkeypatch.setattr(
        module, "validate_sequential_neutra_hmc_xla_receipt", lambda *a, **k: None
    )
    config = module.FixedTransportCandidateRefinementConfig(
        target_scope="candidate_fixture:refinement",
        target_status_trace_policy="none",
        xla_qualification=receipt,
        xla_qualification_required=True,
    )
    candidates = ({"num_leapfrog_steps": 3, "epsilon": 0.20},)

    def interrupted_root(name):
        root = tmp_path / name
        with pytest.raises(KeyboardInterrupt):
            module.refine_fixed_transport_hmc_candidates(
                base_adapter=GaussianAdapter(), fixed_transport=IdentityTransport(),
                initial_state=START, candidates=candidates, config=config,
                runner_factory=InterruptingRefinementRunnerFactory(
                    lambda leapfrog, epsilon: 0.70, fail_on_call=2
                ), checkpoint_dir=root,
            )
        return root

    mismatch = interrupted_root("mismatch")
    changed = module.FixedTransportCandidateRefinementConfig(
        target_scope="candidate_fixture:refinement",
        target_status_trace_policy="none",
        seed_base=(20260809, 9999),
        xla_qualification=receipt,
        xla_qualification_required=True,
    )
    with pytest.raises(RuntimeError, match="run contract mismatch"):
        module.refine_fixed_transport_hmc_candidates(
            base_adapter=GaussianAdapter(), fixed_transport=IdentityTransport(),
            initial_state=START, candidates=candidates, config=changed,
            runner_factory=RefinementRunnerFactory(lambda l, e: 0.70),
            checkpoint_dir=mismatch, resume=True,
        )

    corrupt = interrupted_root("corrupt")
    state_path = next((corrupt / "states").glob("*.tftensor"))
    state_path.write_bytes(state_path.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="byte count mismatch"):
        module.refine_fixed_transport_hmc_candidates(
            base_adapter=GaussianAdapter(), fixed_transport=IdentityTransport(),
            initial_state=START, candidates=candidates, config=config,
            runner_factory=RefinementRunnerFactory(lambda l, e: 0.70),
            checkpoint_dir=corrupt, resume=True,
        )

    orphan = interrupted_root("orphan")
    (orphan / "orphan.bin").write_bytes(b"orphan")
    with pytest.raises(RuntimeError, match="orphan artifacts"):
        module.refine_fixed_transport_hmc_candidates(
            base_adapter=GaussianAdapter(), fixed_transport=IdentityTransport(),
            initial_state=START, candidates=candidates, config=config,
            runner_factory=RefinementRunnerFactory(lambda l, e: 0.70),
            checkpoint_dir=orphan, resume=True,
        )

    terminal = tmp_path / "terminal"
    module.refine_fixed_transport_hmc_candidates(
        base_adapter=GaussianAdapter(), fixed_transport=IdentityTransport(),
        initial_state=START, candidates=candidates, config=config,
        runner_factory=RefinementRunnerFactory(lambda l, e: 0.70),
        checkpoint_dir=terminal,
    )
    with pytest.raises(RuntimeError, match="checkpoint is terminal"):
        module.refine_fixed_transport_hmc_candidates(
            base_adapter=GaussianAdapter(), fixed_transport=IdentityTransport(),
            initial_state=START, candidates=candidates, config=config,
            runner_factory=RefinementRunnerFactory(lambda l, e: 0.70),
            checkpoint_dir=terminal, resume=True,
        )


def test_candidate_refinement_repairs_low_acceptance_once():
    def acceptance(leapfrog, epsilon):
        del leapfrog
        return 0.40 if epsilon < 0.2 else 0.70

    factory = RefinementRunnerFactory(acceptance)
    result = module.refine_fixed_transport_hmc_candidates(
        base_adapter=GaussianAdapter(),
        fixed_transport=IdentityTransport(),
        initial_state=START,
        candidates=({"num_leapfrog_steps": 3, "epsilon": 0.10},),
        config=module.FixedTransportCandidateRefinementConfig(
            target_scope="candidate_fixture:refinement",
            target_status_trace_policy="none",
        ),
        runner_factory=factory,
    )
    assert [stage["num_results"] for stage in result.stages] == [500]
    assert len(result.stages[0]["attempts"]) == 2
    assert result.stages[0]["attempts"][1]["candidates"][0]["epsilon"] == pytest.approx(0.08)
    assert result.final_status == "no_candidate_after_refinement"


def test_explanatory_route_comparison_recurses_nested_status_mapping():
    left = SimpleNamespace(
        samples=tf.zeros((2, 4, 2), tf.float64),
        trace={"target_status_telemetry": {"status_code": tf.zeros((2, 4), tf.int32)}},
    )
    right = SimpleNamespace(
        samples=tf.ones((2, 4, 2), tf.float64),
        trace={"target_status_telemetry": {"status_code": tf.ones((2, 4), tf.int32)}},
    )
    rows = module._explanatory_route_comparisons(
        left,
        right,
        atol=0.0,
        rtol=0.0,
        prefix="nested",
    )
    assert any(row["label"] == "nested.trace.target_status_telemetry.status_code" for row in rows)
    assert all(row["hard_veto"] is False for row in rows)


def test_campaign_comparison_delegates_to_recursive_route_helper():
    source = inspect.getsource(module._campaign_compare_runs)
    assert "_explanatory_route_comparisons(" in source
    assert "tf.convert_to_tensor" not in source
    campaign_source = inspect.getsource(module.run_fixed_transport_hmc_candidate_campaign)
    terminal_block = campaign_source.split("step_comparison =", 1)[1].split(")", 1)[0]
    assert "_explanatory_value_comparison(" in terminal_block
