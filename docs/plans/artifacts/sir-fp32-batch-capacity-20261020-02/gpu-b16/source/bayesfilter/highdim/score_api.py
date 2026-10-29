"""Score API contracts for high-dimensional evidence-class readiness."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import tensorflow as tf

from bayesfilter.highdim.diagnostics import HighDimStatus, freeze_mapping
from bayesfilter.highdim.fixed_branch import BranchHash, BranchIdentity, BranchManifest


_ALLOWED_EVIDENCE_CLASSES = frozenset({"lower_rung", "production"})
_ALLOWED_STABLE_SCORE_ROUTE_LABELS = frozenset(
    {
        "hmc_compatible_deterministic_filtering",
        "gradient_calibration_diagnostic",
        "documented-deviation fixed-design substitute",
    }
)
_ALLOWED_HMC_STATUSES = frozenset(
    {
        "blocked_tier2_tier3_not_run",
        "blocked_missing_score_evidence",
        "blocked_missing_production_filtering_token",
        "not_requested",
    }
)
_P49_ALLOWED_GRADIENT_HMC_STATUSES = frozenset(
    {
        "blocked_tier2_tier3_not_run",
        "blocked_missing_score_evidence",
        "not_requested",
        "promoted",
    }
)
_P49_ALLOWED_HMC_TIERS = frozenset(
    {
        "TIER_1_LOCAL_VALUE_AND_DIRECTIONAL_SCORE",
        "TIER_2_SHORT_CHAIN_DIAGNOSTICS",
        "TIER_3_HAMILTONIAN_LEAPFROG_FOR_HMC",
    }
)


@dataclass(frozen=True)
class ExperimentalScoreAPIResult:
    """Frozen result for a small deterministic score API evaluation."""

    target_id: str
    evidence_class: str
    m1_route_label: str
    parameterization: str
    theta: tf.Tensor
    log_likelihood: tf.Tensor
    score: tf.Tensor
    branch_identity: BranchIdentity
    status: HighDimStatus
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        target_id = _require_text("target_id", self.target_id)
        evidence_class = _require_text("evidence_class", self.evidence_class)
        if evidence_class not in _ALLOWED_EVIDENCE_CLASSES:
            raise ValueError("evidence_class must be lower_rung or production")
        m1_route_label = _require_text("m1_route_label", self.m1_route_label)
        if m1_route_label not in {
            "adaptive route candidate",
            "documented-deviation fixed-design substitute",
        }:
            raise ValueError("invalid M1 route label")
        parameterization = _require_text("parameterization", self.parameterization)
        theta = tf.convert_to_tensor(self.theta, dtype=tf.float64)
        value = tf.convert_to_tensor(self.log_likelihood, dtype=tf.float64)
        score = tf.convert_to_tensor(self.score, dtype=tf.float64)
        if theta.shape.rank != 1 or score.shape != theta.shape or value.shape.rank != 0:
            raise ValueError(f"score API result: {HighDimStatus.INVALID_SHAPE.value}")
        if not bool(
            tf.reduce_all(tf.math.is_finite(theta)).numpy()
            and tf.math.is_finite(value).numpy()
            and tf.reduce_all(tf.math.is_finite(score)).numpy()
        ):
            raise ValueError(f"score API result: {HighDimStatus.NONFINITE_VALUE.value}")
        if not isinstance(self.branch_identity, BranchIdentity):
            raise TypeError("branch_identity must be BranchIdentity")
        if not isinstance(self.status, HighDimStatus):
            raise TypeError("status must be HighDimStatus")
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "evidence_class", evidence_class)
        object.__setattr__(self, "m1_route_label", m1_route_label)
        object.__setattr__(self, "parameterization", parameterization)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "log_likelihood", value)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "diagnostics", freeze_mapping(self.diagnostics))


@dataclass(frozen=True)
class HighDimScoreAPIResult:
    """Stable subpackage-scoped value/score API result.

    The contract is intentionally scoped to ``bayesfilter.highdim``.  A finite
    scalar value and score do not imply HMC readiness or a root-level
    ``bayesfilter`` public API export.
    """

    target_id: str
    evidence_class: str
    route_label: str
    parameterization: str
    theta: tf.Tensor
    log_likelihood: tf.Tensor
    score: tf.Tensor
    branch_identity: BranchIdentity
    status: HighDimStatus
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        target_id = _require_text("target_id", self.target_id)
        evidence_class = _require_text("evidence_class", self.evidence_class)
        if evidence_class not in _ALLOWED_EVIDENCE_CLASSES:
            raise ValueError("evidence_class must be lower_rung or production")
        route_label = _require_text("route_label", self.route_label)
        if route_label not in _ALLOWED_STABLE_SCORE_ROUTE_LABELS:
            raise ValueError("invalid stable score route label")
        parameterization = _require_text("parameterization", self.parameterization)
        theta = tf.convert_to_tensor(self.theta, dtype=tf.float64)
        value = tf.convert_to_tensor(self.log_likelihood, dtype=tf.float64)
        score = tf.convert_to_tensor(self.score, dtype=tf.float64)
        if theta.shape.rank != 1 or score.shape != theta.shape or value.shape.rank != 0:
            raise ValueError(f"score API result: {HighDimStatus.INVALID_SHAPE.value}")
        if not bool(
            tf.reduce_all(tf.math.is_finite(theta)).numpy()
            and tf.math.is_finite(value).numpy()
            and tf.reduce_all(tf.math.is_finite(score)).numpy()
        ):
            raise ValueError(f"score API result: {HighDimStatus.NONFINITE_VALUE.value}")
        if not isinstance(self.branch_identity, BranchIdentity):
            raise TypeError("branch_identity must be BranchIdentity")
        if not isinstance(self.status, HighDimStatus):
            raise TypeError("status must be HighDimStatus")
        diagnostics = freeze_mapping(self.diagnostics)
        if diagnostics.get("api_scope") != "bayesfilter.highdim":
            raise ValueError("stable score API result must declare bayesfilter.highdim scope")
        if diagnostics.get("stable_subpackage_api") is not True:
            raise ValueError("stable score API result must declare stable subpackage API")
        if diagnostics.get("stable_top_level_api") is not False:
            raise ValueError("stable score API result must not claim stable top-level API")
        if diagnostics.get("hmc_readiness") != "not_claimed":
            raise ValueError("stable score API result must not claim HMC readiness")
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "evidence_class", evidence_class)
        object.__setattr__(self, "route_label", route_label)
        object.__setattr__(self, "parameterization", parameterization)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "log_likelihood", value)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "diagnostics", diagnostics)


@dataclass(frozen=True)
class HighDimBatchedScoreAPIResult:
    """Stable subpackage-scoped batched value/score API result."""

    target_id: str
    evidence_class: str
    route_label: str
    parameterization: str
    theta: tf.Tensor
    log_likelihoods: tf.Tensor
    score: tf.Tensor
    branch_identities: tuple[BranchIdentity, ...]
    status: HighDimStatus
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        target_id = _require_text("target_id", self.target_id)
        evidence_class = _require_text("evidence_class", self.evidence_class)
        if evidence_class not in _ALLOWED_EVIDENCE_CLASSES:
            raise ValueError("evidence_class must be lower_rung or production")
        route_label = _require_text("route_label", self.route_label)
        if route_label not in _ALLOWED_STABLE_SCORE_ROUTE_LABELS:
            raise ValueError("invalid stable score route label")
        parameterization = _require_text("parameterization", self.parameterization)
        theta = tf.convert_to_tensor(self.theta, dtype=tf.float64)
        values = tf.convert_to_tensor(self.log_likelihoods, dtype=tf.float64)
        score = tf.convert_to_tensor(self.score, dtype=tf.float64)
        if theta.shape.rank != 1 or values.shape.rank != 1 or score.shape.rank != 2:
            raise ValueError(f"batched score API result: {HighDimStatus.INVALID_SHAPE.value}")
        if int(score.shape[1]) != int(theta.shape[0]):
            raise ValueError(f"batched score API result: {HighDimStatus.INVALID_SHAPE.value}")
        if int(score.shape[0]) != int(values.shape[0]):
            raise ValueError(f"batched score API result: {HighDimStatus.INVALID_SHAPE.value}")
        branch_identities = tuple(self.branch_identities)
        if len(branch_identities) != int(values.shape[0]) or not branch_identities:
            raise ValueError("branch_identities must match nonempty batch size")
        if any(not isinstance(identity, BranchIdentity) for identity in branch_identities):
            raise TypeError("branch_identities must contain BranchIdentity values")
        if not bool(
            tf.reduce_all(tf.math.is_finite(theta)).numpy()
            and tf.reduce_all(tf.math.is_finite(values)).numpy()
            and tf.reduce_all(tf.math.is_finite(score)).numpy()
        ):
            raise ValueError(f"batched score API result: {HighDimStatus.NONFINITE_VALUE.value}")
        if not isinstance(self.status, HighDimStatus):
            raise TypeError("status must be HighDimStatus")
        diagnostics = freeze_mapping(self.diagnostics)
        if diagnostics.get("api_scope") != "bayesfilter.highdim":
            raise ValueError("batched score API result must declare bayesfilter.highdim scope")
        if diagnostics.get("stable_subpackage_api") is not True:
            raise ValueError("batched score API result must declare stable subpackage API")
        if diagnostics.get("stable_top_level_api") is not False:
            raise ValueError("batched score API result must not claim stable top-level API")
        if diagnostics.get("hmc_readiness") != "not_claimed":
            raise ValueError("batched score API result must not claim HMC readiness")
        if diagnostics.get("setup_identity_channel") != "diagnostics_and_branch_manifest":
            raise ValueError("batched score API result must expose setup identity channel")
        if diagnostics.get("batch_identity_mode") not in {
            "shared_setup_identity",
            "per_item_setup_identity",
        }:
            raise ValueError("batched score API result has invalid batch identity mode")
        expected_hashes = tuple(identity.hash.value for identity in branch_identities)
        observed_hashes = tuple(
            str(value) for value in diagnostics.get("per_item_branch_hashes", ())
        )
        if observed_hashes != expected_hashes:
            raise ValueError("batched score API result branch hash diagnostics mismatch")
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "evidence_class", evidence_class)
        object.__setattr__(self, "route_label", route_label)
        object.__setattr__(self, "parameterization", parameterization)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "log_likelihoods", values)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "branch_identities", branch_identities)
        object.__setattr__(self, "diagnostics", diagnostics)


@dataclass(frozen=True)
class ScoreReadinessRow:
    """Readiness-table row with explicit evidence-class boundaries."""

    target_id: str
    evidence_class: str
    upstream_tokens: tuple[str, ...]
    p42_tiers_passed: tuple[str, ...]
    api_status: str
    hmc_status: str
    promoted_claim: str
    forbidden_claims: tuple[str, ...]
    m1_route_label: str

    def __post_init__(self) -> None:
        target_id = _require_text("target_id", self.target_id)
        evidence_class = _require_text("evidence_class", self.evidence_class)
        if evidence_class not in _ALLOWED_EVIDENCE_CLASSES:
            raise ValueError("evidence_class must be lower_rung or production")
        api_status = _require_text("api_status", self.api_status)
        hmc_status = _require_text("hmc_status", self.hmc_status)
        if hmc_status not in _ALLOWED_HMC_STATUSES:
            raise ValueError("unknown HMC status")
        m1_route_label = _require_text("m1_route_label", self.m1_route_label)
        if m1_route_label not in {
            "adaptive route candidate",
            "documented-deviation fixed-design substitute",
        }:
            raise ValueError("invalid M1 route label")
        upstream_tokens = _text_tuple("upstream_tokens", self.upstream_tokens)
        p42_tiers = tuple(str(value).strip() for value in self.p42_tiers_passed)
        forbidden = _text_tuple("forbidden_claims", self.forbidden_claims)
        claim = _require_text("promoted_claim", self.promoted_claim)
        if evidence_class == "production" and "production" not in target_id:
            raise ValueError("production rows must use production target ids")
        if "production" in claim.lower() and evidence_class != "production":
            raise ValueError("lower-rung rows cannot promote production claims")
        claim_lower = claim.lower()
        if "hmc readiness" in claim_lower and "no " not in claim_lower and "not " not in claim_lower:
            raise ValueError("HMC readiness cannot be promoted by this P47-M6 table")
        if api_status.startswith("passed") and "TIER_1_LOCAL_VALUE_AND_DIRECTIONAL_SCORE" not in p42_tiers:
            raise ValueError("passed API score rows require P42 Tier 1 evidence")
        if api_status.startswith("blocked") and p42_tiers:
            raise ValueError("blocked API rows must not list passed P42 tiers")
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "evidence_class", evidence_class)
        object.__setattr__(self, "upstream_tokens", upstream_tokens)
        object.__setattr__(self, "p42_tiers_passed", p42_tiers)
        object.__setattr__(self, "api_status", api_status)
        object.__setattr__(self, "hmc_status", hmc_status)
        object.__setattr__(self, "promoted_claim", claim)
        object.__setattr__(self, "forbidden_claims", forbidden)
        object.__setattr__(self, "m1_route_label", m1_route_label)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "target_id": self.target_id,
            "evidence_class": self.evidence_class,
            "upstream_tokens": self.upstream_tokens,
            "p42_tiers_passed": self.p42_tiers_passed,
            "api_status": self.api_status,
            "hmc_status": self.hmc_status,
            "promoted_claim": self.promoted_claim,
            "forbidden_claims": self.forbidden_claims,
            "m1_route_label": self.m1_route_label,
        }


@dataclass(frozen=True)
class ScoreReadinessManifest:
    """P47-M6 readiness manifest separating API and HMC evidence classes."""

    phase: str
    rows: tuple[ScoreReadinessRow, ...]
    pass_token: str
    nonclaims: tuple[str, ...]

    def __post_init__(self) -> None:
        phase = _require_text("phase", self.phase)
        if phase != "P47-M6":
            raise ValueError("phase must be P47-M6")
        rows = tuple(self.rows)
        if not rows:
            raise ValueError("rows must be nonempty")
        if any(not isinstance(row, ScoreReadinessRow) for row in rows):
            raise TypeError("rows must contain ScoreReadinessRow")
        pass_token = _require_text("pass_token", self.pass_token)
        if pass_token != "PASS_P47_M6_SCORE_HMC_READINESS":
            raise ValueError("unexpected P47-M6 pass token")
        nonclaims = _text_tuple("nonclaims", self.nonclaims)
        text = " ".join(nonclaims).lower()
        for required in (
            "no production hmc readiness",
            "no production score api",
            "lower-rung rows are not production",
        ):
            if required not in text:
                raise ValueError(f"missing nonclaim: {required}")
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "pass_token", pass_token)
        object.__setattr__(self, "nonclaims", nonclaims)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "phase": self.phase,
            "pass_token": self.pass_token,
            "rows": tuple(row.manifest_payload() for row in self.rows),
            "nonclaims": self.nonclaims,
        }


@dataclass(frozen=True)
class GradientLaneEvidenceContract:
    """P49 boundary for deterministic gradient-bearing adaptation evidence."""

    route_label: str
    branch_replay_status: str
    value_gradient_status: str
    likelihood_variance_calibration_status: str
    hmc_readiness_status: str
    source_fidelity_claim: bool
    differentiates_adaptive_random_branch: bool
    required_hmc_tiers: tuple[str, ...]
    nonclaims: tuple[str, ...]

    def __post_init__(self) -> None:
        route_label = _require_text("route_label", self.route_label)
        if route_label != "gradient_bearing_adaptation":
            raise ValueError("gradient lane must use gradient_bearing_adaptation route label")
        branch_replay = _require_text("branch_replay_status", self.branch_replay_status)
        value_gradient = _require_text("value_gradient_status", self.value_gradient_status)
        variance = _require_text(
            "likelihood_variance_calibration_status",
            self.likelihood_variance_calibration_status,
        )
        hmc = _require_text("hmc_readiness_status", self.hmc_readiness_status)
        required_hmc_tiers = tuple(str(item).strip() for item in self.required_hmc_tiers)
        if hmc not in _P49_ALLOWED_GRADIENT_HMC_STATUSES:
            raise ValueError("unknown P49 gradient-lane HMC readiness status")
        if any(not item for item in required_hmc_tiers):
            raise ValueError("required_hmc_tiers must contain nonempty tiers")
        unknown_tiers = sorted(set(required_hmc_tiers).difference(_P49_ALLOWED_HMC_TIERS))
        if unknown_tiers:
            raise ValueError(f"unknown HMC tier: {', '.join(unknown_tiers)}")
        nonclaims = _text_tuple("nonclaims", self.nonclaims)
        if self.source_fidelity_claim:
            raise ValueError("gradient evidence cannot claim source-faithful filtering")
        if self.differentiates_adaptive_random_branch:
            raise ValueError("adaptive random source branches need a separate differentiability contract")
        if hmc == "promoted" and not required_hmc_tiers:
            raise ValueError("HMC readiness promotion requires explicit HMC tiers")
        text = " ".join(nonclaims).lower()
        for required in (
            "no source-faithful filtering claim",
            "no hmc readiness by default",
        ):
            if required not in text:
                raise ValueError(f"missing nonclaim: {required}")
        object.__setattr__(self, "route_label", route_label)
        object.__setattr__(self, "branch_replay_status", branch_replay)
        object.__setattr__(self, "value_gradient_status", value_gradient)
        object.__setattr__(self, "likelihood_variance_calibration_status", variance)
        object.__setattr__(self, "hmc_readiness_status", hmc)
        object.__setattr__(self, "source_fidelity_claim", bool(self.source_fidelity_claim))
        object.__setattr__(
            self,
            "differentiates_adaptive_random_branch",
            bool(self.differentiates_adaptive_random_branch),
        )
        object.__setattr__(self, "required_hmc_tiers", required_hmc_tiers)
        object.__setattr__(self, "nonclaims", nonclaims)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "route_label": self.route_label,
            "branch_replay_status": self.branch_replay_status,
            "value_gradient_status": self.value_gradient_status,
            "likelihood_variance_calibration_status": self.likelihood_variance_calibration_status,
            "hmc_readiness_status": self.hmc_readiness_status,
            "source_fidelity_claim": bool(self.source_fidelity_claim),
            "differentiates_adaptive_random_branch": bool(
                self.differentiates_adaptive_random_branch
            ),
            "required_hmc_tiers": self.required_hmc_tiers,
            "nonclaims": self.nonclaims,
        }


def evaluate_experimental_score_api(
    target_id: str,
    evidence_class: str,
    m1_route_label: str,
    parameterization: str,
    theta: tf.Tensor,
    value_fn: Callable[[tf.Tensor], tf.Tensor],
    diagnostics: Mapping[str, object] | None = None,
) -> ExperimentalScoreAPIResult:
    """Evaluate a deterministic value function and package its gradient.

    This is an experimental subpackage helper for fixed, smooth, tiny P42
    Tier-1 fixtures.  It does not create a stable top-level score API or HMC
    target.
    """

    theta_tensor = tf.convert_to_tensor(theta, dtype=tf.float64)
    if theta_tensor.shape.rank != 1:
        raise ValueError(f"theta: {HighDimStatus.INVALID_SHAPE.value}")
    with tf.GradientTape() as tape:
        tape.watch(theta_tensor)
        value = tf.convert_to_tensor(value_fn(theta_tensor), dtype=tf.float64)
    score = tape.gradient(value, theta_tensor)
    if score is None:
        raise ValueError("score gradient is None")
    payload = {
        "target_id": target_id,
        "evidence_class": evidence_class,
        "m1_route_label": m1_route_label,
        "parameterization": parameterization,
        "theta": theta_tensor,
        "log_likelihood": value,
        "score": score,
        "fixed_branch_only": True,
        "experimental_subpackage_only": True,
        "stable_top_level_api": False,
    }
    identity = BranchIdentity(
        manifest=BranchManifest("experimental_score_api_result.v1", payload),
        hash=BranchManifest("experimental_score_api_result.v1", payload).sha256(),
    )
    return ExperimentalScoreAPIResult(
        target_id=target_id,
        evidence_class=evidence_class,
        m1_route_label=m1_route_label,
        parameterization=parameterization,
        theta=theta_tensor,
        log_likelihood=value,
        score=score,
        branch_identity=identity,
        status=HighDimStatus.OK,
        diagnostics={
            **dict(diagnostics or {}),
            "fixed_branch_only": True,
            "experimental_subpackage_only": True,
            "stable_top_level_api": False,
            "hmc_readiness": "not_claimed",
        },
    )


def evaluate_highdim_score_api(
    target_id: str,
    evidence_class: str,
    route_label: str,
    parameterization: str,
    theta: tf.Tensor,
    value_fn: Callable[[tf.Tensor], tf.Tensor],
    diagnostics: Mapping[str, object] | None = None,
    *,
    setup_identity: Mapping[str, object] | None = None,
) -> HighDimScoreAPIResult:
    """Evaluate a deterministic scalar value function and its score.

    This is the stable ``bayesfilter.highdim`` subpackage contract for
    fixed, deterministic, TensorFlow float64 score evaluations.  It is not a
    root-level ``bayesfilter`` export and it does not certify HMC readiness.
    """

    theta_tensor = tf.convert_to_tensor(theta, dtype=tf.float64)
    if theta_tensor.shape.rank != 1:
        raise ValueError(f"theta: {HighDimStatus.INVALID_SHAPE.value}")
    with tf.GradientTape() as tape:
        tape.watch(theta_tensor)
        value = tf.convert_to_tensor(value_fn(theta_tensor), dtype=tf.float64)
    score = tape.gradient(value, theta_tensor)
    if score is None:
        raise ValueError("score gradient is None")
    payload = {
        "target_id": target_id,
        "evidence_class": evidence_class,
        "route_label": route_label,
        "parameterization": parameterization,
        "theta": theta_tensor,
        "log_likelihood": value,
        "score": score,
        "api_scope": "bayesfilter.highdim",
        "stable_subpackage_api": True,
        "stable_top_level_api": False,
        "hmc_readiness": "not_claimed",
    }
    if setup_identity is not None:
        payload["setup_identity"] = _nonempty_mapping("setup_identity", setup_identity)
        payload["setup_identity_channel"] = "branch_manifest"
    manifest = BranchManifest("highdim_score_api_result.v1", payload)
    identity = BranchIdentity(manifest=manifest, hash=manifest.sha256())
    result_diagnostics = {
        **dict(diagnostics or {}),
        "api_scope": "bayesfilter.highdim",
        "stable_subpackage_api": True,
        "stable_top_level_api": False,
        "hmc_readiness": "not_claimed",
    }
    if setup_identity is not None:
        result_diagnostics["setup_identity_channel"] = "branch_manifest"
    return HighDimScoreAPIResult(
        target_id=target_id,
        evidence_class=evidence_class,
        route_label=route_label,
        parameterization=parameterization,
        theta=theta_tensor,
        log_likelihood=value,
        score=score,
        branch_identity=identity,
        status=HighDimStatus.OK,
        diagnostics=result_diagnostics,
    )


def evaluate_batched_highdim_score_api(
    target_id: str,
    evidence_class: str,
    route_label: str,
    parameterization: str,
    theta: tf.Tensor,
    value_fns: Sequence[Callable[[tf.Tensor], tf.Tensor]],
    diagnostics: Mapping[str, object] | None = None,
    *,
    shared_setup_identity: Mapping[str, object] | None = None,
    per_item_setup_identities: Sequence[Mapping[str, object]] | None = None,
) -> HighDimBatchedScoreAPIResult:
    """Evaluate deterministic scalar value functions under one fixed theta."""

    theta_tensor = tf.convert_to_tensor(theta, dtype=tf.float64)
    if theta_tensor.shape.rank != 1:
        raise ValueError(f"theta: {HighDimStatus.INVALID_SHAPE.value}")
    functions = tuple(value_fns)
    if not functions:
        raise ValueError("value_fns must be nonempty")
    if any(not callable(value_fn) for value_fn in functions):
        raise TypeError("value_fns must contain callables")
    setup_items, identity_mode = _batched_setup_identities(
        len(functions),
        shared_setup_identity=shared_setup_identity,
        per_item_setup_identities=per_item_setup_identities,
    )
    values: list[tf.Tensor] = []
    scores: list[tf.Tensor] = []
    identities: list[BranchIdentity] = []
    for index, (value_fn, setup_identity) in enumerate(zip(functions, setup_items)):
        with tf.GradientTape() as tape:
            tape.watch(theta_tensor)
            value = tf.convert_to_tensor(value_fn(theta_tensor), dtype=tf.float64)
        if value.shape.rank != 0:
            raise ValueError(f"value_fns[{index}]: {HighDimStatus.INVALID_SHAPE.value}")
        score = tape.gradient(value, theta_tensor)
        if score is None:
            raise ValueError(f"value_fns[{index}] score gradient is None")
        score = tf.convert_to_tensor(score, dtype=tf.float64)
        if score.shape != theta_tensor.shape:
            raise ValueError(f"value_fns[{index}] score: {HighDimStatus.INVALID_SHAPE.value}")
        item_payload = {
            "target_id": target_id,
            "evidence_class": evidence_class,
            "route_label": route_label,
            "parameterization": parameterization,
            "theta": theta_tensor,
            "log_likelihood": value,
            "score": score,
            "api_scope": "bayesfilter.highdim",
            "stable_subpackage_api": True,
            "stable_top_level_api": False,
            "hmc_readiness": "not_claimed",
            "setup_identity_channel": "diagnostics_and_branch_manifest",
            "batch_identity_mode": identity_mode,
            "batch_index": index,
            "batch_size": len(functions),
            "setup_identity": setup_identity,
        }
        manifest = BranchManifest("highdim_batched_score_api_item.v1", item_payload)
        values.append(value)
        scores.append(score)
        identities.append(BranchIdentity(manifest=manifest, hash=manifest.sha256()))
    branch_hashes = tuple(identity.hash.value for identity in identities)
    result_diagnostics = {
        **dict(diagnostics or {}),
        "api_scope": "bayesfilter.highdim",
        "stable_subpackage_api": True,
        "stable_top_level_api": False,
        "hmc_readiness": "not_claimed",
        "setup_identity_channel": "diagnostics_and_branch_manifest",
        "batch_identity_mode": identity_mode,
        "batch_size": len(functions),
        "per_item_branch_hashes": branch_hashes,
    }
    return HighDimBatchedScoreAPIResult(
        target_id=target_id,
        evidence_class=evidence_class,
        route_label=route_label,
        parameterization=parameterization,
        theta=theta_tensor,
        log_likelihoods=tf.stack(values),
        score=tf.stack(scores),
        branch_identities=tuple(identities),
        status=HighDimStatus.OK,
        diagnostics=result_diagnostics,
    )


def score_readiness_branch_hash(manifest: ScoreReadinessManifest) -> BranchHash:
    """Return an auditable hash for a score-readiness manifest."""

    return BranchManifest("p47_score_readiness_manifest.v1", manifest.manifest_payload()).sha256()


def _require_text(name: str, value: object) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be nonempty")
    return text


def _text_tuple(name: str, values: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values)
    if not normalized or any(not value for value in normalized):
        raise ValueError(f"{name} must contain nonempty strings")
    return normalized


def _nonempty_mapping(name: str, values: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    normalized = {str(key): value for key, value in values.items()}
    if not normalized or any(not key for key in normalized):
        raise ValueError(f"{name} must be a nonempty mapping")
    return normalized


def _batched_setup_identities(
    batch_size: int,
    *,
    shared_setup_identity: Mapping[str, object] | None,
    per_item_setup_identities: Sequence[Mapping[str, object]] | None,
) -> tuple[tuple[Mapping[str, object], ...], str]:
    if shared_setup_identity is not None and per_item_setup_identities is not None:
        raise ValueError("provide shared_setup_identity or per_item_setup_identities, not both")
    if shared_setup_identity is None and per_item_setup_identities is None:
        raise ValueError("batched score API requires setup identity metadata")
    if shared_setup_identity is not None:
        setup = _nonempty_mapping("shared_setup_identity", shared_setup_identity)
        return tuple(setup for _ in range(batch_size)), "shared_setup_identity"
    assert per_item_setup_identities is not None
    setup_items = tuple(
        _nonempty_mapping(f"per_item_setup_identities[{index}]", item)
        for index, item in enumerate(per_item_setup_identities)
    )
    if len(setup_items) != batch_size:
        raise ValueError("per_item_setup_identities must match value_fns length")
    return setup_items, "per_item_setup_identity"
