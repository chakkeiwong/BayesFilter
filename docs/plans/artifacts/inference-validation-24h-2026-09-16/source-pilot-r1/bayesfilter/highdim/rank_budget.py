"""Memory-bounded fixed-rank preflight contracts for high-dimensional routes."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, sqrt
from typing import Mapping, Sequence


P52_GIB = 1024**3
P52_DEFAULT_CANDIDATE_RANKS = (2, 4, 8, 16, 32)
P52_DEFAULT_PHYSICAL_CAP_BYTES = 32 * P52_GIB
P52_DEFAULT_ALGORITHM_CAP_BYTES = 16 * P52_GIB
P52_DEFAULT_STEP_CAP_BYTES = 8 * P52_GIB
P52_MEMORY_PREFLIGHT_CLAIM = "memory and rank feasibility forecast only"
P52_ALLOWED_REFF_SOURCES = frozenset(
    {
        "measured_lower_rung",
        "operator_core_estimate",
        "conservative_declared_bound",
    }
)
P53_RANK_SELECTION_CLAIM = "fixed_rank_selection_from_admitted_scaling_route_metadata"
P57_SOURCE_FAITHFUL_RANK_POLICY_CLAIM = (
    "source_faithful_fixed_ttsirt_rank_policy"
)
P57_AUTHOR_SIR_D18_RANK_LADDER = (10, 20, 40)
P57_FIXED_TTSIRT_MEMORY_TERMS = (
    "tt_cores",
    "mass_contractions",
    "cdf_kr_state",
    "sample_batches",
    "autodiff_workspace",
    "retained_objects",
)
P57_UKF_DIAGNOSTIC_ROLE = "diagnostic_scout_only_not_rank_truth"
P57_ALLOWED_UKF_CLAIM_CLASS = "scout_not_truth"
P57_RANK_PROMOTION_LOG_LIK_PER_OBS_TOL = 1e-3
P57_RANK_PROMOTION_MEAN_SCALED_RMSE_TOL = 5e-2
P57_RANK_PROMOTION_COV_REL_FRO_TOL = 1e-1
P57_RANK_PROMOTION_REPLAY_RESIDUAL_TOL = 0.0
P57_RANK_PROMOTION_GRADIENT_COSINE_MIN = 0.995
P57_RANK_PROMOTION_GRADIENT_REL_SCORE_TOL = 5e-2
P57_FIXED_TTSIRT_ROUTE_CLASS = "fixed_ttsirt_source_route"
P57_RANK_COMPARATOR_KINDS = frozenset(
    {
        "dense_lower_rung",
        "same_route_higher_rank",
        "none",
    }
)


@dataclass(frozen=True)
class RankBudgetConfig:
    """Inputs for a deterministic fixed-rank memory preflight.

    This object is a pre-run budget contract.  It does not evaluate a filter,
    validate correctness, or certify HMC readiness.
    """

    dimension: int
    basis_size: int
    dtype_bytes: int = 8
    effective_transition_rank_multiplier: int = 16
    workspace_multiplier: int = 8
    physical_cap_bytes: int = P52_DEFAULT_PHYSICAL_CAP_BYTES
    algorithm_cap_bytes: int = P52_DEFAULT_ALGORITHM_CAP_BYTES
    step_cap_bytes: int = P52_DEFAULT_STEP_CAP_BYTES
    candidate_ranks: tuple[int, ...] = P52_DEFAULT_CANDIDATE_RANKS
    reff_source: str = "conservative_declared_bound"
    claim_class: str = P52_MEMORY_PREFLIGHT_CLAIM

    def __post_init__(self) -> None:
        positive_fields = {
            "dimension": self.dimension,
            "basis_size": self.basis_size,
            "dtype_bytes": self.dtype_bytes,
            "effective_transition_rank_multiplier": (
                self.effective_transition_rank_multiplier
            ),
            "workspace_multiplier": self.workspace_multiplier,
            "physical_cap_bytes": self.physical_cap_bytes,
            "algorithm_cap_bytes": self.algorithm_cap_bytes,
            "step_cap_bytes": self.step_cap_bytes,
        }
        for name, value in positive_fields.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        ranks = tuple(int(rank) for rank in self.candidate_ranks)
        if not ranks or any(rank <= 0 for rank in ranks):
            raise ValueError("candidate_ranks must contain positive ranks")
        if tuple(sorted(ranks)) != ranks:
            raise ValueError("candidate_ranks must be sorted increasingly")
        if len(set(ranks)) != len(ranks):
            raise ValueError("candidate_ranks must not contain duplicates")
        if self.reff_source not in P52_ALLOWED_REFF_SOURCES:
            raise ValueError("reff_source must be explicit and approved")
        if self.claim_class != P52_MEMORY_PREFLIGHT_CLAIM:
            raise ValueError("rank budget preflight cannot promote stronger claims")
        if int(self.step_cap_bytes) > int(self.algorithm_cap_bytes):
            raise ValueError("step_cap_bytes cannot exceed algorithm_cap_bytes")
        if int(self.algorithm_cap_bytes) > int(self.physical_cap_bytes):
            raise ValueError("algorithm_cap_bytes cannot exceed physical_cap_bytes")
        object.__setattr__(self, "dimension", int(self.dimension))
        object.__setattr__(self, "basis_size", int(self.basis_size))
        object.__setattr__(self, "dtype_bytes", int(self.dtype_bytes))
        object.__setattr__(
            self,
            "effective_transition_rank_multiplier",
            int(self.effective_transition_rank_multiplier),
        )
        object.__setattr__(self, "workspace_multiplier", int(self.workspace_multiplier))
        object.__setattr__(self, "physical_cap_bytes", int(self.physical_cap_bytes))
        object.__setattr__(self, "algorithm_cap_bytes", int(self.algorithm_cap_bytes))
        object.__setattr__(self, "step_cap_bytes", int(self.step_cap_bytes))
        object.__setattr__(self, "candidate_ranks", ranks)


@dataclass(frozen=True)
class RankBudgetForecast:
    """Forecast for one rank candidate under a fixed budget config."""

    rank: int
    state_memory_bytes: int
    step_memory_bytes: int
    within_step_cap: bool

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "rank": self.rank,
            "state_memory_bytes": self.state_memory_bytes,
            "step_memory_bytes": self.step_memory_bytes,
            "within_step_cap": self.within_step_cap,
        }


@dataclass(frozen=True)
class RankBudgetPreflight:
    """Deterministic M2 rank-budget decision for one dimension."""

    config: RankBudgetConfig
    r_max: int
    forecasts: tuple[RankBudgetForecast, ...]
    feasible_ranks: tuple[int, ...]
    status: str
    blocker: str | None
    nonclaims: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in {
            "PASS_P52_MEMORY_PREFLIGHT",
            "BLOCK_P52_RANK_BUDGET_EMPTY",
        }:
            raise ValueError("unknown rank-budget preflight status")
        if self.status.startswith("PASS") and self.blocker is not None:
            raise ValueError("passing preflight cannot carry a blocker")
        if self.status.startswith("BLOCK") and not self.blocker:
            raise ValueError("blocking preflight requires a blocker")
        required = {
            "memory feasibility only",
            "no filtering correctness",
            "no HMC readiness",
            "no d=100 filtering correctness",
        }
        if not required <= set(self.nonclaims):
            raise ValueError("missing rank-budget nonclaims")

    def manifest_payload(self) -> Mapping[str, object]:
        cfg = self.config
        return {
            "dimension": cfg.dimension,
            "basis_size": cfg.basis_size,
            "dtype_bytes": cfg.dtype_bytes,
            "effective_transition_rank_multiplier": (
                cfg.effective_transition_rank_multiplier
            ),
            "workspace_multiplier": cfg.workspace_multiplier,
            "physical_cap_bytes": cfg.physical_cap_bytes,
            "algorithm_cap_bytes": cfg.algorithm_cap_bytes,
            "step_cap_bytes": cfg.step_cap_bytes,
            "candidate_ranks": cfg.candidate_ranks,
            "reff_source": cfg.reff_source,
            "claim_class": cfg.claim_class,
            "r_max": self.r_max,
            "feasible_ranks": self.feasible_ranks,
            "status": self.status,
            "blocker": self.blocker,
            "forecasts": [dict(row.manifest_payload()) for row in self.forecasts],
            "nonclaims": self.nonclaims,
        }


@dataclass(frozen=True)
class P57RankComparatorEvidence:
    """Comparator evidence for one fixed TT/SIRT source-route rank.

    The evidence row is deliberately tied to the source-route transport class.
    It cannot be constructed from the old local/operator ``R_eff`` route.
    """

    rank: int
    comparator_kind: str
    per_observation_log_likelihood_error: float | None
    filtered_mean_scaled_rmse: float | None
    covariance_relative_frobenius_error: float | None
    replay_residual: float | None
    comparator_rank: int | None = None
    gradient_reference_available: bool = False
    gradient_directional_cosine: float | None = None
    gradient_relative_score_error: float | None = None
    route_class: str = P57_FIXED_TTSIRT_ROUTE_CLASS
    source_anchor: str = (
        "third_party/audit/zhao_cui_tensor_ssm_p10/source/eg3_sir/"
        "mainscript.m:48-51"
    )

    def __post_init__(self) -> None:
        rank = int(self.rank)
        if rank <= 0:
            raise ValueError("rank must be positive")
        if self.route_class != P57_FIXED_TTSIRT_ROUTE_CLASS:
            raise ValueError("P57 rank evidence requires fixed_ttsirt_source_route")
        if self.comparator_kind not in P57_RANK_COMPARATOR_KINDS:
            raise ValueError("unknown P57 rank comparator kind")
        if not str(self.source_anchor).strip():
            raise ValueError("source_anchor must be nonempty")
        comparator_rank = (
            None if self.comparator_rank is None else int(self.comparator_rank)
        )
        if self.comparator_kind == "same_route_higher_rank":
            if comparator_rank is None or comparator_rank <= rank:
                raise ValueError("same-route comparator must use a higher rank")
        elif comparator_rank is not None:
            raise ValueError("comparator_rank is only valid for same-route evidence")
        if self.comparator_kind == "none":
            for name in (
                "per_observation_log_likelihood_error",
                "filtered_mean_scaled_rmse",
                "covariance_relative_frobenius_error",
                "replay_residual",
            ):
                if getattr(self, name) is not None:
                    raise ValueError("no-comparator evidence cannot carry errors")
        else:
            for name in (
                "per_observation_log_likelihood_error",
                "filtered_mean_scaled_rmse",
                "covariance_relative_frobenius_error",
                "replay_residual",
            ):
                _require_finite_nonnegative(name, getattr(self, name))
        if self.gradient_reference_available:
            _require_finite_nonnegative(
                "gradient_relative_score_error",
                self.gradient_relative_score_error,
            )
            cosine = self.gradient_directional_cosine
            if cosine is None or not (-1.0 <= float(cosine) <= 1.0):
                raise ValueError("gradient_directional_cosine must be in [-1, 1]")
        elif (
            self.gradient_directional_cosine is not None
            or self.gradient_relative_score_error is not None
        ):
            raise ValueError("gradient metrics require gradient_reference_available")
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "comparator_rank", comparator_rank)

    @property
    def has_comparator(self) -> bool:
        return self.comparator_kind != "none"

    @property
    def passes_value_tolerances(self) -> bool:
        if not self.has_comparator:
            return False
        return (
            float(self.per_observation_log_likelihood_error)
            <= P57_RANK_PROMOTION_LOG_LIK_PER_OBS_TOL
            and float(self.filtered_mean_scaled_rmse)
            <= P57_RANK_PROMOTION_MEAN_SCALED_RMSE_TOL
            and float(self.covariance_relative_frobenius_error)
            <= P57_RANK_PROMOTION_COV_REL_FRO_TOL
            and float(self.replay_residual)
            <= P57_RANK_PROMOTION_REPLAY_RESIDUAL_TOL
        )

    @property
    def passes_gradient_tolerances_or_not_required(self) -> bool:
        if not self.gradient_reference_available:
            return True
        return (
            float(self.gradient_directional_cosine)
            >= P57_RANK_PROMOTION_GRADIENT_COSINE_MIN
            and float(self.gradient_relative_score_error)
            <= P57_RANK_PROMOTION_GRADIENT_REL_SCORE_TOL
        )

    @property
    def passes_promotion_rule(self) -> bool:
        return (
            self.passes_value_tolerances
            and self.passes_gradient_tolerances_or_not_required
        )

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "rank": self.rank,
            "route_class": self.route_class,
            "comparator_kind": self.comparator_kind,
            "comparator_rank": self.comparator_rank,
            "per_observation_log_likelihood_error": (
                self.per_observation_log_likelihood_error
            ),
            "filtered_mean_scaled_rmse": self.filtered_mean_scaled_rmse,
            "covariance_relative_frobenius_error": (
                self.covariance_relative_frobenius_error
            ),
            "replay_residual": self.replay_residual,
            "gradient_reference_available": self.gradient_reference_available,
            "gradient_directional_cosine": self.gradient_directional_cosine,
            "gradient_relative_score_error": self.gradient_relative_score_error,
            "passes_promotion_rule": self.passes_promotion_rule,
            "source_anchor": self.source_anchor,
        }


@dataclass(frozen=True)
class P57SourceFaithfulRankSelectionResult:
    """P57 source-route rank-policy result.

    A passing result means the rank policy and comparator gate are satisfied for
    the supplied evidence.  It is still not a filtering-correctness claim.
    """

    candidate_ranks: tuple[int, ...]
    feasible_ranks: tuple[int, ...]
    evidence: tuple[P57RankComparatorEvidence, ...]
    selected_rank: int | None
    status: str
    blocker: str | None
    ukf_claim_class: str
    author_rank_ladder: tuple[int, ...] = P57_AUTHOR_SIR_D18_RANK_LADDER
    claim_class: str = P57_SOURCE_FAITHFUL_RANK_POLICY_CLAIM
    memory_terms: tuple[str, ...] = P57_FIXED_TTSIRT_MEMORY_TERMS
    ukf_role: str = P57_UKF_DIAGNOSTIC_ROLE
    nonclaims: tuple[str, ...] = (
        "rank policy only",
        "no filtering correctness",
        "no HMC readiness",
        "no spatial SIR d18 success",
        "UKF is diagnostic only",
    )

    def __post_init__(self) -> None:
        if self.status not in {
            "PASS_P57_M7_SOURCE_FAITHFUL_RANK_UKF_CALIBRATION",
            "BLOCK_P57_M7_RANK_COMPARATOR_MISSING",
            "BLOCK_P57_M7_RANK_TOLERANCE_FAILURE",
        }:
            raise ValueError("unknown P57 rank-selection status")
        if self.claim_class != P57_SOURCE_FAITHFUL_RANK_POLICY_CLAIM:
            raise ValueError("P57 rank policy cannot promote stronger claims")
        if self.ukf_claim_class != P57_ALLOWED_UKF_CLAIM_CLASS:
            raise ValueError("UKF can only enter P57 rank policy as scout_not_truth")
        candidates = tuple(int(rank) for rank in self.candidate_ranks)
        feasible = tuple(int(rank) for rank in self.feasible_ranks)
        for name, ranks in (("candidate_ranks", candidates), ("feasible_ranks", feasible)):
            if not ranks or any(rank <= 0 for rank in ranks):
                raise ValueError(f"{name} must contain positive ranks")
            if tuple(sorted(ranks)) != ranks or len(set(ranks)) != len(ranks):
                raise ValueError(f"{name} must be unique and sorted")
        if not set(feasible) <= set(candidates):
            raise ValueError("feasible_ranks must be a subset of candidate_ranks")
        if not all(
            isinstance(row, P57RankComparatorEvidence) for row in self.evidence
        ):
            raise TypeError("evidence must contain P57RankComparatorEvidence")
        if self.status.startswith("PASS"):
            if self.selected_rank is None or int(self.selected_rank) not in feasible:
                raise ValueError("passing P57 rank policy requires feasible selected_rank")
            if self.blocker is not None:
                raise ValueError("passing P57 rank policy cannot carry blocker")
        else:
            if self.selected_rank is not None:
                raise ValueError("blocking P57 rank policy cannot select rank")
            if not self.blocker:
                raise ValueError("blocking P57 rank policy requires blocker")
        required = {
            "rank policy only",
            "no filtering correctness",
            "no HMC readiness",
            "UKF is diagnostic only",
        }
        if not required <= set(self.nonclaims):
            raise ValueError("missing P57 rank-policy nonclaims")
        if set(self.memory_terms) != set(P57_FIXED_TTSIRT_MEMORY_TERMS):
            raise ValueError("P57 memory terms must cover fixed TT/SIRT state")
        object.__setattr__(self, "candidate_ranks", candidates)
        object.__setattr__(self, "feasible_ranks", feasible)
        object.__setattr__(self, "author_rank_ladder", tuple(self.author_rank_ladder))
        if self.selected_rank is not None:
            object.__setattr__(self, "selected_rank", int(self.selected_rank))

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "candidate_ranks": self.candidate_ranks,
            "feasible_ranks": self.feasible_ranks,
            "author_rank_ladder": self.author_rank_ladder,
            "selected_rank": self.selected_rank,
            "status": self.status,
            "blocker": self.blocker,
            "claim_class": self.claim_class,
            "memory_terms": self.memory_terms,
            "ukf_claim_class": self.ukf_claim_class,
            "ukf_role": self.ukf_role,
            "promotion_tolerances": p57_rank_promotion_tolerances(),
            "evidence": [row.manifest_payload() for row in self.evidence],
            "nonclaims": self.nonclaims,
        }


def p57_rank_promotion_tolerances() -> Mapping[str, object]:
    """Return the source-route rank-promotion tolerances from P57-M7."""

    return {
        "per_observation_log_likelihood_error": (
            P57_RANK_PROMOTION_LOG_LIK_PER_OBS_TOL
        ),
        "filtered_mean_scaled_rmse": P57_RANK_PROMOTION_MEAN_SCALED_RMSE_TOL,
        "covariance_relative_frobenius_error": P57_RANK_PROMOTION_COV_REL_FRO_TOL,
        "replay_residual": P57_RANK_PROMOTION_REPLAY_RESIDUAL_TOL,
        "gradient_directional_cosine_min": P57_RANK_PROMOTION_GRADIENT_COSINE_MIN,
        "gradient_relative_score_error": P57_RANK_PROMOTION_GRADIENT_REL_SCORE_TOL,
        "promotion_rule": (
            "smallest feasible fixed TT/SIRT rank passing dense lower-rung or "
            "same-route higher-rank comparator; never largest-rank self-promotion"
        ),
    }


def p57_select_source_faithful_rank(
    *,
    candidate_ranks: tuple[int, ...],
    feasible_ranks: tuple[int, ...],
    evidence: tuple[P57RankComparatorEvidence, ...],
    ukf_claim_class: str = P57_ALLOWED_UKF_CLAIM_CLASS,
    author_rank_ladder: tuple[int, ...] = P57_AUTHOR_SIR_D18_RANK_LADDER,
) -> P57SourceFaithfulRankSelectionResult:
    """Select a source-faithful fixed TT/SIRT rank from comparator evidence."""

    candidates = tuple(int(rank) for rank in candidate_ranks)
    feasible = tuple(int(rank) for rank in feasible_ranks)
    rows = tuple(evidence)
    comparable = tuple(row for row in rows if row.rank in feasible and row.has_comparator)
    passing = tuple(row for row in comparable if row.passes_promotion_rule)
    if passing:
        selected_rank = min(row.rank for row in passing)
        status = "PASS_P57_M7_SOURCE_FAITHFUL_RANK_UKF_CALIBRATION"
        blocker = None
    elif not comparable:
        selected_rank = None
        status = "BLOCK_P57_M7_RANK_COMPARATOR_MISSING"
        blocker = (
            "no dense lower-rung or same-route higher-rank comparator exists "
            "for a feasible fixed TT/SIRT rank"
        )
    else:
        selected_rank = None
        status = "BLOCK_P57_M7_RANK_TOLERANCE_FAILURE"
        blocker = "comparator evidence exists but no feasible rank passes tolerances"
    return P57SourceFaithfulRankSelectionResult(
        candidate_ranks=candidates,
        feasible_ranks=feasible,
        evidence=rows,
        selected_rank=selected_rank,
        status=status,
        blocker=blocker,
        ukf_claim_class=str(ukf_claim_class),
        author_rank_ladder=tuple(int(rank) for rank in author_rank_ladder),
    )


def p57_fixed_ttsirt_memory_terms() -> tuple[str, ...]:
    """Return memory terms that P57 rank budgets must account for."""

    return P57_FIXED_TTSIRT_MEMORY_TERMS


def state_memory_bytes(config: RankBudgetConfig, rank: int) -> int:
    """Return ``bytes * d * n * r^2`` for TT state-core storage."""

    _require_positive_rank(rank)
    return config.dtype_bytes * config.dimension * config.basis_size * int(rank) ** 2


def step_memory_bytes(config: RankBudgetConfig, rank: int) -> int:
    """Return ``bytes * d * n * (R_eff * r)^2 * omega`` for one step."""

    _require_positive_rank(rank)
    reff = config.effective_transition_rank_multiplier
    return (
        config.dtype_bytes
        * config.dimension
        * config.basis_size
        * (reff * int(rank)) ** 2
        * config.workspace_multiplier
    )


def rank_ceiling(config: RankBudgetConfig) -> int:
    """Return the hard rank ceiling implied by the step memory cap."""

    denominator = (
        config.dtype_bytes
        * config.dimension
        * config.basis_size
        * config.workspace_multiplier
        * config.effective_transition_rank_multiplier**2
    )
    return int(floor(sqrt(config.step_cap_bytes / denominator)))


def evaluate_rank_budget(config: RankBudgetConfig) -> RankBudgetPreflight:
    """Evaluate candidate ranks and classify the memory preflight."""

    ceiling = rank_ceiling(config)
    forecasts = tuple(
        RankBudgetForecast(
            rank=rank,
            state_memory_bytes=state_memory_bytes(config, rank),
            step_memory_bytes=step_memory_bytes(config, rank),
            within_step_cap=rank <= ceiling,
        )
        for rank in config.candidate_ranks
    )
    feasible = tuple(row.rank for row in forecasts if row.within_step_cap)
    if feasible:
        status = "PASS_P52_MEMORY_PREFLIGHT"
        blocker = None
    else:
        status = "BLOCK_P52_RANK_BUDGET_EMPTY"
        blocker = "no candidate rank is below the hard memory ceiling"
    return RankBudgetPreflight(
        config=config,
        r_max=ceiling,
        forecasts=forecasts,
        feasible_ranks=feasible,
        status=status,
        blocker=blocker,
        nonclaims=(
            "memory feasibility only",
            "no filtering correctness",
            "no HMC readiness",
            "no d=100 filtering correctness",
        ),
    )


def p52_spatial_sir_rank_budget_manifest(
    *,
    dimensions: Sequence[int] = (18, 50, 100),
    basis_size: int = 3,
    effective_transition_rank_multiplier: int = 16,
    workspace_multiplier: int = 8,
    reff_source: str = "conservative_declared_bound",
    candidate_ranks: tuple[int, ...] = P52_DEFAULT_CANDIDATE_RANKS,
) -> Mapping[str, object]:
    """Build the P52-M2 manifest payload for spatial SIR rank preflight."""

    rows = []
    for dimension in dimensions:
        config = RankBudgetConfig(
            dimension=int(dimension),
            basis_size=int(basis_size),
            effective_transition_rank_multiplier=int(
                effective_transition_rank_multiplier
            ),
            workspace_multiplier=int(workspace_multiplier),
            reff_source=reff_source,
            candidate_ranks=candidate_ranks,
        )
        preflight = evaluate_rank_budget(config)
        rows.append(dict(preflight.manifest_payload()))
    return {
        "schema_version": "p52.rank_budget_preflight.v1",
        "phase": "P52-M2",
        "status": "PASS_P52_M2_MEMORY_RANK_CEILING",
        "claim_class": P52_MEMORY_PREFLIGHT_CLAIM,
        "dimensions": tuple(int(value) for value in dimensions),
        "rows": rows,
        "nonclaims": (
            "memory feasibility only",
            "no filtering correctness",
            "no production spatial SIR readiness",
            "no HMC readiness",
            "no GPU readiness",
            "no d=100 filtering correctness",
        ),
    }


@dataclass(frozen=True)
class P53RankSelectionResult:
    """Fixed-rank selection outcome for the admitted P53 scaling route."""

    route_id: str
    route_class: str
    admission_token: str
    dimension: int
    basis_size: int
    effective_transition_rank_multiplier: int
    candidate_ranks: tuple[int, ...]
    feasible_ranks: tuple[int, ...]
    selected_rank: int | None
    status: str
    blocker: str | None
    rank_mutation_allowed_in_likelihood: bool
    preflight: RankBudgetPreflight
    claim_class: str = P53_RANK_SELECTION_CLAIM
    nonclaims: tuple[str, ...] = (
        "rank selection only",
        "no filtering correctness",
        "no d=18 spatial SIR result",
        "no HMC readiness",
        "no GPU readiness",
    )

    def __post_init__(self) -> None:
        if self.status not in {
            "PASS_P53_M5_RANK_SELECTION_INTEGRATION",
            "BLOCK_P53_M5_RANK_SELECTION_INTEGRATION",
        }:
            raise ValueError("unknown P53 rank-selection status")
        if self.claim_class != P53_RANK_SELECTION_CLAIM:
            raise ValueError("P53 rank selection cannot promote stronger claims")
        if self.route_class != "scaling_route":
            raise ValueError("P53 rank selection requires a scaling_route")
        if self.admission_token != "PASS_P53_M4D_SCALING_ROUTE_ADMISSION":
            raise ValueError("P53 rank selection requires M4D admission")
        if self.rank_mutation_allowed_in_likelihood:
            raise ValueError("rank mutation inside likelihood is forbidden")
        if self.status.startswith("PASS"):
            if self.selected_rank is None:
                raise ValueError("passing P53 rank selection requires selected_rank")
            if int(self.selected_rank) not in tuple(int(rank) for rank in self.feasible_ranks):
                raise ValueError("selected_rank must be feasible")
            if self.blocker is not None:
                raise ValueError("passing P53 rank selection cannot carry blocker")
        else:
            if self.selected_rank is not None:
                raise ValueError("blocking P53 rank selection cannot select rank")
            if not self.blocker:
                raise ValueError("blocking P53 rank selection requires blocker")
        required = {
            "rank selection only",
            "no filtering correctness",
            "no d=18 spatial SIR result",
            "no HMC readiness",
        }
        if not required <= set(self.nonclaims):
            raise ValueError("missing P53 rank-selection nonclaims")
        object.__setattr__(self, "candidate_ranks", tuple(int(rank) for rank in self.candidate_ranks))
        object.__setattr__(self, "feasible_ranks", tuple(int(rank) for rank in self.feasible_ranks))
        if self.selected_rank is not None:
            object.__setattr__(self, "selected_rank", int(self.selected_rank))

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "route_id": self.route_id,
            "route_class": self.route_class,
            "admission_token": self.admission_token,
            "dimension": int(self.dimension),
            "basis_size": int(self.basis_size),
            "effective_transition_rank_multiplier": int(
                self.effective_transition_rank_multiplier
            ),
            "candidate_ranks": self.candidate_ranks,
            "feasible_ranks": self.feasible_ranks,
            "selected_rank": self.selected_rank,
            "status": self.status,
            "blocker": self.blocker,
            "rank_mutation_allowed_in_likelihood": self.rank_mutation_allowed_in_likelihood,
            "claim_class": self.claim_class,
            "preflight": dict(self.preflight.manifest_payload()),
            "nonclaims": self.nonclaims,
        }


def p53_select_fixed_rank_from_admitted_route(
    route_metadata: Mapping[str, object],
    *,
    dimension: int,
    candidate_ranks: tuple[int, ...] = P52_DEFAULT_CANDIDATE_RANKS,
    admission_token: str | None = None,
    basis_size: int | None = None,
    workspace_multiplier: int = 8,
    selection_policy: str = "largest_feasible",
) -> P53RankSelectionResult:
    """Select a fixed rank from admitted P53 route metadata.

    The selected rank is a pre-likelihood configuration value.  This helper
    does not run filtering and does not adapt ranks during likelihood calls.
    """

    token = admission_token or str(route_metadata.get("admission_token", ""))
    if token != "PASS_P53_M4D_SCALING_ROUTE_ADMISSION":
        raise ValueError("P53 rank selection requires PASS_P53_M4D_SCALING_ROUTE_ADMISSION")
    route_class = str(route_metadata.get("route_class", ""))
    if route_class != "scaling_route":
        raise ValueError("P53 rank selection requires scaling_route metadata")
    route_id = str(route_metadata.get("route_id", ""))
    if not route_id:
        raise ValueError("route_id must be present")
    if selection_policy != "largest_feasible":
        raise ValueError("unknown P53 rank selection policy")
    effective = int(
        route_metadata.get(
            "R_eff",
            route_metadata.get(
                "effective_transition_rank_multiplier",
                route_metadata.get("example_j3_R_eff", 0),
            ),
        )
    )
    if effective <= 0:
        raise ValueError("route metadata must provide positive R_eff")
    basis = int(basis_size or route_metadata.get("basis_order", 0))
    if basis <= 0:
        raise ValueError("basis_size or route basis_order must be positive")
    config = RankBudgetConfig(
        dimension=int(dimension),
        basis_size=basis,
        effective_transition_rank_multiplier=effective,
        workspace_multiplier=int(workspace_multiplier),
        candidate_ranks=candidate_ranks,
        reff_source="conservative_declared_bound",
    )
    preflight = evaluate_rank_budget(config)
    feasible = preflight.feasible_ranks
    if feasible:
        status = "PASS_P53_M5_RANK_SELECTION_INTEGRATION"
        selected_rank = max(feasible)
        blocker = None
    else:
        status = "BLOCK_P53_M5_RANK_SELECTION_INTEGRATION"
        selected_rank = None
        blocker = preflight.blocker
    return P53RankSelectionResult(
        route_id=route_id,
        route_class=route_class,
        admission_token=token,
        dimension=int(dimension),
        basis_size=basis,
        effective_transition_rank_multiplier=effective,
        candidate_ranks=candidate_ranks,
        feasible_ranks=feasible,
        selected_rank=selected_rank,
        status=status,
        blocker=blocker,
        rank_mutation_allowed_in_likelihood=False,
        preflight=preflight,
    )


def _require_positive_rank(rank: int) -> None:
    if int(rank) <= 0:
        raise ValueError("rank must be positive")


def _require_finite_nonnegative(name: str, value: float | None) -> None:
    if value is None:
        raise ValueError(f"{name} is required")
    numeric = float(value)
    if numeric < 0.0 or numeric in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite and nonnegative")
    if numeric != numeric:
        raise ValueError(f"{name} must be finite and nonnegative")
