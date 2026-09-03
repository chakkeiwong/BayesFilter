from __future__ import annotations

import ast
import importlib
import inspect
import runpy
from dataclasses import replace
from pathlib import Path

import pytest

from bayesfilter.inference.hmc import SequentialRHatHMCVerificationConfig
from bayesfilter.inference.tuning_contract import (
    HMC_TUNING_CAPABILITY_REGISTRY_SCHEMA,
    HMC_TUNING_INTERFACE_CAPABILITIES,
    HMC_TUNING_ORDINARY_RHAT_THRESHOLD,
    HMC_TUNING_RUNNER_BINDING_SCHEMA,
    HMC_TUNING_ROUTE_REGISTRY,
    active_hmc_tuning_routes,
    hmc_tuning_capability_registry_payload,
    hmc_tuning_interface_capability,
    validate_hmc_tuning_interface_capabilities,
)
from bayesfilter.hmc_route_contract import (
    HMC_TOP_LEVEL_SELECTION_STAGE,
    OPERATIONAL_FIXED_TRAJECTORY_ALGORITHM_ID,
    resolve_hmc_algorithm_route,
)
from bayesfilter.inference.hmc_kernel_tuning import (
    HMCKernelTuningConfig,
    _public_resolved_policy_payload,
)
import bayesfilter.inference.hmc_kernel_tuning as hmc_kernel_tuning_module
import bayesfilter.inference.hmc_tuning_dispatch as hmc_tuning_dispatch
from scripts.render_hmc_tuning_interface_docs import (
    OUTPUT_PATHS,
    render_markdown,
    write_or_check,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GUIDE_PATH = REPO_ROOT / "docs/reference/hmc-tuning-interface.md"
CHAPTER_PATH = REPO_ROOT / "docs/chapters/ch21b_hmc_tuning_interfaces.tex"
EXAMPLE_PATHS = (
    REPO_ROOT / "docs/examples/hmc_tuning_route_selection.py",
    REPO_ROOT / "docs/examples/hmc_tuning_ordinary.py",
    REPO_ROOT / "docs/examples/hmc_tuning_covariance_first.py",
    REPO_ROOT / "docs/examples/hmc_tuning_neural_force_binding.py",
    REPO_ROOT / "docs/examples/hmc_tuning_fixed_transport.py",
)


def test_capability_registry_covers_routes_and_has_resolvable_evidence() -> None:
    payload = hmc_tuning_capability_registry_payload()

    assert payload["schema"] == HMC_TUNING_CAPABILITY_REGISTRY_SCHEMA
    assert len(payload["interfaces"]) == len(HMC_TUNING_INTERFACE_CAPABILITIES)
    public_tuners = tuple(
        record
        for record in HMC_TUNING_INTERFACE_CAPABILITIES
        if record.interface_kind == "public_tuner"
    )
    assert len(public_tuners) == 10
    assert len(active_hmc_tuning_routes()) == 2
    assert sum(
        record.interface_kind == "public_tuner"
        and record.capability_status in {"diagnostic_only", "historical_only"}
        for record in HMC_TUNING_INTERFACE_CAPABILITIES
    ) == 8
    assert sum(
        record.role == "active" and record.artifact_authority
        for record in HMC_TUNING_ROUTE_REGISTRY
    ) == 2
    assert {record.qualified_name for record in public_tuners} == {
        record.qualified_name for record in active_hmc_tuning_routes()
    } | {
        record.qualified_name
        for record in HMC_TUNING_INTERFACE_CAPABILITIES
        if record.interface_kind == "public_tuner"
        and record.capability_status in {"diagnostic_only", "historical_only"}
    }
    for record in HMC_TUNING_INTERFACE_CAPABILITIES:
        assert record.evidence_anchors
        for anchor in record.evidence_anchors:
            path_text = anchor.split("::", 1)[0]
            if path_text.startswith("bayesfilter/") or path_text.startswith(
                ("tests/", "scripts/")
            ):
                path_text = path_text.split("::", 1)[0]
                assert (REPO_ROOT / path_text).is_file(), anchor


def test_ordinary_capability_matches_public_signature() -> None:
    ordinary = hmc_tuning_interface_capability("tune_hmc_kernel")
    fixed_transport = hmc_tuning_interface_capability(
        "tune_fixed_transport_hmc_kernel"
    )
    ordinary_function = getattr(importlib.import_module(ordinary.module), ordinary.interface_name)
    fixed_function = getattr(
        importlib.import_module(fixed_transport.module), fixed_transport.interface_name
    )

    ordinary_signature = inspect.signature(ordinary_function)
    fixed_signature = inspect.signature(fixed_function)
    assert "runner_binding" in ordinary_signature.parameters
    assert "run_full_chain" not in ordinary_signature.parameters
    assert "run_full_chain" in fixed_signature.parameters
    assert ordinary.mass_capability == "owned"
    assert ordinary.step_size_capability == "owned"
    assert ordinary.trajectory_capability == "owned"
    assert ordinary.fresh_verification_required is True
    assert ordinary.acceptance_alone_can_handoff is False
    assert fixed_transport.requires_frozen_transport is True
    assert fixed_transport.mass_capability == "fixed"


def test_compatibility_delegate_cannot_create_a_second_public_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    observed: dict[str, object] = {}

    def fake_dispatch(**kwargs: object) -> object:
        observed.update(kwargs)
        return sentinel

    monkeypatch.setattr(hmc_tuning_dispatch, "tune_hmc_kernel", fake_dispatch)
    result = hmc_kernel_tuning_module.tune_hmc_kernel(
        adapter="adapter",
        initial_position="position",
    )

    assert result is sentinel
    assert observed["adapter"] == "adapter"
    assert observed["initial_position"] == "position"
    assert observed["config"] is None
    assert observed["runner_binding"] is None
    assert "Compatibility delegate" in (
        hmc_kernel_tuning_module.tune_hmc_kernel.__doc__ or ""
    )


def test_every_documented_interface_resolves() -> None:
    for record in HMC_TUNING_INTERFACE_CAPABILITIES:
        module = importlib.import_module(record.module)
        assert hasattr(module, record.interface_name), record.qualified_name


def test_generated_route_tables_are_current() -> None:
    assert write_or_check(REPO_ROOT, check=True) == ()


def test_renderer_check_detects_temporary_capability_drift(tmp_path: Path) -> None:
    for relative_path in OUTPUT_PATHS:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text((REPO_ROOT / relative_path).read_text(encoding="utf-8"), encoding="utf-8")

    ordinary = hmc_tuning_interface_capability("tune_hmc_kernel")
    mutated_records = tuple(
        replace(record, mass_policy="unreviewed changed mass claim")
        if record is ordinary
        else record
        for record in HMC_TUNING_INTERFACE_CAPABILITIES
    )
    mutated = render_markdown(mutated_records)
    assert mutated != render_markdown()
    (tmp_path / OUTPUT_PATHS[0]).write_text(mutated, encoding="utf-8")

    assert write_or_check(tmp_path, check=True) == (OUTPUT_PATHS[0],)


def test_false_claim_mutations_fail_registry_validation() -> None:
    ordinary = hmc_tuning_interface_capability("tune_hmc_kernel")
    fixed = hmc_tuning_interface_capability("tune_fixed_transport_hmc_kernel")
    neural_runner = hmc_tuning_interface_capability(
        "run_full_chain_neural_force_hmc"
    )

    with pytest.raises(ValueError, match="acceptance alone"):
        replace(ordinary, acceptance_alone_can_handoff=True)
    with pytest.raises(ValueError, match="fresh verification"):
        replace(ordinary, fresh_verification_required=False)

    false_mass = tuple(
        replace(record, mass_capability="owned")
        if record is neural_runner
        else record
        for record in HMC_TUNING_INTERFACE_CAPABILITIES
    )
    with pytest.raises(ValueError, match="cannot own tuning choices"):
        validate_hmc_tuning_interface_capabilities(false_mass)

    false_dual_averaging_l = tuple(
        replace(record, trajectory_capability="owned")
        if record is neural_runner
        else record
        for record in HMC_TUNING_INTERFACE_CAPABILITIES
    )
    with pytest.raises(ValueError, match="cannot own tuning choices"):
        validate_hmc_tuning_interface_capabilities(false_dual_averaging_l)

    false_transport = tuple(
        replace(record, requires_frozen_transport=False)
        if record is fixed
        else record
        for record in HMC_TUNING_INTERFACE_CAPABILITIES
    )
    with pytest.raises(ValueError, match="mass capability|frozen transport"):
        validate_hmc_tuning_interface_capabilities(false_transport)

    false_export_role = tuple(
        replace(
            record,
            interface_kind="public_tuner",
            capability_status="tested_supported",
            artifact_authority=True,
            replacement=None,
            fresh_verification_required=True,
        )
        if record is neural_runner
        else record
        for record in HMC_TUNING_INTERFACE_CAPABILITIES
    )
    with pytest.raises(ValueError, match="lacks route classification|mass capability"):
        validate_hmc_tuning_interface_capabilities(false_export_role)


def test_ordinary_ess_admission_is_explicitly_disabled() -> None:
    ordinary = hmc_tuning_interface_capability("tune_hmc_kernel")

    assert ordinary.ess_admission_policy == (
        "disabled for ordinary tuning admission; retained posterior ESS is separate"
    )
    assert "bulk_ess" not in inspect.signature(
        getattr(importlib.import_module(ordinary.module), ordinary.interface_name)
    ).parameters


def test_ordinary_rhat_threshold_has_one_exported_implementation_anchor() -> None:
    ordinary = hmc_tuning_interface_capability("tune_hmc_kernel")
    default = SequentialRHatHMCVerificationConfig.__dataclass_fields__[
        "rhat_threshold"
    ].default

    assert default == HMC_TUNING_ORDINARY_RHAT_THRESHOLD
    assert f"{HMC_TUNING_ORDINARY_RHAT_THRESHOLD:.2f}" in (
        ordinary.fresh_verification_policy
    )


def test_normative_chapter_and_agent_guide_are_wired_to_registry() -> None:
    guide = GUIDE_PATH.read_text(encoding="utf-8")
    chapter = CHAPTER_PATH.read_text(encoding="utf-8")
    normalized_guide = " ".join(guide.split())
    main = (REPO_ROOT / "docs/main.tex").read_text(encoding="utf-8")
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert main.index("\\input{chapters/ch21_hmc_for_state_space}") < main.index(
        "\\input{chapters/ch21b_hmc_tuning_interfaces}"
    ) < main.index("\\input{chapters/ch22_mass_matrices}")
    assert "\\input{generated/hmc_tuning_route_table}" in chapter
    assert HMC_TUNING_CAPABILITY_REGISTRY_SCHEMA in guide
    assert HMC_TUNING_RUNNER_BINDING_SCHEMA in guide
    assert "HMC_TUNING_INTERFACE_CAPABILITIES" in guide
    assert "docs/reference/hmc-tuning-interface.md" in agents

    for term in (
        "tune_hmc_kernel",
        "tune_fixed_transport_hmc_kernel",
        "bind_neural_force_hmc_tuning_runner",
        "run_full_chain_neural_force_hmc",
        "fixed `M=I`, fixed `L=1`",
        "artifact-authoritative fixed-transport policy",
        "Bulk and tail ESS are disabled",
        "R-hat values at or below",
        "negative_hessian",
        "initial_covariance",
        "parameter_scales",
        "explicit initial-position bank",
        "initial_position_was_replicated=True",
        "Durable typed TensorFlow replay",
        "build_retained_bound_hmc_archive_runner_from_tuning_result",
        "continuation_manifest",
        "Durable ordinary replay",
        "Replay roles and authority",
        "build_mechanics_only_frozen_kernel_hmc_adapter_from_tuning_payload",
        "build_claim_bearing_retained_frozen_kernel_hmc_adapter_from_tuning_payload",
        "claim_bearing_blockers",
        "caller-edited authority flag or blocker list cannot grant authority",
        "tuning_config",
        "mechanics_only_nonclaiming",
        "admission_supported=False",
        "posterior_admission_authority=False",
        "same frozen transition",
        "mass-signature change invalidates the selection and its calibration",
        "retained R-hat and ESS are explanatory",
        "short `interface_name`",
        "do not transfer to this route",
        "audit_ordinary_hmc_migration_surface.py",
        "unknown_dynamic_import",
        "unresolved_dynamic_attribute",
    ):
        assert term in normalized_guide
    assert (
        "does not relabel a non-gradient field as the exact adapter score"
        in normalized_guide
    )
    assert "\\label{eq:bf-neural-force-endpoint-correction}" in chapter
    assert "artifact\\_authority" in chapter
    assert "posterior\\_admission\\_authority" in chapter
    assert "admission\\_supported" in chapter
    assert "ordinary\\_hmc" in chapter
    assert "operational\\_paired\\_fixed\\_trajectory\\_selection\\_v3" in chapter
    assert "shared/frozen epsilon" in chapter
    assert "public artifact-authority boundary rejects" in chapter
    assert "frozen mechanics" in chapter
    assert "Replay roles and claim authority" in chapter
    assert "build_claim_bearing_retained_frozen_kernel_hmc_adapter_from_mechanics_payload" in chapter
    assert "artifact-authoritative fixed-transport tuning policy" in chapter
    assert "not posterior convergence" in chapter


def test_examples_are_exact_listings_and_public_imports_resolve() -> None:
    chapter = CHAPTER_PATH.read_text(encoding="utf-8")
    for example_path in EXAMPLE_PATHS:
        relative = example_path.relative_to(REPO_ROOT / "docs").as_posix()
        assert f"{{{relative}}}" in chapter
        source = example_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(example_path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            if not node.module.startswith("bayesfilter"):
                continue
            module = importlib.import_module(node.module)
            for imported in node.names:
                assert hasattr(module, imported.name), (
                    f"{example_path.name}: {node.module}.{imported.name}"
                )


def test_construction_examples_execute_without_tuning_or_hmc() -> None:
    covariance_namespace = runpy.run_path(
        str(REPO_ROOT / "docs/examples/hmc_tuning_covariance_first.py")
    )
    neural_namespace = runpy.run_path(
        str(REPO_ROOT / "docs/examples/hmc_tuning_neural_force_binding.py")
    )

    assert covariance_namespace["main"]()["status"] == (
        "arguments_bound_without_hmc"
    )
    binding_payload = neural_namespace["main"]()
    assert binding_payload["artifact_authority"] is False
    assert binding_payload["tensor_kernel_factory_available"] is True


def test_guide_rejects_the_observed_low_level_runner_misclassification() -> None:
    guide = GUIDE_PATH.read_text(encoding="utf-8")
    normalized = " ".join(guide.split())

    assert "A chain runner or stage helper is not a complete tuner" in normalized
    assert "It does not tune mass or choose `L`" in normalized
    assert "Acceptance by itself" not in guide
    assert "Do not treat acceptance alone as convergence or handoff evidence" in normalized
    assert "A failed verifier must have no final kernel" in normalized


def test_guide_binds_the_executable_ordinary_default_policy() -> None:
    config = HMCKernelTuningConfig.standard()
    route = resolve_hmc_algorithm_route(
        algorithm_id=config.algorithm_id,
        stage=HMC_TOP_LEVEL_SELECTION_STAGE,
        chain_execution_mode=config.chain_execution_mode,
        use_xla=config.use_xla,
    )
    resolved = _public_resolved_policy_payload(config)

    assert route.algorithm_id == OPERATIONAL_FIXED_TRAJECTORY_ALGORITHM_ID
    assert route.operational_authority is True
    assert route.artifact_authority is True
    assert route.scientific_promotion_authority is False
    assert resolved["config_variant"] == "ordinary_hmc"
    assert resolved["algorithm_id"] == OPERATIONAL_FIXED_TRAJECTORY_ALGORITHM_ID
    assert resolved["claim_bearing_artifact_authority"] is False
    assert resolved["claim_bearing_blocker"] == (
        "ordinary_runtime_numpy_policy_pending"
    )

    guide = " ".join(GUIDE_PATH.read_text(encoding="utf-8").split())
    assert "ordinary_hmc" in guide
    assert OPERATIONAL_FIXED_TRAJECTORY_ALGORITHM_ID in guide
    assert "shared/frozen epsilon" in guide
    assert "explicit `joint_l_epsilon_grid_fixed_mass_hmc` identifier" in guide
    assert "claim_bearing_artifact_authority=True" in guide


def test_ordinary_module_prose_does_not_promote_the_legacy_joint_grid() -> None:
    source = (
        REPO_ROOT / "bayesfilter/inference/hmc_kernel_tuning.py"
    ).read_text(encoding="utf-8")

    assert "operational fixed-trajectory screen" in source
    assert "shared/frozen epsilon" in source
    assert "legacy or internal construction" in source
    assert "diagnostic/non-promoting" in source
    assert "promoted fixed-mass joint" not in source
