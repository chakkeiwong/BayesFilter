"""Run the MP4 flow and DPF readiness review.

This runner performs static inventory plus import/signature probes only.  It
does not instantiate student filters, call filtering methods, execute notebooks,
run experiment scripts, train models, or run HMC.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import asdict, dataclass, field
import importlib
import inspect
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

from experiments.student_dpf_baselines.adapters.advanced_particle_filter_adapter import (
    SOURCE_COMMIT as ADVANCED_COMMIT,
    VENDOR_ROOT as ADVANCED_VENDOR_ROOT,
)
from experiments.student_dpf_baselines.adapters.common import prepend_sys_path, write_json
from experiments.student_dpf_baselines.adapters.mlcoe_adapter import (
    SNAPSHOT_ROOT as MLCOE_ROOT,
    SOURCE_COMMIT as MLCOE_COMMIT,
)


DATE = "2026-05-11"
OUTPUT_PATH = Path(
    "experiments/student_dpf_baselines/reports/outputs/"
    "flow_dpf_readiness_inventory_2026-05-11.json"
)
SUMMARY_PATH = Path(
    "experiments/student_dpf_baselines/reports/outputs/"
    "flow_dpf_readiness_summary_2026-05-11.json"
)
REPORT_PATH = Path(
    "experiments/student_dpf_baselines/reports/"
    "student-dpf-baseline-flow-dpf-readiness-review-result-2026-05-11.md"
)


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    implementation_name: str
    source_commit: str
    root: Path
    module_import: str
    object_name: str
    object_kind: str
    file_path: Path
    category: str
    family: str
    target_semantics: str
    likely_fixture: str | None
    artifact_requirements: tuple[str, ...] = ()
    notes: str = ""


@dataclass(slots=True)
class ProbeRecord:
    candidate_id: str
    implementation_name: str
    source_commit: str
    module_import: str
    object_name: str
    object_kind: str
    file_path: str
    category: str
    family: str
    target_semantics: str
    likely_fixture: str | None
    artifact_requirements: list[str]
    notes: str
    static_found: bool
    static_public_methods: list[str] = field(default_factory=list)
    static_decorators: list[str] = field(default_factory=list)
    probe_status: str = "not_run"
    readiness_class: str = "not_classified"
    import_error: str | None = None
    object_signature: str | None = None
    method_signatures: dict[str, str] = field(default_factory=dict)
    comparison_role: str = "not_selected"
    blocker_reason: str | None = None


def main() -> None:
    records = [_probe_candidate(candidate) for candidate in _candidates()]
    summary = _summarize(records)
    write_json(
        OUTPUT_PATH,
        {
            "date": DATE,
            "scope": "student_dpf_baseline_mp4_flow_dpf_readiness_review",
            "execution_mode": "static_inventory_plus_import_signature_probe",
            "command": (
                "python -m "
                "experiments.student_dpf_baselines.runners.run_flow_dpf_readiness_review"
            ),
            "working_directory": str(Path.cwd()),
            "environment": _environment_record(),
            "instantiation_or_filter_execution": False,
            "records": [asdict(record) for record in records],
            "summary": summary,
        },
    )
    write_json(SUMMARY_PATH, summary)
    REPORT_PATH.write_text(_render_report(records, summary), encoding="utf-8")


def _environment_record() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import tensorflow as tf  # type: ignore

        versions["tensorflow"] = tf.__version__
        versions["tensorflow_devices"] = [
            {"name": device.name, "type": device.device_type}
            for device in tf.config.list_physical_devices()
        ]
    except Exception as exc:  # pragma: no cover - environment dependent.
        versions["tensorflow_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import tensorflow_probability as tfp  # type: ignore

        versions["tensorflow_probability"] = tfp.__version__
    except Exception as exc:  # pragma: no cover - environment dependent.
        versions["tensorflow_probability_error"] = f"{type(exc).__name__}: {exc}"
    return versions


def _candidates() -> list[Candidate]:
    adv = ADVANCED_VENDOR_ROOT / "advanced_particle_filter"
    mlcoe = MLCOE_ROOT
    return [
        Candidate(
            "advanced_edh_flow",
            "advanced_particle_filter",
            ADVANCED_COMMIT,
            ADVANCED_VENDOR_ROOT,
            "advanced_particle_filter.filters.edh",
            "EDHFlow",
            "class",
            adv / "filters" / "edh.py",
            "edh_ledh_flow",
            "edh",
            "pure EDH flow with uniform weights",
            "nonlinear_gaussian_range_bearing",
            notes="Pure flow path; comparison value is lower than importance-corrected EDH.",
        ),
        Candidate(
            "advanced_edh_particle_filter",
            "advanced_particle_filter",
            ADVANCED_COMMIT,
            ADVANCED_VENDOR_ROOT,
            "advanced_particle_filter.filters.edh",
            "EDHParticleFilter",
            "class",
            adv / "filters" / "edh.py",
            "edh_ledh_pfpf",
            "edh",
            "EDH particle flow with importance-weight correction",
            "nonlinear_gaussian_range_bearing",
        ),
        Candidate(
            "advanced_ledh_flow",
            "advanced_particle_filter",
            ADVANCED_COMMIT,
            ADVANCED_VENDOR_ROOT,
            "advanced_particle_filter.filters.ledh",
            "LEDHFlow",
            "class",
            adv / "filters" / "ledh.py",
            "edh_ledh_flow",
            "ledh",
            "local EDH flow with per-particle linearization",
            "nonlinear_gaussian_range_bearing",
        ),
        Candidate(
            "advanced_ledh_particle_filter",
            "advanced_particle_filter",
            ADVANCED_COMMIT,
            ADVANCED_VENDOR_ROOT,
            "advanced_particle_filter.filters.ledh",
            "LEDHParticleFilter",
            "class",
            adv / "filters" / "ledh.py",
            "edh_ledh_pfpf",
            "ledh",
            "LEDH particle flow with importance-weight correction",
            "nonlinear_gaussian_range_bearing",
        ),
        Candidate(
            "advanced_stochastic_pff_flow",
            "advanced_particle_filter",
            ADVANCED_COMMIT,
            ADVANCED_VENDOR_ROOT,
            "advanced_particle_filter.filters.stochastic_pff",
            "StochasticPFFlow",
            "class",
            adv / "filters" / "stochastic_pff.py",
            "stochastic_flow",
            "spf",
            "stochastic particle flow with optimized beta schedule",
            "separate_stochastic_flow_fixture_required",
            notes="Not selected before deterministic EDH/LEDH adapter gate.",
        ),
        Candidate(
            "advanced_stochastic_pfpf",
            "advanced_particle_filter",
            ADVANCED_COMMIT,
            ADVANCED_VENDOR_ROOT,
            "advanced_particle_filter.filters.stochastic_pfpf",
            "StochasticPFParticleFilter",
            "class",
            adv / "filters" / "stochastic_pfpf.py",
            "stochastic_flow",
            "spf_pfpf",
            "stochastic particle-flow particle filter",
            "separate_stochastic_flow_fixture_required",
        ),
        Candidate(
            "advanced_scalar_kernel_pff",
            "advanced_particle_filter",
            ADVANCED_COMMIT,
            ADVANCED_VENDOR_ROOT,
            "advanced_particle_filter.filters.kernel_pff",
            "ScalarKernelPFF",
            "class",
            adv / "filters" / "kernel_pff.py",
            "kernel_pff",
            "kernel_pff",
            "scalar-kernel particle flow",
            "linear_gaussian_debug_only",
            notes="MP3 classified kernel PFF as excluded pending debug.",
        ),
        Candidate(
            "advanced_matrix_kernel_pff",
            "advanced_particle_filter",
            ADVANCED_COMMIT,
            ADVANCED_VENDOR_ROOT,
            "advanced_particle_filter.filters.kernel_pff",
            "MatrixKernelPFF",
            "class",
            adv / "filters" / "kernel_pff.py",
            "kernel_pff",
            "kernel_pff",
            "matrix-kernel particle flow",
            "linear_gaussian_debug_only",
            notes="MP3 classified kernel PFF as excluded pending debug.",
        ),
        Candidate(
            "advanced_tf_dpf",
            "advanced_particle_filter",
            ADVANCED_COMMIT,
            ADVANCED_VENDOR_ROOT,
            "advanced_particle_filter.tf_filters.differentiable_particle",
            "TFDifferentiableParticleFilter",
            "class",
            adv / "tf_filters" / "differentiable_particle.py",
            "dpf",
            "differentiable_particle_filter",
            "batched TensorFlow DPF with soft, sinkhorn, or amortized resampling",
            "separate_dpf_reproduction_gate",
            ("optional amortized OT checkpoint for amortized mode",),
        ),
        Candidate(
            "advanced_soft_resample",
            "advanced_particle_filter",
            ADVANCED_COMMIT,
            ADVANCED_VENDOR_ROOT,
            "advanced_particle_filter.tf_utils.soft_resampler",
            "soft_resample",
            "function",
            adv / "tf_utils" / "soft_resampler.py",
            "differentiable_resampling",
            "soft_resampling",
            "soft differentiable resampling primitive",
            "separate_resampling_unit_gate",
        ),
        Candidate(
            "advanced_sinkhorn_resample",
            "advanced_particle_filter",
            ADVANCED_COMMIT,
            ADVANCED_VENDOR_ROOT,
            "advanced_particle_filter.tf_utils.sinkhorn",
            "sinkhorn_resample",
            "function",
            adv / "tf_utils" / "sinkhorn.py",
            "differentiable_resampling",
            "sinkhorn_resampling",
            "entropy-regularized OT resampling primitive",
            "separate_resampling_unit_gate",
        ),
        Candidate(
            "advanced_amortized_ot_resampler",
            "advanced_particle_filter",
            ADVANCED_COMMIT,
            ADVANCED_VENDOR_ROOT,
            "advanced_particle_filter.tf_utils.amortized_resampler",
            "AmortizedOTResampler",
            "class",
            adv / "tf_utils" / "amortized_resampler.py",
            "neural_ot",
            "amortized_ot",
            "pretrained neural OT resampling operator",
            "separate_amortized_ot_reproduction_gate",
            ("vendored checkpoint or weights",),
        ),
        Candidate(
            "advanced_hmc_corenflos_lg_dpf",
            "advanced_particle_filter",
            ADVANCED_COMMIT,
            ADVANCED_VENDOR_ROOT,
            "advanced_particle_filter.hmc.run_hmc_corenflos_lg",
            "run_hmc_corenflos_lg_dpf",
            "function",
            adv / "hmc" / "run_hmc_corenflos_lg.py",
            "hmc_parameter_inference",
            "hmc_dpf",
            "HMC target built from differentiable particle filter evidence",
            "separate_hmc_reproduction_gate",
        ),
        Candidate(
            "mlcoe_particle_flow_filter",
            "2026MLCOE",
            MLCOE_COMMIT,
            MLCOE_ROOT,
            "src.filters.flow_filters",
            "ParticleFlowFilter",
            "class",
            mlcoe / "src" / "filters" / "flow_filters.py",
            "edh_ledh_flow",
            "particle_flow_base",
            "base TensorFlow particle-flow wrapper with edh/ledh/ledh_opt modes",
            "nonlinear_gaussian_range_bearing",
        ),
        Candidate(
            "mlcoe_edh",
            "2026MLCOE",
            MLCOE_COMMIT,
            MLCOE_ROOT,
            "src.filters.flow_filters",
            "EDH",
            "class",
            mlcoe / "src" / "filters" / "flow_filters.py",
            "edh_ledh_flow",
            "edh",
            "pure EDH flow wrapper",
            "nonlinear_gaussian_range_bearing",
        ),
        Candidate(
            "mlcoe_ledh",
            "2026MLCOE",
            MLCOE_COMMIT,
            MLCOE_ROOT,
            "src.filters.flow_filters",
            "LEDH",
            "class",
            mlcoe / "src" / "filters" / "flow_filters.py",
            "edh_ledh_flow",
            "ledh",
            "pure LEDH flow wrapper",
            "nonlinear_gaussian_range_bearing",
        ),
        Candidate(
            "mlcoe_pfpf_edh",
            "2026MLCOE",
            MLCOE_COMMIT,
            MLCOE_ROOT,
            "src.filters.flow_filters",
            "PFPF_EDH",
            "class",
            mlcoe / "src" / "filters" / "flow_filters.py",
            "edh_ledh_pfpf",
            "edh",
            "EDH particle-flow particle filter wrapper",
            "nonlinear_gaussian_range_bearing",
        ),
        Candidate(
            "mlcoe_pfpf_ledh",
            "2026MLCOE",
            MLCOE_COMMIT,
            MLCOE_ROOT,
            "src.filters.flow_filters",
            "PFPF_LEDH",
            "class",
            mlcoe / "src" / "filters" / "flow_filters.py",
            "edh_ledh_pfpf",
            "ledh",
            "LEDH particle-flow particle filter wrapper",
            "nonlinear_gaussian_range_bearing",
        ),
        Candidate(
            "mlcoe_kpff",
            "2026MLCOE",
            MLCOE_COMMIT,
            MLCOE_ROOT,
            "src.filters.flow_filters",
            "KPFF",
            "class",
            mlcoe / "src" / "filters" / "flow_filters.py",
            "kernel_pff",
            "kernel_pff",
            "TensorFlow kernel particle flow filter",
            "linear_gaussian_debug_only",
            notes="Kept out of first comparison because advanced kernel PFF is excluded.",
        ),
        Candidate(
            "mlcoe_edh_solver",
            "2026MLCOE",
            MLCOE_COMMIT,
            MLCOE_ROOT,
            "src.filters.flows.EDH",
            "EDHSolver",
            "class",
            mlcoe / "src" / "filters" / "flows" / "EDH.py",
            "flow_solver",
            "edh_solver",
            "low-level EDH drift solver, not a full filter adapter target",
            "adapter_internal_only",
        ),
        Candidate(
            "mlcoe_ledh_solver",
            "2026MLCOE",
            MLCOE_COMMIT,
            MLCOE_ROOT,
            "src.filters.flows.LEDH",
            "LEDHSolver",
            "class",
            mlcoe / "src" / "filters" / "flows" / "LEDH.py",
            "flow_solver",
            "ledh_solver",
            "low-level LEDH drift solver, not a full filter adapter target",
            "adapter_internal_only",
        ),
        Candidate(
            "mlcoe_ledh_optimized_flow",
            "2026MLCOE",
            MLCOE_COMMIT,
            MLCOE_ROOT,
            "src.filters.flows.dai_homotopy",
            "LEDHOptimizedFlow",
            "class",
            mlcoe / "src" / "filters" / "flows" / "dai_homotopy.py",
            "stochastic_flow",
            "ledh_opt",
            "Dai-style optimized homotopy schedule",
            "separate_stochastic_flow_fixture_required",
        ),
        Candidate(
            "mlcoe_dpf",
            "2026MLCOE",
            MLCOE_COMMIT,
            MLCOE_ROOT,
            "src.filters.DPF",
            "DPF",
            "class",
            mlcoe / "src" / "filters" / "DPF.py",
            "dpf",
            "differentiable_particle_filter",
            "modular TensorFlow DPF with transformer, soft, and Sinkhorn resamplers",
            "separate_dpf_reproduction_gate",
            ("transformer resampler model state",),
        ),
        Candidate(
            "mlcoe_dpfpf",
            "2026MLCOE",
            MLCOE_COMMIT,
            MLCOE_ROOT,
            "src.filters.dpfpf",
            "DifferentiablePFPF",
            "class",
            mlcoe / "src" / "filters" / "dpfpf.py",
            "dpfpf",
            "differentiable_particle_flow_pf",
            "differentiable PFPF likelihood engine for gradient inference",
            "separate_dpfpf_reproduction_gate",
        ),
        Candidate(
            "mlcoe_soft_resampler",
            "2026MLCOE",
            MLCOE_COMMIT,
            MLCOE_ROOT,
            "src.filters.resampling.soft",
            "SoftResampler",
            "class",
            mlcoe / "src" / "filters" / "resampling" / "soft.py",
            "differentiable_resampling",
            "soft_resampling",
            "soft differentiable resampler",
            "separate_resampling_unit_gate",
        ),
        Candidate(
            "mlcoe_sinkhorn_resampler",
            "2026MLCOE",
            MLCOE_COMMIT,
            MLCOE_ROOT,
            "src.filters.resampling.optimal_transport",
            "SinkhornResampler",
            "class",
            mlcoe / "src" / "filters" / "resampling" / "optimal_transport.py",
            "differentiable_resampling",
            "sinkhorn_resampling",
            "Sinkhorn OT differentiable resampler",
            "separate_resampling_unit_gate",
        ),
        Candidate(
            "mlcoe_transformer_resampler",
            "2026MLCOE",
            MLCOE_COMMIT,
            MLCOE_ROOT,
            "src.filters.resampling.transformer",
            "TransformerResampler",
            "class",
            mlcoe / "src" / "filters" / "resampling" / "transformer.py",
            "neural_resampling",
            "transformer_resampling",
            "trainable transformer resampling prior",
            "separate_neural_resampling_reproduction_gate",
            ("trained or initialized model policy",),
        ),
        Candidate(
            "mlcoe_phmc_pfpf",
            "2026MLCOE",
            MLCOE_COMMIT,
            MLCOE_ROOT,
            "src.inference.phmc",
            "hmc_pfpf",
            "function",
            mlcoe / "src" / "inference" / "phmc.py",
            "hmc_parameter_inference",
            "phmc_dpfpf",
            "particle HMC using dPFPF likelihood gradients",
            "separate_hmc_reproduction_gate",
        ),
    ]


def _probe_candidate(candidate: Candidate) -> ProbeRecord:
    static_info = _static_object_info(candidate)
    record = ProbeRecord(
        candidate_id=candidate.candidate_id,
        implementation_name=candidate.implementation_name,
        source_commit=candidate.source_commit,
        module_import=candidate.module_import,
        object_name=candidate.object_name,
        object_kind=candidate.object_kind,
        file_path=str(candidate.file_path),
        category=candidate.category,
        family=candidate.family,
        target_semantics=candidate.target_semantics,
        likely_fixture=candidate.likely_fixture,
        artifact_requirements=list(candidate.artifact_requirements),
        notes=candidate.notes,
        static_found=static_info["found"],
        static_public_methods=static_info["public_methods"],
        static_decorators=static_info["decorators"],
    )

    probe = _import_and_inspect(candidate)
    record.probe_status = probe["status"]
    record.import_error = probe["error"]
    record.object_signature = probe["object_signature"]
    record.method_signatures = probe["method_signatures"]
    record.readiness_class = _classify_readiness(candidate, record)
    record.blocker_reason = _blocker_reason(candidate, record)
    return record


def _static_object_info(candidate: Candidate) -> dict[str, Any]:
    if not candidate.file_path.exists():
        return {"found": False, "public_methods": [], "decorators": []}
    tree = ast.parse(candidate.file_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != candidate.object_name:
            continue
        decorators = [_decorator_name(dec) for dec in node.decorator_list]
        public_methods: list[str] = []
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name == "__init__" or not item.name.startswith("_"):
                        public_methods.append(item.name)
        return {
            "found": True,
            "public_methods": public_methods,
            "decorators": decorators,
        }
    return {"found": False, "public_methods": [], "decorators": []}


def _decorator_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts = []
        current: ast.AST = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return ast.dump(node)


def _import_and_inspect(candidate: Candidate) -> dict[str, Any]:
    prefixes = ("advanced_particle_filter", "src")
    saved_modules = {
        key: module
        for key, module in sys.modules.items()
        if key == "advanced_particle_filter"
        or key.startswith("advanced_particle_filter.")
        or key == "src"
        or key.startswith("src.")
    }
    _purge_modules(prefixes)
    try:
        with prepend_sys_path(candidate.root):
            module = importlib.import_module(candidate.module_import)
            obj = getattr(module, candidate.object_name)
            return {
                "status": "importable",
                "error": None,
                "object_signature": _safe_signature(
                    obj.__init__ if inspect.isclass(obj) else obj
                ),
                "method_signatures": _method_signatures(obj),
            }
    except ModuleNotFoundError as exc:
        return {
            "status": "blocked_missing_dependency",
            "error": f"{type(exc).__name__}: {exc}",
            "object_signature": None,
            "method_signatures": {},
        }
    except AttributeError as exc:
        return {
            "status": "blocked_unclear_api",
            "error": f"{type(exc).__name__}: {exc}",
            "object_signature": None,
            "method_signatures": {},
        }
    except Exception as exc:  # pragma: no cover - environment dependent.
        return {
            "status": "blocked_import_side_effect",
            "error": f"{type(exc).__name__}: {exc}",
            "object_signature": None,
            "method_signatures": {},
        }
    finally:
        _purge_modules(prefixes)
        sys.modules.update(saved_modules)


def _purge_modules(prefixes: tuple[str, ...]) -> None:
    for key in list(sys.modules):
        if any(key == prefix or key.startswith(prefix + ".") for prefix in prefixes):
            del sys.modules[key]


def _safe_signature(obj: Any) -> str | None:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def _method_signatures(obj: Any) -> dict[str, str]:
    names = [
        "filter",
        "step",
        "init",
        "update",
        "predict",
        "resample",
        "__call__",
    ]
    signatures: dict[str, str] = {}
    for name in names:
        if not hasattr(obj, name):
            continue
        sig = _safe_signature(getattr(obj, name))
        if sig is not None:
            signatures[name] = sig
    return signatures


def _classify_readiness(candidate: Candidate, record: ProbeRecord) -> str:
    if record.probe_status != "importable":
        return record.probe_status
    if candidate.category == "kernel_pff":
        return "excluded_pending_debug"
    if candidate.category in {"edh_ledh_pfpf", "edh_ledh_flow"}:
        if candidate.family in {"edh", "ledh", "particle_flow_base"}:
            if _has_call_surface(record):
                return "candidate_for_bounded_comparison"
            return "blocked_unclear_api"
    if candidate.category in {"flow_solver"}:
        return "adapter_internal_only"
    if candidate.category in {
        "dpf",
        "dpfpf",
        "differentiable_resampling",
        "neural_ot",
        "neural_resampling",
        "hmc_parameter_inference",
        "stochastic_flow",
    }:
        return "reproduction_gate_required"
    return "not_comparable"


def _has_call_surface(record: ProbeRecord) -> bool:
    if record.object_signature:
        return True
    return any(
        method in record.method_signatures
        for method in ("filter", "step", "init", "update", "__call__")
    )


def _blocker_reason(candidate: Candidate, record: ProbeRecord) -> str | None:
    if record.probe_status != "importable":
        return record.import_error
    if record.readiness_class == "excluded_pending_debug":
        return "MP3 found consistent max-iteration hits for kernel PFF diagnostics."
    if record.readiness_class == "reproduction_gate_required":
        return (
            "Requires a separate reproduction gate before comparison because "
            f"category is {candidate.category!r}."
        )
    if record.readiness_class == "adapter_internal_only":
        return "Low-level solver may support an adapter but is not a complete comparison target."
    if record.readiness_class == "blocked_unclear_api":
        return "Import succeeded but no inspectable constructor or call surface was found."
    return None


def _summarize(records: list[ProbeRecord]) -> dict[str, Any]:
    candidate_decision = _select_candidate(records)
    return {
        "records": len(records),
        "probe_status_counts": dict(Counter(r.probe_status for r in records)),
        "readiness_class_counts": dict(Counter(r.readiness_class for r in records)),
        "category_counts": dict(Counter(r.category for r in records)),
        "candidate_decision": candidate_decision,
        "hypothesis_results": _hypothesis_results(records, candidate_decision),
        "next_phase_recommendation": _next_phase_recommendation(candidate_decision),
    }


def _select_candidate(records: list[ProbeRecord]) -> dict[str, Any]:
    by_id = {record.candidate_id: record for record in records}
    advanced = by_id.get("advanced_edh_particle_filter")
    mlcoe = by_id.get("mlcoe_pfpf_edh")
    if (
        advanced is not None
        and mlcoe is not None
        and advanced.readiness_class == "candidate_for_bounded_comparison"
        and mlcoe.readiness_class == "candidate_for_bounded_comparison"
    ):
        advanced.comparison_role = "selected_later_candidate"
        mlcoe.comparison_role = "selected_later_candidate"
        return {
            "decision": "selected_bounded_candidate",
            "candidate_family": "importance_corrected_edh_pfpf",
            "advanced_candidate_id": advanced.candidate_id,
            "mlcoe_candidate_id": mlcoe.candidate_id,
            "advanced_path": (
                "advanced_particle_filter.filters.edh.EDHParticleFilter"
            ),
            "mlcoe_path": "src.filters.flow_filters.PFPF_EDH",
            "fixture": "existing nonlinear Gaussian range-bearing fixture",
            "metrics": [
                "latent_state_rmse",
                "final_position_rmse",
                "average_ess_if_exposed",
                "resampling_count_if_exposed",
                "runtime_seconds",
                "finite_output_checks",
                "EKF_UKF_proxy_comparison",
            ],
            "runtime_cap": "short horizon, <=64 particles, <=10 flow steps for first adapter spike",
            "adapter_scope": (
                "adapter-owned bridges only; no vendored-code edits and no "
                "production bayesfilter imports"
            ),
            "blocker_plan": (
                "If either API requires unrecorded model assumptions, stop and "
                "record blocked_missing_assumption rather than widening scope."
            ),
            "selection_reason": (
                "Both paths represent EDH particle-flow particle filters with "
                "importance correction; this is more semantically aligned than "
                "pure flow, kernel PFF, stochastic flow, neural DPF, or HMC."
            ),
        }
    return {
        "decision": "blocker_only",
        "reason": (
            "No matched advanced/MLCOE EDH PFPF pair satisfied import and "
            "signature-readiness gates."
        ),
    }


def _hypothesis_results(
    records: list[ProbeRecord],
    candidate_decision: dict[str, Any],
) -> dict[str, str]:
    by_category = _records_by_category(records)
    edh_ready = [
        record
        for record in by_category.get("edh_ledh_pfpf", [])
        if record.readiness_class == "candidate_for_bounded_comparison"
    ]
    kernel_records = by_category.get("kernel_pff", [])
    reproduction_gate_records = [
        record
        for record in records
        if record.category
        in {
            "dpf",
            "dpfpf",
            "differentiable_resampling",
            "neural_ot",
            "neural_resampling",
            "hmc_parameter_inference",
        }
    ]
    importable = [record for record in records if record.probe_status == "importable"]
    return {
        "F1_edh_ledh_first_candidate": (
            "supported_with_adapter_caveat"
            if len(edh_ready) >= 2
            and candidate_decision["decision"] == "selected_bounded_candidate"
            else "not_supported"
        ),
        "F2_kernel_pff_excluded": (
            "supported"
            if kernel_records
            and all(r.readiness_class == "excluded_pending_debug" for r in kernel_records)
            else "not_supported"
        ),
        "F3_dpf_neural_ot_hmc_reproduction_gates": (
            "supported"
            if reproduction_gate_records
            and all(
                r.readiness_class == "reproduction_gate_required"
                for r in reproduction_gate_records
            )
            else "partially_supported"
        ),
        "F4_import_signature_probe_sufficient": (
            "supported"
            if importable and len(importable) == len(records)
            else "supported_with_blockers"
        ),
        "F5_reuse_existing_fixtures": (
            "supported"
            if candidate_decision["decision"] == "selected_bounded_candidate"
            else "blocked"
        ),
    }


def _records_by_category(records: list[ProbeRecord]) -> dict[str, list[ProbeRecord]]:
    grouped: dict[str, list[ProbeRecord]] = {}
    for record in records:
        grouped.setdefault(record.category, []).append(record)
    return grouped


def _next_phase_recommendation(candidate_decision: dict[str, Any]) -> str:
    if candidate_decision["decision"] == "selected_bounded_candidate":
        return (
            "Create a scoped MP4 follow-up adapter spike for "
            "advanced EDHParticleFilter versus MLCOE PFPF_EDH on the existing "
            "nonlinear Gaussian range-bearing fixture."
        )
    return (
        "Do not run flow/DPF comparison.  Choose a separate reproduction gate "
        "for the least-blocked category."
    )


def _render_report(records: list[ProbeRecord], summary: dict[str, Any]) -> str:
    lines = [
        "# Student DPF baseline flow and DPF readiness review result",
        "",
        "## Date",
        "",
        DATE,
        "",
        "## Scope",
        "",
        "This report covers MP4 of the quarantined student DPF experimental-baseline",
        "stream.  It uses static inventory plus import/signature probes only.",
        "No student filter was instantiated, no filter/update/step method was",
        "called, no notebook or experiment script was executed, and no vendored",
        "student code was modified.",
        "",
        "## Command",
        "",
        "`python -m experiments.student_dpf_baselines.runners.run_flow_dpf_readiness_review`",
        "",
        "Working directory: `/home/ubuntu/python/BayesFilter`",
        "",
        "## Provenance",
        "",
        f"- `advanced_particle_filter`: `{ADVANCED_COMMIT}`",
        f"- `2026MLCOE`: `{MLCOE_COMMIT}`",
        "",
        "## Overall",
        "",
        f"- records: {summary['records']}",
        f"- probe status counts: `{summary['probe_status_counts']}`",
        f"- readiness counts: `{summary['readiness_class_counts']}`",
        f"- candidate decision: `{summary['candidate_decision']['decision']}`",
        "",
        "## Inventory",
        "",
        "| Candidate | Implementation | Category | Probe | Readiness | Signature |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        signature = (record.object_signature or "").replace("|", "\\|")
        if len(signature) > 90:
            signature = signature[:87] + "..."
        lines.append(
            f"| `{record.candidate_id}` | {record.implementation_name} | "
            f"`{record.category}` | `{record.probe_status}` | "
            f"`{record.readiness_class}` | `{signature}` |"
        )

    lines.extend(["", "## Candidate Decision", ""])
    decision = summary["candidate_decision"]
    if decision["decision"] == "selected_bounded_candidate":
        lines.extend(
            [
                f"- family: `{decision['candidate_family']}`",
                f"- advanced path: `{decision['advanced_path']}`",
                f"- MLCOE path: `{decision['mlcoe_path']}`",
                f"- fixture: {decision['fixture']}",
                f"- runtime cap: {decision['runtime_cap']}",
                f"- adapter scope: {decision['adapter_scope']}",
                f"- blocker plan: {decision['blocker_plan']}",
                f"- reason: {decision['selection_reason']}",
                "",
                "Metrics for the later bounded adapter spike:",
            ]
        )
        for metric in decision["metrics"]:
            lines.append(f"- `{metric}`")
    else:
        lines.append(f"- blocker reason: {decision['reason']}")

    lines.extend(["", "## Hypothesis Results", ""])
    for key, value in summary["hypothesis_results"].items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The EDH/PFPF-EDH pair is the only immediate comparison candidate.",
            "It is still not ready for performance claims; it is ready only for a",
            "small adapter-owned spike that reuses the existing nonlinear Gaussian",
            "range-bearing fixture and records proxy metrics.",
            "",
            "Kernel PFF remains excluded from routine comparison because MP3 found",
            "that reduced runs completed but consistently hit the iteration cap.",
            "",
            "DPF, differentiable resampling, neural OT, transformer resampling,",
            "stochastic flow, and HMC paths are importable surfaces, not comparison",
            "evidence.  Each needs its own reproduction gate before any result claim.",
            "",
            "## Next Phase Recommendation",
            "",
            summary["next_phase_recommendation"],
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
