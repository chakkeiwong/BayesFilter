"""Read-only AST discovery of Kalman/UKF routes and their local import closure.

This inventories Python iteration (including comprehensions), TensorFlow loops,
compilation declarations and Python/NumPy boundaries. It is not a compiler or
proof of callback purity: dynamic callbacks require traced-graph qualification.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKENS = re.compile(r"kalman|ukf|sigma_point", re.IGNORECASE)
ITERATION = (
    ast.For,
    ast.While,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)
# These files define the generic numerical filter closure and owned model adapters.
GUARDED_FILES = (
    "bayesfilter/linear/kalman_tf.py",
    "bayesfilter/linear/kalman_qr_tf.py",
    "bayesfilter/linear/kalman_qr_derivatives_tf.py",
    "bayesfilter/linear/kalman_second_order_tf.py",
    "bayesfilter/linear/stationary_lgssm_derivatives_tf.py",
    "bayesfilter/linear/dtypes_tf.py",
    "bayesfilter/linear/types_tf.py",
    "bayesfilter/linear/svd_factor_tf.py",
    "bayesfilter/linear/kalman_svd_tf.py",
    "bayesfilter/linear/kalman_svd_derivatives_tf.py",
    "bayesfilter/linear/batched_kalman_svd_derivatives_tf.py",
    "bayesfilter/linear/kalman_covariance_derivatives_tf.py",
    "bayesfilter/linear/correlated_kalman_tf.py",
    "bayesfilter/linear/experimental_batched_kalman_tf.py",
    "bayesfilter/linear/qr_factor_tf.py",
    "bayesfilter/linear/stack_qr_tf.py",
    "bayesfilter/linear/rectangular_factor_tf.py",
    "bayesfilter/linear/block_qr_conditional_tf.py",
    "bayesfilter/linear/lower_rank_downdate_tf.py",
    "bayesfilter/nonlinear/factor_srukf_tf.py",
    "bayesfilter/nonlinear/rectangular_srukf_tf.py",
    "bayesfilter/nonlinear/sigma_points_tf.py",
    "bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py",
    "bayesfilter/nonlinear/srukf_factor_tf.py",
    "bayesfilter/nonlinear/factor_srukf_compat.py",
    "bayesfilter/nonlinear/experimental_batched_svd_sigma_point_tf.py",
    "bayesfilter/nonlinear/ssl_lstm_sgqf_ukf_adapters.py",
    "bayesfilter/testing/direct_factor_srukf_adapters_tf.py",
    "bayesfilter/highdim/actual_sv_srukf_tf.py",
    "bayesfilter/highdim/ukf_scout.py",
    "bayesfilter/linear/compiled_recurrence_tf.py",
    "bayesfilter/nonlinear/fixed_sgqf_compiled_tf.py",
    "bayesfilter/nonlinear/fixed_sgqf_derivatives_tf.py",
    "bayesfilter/nonlinear/fixed_sgqf_structural_adapter_tf.py",
    "bayesfilter/nonlinear/compiled_value_paths.py",
    "bayesfilter/nonlinear/batched_svd_sigma_point_tf.py",
    "bayesfilter/nonlinear/ssl_lstm_precision_experiment_tf.py",
    "bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py",
    "bayesfilter/testing/ksc_ukf_neutra_target_tf.py",
    "bayesfilter/testing/ksc_gaussian_sum_ukf_neutra_target_tf.py",
    "bayesfilter/testing/predator_prey_ukf_neutra_target_tf.py",
    "bayesfilter/testing/structural_ukf_neutra_target_design_tf.py",
    "bayesfilter/testing/lgssm_generic_target_adapter_tf.py",
    "bayesfilter/testing/tf_hmc_readiness.py",
    "bayesfilter/testing/simple_nonlinear_generic_target_adapter_tf.py",
)
# Mixed-purpose modules are guarded at their numerical entry points and callbacks.
GUARDED_SCOPES = {
    "bayesfilter/nonlinear/ssl_lstm_posterior_tf.py": ("SSLLSTMParameterMask.embed",),
    "bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py": (
        "SSLLSTMComplexityPosteriorTarget.full_theta",
    ),
    "bayesfilter/nonlinear/fixed_sgqf_tf.py": (
        "tf_fixed_sgqf_filter",
        "_fixed_sgqf_tensor_step_diagnostics",
    ),
    "bayesfilter/highdim/models.py": (
        "SpatialSIRSSM.transition_mean",
        "ParameterizedZhaoCuiSIRSSM.transition_mean_parameter_jacobian",
        "PredatorPreySSM.transition_mean",
        "PredatorPreySSM.transition_mean_parameter_jacobian",
        "_zhao_cui_sir_austria_transition_mean_xla",
    ),
    "bayesfilter/highdim/filtering.py": ("_compiled_linear_gaussian_moment_history",),
    "bayesfilter/highdim/sv_mixture_cut4.py": (
        "independent_panel_sv_mixture_ukf_filter",
        "independent_panel_sv_mixture_ukf_score",
        "_compiled_panel_mixture_ukf",
        "_panel_transformed_sv_component_structural_model",
        "_panel_transformed_sv_component_ukf_structural_model",
        "_panel_transformed_sv_component_ukf_structural_derivatives",
        "_ukf_component_score_update",
        "actual_transformed_sv_independent_panel_augmented_noise_ukf_filter",
        "_actual_transformed_sv_augmented_noise_ukf_structural_model",
        "_actual_transformed_sv_augmented_noise_ukf_structural_derivatives",
        "_gamma_seed_covariance_derivatives",
        "_affine_observation_offset_theta_derivatives",
        "_collapse_gaussian_components",
        "_collapse_gaussian_components_with_derivatives",
        "_embed_axis_mean_derivatives",
        "_embed_axis_variance_derivatives",
        "_physical_theta_jacobian",
    ),
}
EXCEPTIONS_PATH = ROOT / "scripts/kalman_ukf_runtime_loop_exceptions.json"


def is_guarded(path, function):
    return path in GUARDED_FILES or any(
        function == scope or function.startswith(scope + ".")
        for scope in GUARDED_SCOPES.get(path, ())
    )


def loop_digest(node):
    # Lint exception identity, not execution authorization. A changed loop body
    # must be inspected again; function-name allowlists could hide new numerics.
    # Python 3.13 began omitting empty fields by default. Keep the 3.12 dump
    # representation so unchanged reviewed loops retain their lint identity.
    options = {"show_empty": True} if sys.version_info >= (3, 13) else {}
    payload = ast.dump(node, include_attributes=False, **options)
    return hashlib.sha256(payload.encode()).hexdigest()


def classify(path, function, node, exceptions):
    if is_guarded(path, function):
        exception = exceptions.get((path, function, loop_digest(node)))
        return exception["role"] if exception else "runtime_python_iteration"
    if (
        path.startswith("bayesfilter/filters/")
        or "/references/" in path
        or "/vendor/" in path
        or "_reference" in path
        or path.endswith("_numpy.py")
    ):
        return "independent_reference_or_vendor"
    if path == "bayesfilter/adapters/macrofinance.py":
        return "independent_numpy_adapter_and_host_diagnostics"
    if path.endswith("ledh_pfpf_alg1_ukf_tf.py") or "/ledh_contract_e_" in path:
        return "historical_ledh_or_reference_not_canonical_ukf_admission"
    if "/runners/" in path or path.startswith("scripts/"):
        return "host_experiment_orchestration"
    if path.endswith("ukf_initializer.py"):
        return "offline_tt_initializer_not_filter_recurrence"
    if path.endswith("sv_mixture_cut4.py"):
        if "reference" in function or "kalman_filter" in function:
            return "independent_tiny_reference"
        return "adjacent_sgqf_cut4_tt_comparator_or_host_preparation"
    if path.endswith("models.py"):
        return "model_construction_metadata_or_data_simulation"
    if "/highdim/" in path:
        return "adjacent_tensor_train_or_transport_dependency"
    return "schema_artifact_or_other_dependency_inspected_separately"


def inspect_source(path, source, exceptions=None):
    exceptions = exceptions or {}
    tree = ast.parse(source)
    iterations, functions, boundaries = [], [], []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.stack = []

        def visit_ClassDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node):
            self.stack.append(node.name)
            functions.append(
                {
                    "function": ".".join(self.stack),
                    "line": node.lineno,
                    "decorators": [ast.unparse(d) for d in node.decorator_list],
                }
            )
            self.generic_visit(node)
            self.stack.pop()

        def generic_visit(self, node):
            name = ".".join(self.stack)
            if isinstance(node, ITERATION):
                iterations.append(
                    {
                        "function": name,
                        "line": node.lineno,
                        "kind": type(node).__name__,
                        "expression": ast.unparse(node)[:500],
                        "classification": classify(path, name, node, exceptions),
                        "ast_sha256": loop_digest(node),
                    }
                )
            if isinstance(node, ast.Call):
                call = ast.unparse(node.func)
                if call in {
                    "tf.while_loop",
                    "tf.scan",
                    "tf.map_fn",
                    "tf.vectorized_map",
                    "tf.py_function",
                    "tf.numpy_function",
                    "tf.function",
                    "compiled_tensor_recurrence",
                } or call.endswith(".numpy"):
                    boundaries.append(
                        {
                            "function": name,
                            "line": node.lineno,
                            "call": call,
                            "keywords": {
                                kw.arg: ast.unparse(kw.value)
                                for kw in node.keywords
                                if kw.arg
                            },
                        }
                    )
            super().generic_visit(node)

    Visitor().visit(tree)
    imports = [
        n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
    ]
    imports += [
        alias.name
        for n in ast.walk(tree)
        if isinstance(n, ast.Import)
        for alias in n.names
    ]
    lazy_exports = {}
    if path in {"bayesfilter/linear/__init__.py", "bayesfilter/nonlinear/__init__.py"}:
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "_EXPORT_MODULES"
                for target in node.targets
            ):
                lazy_exports = ast.literal_eval(node.value)
    return {
        "path": path,
        "sha256": hashlib.sha256(source.encode()).hexdigest(),
        "iterations": iterations,
        "functions": functions,
        "numerical_boundaries": boundaries,
        "imports": sorted(set(imports)),
        "lazy_exports": lazy_exports,
    }


def read_exceptions():
    return (
        {
            (row["path"], row["function"], row["ast_sha256"]): row
            for row in json.loads(EXCEPTIONS_PATH.read_text())["exceptions"]
        }
        if EXCEPTIONS_PATH.exists()
        else {}
    )


def audit(root=ROOT, baseline=None):
    exceptions = read_exceptions()
    paths = sorted(
        p
        for base in ("bayesfilter", "experiments/dpf_implementation/tf_tfp", "scripts")
        for p in (root / base).rglob("*.py")
    )
    sources = {str(p.relative_to(root)): p.read_text() for p in paths}
    if baseline:
        for path in list(sources):
            result = subprocess.run(
                ["git", "show", f"{baseline}:{path}"],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                sources[path] = result.stdout
            else:
                del sources[path]
    roots = {
        p
        for p, s in sources.items()
        if TOKENS.search(p)
        or re.search(r"def \w*(?:kalman|ukf)\w*\(", s, re.IGNORECASE)
    }
    roots.update((set(GUARDED_FILES) | set(GUARDED_SCOPES)) & sources.keys())
    roots.update(
        {"bayesfilter/linear/__init__.py", "bayesfilter/nonlinear/__init__.py"}
        & sources.keys()
    )
    selected = set(roots)
    pending = list(roots)
    parsed = {}
    while pending:
        path = pending.pop()
        row = parsed[path] = inspect_source(path, sources[path], exceptions)
        for module in set(row["imports"]) | set(row["lazy_exports"].values()):
            candidate = module.replace(".", "/") + ".py"
            if candidate not in sources:
                candidate = module.replace(".", "/") + "/__init__.py"
            if candidate in sources and candidate not in selected:
                selected.add(candidate)
                pending.append(candidate)
    violations = [
        dict(path=p, **entry)
        for p, row in parsed.items()
        for entry in row["iterations"]
        if entry["classification"] == "runtime_python_iteration"
    ]
    present = {
        (p, entry["function"], entry["ast_sha256"])
        for p, row in parsed.items()
        for entry in row["iterations"]
    }
    stale = (
        [
            {"path": p, "function": f, "ast_sha256": digest}
            for p, f, digest in sorted(exceptions.keys() - present)
        ]
        if baseline is None
        else []
    )
    return {
        "schema": "kalman_ukf_runtime_ast_audit.v2",
        "baseline": baseline,
        "python_files_discovered": len(paths),
        "root_modules": sorted(roots),
        "modules_in_import_closure": len(parsed),
        "guarded_files": list(GUARDED_FILES),
        "guarded_scopes": GUARDED_SCOPES,
        "guard_violations": violations,
        "stale_exceptions": stale,
        "modules": [parsed[p] for p in sorted(parsed)],
        "limits": [
            "Static imports and public lazy exports include unused dependencies and do not resolve dynamic callbacks.",
            "Historical/reference/preparation loops are inventoried, not promoted.",
            "Absence of Python iteration or presence of jit_compile is not XLA execution evidence.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = audit(baseline=args.baseline)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "python_files_discovered",
                    "modules_in_import_closure",
                    "guard_violations",
                    "stale_exceptions",
                )
            },
            indent=2,
        )
    )
    return int(
        args.check and bool(report["guard_violations"] or report["stale_exceptions"])
    )


if __name__ == "__main__":
    raise SystemExit(main())
