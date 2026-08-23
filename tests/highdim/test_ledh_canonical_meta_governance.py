"""Meta-governance gates (added 2026-08-24 after the fidelity audit).

D-class: structural defects Python permits silently.
A-class: referent-coverage registry — every model declares a STATUS for
every gate class; silence is a failure. 'pending' cells are allowed but
must be explicit, so honesty is mechanical, not narrative.
"""

from __future__ import annotations

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]

CANONICAL_MODULES = [
    "bayesfilter/highdim/ledh_canonical_filter_tf.py",
    "bayesfilter/highdim/ledh_canonical_score_tf.py",
    "bayesfilter/highdim/ledh_canonical_score_stages_tf.py",
    "bayesfilter/highdim/ledh_canonical_batch_tf.py",
    "bayesfilter/highdim/ledh_canonical_batch_fused_tf.py",
    "bayesfilter/highdim/ledh_canonical_models_tf.py",
    "bayesfilter/highdim/ledh_canonical_neutra_targets_tf.py",
    "bayesfilter/highdim/ledh_ukf_lifecycle_tf.py",
    "bayesfilter/highdim/ledh_flow_perparticle_tf.py",
    "bayesfilter/highdim/ledh_alg1_contract.py",
]


def test_no_duplicate_toplevel_definitions():
    """The stale-duplicate class: Python silently keeps the LAST of two
    same-named top-level definitions; a fix can be masked by a leftover.
    (This exact bug briefly masked the gen-SV density fix.)"""

    offenders = []
    for relative in CANONICAL_MODULES:
        tree = ast.parse((REPO / relative).read_text(encoding="utf-8"))
        seen: dict[str, int] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in seen:
                    offenders.append(
                        f"{relative}: duplicate top-level '{node.name}' "
                        f"(lines {seen[node.name]} and {node.lineno})"
                    )
                seen[node.name] = node.lineno
    assert not offenders, offenders


# --- A-class: referent-coverage registry -------------------------------
# Gate classes per model. Values are either the covering test function
# name (verified to exist) or an EXPLICIT pending marker with reason.
# Adding a model without a registry row fails the completeness check.

PENDING = "PENDING:"

MODEL_GATE_REGISTRY = {
    "austria_sir": {
        "oracle_score": "test_austria_analytical_score_direction0_matches_oracle",
        "independent_fidelity": "test_austria_dynamics_match_vendored_reference_adapter",
        "invariance_or_reduction": (
            PENDING + " reduction slice (kappa->0 diffusion limit) queued"
        ),
        "statistical_identity": (
            PENDING + " Fisher gate queued with the P6 GPU campaign "
            "(18-dim UKF replication cost)"
        ),
    },
    "predator_prey": {
        "oracle_score": "test_predator_prey_onboarding_score_gate",
        "independent_fidelity": "test_predator_prey_dynamics_match_vendored_reference_adapter",
        "invariance_or_reduction": PENDING + " trivial-dynamics slice queued",
        "statistical_identity": "test_predator_prey_fisher_identity",
    },
    "diagonal_lgssm": {
        "oracle_score": "test_diagonal_lgssm_onboarding_score_gate",
        "independent_fidelity": "test_diagonal_lgssm_observation_matrix_matches_reference",
        "invariance_or_reduction": (
            "test_s1_lgssm_kalman_exactness"  # exact-math referent
        ),
        "statistical_identity": "test_diagonal_lgssm_fisher_identity",
    },
    "ksc_sv": {
        "oracle_score": "test_ksc_sv_onboarding_score_gate",
        "independent_fidelity": "test_ksc_mixture_density_matches_vendored_reference",
        "invariance_or_reduction": "test_ksc_equals_actual_sv_up_to_constant",
        "statistical_identity": "test_ksc_sv_fisher_identity",
    },
    "generalized_sv": {
        "oracle_score": "test_generalized_sv_onboarding_score_gate",
        "independent_fidelity": "test_generalized_sv_densities_match_native_reference",
        "invariance_or_reduction": "test_generalized_sv_reduction_slice_matches_kalman",
        "statistical_identity": "test_generalized_sv_fisher_identity",
    },
}

GATE_TEST_FILES = [
    "tests/highdim/test_ledh_canonical_models.py",
    "tests/highdim/test_ledh_canonical_model_fidelity.py",
    "tests/highdim/test_ledh_canonical_filter.py",
    "tests/highdim/test_ledh_canonical_fisher_identity.py",
]


def _all_test_function_names() -> set[str]:
    names: set[str] = set()
    for relative in GATE_TEST_FILES:
        path = REPO / relative
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                names.add(node.name)
    return names


def test_every_model_declares_every_gate_class():
    available = _all_test_function_names()
    problems = []
    required_classes = (
        "oracle_score",
        "independent_fidelity",
        "invariance_or_reduction",
        "statistical_identity",
    )
    for model, cells in MODEL_GATE_REGISTRY.items():
        for gate_class in required_classes:
            if gate_class not in cells:
                problems.append(f"{model}: MISSING declaration for {gate_class}")
                continue
            value = cells[gate_class]
            if value.startswith(PENDING):
                continue  # explicit, honest pending
            if value not in available:
                problems.append(
                    f"{model}: {gate_class} claims test '{value}' which "
                    "does not exist"
                )
    assert not problems, problems


def test_registry_covers_all_onboarded_models():
    from bayesfilter.highdim import ledh_canonical_models_tf as models

    factories = [
        name for name in dir(models) if name.endswith("_canonical_model")
    ]
    registered_prefixes = set(MODEL_GATE_REGISTRY)
    missing = []
    for factory in factories:
        stem = factory.replace("_canonical_model", "")
        if not any(stem.startswith(p) or p.startswith(stem) for p in registered_prefixes):
            missing.append(factory)
    assert not missing, (
        f"onboarded models without a gate-registry row: {missing}"
    )
