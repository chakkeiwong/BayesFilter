"""Resolve M17 matrix proposals without starting a scientific experiment."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "m16-r1/source-r1"
sys.path.insert(0, str(SOURCE))
from bayesfilter.testing.inference_validation.designs import ScenarioSpec, ValidationDesign, seed_for
from bayesfilter.testing.inference_validation.execution import plan_suite
from bayesfilter.testing.inference_validation.storage import write_json
from bayesfilter.testing.inference_validation.targets import ValidationTarget
from bayesfilter.inference.neutra_artifacts import finalize_dense_iaf_neutra_artifact_payload

PLAN = "docs/plans/bayesfilter-hmc-repair-m17-design-2026-09-21.md"
COUNTS = {"warmup_min_results": 2000, "warmup_check_window_results": 1000,
          "warmup_chunk_results": 500, "warmup_max_results": 10000,
          "retained_min_results": 1000, "retained_chunk_results": 500, "retained_max_results": 10000}


def nonlinear_payload(target):
    # Fixed codec fixture, independently checked in its existing loader tests.
    return finalize_dense_iaf_neutra_artifact_payload({
        "schema": "bayesfilter.neutra.dense_iaf_frozen_transport.v1",
        "transport_id": "m17-fixed-nonlinear-mechanics", "dimension": 2,
        "target_signature": target.adapter_signature(), "log_jacobian_available": True,
        "component_order": ("dense",), "components": ({
            "component_id": "dense", "kind": "dense_autoregressive_iaf", "dim": 2,
            "hidden_layers": (2,), "activation": "tanh", "s_max": 1.,
            "masks_policy": "legacy_degree_masks_v1", "dtype": "float64",
            "weights": (((.5, -.25), (.75, .1)),
                        ((.2, -.4, .3, -.2), (.1, .6, -.5, .7))),
            "biases": ((.05, -.1), (.02, -.03, .04, -.05)),},),
        "training_state_hash": "sha256:m17-fixed-fixture-no-training",
        "nonclaims": ("fixed transform mechanics only", "no training or transport-quality claim")})


def common(ordinary=False):
    result = {"plan_file": PLAN, "member_rule": "first_verified",
              "posterior_members": "selected", "posterior_settings": COUNTS}
    if ordinary:
        result.update(preparation_preset="standard", metric_evidence_policy="finite_window",
            bootstrap_initialization_rounds=20, metric_probe_num_results=16,
            preparation_max_restarts=3, preparation_bound_expansion_steps=1, native_search=True)
    return result


def main():
    for device in ("cpu_reference", "gpu"):
        rows = []
        geometries = [
            ("rotated", "rotated_gaussian", "ordinary", "dispersed", {"condition": 100., "angle": .6}),
            ("centered", "funnel", "ordinary", "dispersed", {"scale": 3.}),
            ("noncentered", "funnel_noncentered", "ordinary", "dispersed", {"scale": 3.}),
            ("cauchy", "cauchy", "ordinary", "dispersed", {}),
            ("mixture-single", "mixture", "prepared", "single_mode", {"separation": 5., "weight": .3}),
            ("mixture-dispersed", "mixture", "prepared", "mode_dispersed", {"separation": 5., "weight": .3})]
        for label, model, route, start, parameters in geometries:
            name = "m17-" + device + "-" + label
            options = common(route == "ordinary")
            if model == "mixture":
                options["global_quantities"] = ["left_mode_probability"]
            rows.append(ValidationDesign(design_id=name, engine="accuracy",
                scenario=ScenarioSpec(model, route, start=start, parameters=parameters),
                device=device, replications=1, draws=500, posterior_cap=10000,
                budget_seconds=800 if device == "gpu" else 600, mcse_tolerance=.05,
                seed=seed_for(2026092197, name)[0], step_size=.5,
                purpose="difficult-geometry integration; caps and global errors retained",
                numerical_provenance=PLAN, options=options))
        for label, model, nonlinear, parameters in (
                ("affine-gaussian", "gaussian", False, {}),
                ("affine-banana", "banana", False, {"bend": .5}),
                ("nonlinear-banana", "banana", True, {"bend": .5}),
                ("nonlinear-dirichlet", "dirichlet", True, {"concentration": [2., 3., 4.]})):
            name = "m17-" + device + "-" + label
            options = common()
            if nonlinear:
                target = ValidationTarget(model, parameters, jit_compile=False)
                options["transport_payload"] = nonlinear_payload(target)
            rows.append(ValidationDesign(design_id=name, engine="accuracy",
                scenario=ScenarioSpec(model, "fixed_transport", parameters=parameters),
                device=device, replications=1, draws=500, posterior_cap=10000,
                budget_seconds=550 if device == "gpu" else 500, mcse_tolerance=.05,
                seed=seed_for(2026092197, name)[0], step_size=.5,
                purpose="frozen-transform public composition; no transport-quality claim",
                numerical_provenance=PLAN, options=options))
        if device == "cpu_reference":
            name = "m17-cpu_reference-student-t"
            rows.append(ValidationDesign(design_id=name, engine="accuracy",
                scenario=ScenarioSpec("student_t", "ordinary", parameters={"df": 5.}),
                device=device, replications=1, draws=500, posterior_cap=10000,
                budget_seconds=600, mcse_tolerance=.05, step_size=.5,
                seed=seed_for(2026092197, name)[0], options=common(True),
                purpose="finite-moment heavy-tail counterpart", numerical_provenance=PLAN))
        suite = {"schema": "bayesfilter.inference_validation_suite.v1",
                 "suite_id": "m17-matrix-" + device, "profile": "master",
                 "profiles": {"master": ["accuracy"]}, "designs": [d.payload() for d in rows]}
        plan = plan_suite(suite)
        write_json(ROOT / ("matrix-" + device + ".json"), suite)
        write_json(ROOT / ("matrix-" + device + "-plan.json"), plan)
        print(device, len(rows), plan["budget_by_device"])


if __name__ == "__main__":
    main()
