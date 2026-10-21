"""Explicit full-model research consumers; shared tuners own all membership.

Frozen designs are hypotheses, not evidence of an adaptation method or a new
entry-point authority. Posterior and reference checks never select members.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

METHOD_PROFILES = {"fixed_hmc": ("hmc", "fixed"),
    "randomized_hmc": ("hmc", "uniform_duration"),
    "snaper": ("hmc", "uniform_duration"), "mams": ("mams", "uniform_duration")}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(path)


def full_model_binding(*, target, geometry_directory, method, seed, execution,
                       epsilon_domain, repair_factor, max_repairs_per_family, source_paths):
    """Reissue the current target, identical local geometry and declared policy."""
    import tensorflow as tf
    from bayesfilter.inference import PrecomputedMassArtifact, bind_hmc_candidate_set_execution
    from bayesfilter.inference.posterior_adapter import value_score_capability
    from .modern_prepared_nonlinear import prepared_binding

    profile = execution.transition_profile
    if (method not in METHOD_PROFILES or profile is None
            or (profile.family, profile.trajectory) != METHOD_PROFILES[method]):
        raise ValueError("method label disagrees with frozen transition profile")
    base, fixture, metadata = prepared_binding(target, geometry_directory,
        geometry_kind="local", seed=seed, use_xla=execution.use_xla,
        source_paths=(__file__, *source_paths), geometry_source_policy="warm_start_only",
        transition_profile=profile, epsilon_domain=epsilon_domain,
        max_repairs_per_family=max_repairs_per_family)
    horizon, dimension = (100, 5) if target == "NL-STR" else (20, 3 if target == "NL-SIR" else 6)
    if (metadata["horizon"] != horizon or metadata["full_horizon"] != horizon
            or metadata["parameter_dim"] != dimension):
        raise ValueError("full-model consumer requires the declared full horizon and dimension")
    factor = metadata["geometry_factor"]
    mass = PrecomputedMassArtifact(position=metadata["geometry_center"],
        factor=factor, covariance=factor@tf.transpose(factor),
        adapter_signature=fixture.adapter.adapter_signature(), position_role="reference_center",
        covariance_source="diagnostic_local_curvature_proposal")
    binding = bind_hmc_candidate_set_execution(adapter=fixture.adapter,
        initial_position=metadata["start_positions"],
        target_scope=value_score_capability(fixture.adapter).target_scope,
        target_lineage=dict(model=target, data=metadata["data_sha256"],
            geometry=metadata["geometry_sha256"], geometry_source_policy="warm_start_only"),
        config=execution, source_paths=tuple(base._spec["source_closure"]),
        scope_id=f"modern-full-{target.lower()}-{method}", search_id=f"seed-{seed}",
        epsilon_domain=epsilon_domain, repair_factor=repair_factor,
        max_repairs_per_family=max_repairs_per_family, mass_artifact=mass)
    return binding, fixture, metadata


def validate_reference_controls(mcse_multiplier, normalized_tolerance):
    """Reject invalid operational thresholds before numerical work."""
    if any(type(x) not in (float, int) or not math.isfinite(x) or x <= 0
           for x in (mcse_multiplier, normalized_tolerance)):
        raise ValueError("explicit positive finite reference-screen controls required")


def reference_screen(values, reference, *, mcse_multiplier, normalized_tolerance, precision_policy):
    """Operational finite-target screen; no nominal stopped-interval coverage."""
    import tensorflow as tf
    from bayesfilter.inference.hmc_precision import HMCPrecisionPolicy, mean_precision
    validate_reference_controls(mcse_multiplier, normalized_tolerance)
    if (not isinstance(precision_policy, HMCPrecisionPolicy)
            or [p.name for p in precision_policy.targets] != reference["quantities"]
            or any(p.kind != "mean" for p in precision_policy.targets)):
        raise ValueError("reference screen requires matching mean precision targets")
    result = dict(passed=False, status="no_retained_draws", tuning_membership_authority=False,
                  nominal_coverage_established=False, precision_policy=precision_policy.payload())
    if values is None:
        return result
    values = tf.convert_to_tensor(values, tf.float64)
    if values.shape.rank != 3 or values.shape[1] != 4 or values.shape[-1] != len(reference["quantities"]):
        raise ValueError("reference quantities or chain dimensions mismatch")
    if values.shape[0] < 4:
        return result
    if not bool(tf.reduce_all(tf.math.is_finite(values))):
        return dict(result, status="nonfinite_retained_values")
    precision = mean_precision(values, method=precision_policy.method,
        batch_size=precision_policy.batch_size, min_batches=precision_policy.min_batches,
        lugsail_r=precision_policy.lugsail_r, lugsail_c=precision_policy.lugsail_c,
        jit_compile=precision_policy.jit_compile)
    if not bool(tf.reduce_all(precision["valid"]) & tf.reduce_all(precision["mcse"] > 0)):
        return dict(result, status="uncertainty_unavailable")
    sd = tf.constant(reference["sd"], tf.float64)
    sensitivity = tf.constant(reference["uncertainty"], tf.float64)
    expectation = tf.constant(reference["mean"], tf.float64)
    for vector in (sd, sensitivity, expectation):
        tf.debugging.assert_equal(tf.shape(vector), [len(reference["quantities"])])
    tf.debugging.assert_positive(sd)
    tf.debugging.assert_non_negative(sensitivity)
    tf.debugging.assert_all_finite(sd, "invalid reference SD")
    tf.debugging.assert_all_finite(sensitivity, "invalid reference sensitivity")
    tf.debugging.assert_all_finite(expectation, "invalid reference expectation")
    errors = tf.abs(precision["estimate"]-expectation)
    screen = (errors+mcse_multiplier*precision["mcse"]+sensitivity)/sd
    passed = bool(tf.reduce_all(tf.math.is_finite(screen)) & tf.reduce_all(screen <= normalized_tolerance))
    return dict(result, passed=passed,
        status="agreement_screen_passed" if passed else "agreement_not_established",
        estimate=precision["estimate"].numpy().tolist(), mcse=precision["mcse"].numpy().tolist(),
        absolute_error=errors.numpy().tolist(), normalized_error_screen=screen.numpy().tolist(),
        mcse_multiplier=mcse_multiplier, normalized_tolerance=normalized_tolerance)


def assess_full_model_members(*, run, members, output_dir, target, reference,
                              posterior_config_factory, mcse_multiplier,
                              normalized_tolerance, use_xla, check_inputs):
    """Assess every verified identity in creation order, retaining all failures."""
    import tensorflow as tf
    from bayesfilter.inference import run_hmc_posterior
    from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint
    from bayesfilter.runtime.execution_budget import ExecutionBudgetExceeded, require_execution_budget
    from scripts.audit_mams_reference_compatibility import model_quantities, quantity_names
    if set(members) != set(run.result.verified_candidate_ids):
        raise ValueError("posterior member inventory differs from public verified set")
    if reference["target"] != target or reference["quantities"] != quantity_names(target):
        raise ValueError("reference target or quantity chart mismatch")
    validate_reference_controls(mcse_multiplier, normalized_tolerance)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    rows = [dict(candidate_id=c.candidate_id, parent_candidate_id=c.parent_candidate_id,
                 tuning_state=run.result.candidate_states[c.candidate_id],
                 status="posterior_not_started" if c.candidate_id in members else "not_verified")
            for c in run.result.candidates]
    report = dict(status="running", rows=rows,
        verified_candidate_ids=list(run.result.verified_candidate_ids),
        tuning_membership_unchanged=True, method_ranking=None, release_qualified=False)
    def save():
        write_json(directory/"result.json", report)
    save()
    try:
        for index, row in enumerate(rows):
            if row["status"] == "not_verified":
                continue
            require_execution_budget()
            check_inputs()
            member = members[row["candidate_id"]]
            config = posterior_config_factory(member, index)
            policy = config.payload(chain_count=4)
            if bool(policy["jit_compile"]) != use_xla:
                raise ValueError("posterior and bound execution compilation policy differ")
            destination = directory/f"member-{index:03d}"
            destination.mkdir()
            write_json(destination/"policy.json", policy)
            row.update(status="posterior_running", result_file=str(destination/"result.json"))
            save()
            with DurableTensorCheckpoint(destination/"chunks", dict(member_hash=member.member_hash,
                    quantities=reference["quantities"], policy=policy)) as store:
                result = run_hmc_posterior(member=member, config=config,
                    model_transform=lambda q: model_quantities(q, target=target),
                    parameter_names=reference["quantities"], checkpoint_store=store,
                    budget_check=lambda _: require_execution_budget())
            for key in ("private_warmup_raw", "private_retained_raw"):
                if result.get(key) is not None:
                    tf.io.write_file(str(destination/(key.removeprefix("private_")+".tensor")),
                                     tf.io.serialize_tensor(result[key]))
            write_json(destination/"result.json", {k:v for k,v in result.items() if not k.startswith("private_")})
            comparison = reference_screen(result.get("private_retained_raw"), reference,
                mcse_multiplier=mcse_multiplier, normalized_tolerance=normalized_tolerance,
                precision_policy=config.assessment_policy.precision)
            row.update(status="posterior_checks_passed" if result["passed"] else "posterior_incomplete",
                posterior_hard_vetoes=result.get("hard_vetoes", []), reference=comparison,
                delivered=bool(result["passed"] and not result.get("hard_vetoes") and comparison["passed"]))
            save()
        check_inputs()
        report["status"] = "completed"
    except ExecutionBudgetExceeded:
        report["status"] = "resource_stopped"
        for row in rows:
            if row["status"] in ("posterior_running", "posterior_not_started"):
                row["status"] += "_resource_stopped"
    except BaseException as exc:
        report.update(status="repair_required", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["delivered_count"] = sum(row.get("delivered", False) for row in rows)
        save()
    return report
