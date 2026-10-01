"""Host reconstruction of already-completed quadratic refinement records."""

import tensorflow as tf

from bayesfilter.inference.quadratic_rounds_tf import ROLES, STATUS


def paired_quadratic_result(computed, config, dimension, *, jit_compile, trace_count):
    """Copy complete arrays once, then rebuild the original reporting schema.

    All numerical decisions, callback calls, fitted models and accepted factors
    have already completed. Host iteration only formats active record slots.
    """
    from bayesfilter.inference.batched_quadratic_center import (
        BatchedQuadraticCenterResult,
    )

    def host(value):
        return value.numpy().tolist()

    status = STATUS[int(computed["status"])]
    accepted = status == "local_center_candidate"
    history, rounds = tf.nest.map_structure(host, (computed["history"], computed["rounds"]))
    reports = []
    for index, present in enumerate(rounds["model_present"]):
        if not present:
            continue
        report = {key: rounds[key][index] for key in ("anchor", "anchor_value", "radius", "cloud_changed_incumbent")}
        report["model"] = {key: value[index] for key, value in rounds["models"].items()}
        if rounds["centeredness_present"][index]:
            report["centeredness"] = rounds["centeredness"][index]
        if rounds["step_present"][index]:
            report.update({key: rounds[key][index] for key in ("actual_improvement", "ratio", "trust_step_accepted")})
            report["step"] = {key: value[index] for key, value in rounds["steps"].items()}
        reports.append(report)
    details = {key: int(computed[key]) for key in ("physical_rows", "padded_rows", "invalid_rows", "callback_batches")}
    details.update(
        planned_physical_rows=config.planned_rows(dimension), selected_evaluation_index=int(computed["selected_index"]),
        seed=config.seed, pilot_method=config.pilot_method, paired_steps=config.paired_steps,
        full_initializer_xla=False, jit_compile_refinement=jit_compile, jit_compile_trust=jit_compile,
        jit_compile_fit=jit_compile, refinement_trace_count=trace_count,
        fit_calls=int(computed["fit_calls"]), trust_calls=int(computed["trust_calls"]), rounds=reports,
        candidate_batches=[{"role": ROLES[history["roles"][index]], "first_index": history["first_index"][index],
            **{key: history[key][index] for key in ("positions", "values", "scores", "valid")}}
            for index in range(details["callback_batches"])],
    )
    if accepted:
        # All branches are traced even if no trust solve actually executes.
        details.update(fit_trace_count=1, trust_trace_count=1)
    return BatchedQuadraticCenterResult(accepted, status, computed["center"], computed["center_value"], computed["center_score"],
        computed["factor"] if accepted else None, computed["precision"] if accepted else None, details)
