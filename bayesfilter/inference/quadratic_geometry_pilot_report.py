"""Formatting for native geometry pilot diagnostics and exact candidate rows."""

from bayesfilter.inference._exact_incumbent import candidates_from_rows


def geometry_pilot_report(raw, *, rank, requested_direction_count, batched, start_index=1):
    if not rank:
        return raw["q_basis"], {"positive_curvature_count": 0}, ()
    positive = int(raw["positive_count"])
    diagnostics = {"pilot_direction_count": requested_direction_count,
        "finite_positive_curvature_count": positive,
        "curvature_min": float(raw["curvature_min"]) if positive else None,
        "curvature_max": float(raw["curvature_max"]) if positive else None,
        "curvature_source": "central_score_difference_directional_curvature",
        "center_score_norm": float(raw["center_score_norm"]),
        "sketch_eigenvalues": tuple(raw["eigenvalues"].numpy().tolist()),
        "basis_source": "directional_curvature_sketch" if positive else "identity_fallback",
        "evaluation_route": "batched_value_and_score" if batched else "scalar_value_and_score_loop",
        "evaluation_batch_size": raw["values"].shape[0] if batched else 1}
    candidates = candidates_from_rows(raw["positions"], raw["values"], raw["scores"],
        start_index=start_index, source_role="pilot", eligibility=None if batched else raw["valid"])
    return raw["q_basis"], diagnostics, candidates
